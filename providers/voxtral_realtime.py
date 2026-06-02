"""STT provider for vLLM's Voxtral Realtime WebSocket API.

vLLM's protocol differs from the OpenAI Realtime Transcription API in three ways:
- session.update: model is at the top level, not nested inside session.audio
- No server-side VAD: client must send input_audio_buffer.commit to trigger generation
- Response events: transcription.delta / transcription.done (not conversation.item.*)

Audio must be PCM16, 16 kHz, mono, base64-encoded.
"""

import asyncio
import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field

import aiohttp
import numpy as np
from livekit import rtc
from livekit.agents import stt

from providers.base import BaseSttAgent, BaseSttConfig

_SILENCE_THRESHOLD_RMS = float(os.getenv("VOXTRAL_SILENCE_THRESHOLD_RMS", "500"))
_SILENCE_DURATION_S = float(os.getenv("VOXTRAL_SILENCE_DURATION_S", "0.8"))
_MAX_BUFFER_DURATION_S = float(os.getenv("VOXTRAL_MAX_BUFFER_DURATION_S", "12.0"))
_TARGET_SAMPLE_RATE = int(os.getenv("VOXTRAL_TARGET_SAMPLE_RATE", "16000"))
_TRANSCRIPTION_TIMEOUT_S = float(os.getenv("VOXTRAL_TRANSCRIPTION_TIMEOUT_S", "30.0"))


@dataclass
class VoxtralRealtimeConfig(BaseSttConfig):
    api_key: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_REALTIME_API_KEY")
    )
    model: str = field(
        default_factory=lambda: os.getenv(
            "OPENAI_REALTIME_MODEL", "mistralai/Voxtral-Mini-4B-Realtime-2602"
        )
    )
    base_url: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_REALTIME_BASE_URL", None)
    )


voxtral_realtime_config = VoxtralRealtimeConfig()


class VoxtralRealtimeSttAgent(BaseSttAgent):
    def __init__(self, config: VoxtralRealtimeConfig):
        super().__init__(config)
        self._http_session: aiohttp.ClientSession | None = None

    def _get_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None:
            self._http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, connect=15)
            )
        return self._http_session

    def _build_ws_url(self) -> str:
        base = (self.config.base_url or "https://api.openai.com/v1").rstrip("/")
        base = base.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        return f"{base}/realtime?intent=transcription"

    def _create_stt_stream(self, locale: str) -> stt.SpeechStream:
        raise NotImplementedError("VoxtralRealtime uses a custom pipeline")

    def _update_stream_locale(self, user_id: str, locale: str):
        provider = self.participant_settings.get(user_id, {}).get("provider", "voxtral-realtime")
        self.stop_transcription_for_user(user_id)
        self.start_transcription_for_user(user_id, locale, provider)

    def start_transcription_for_user(self, user_id: str, locale: str, provider: str):
        settings = self.participant_settings.setdefault(user_id, {})
        settings["locale"] = locale
        settings["provider"] = provider

        participant = self._find_participant(user_id)
        if not participant:
            logging.error(f"Cannot start transcription, participant {user_id} not found.")
            return

        track = self._find_audio_track(participant)
        if not track:
            logging.warning(
                f"Won't start transcription yet, no audio track found for {user_id}."
            )
            return

        if participant.identity in self.processing_info:
            logging.debug(
                f"Transcription already running for {participant.identity}, ignoring."
            )
            return

        language = self._sanitize_locale(locale)
        task = asyncio.create_task(
            self._run_transcription_pipeline(participant, track, language)
        )
        self.processing_info[participant.identity] = {"task": task}
        logging.info(
            f"Started Voxtral Realtime transcription for {participant.identity} ({locale})."
        )

    async def _cleanup(self):
        await super()._cleanup()
        if self._http_session:
            await self._http_session.close()
            self._http_session = None

    async def _run_transcription_pipeline(
        self,
        participant: rtc.RemoteParticipant,
        track: rtc.Track,
        language: str,
    ):
        ws_url = self._build_ws_url()
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        open_time = time.time()
        self.open_time = open_time
        audio_stream = rtc.AudioStream(track)

        try:
            async with self._get_http_session().ws_connect(ws_url, headers=headers) as ws:
                msg = await asyncio.wait_for(ws.receive(), timeout=10.0)
                if msg.type != aiohttp.WSMsgType.TEXT:
                    logging.error("Voxtral WS: expected text for session.created")
                    return
                data = json.loads(msg.data)
                if data.get("type") != "session.created":
                    logging.error(f"Voxtral WS: unexpected first message: {data}")
                    return
                logging.info(f"Voxtral WS session created for {participant.identity}")

                # vLLM expects model at top level of session.update
                await ws.send_json({"type": "session.update", "model": self.config.model})

                await self._vad_loop(ws, audio_stream, participant, language, open_time)

        except asyncio.CancelledError:
            logging.info(f"Voxtral Realtime transcription for {participant.identity} cancelled.")
        except Exception as e:
            logging.error(
                f"Voxtral Realtime error for {participant.identity}: {e}", exc_info=True
            )
        finally:
            self.processing_info.pop(participant.identity, None)
            await audio_stream.aclose()

    async def _vad_loop(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        audio_stream: rtc.AudioStream,
        participant: rtc.RemoteParticipant,
        language: str,
        open_time: float,
    ):
        speech_buffer: list[rtc.AudioFrame] = []
        buffer_duration = 0.0
        silence_duration = 0.0
        was_speaking = False
        speech_start_time = 0.0

        async def flush_segment(frames: list[rtc.AudioFrame], seg_start: float) -> None:
            if not frames:
                return
            try:
                combined = rtc.combine_audio_frames(frames)
                pcm_bytes = _to_pcm16_16k(combined)

                chunk_size = _TARGET_SAMPLE_RATE // 10 * 2  # 100 ms of int16
                for i in range(0, len(pcm_bytes), chunk_size):
                    await ws.send_json({
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm_bytes[i : i + chunk_size]).decode(),
                    })

                # Start generation, then signal end of this segment
                await ws.send_json({"type": "input_audio_buffer.commit"})
                await ws.send_json({"type": "input_audio_buffer.commit", "final": True})

                text = await _collect_transcription(ws)
                if text:
                    seg_end = time.time() - open_time
                    event = stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=[
                            stt.SpeechData(
                                text=text,
                                language=language,
                                start_time=seg_start,
                                end_time=seg_end,
                            )
                        ],
                    )
                    self.emit(
                        "final_transcript",
                        participant=participant,
                        event=event,
                        open_time=open_time,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logging.error(
                    f"Voxtral: error transcribing segment for {participant.identity}: {e}"
                )

        async for audio_event in audio_stream:
            frame = audio_event.frame
            samples = np.frombuffer(frame.data, dtype=np.int16)
            rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
            is_speaking = rms > _SILENCE_THRESHOLD_RMS
            frame_duration = frame.samples_per_channel / frame.sample_rate

            if is_speaking:
                if not was_speaking:
                    speech_start_time = time.time() - open_time
                speech_buffer.append(frame)
                buffer_duration += frame_duration
                silence_duration = 0.0
                was_speaking = True

                if buffer_duration >= _MAX_BUFFER_DURATION_S:
                    await flush_segment(speech_buffer[:], speech_start_time)
                    speech_buffer.clear()
                    buffer_duration = 0.0
                    speech_start_time = time.time() - open_time
            elif was_speaking:
                speech_buffer.append(frame)
                buffer_duration += frame_duration
                silence_duration += frame_duration

                if (
                    silence_duration >= _SILENCE_DURATION_S
                    or buffer_duration >= _MAX_BUFFER_DURATION_S
                ):
                    await flush_segment(speech_buffer[:], speech_start_time)
                    speech_buffer.clear()
                    buffer_duration = 0.0
                    silence_duration = 0.0
                    was_speaking = False

        await flush_segment(speech_buffer[:], speech_start_time)


def _to_pcm16_16k(frame: rtc.AudioFrame) -> bytes:
    """Resample an AudioFrame to 16 kHz mono PCM16."""
    samples = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32)

    if frame.num_channels > 1:
        samples = samples.reshape(-1, frame.num_channels).mean(axis=1)

    if frame.sample_rate != _TARGET_SAMPLE_RATE:
        n_orig = len(samples)
        n_target = int(round(n_orig * _TARGET_SAMPLE_RATE / frame.sample_rate))
        samples = np.interp(
            np.linspace(0, n_orig - 1, n_target),
            np.arange(n_orig),
            samples,
        )

    return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()


async def _collect_transcription(
    ws: aiohttp.ClientWebSocketResponse,
    timeout: float = _TRANSCRIPTION_TIMEOUT_S,
) -> str:
    """Read WebSocket messages until transcription.done and return the text."""
    text = ""
    try:
        async def _read() -> str:
            nonlocal text
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    logging.warning("Voxtral WS closed while collecting transcription")
                    return text
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                data = json.loads(msg.data)
                msg_type = data.get("type")
                if msg_type == "transcription.delta":
                    text += data.get("delta", "")
                elif msg_type == "transcription.done":
                    return data.get("text", text).strip()
                elif msg_type == "error":
                    logging.error(f"Voxtral WS error event: {data}")
                    return text

        return await asyncio.wait_for(_read(), timeout=timeout)
    except asyncio.TimeoutError:
        logging.warning("Voxtral: transcription timed out")
        return text
