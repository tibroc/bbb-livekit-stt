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

import metrics
from providers.base import BaseSttAgent, BaseSttConfig

_SILENCE_THRESHOLD_RMS = float(os.getenv("VOXTRAL_SILENCE_THRESHOLD_RMS", "500"))
_SILENCE_DURATION_S = float(os.getenv("VOXTRAL_SILENCE_DURATION_S", "0.6"))
_MAX_BUFFER_DURATION_S = float(os.getenv("VOXTRAL_MAX_BUFFER_DURATION_S", "8.0"))
_TARGET_SAMPLE_RATE = int(os.getenv("VOXTRAL_TARGET_SAMPLE_RATE", "16000"))
_TRANSCRIPTION_TIMEOUT_S = float(os.getenv("VOXTRAL_TRANSCRIPTION_TIMEOUT_S", "10.0"))
# Commit audio to the server every N seconds during continuous speech so text
# appears while the speaker is still talking.  The commit and audio streaming
# run concurrently so the audio loop is never blocked.  Set to 0 to disable.
_PROGRESSIVE_FLUSH_INTERVAL_S = float(
    os.getenv("VOXTRAL_PROGRESSIVE_FLUSH_INTERVAL_S", "2.5")
)


@dataclass
class VoxtralRealtimeConfig(BaseSttConfig):
    api_key: str | None = field(default_factory=lambda: os.getenv("VOXTRAL_API_KEY"))
    model: str = field(
        default_factory=lambda: os.getenv(
            "VOXTRAL_MODEL", "mistralai/Voxtral-Mini-4B-Realtime-2602"
        )
    )
    base_url: str | None = field(
        default_factory=lambda: os.getenv("VOXTRAL_BASE_URL", None)
    )
    interim_results: bool = field(
        default_factory=lambda: (
            os.getenv("VOXTRAL_INTERIM_RESULTS", "true").lower() != "false"
        )
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
        provider = self.participant_settings.get(user_id, {}).get(
            "provider", "voxtral-realtime"
        )
        self.stop_transcription_for_user(user_id)
        self.start_transcription_for_user(user_id, locale, provider)

    def start_transcription_for_user(self, user_id: str, locale: str, provider: str):
        settings = self.participant_settings.setdefault(user_id, {})
        settings["locale"] = locale
        settings["provider"] = provider

        participant = self._find_participant(user_id)
        if not participant:
            logging.error(
                f"Cannot start transcription, participant {user_id} not found."
            )
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
        retry_delay = 1.0
        metrics.ACTIVE_SESSIONS.inc()

        try:
            while True:
                session_start = 0.0
                audio_stream = rtc.AudioStream(track)
                try:
                    async with self._get_http_session().ws_connect(
                        ws_url, headers=headers
                    ) as ws:
                        msg = await asyncio.wait_for(ws.receive(), timeout=10.0)
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            logging.error(
                                "Voxtral WS: expected text for session.created"
                            )
                            return
                        data = json.loads(msg.data)
                        if data.get("type") != "session.created":
                            logging.error(
                                f"Voxtral WS: unexpected first message: {data}"
                            )
                            return
                        logging.info(
                            f"Voxtral WS session created for {participant.identity}"
                        )
                        session_start = time.monotonic()

                        # vLLM expects model at top level of session.update
                        await ws.send_json(
                            {"type": "session.update", "model": self.config.model}
                        )

                        await self._vad_loop(
                            ws, audio_stream, participant, language, open_time
                        )
                        return  # clean exit — audio stream finished normally

                except asyncio.CancelledError:
                    raise
                except aiohttp.ClientError as e:
                    metrics.RECONNECTS_TOTAL.inc()
                    logging.warning(
                        f"Voxtral WS connection lost for {participant.identity} "
                        f"({type(e).__name__}: {e}), reconnecting in {retry_delay:.0f}s"
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 30.0)
                except Exception as e:
                    logging.error(
                        f"Voxtral Realtime error for {participant.identity}: {e}",
                        exc_info=True,
                    )
                    return
                finally:
                    if session_start:
                        metrics.SESSION_DURATION.observe(
                            time.monotonic() - session_start
                        )
                    await audio_stream.aclose()

        except asyncio.CancelledError:
            logging.info(
                f"Voxtral Realtime transcription for {participant.identity} cancelled."
            )
        finally:
            metrics.ACTIVE_SESSIONS.dec()
            self.processing_info.pop(participant.identity, None)

    async def _vad_loop(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        audio_stream: rtc.AudioStream,
        participant: rtc.RemoteParticipant,
        language: str,
        open_time: float,
    ):
        chunk_size = _TARGET_SAMPLE_RATE // 10 * 2  # 100 ms of int16
        send_buffer_bytes = b""
        buffer_duration = 0.0
        silence_duration = 0.0
        was_speaking = False
        # flush_start_time is reset to the current time after every commit so that
        # each chunk gets its own start_time → distinct transcriptId in BBB.
        # Progressive and tail chunks are therefore independent caption lines rather
        # than one overwriting the other.
        flush_start_time = 0.0
        # Tracks the in-flight background transcription task for progressive flushes.
        # Only one may be outstanding at a time to avoid concurrent WS reads.
        prog_task: asyncio.Task | None = None
        # Set to True by _collect_transcript the moment it reads transcription.done
        # from the WS buffer.  Checked after prog_task cancellation to decide
        # whether the final flush must skip one done event in the stream.
        prog_done_consumed = False

        async def _collect_transcript(
            seg_start: float,
            audio_dur: float,
            is_final: bool,
            skip_done_count: int = 0,
        ) -> None:
            """Read transcription events from the WS and emit them.

            Called as a background task for progressive flushes so audio streaming
            is never blocked.  Called with await for the final silence flush.
            Emits INTERIM events for progressive chunks, FINAL for the last one.

            skip_done_count: discard this many transcription.done events (and all
            deltas preceding them) before treating the next one as ours.  Set to 1
            when a progressive commit was cancelled and the final commit was sent
            immediately after — the server responds to both in order, so we skip
            the leftover progressive response and take the final one.
            """
            nonlocal prog_done_consumed
            commit_time = time.monotonic()
            text = ""
            delta_count = 0
            last_delta_time = 0.0
            got_done = False
            skip_remaining = skip_done_count
            try:
                async for msg in _stream_transcription(ws, _TRANSCRIPTION_TIMEOUT_S):
                    msg_type = msg.get("type")
                    if msg_type == "transcription.delta":
                        if skip_remaining > 0:
                            continue
                        now = time.monotonic()
                        if delta_count == 0:
                            metrics.COMMIT_TO_FIRST_DELTA.observe(now - commit_time)
                        else:
                            metrics.DELTA_INTERVAL.observe(now - last_delta_time)
                        last_delta_time = now
                        delta_count += 1

                        text += msg.get("delta", "")
                        if text and self.config.interim_results:
                            self.emit(
                                "interim_transcript",
                                participant=participant,
                                event=stt.SpeechEvent(
                                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                                    alternatives=[
                                        stt.SpeechData(
                                            text=text,
                                            language=language,
                                            start_time=seg_start,
                                            end_time=time.time() - open_time,
                                        )
                                    ],
                                ),
                                open_time=open_time,
                            )
                    elif msg_type == "transcription.done":
                        if skip_remaining > 0:
                            skip_remaining -= 1
                            text = ""
                            continue
                        prog_done_consumed = True
                        got_done = True
                        text = msg.get("text", text).strip()
                        break
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logging.error(
                    f"Voxtral: error transcribing segment for {participant.identity}: {e}"
                )
                metrics.SEGMENTS_TOTAL.labels(outcome="error").inc()
                return

            metrics.COMMIT_TO_DONE.observe(time.monotonic() - commit_time)
            metrics.DELTAS_PER_SEGMENT.observe(delta_count)
            metrics.SEGMENT_AUDIO_DURATION.observe(audio_dur)

            if text:
                metrics.SEGMENTS_TOTAL.labels(outcome="success").inc()
                # Every completed chunk is emitted as FINAL so that each gets its
                # own committed caption in BBB.  Progressive and tail chunks have
                # distinct flush_start_time values → distinct transcriptIds → they
                # appear as separate caption lines rather than one replacing the other.
                self.emit(
                    "final_transcript",
                    participant=participant,
                    event=stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=[
                            stt.SpeechData(
                                text=text,
                                language=language,
                                start_time=seg_start,
                                end_time=time.time() - open_time,
                            )
                        ],
                    ),
                    open_time=open_time,
                )
            elif not got_done:
                metrics.SEGMENTS_TOTAL.labels(outcome="timeout").inc()
            else:
                metrics.SEGMENTS_TOTAL.labels(outcome="empty").inc()

        async def _send_commit(tail_bytes: bytes, final: bool) -> None:
            """Send tail audio + commit message(s). Writes only — no WS reads."""
            if tail_bytes:
                await ws.send_json(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(tail_bytes).decode(),
                    }
                )
            await ws.send_json({"type": "input_audio_buffer.commit"})
            if final:
                await ws.send_json({"type": "input_audio_buffer.commit", "final": True})

        async def _cancel_prog_task() -> int:
            """Cancel the in-flight progressive task and return skip_done_count.

            Awaits the task after cancelling so it is fully stopped before the
            caller touches ws.receive() again — aiohttp forbids concurrent reads.
            The wait is near-instant: CancelledError propagates on the next event
            loop cycle when the task is blocked inside asyncio.wait_for(ws.receive()).

            Returns 1 if transcription.done for the progressive commit has NOT yet
            been read from the WS buffer (final flush must skip past it).
            Returns 0 if the buffer is already clean.
            """
            nonlocal prog_task, prog_done_consumed
            if prog_task and not prog_task.done():
                prog_task.cancel()
                try:
                    await prog_task
                except (asyncio.CancelledError, Exception):
                    pass
                prog_task = None
                # prog_done_consumed is set synchronously the moment _collect_transcript
                # reads transcription.done — no await between that set and here, so
                # the value is authoritative.
                skip = 0 if prog_done_consumed else 1
                prog_done_consumed = False
                return skip
            prog_done_consumed = False
            return 0

        async for audio_event in audio_stream:
            frame = audio_event.frame
            samples = np.frombuffer(frame.data, dtype=np.int16)
            rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
            is_speaking = rms > _SILENCE_THRESHOLD_RMS
            frame_duration = frame.samples_per_channel / frame.sample_rate

            if is_speaking:
                if not was_speaking:
                    flush_start_time = time.time() - open_time
                was_speaking = True
                silence_duration = 0.0

                send_buffer_bytes += _to_pcm16_16k(frame)
                buffer_duration += frame_duration

                while len(send_buffer_bytes) >= chunk_size:
                    await ws.send_json(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(
                                send_buffer_bytes[:chunk_size]
                            ).decode(),
                        }
                    )
                    send_buffer_bytes = send_buffer_bytes[chunk_size:]

                if (
                    _PROGRESSIVE_FLUSH_INTERVAL_S > 0
                    and buffer_duration >= _PROGRESSIVE_FLUSH_INTERVAL_S
                    and (prog_task is None or prog_task.done())
                ):
                    prog_done_consumed = False
                    await _send_commit(send_buffer_bytes, final=False)
                    prog_task = asyncio.create_task(
                        _collect_transcript(
                            flush_start_time, buffer_duration, is_final=False
                        )
                    )
                    send_buffer_bytes = b""
                    buffer_duration = 0.0
                    flush_start_time = time.time() - open_time
                elif buffer_duration >= _MAX_BUFFER_DURATION_S:
                    skip = await _cancel_prog_task()
                    prog_done_consumed = False
                    await _send_commit(send_buffer_bytes, final=False)
                    prog_task = asyncio.create_task(
                        _collect_transcript(
                            flush_start_time,
                            buffer_duration,
                            is_final=False,
                            skip_done_count=skip,
                        )
                    )
                    send_buffer_bytes = b""
                    buffer_duration = 0.0
                    flush_start_time = time.time() - open_time

            elif was_speaking:
                send_buffer_bytes += _to_pcm16_16k(frame)
                buffer_duration += frame_duration
                silence_duration += frame_duration

                while len(send_buffer_bytes) >= chunk_size:
                    await ws.send_json(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(
                                send_buffer_bytes[:chunk_size]
                            ).decode(),
                        }
                    )
                    send_buffer_bytes = send_buffer_bytes[chunk_size:]

                if (
                    silence_duration >= _SILENCE_DURATION_S
                    or buffer_duration >= _MAX_BUFFER_DURATION_S
                ):
                    skip = await _cancel_prog_task()
                    await _send_commit(send_buffer_bytes, final=True)
                    await _collect_transcript(
                        flush_start_time,
                        buffer_duration,
                        is_final=True,
                        skip_done_count=skip,
                    )
                    send_buffer_bytes = b""
                    buffer_duration = 0.0
                    silence_duration = 0.0
                    was_speaking = False
                    flush_start_time = 0.0
                    prog_task = None

        if was_speaking:
            skip = await _cancel_prog_task()
            await _send_commit(send_buffer_bytes, final=True)
            await _collect_transcript(
                flush_start_time,
                buffer_duration,
                is_final=True,
                skip_done_count=skip,
            )


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


async def _stream_transcription(
    ws: aiohttp.ClientWebSocketResponse,
    timeout: float = _TRANSCRIPTION_TIMEOUT_S,
):
    """Yield parsed message dicts from the WebSocket until transcription.done or timeout."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logging.warning("Voxtral: transcription timed out")
            return
        try:
            msg = await asyncio.wait_for(ws.receive(), timeout=remaining)
        except asyncio.TimeoutError:
            logging.warning("Voxtral: transcription timed out")
            return
        if msg.type in (
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSING,
        ):
            logging.warning("Voxtral WS closed while collecting transcription")
            return
        if msg.type != aiohttp.WSMsgType.TEXT:
            continue
        data = json.loads(msg.data)
        msg_type = data.get("type")
        if msg_type == "transcription.delta":
            yield data
        elif msg_type == "transcription.done":
            yield data
            return
        elif msg_type == "error":
            logging.error(f"Voxtral WS error event: {data}")
            return
