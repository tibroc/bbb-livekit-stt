import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import List

from livekit.agents import stt
from livekit.plugins.mistralai import STT as MistralSTT

from config import _get_bool_env, _get_float_env, _get_json_env
from providers.base import BaseSttAgent, BaseSttConfig

# vLLM serves the Voxtral realtime WebSocket at /v1/realtime, while the
# official Mistral SDK hardcodes /v1/audio/transcriptions/realtime.
_VLLM_REALTIME_PATH = "/v1/realtime"


@dataclass
class MistralConfig(BaseSttConfig):
    api_key: str | None = field(default_factory=lambda: os.getenv("MISTRAL_API_KEY"))
    model: str = field(
        default_factory=lambda: os.getenv("MISTRAL_MODEL", "voxtral-mini-latest")
    )
    language: str | None = field(
        default_factory=lambda: os.getenv("MISTRAL_LANGUAGE", None)
    )
    realtime: bool = field(
        default_factory=lambda: _get_bool_env("MISTRAL_REALTIME", False)
    )
    streaming_delay_ms: int | None = field(
        default_factory=lambda: (
            int(os.getenv("MISTRAL_STREAMING_DELAY"))
            if os.getenv("MISTRAL_STREAMING_DELAY")
            else None
        )
    )
    vad_enabled: bool = field(
        default_factory=lambda: _get_bool_env("MISTRAL_VAD_ENABLED", True)
    )
    vad_aggressiveness: int = field(
        default_factory=lambda: int(os.getenv("MISTRAL_VAD_AGGRESSIVENESS", "2"))
    )
    context_bias: List[str] | None = field(
        default_factory=lambda: _get_json_env("MISTRAL_CONTEXT_BIAS")
    )
    interim_results: bool | None = field(
        default_factory=lambda: _get_bool_env("MISTRAL_INTERIM_RESULTS", None)
    )
    min_confidence_interim: float = field(
        default_factory=lambda: _get_float_env("MISTRAL_MIN_CONFIDENCE_INTERIM", 0.0)
    )
    min_confidence_final: float = field(
        default_factory=lambda: _get_float_env("MISTRAL_MIN_CONFIDENCE_FINAL", 0.0)
    )
    custom_endpoint: str | None = field(
        default_factory=lambda: os.getenv("MISTRAL_CUSTOM_ENDPOINT")
    )

    def to_stt_kwargs(self) -> dict:
        """Build kwargs for the MistralAI STT plugin constructor.

        Only includes parameters that are explicitly set so the plugin
        can apply its own defaults for omitted values.
        """
        data = {}

        if self.api_key is not None:
            data["api_key"] = self.api_key
        if self.model:
            data["model"] = self.model
        if self.language is not None:
            data["language"] = self.language
        if self.context_bias is not None:
            data["context_bias"] = self.context_bias
        if self.streaming_delay_ms is not None:
            data["target_streaming_delay_ms"] = self.streaming_delay_ms

        if self.realtime and self.vad_enabled:
            from livekit.plugins.silero import VAD as SileroVAD

            data["vad"] = SileroVAD.load(
                min_speech_duration=0.1,
                activation_threshold=0.25 + (self.vad_aggressiveness * 0.1),
            )

        return data


mistral_config = MistralConfig()


async def _connect_vllm_ws(
    server_url: str,
    model: str,
    *,
    api_key: str | None = None,
    http_headers: dict[str, str] | None = None,
    target_streaming_delay_ms: int | None = None,
    timeout: float = 10.0,
):
    """Open a WebSocket to a vLLM/voxtral realtime endpoint and perform handshake.

    The official Mistral SDK hardcodes the WebSocket path as
    ``/v1/audio/transcriptions/realtime`` and expects the server to send a
    ``session.created`` message immediately after the TCP handshake.  vLLM
    (and compatible servers) expose the endpoint at ``/v1/realtime`` and may
    require the client to send an initial ``session.update`` message before
    the server responds with ``session.created``.

    This function bypasses the SDK's ``RealtimeTranscription.connect()``
    entirely to avoid fragile subclass overrides and directly:
      1. Opens the WebSocket to ``<server_url>/v1/realtime?model=<model>``
      2. Sends a ``session.update`` message with the model field to trigger the handshake
      3. Waits for the ``session.created`` response
      4. Returns a ``RealtimeConnection`` the LiveKit plugin can use
    """
    from mistralai.extra.realtime.connection import RealtimeConnection
    from mistralai.extra.realtime.transcription import _recv_handshake
    from websockets.asyncio.client import connect

    # Build the WebSocket URL
    base = server_url.rstrip("/")
    ws_url = f"{base}{_VLLM_REALTIME_PATH}?model={model}"

    # Convert http(s) to ws(s)
    ws_url = ws_url.replace("https://", "wss://").replace("http://", "ws://")

    headers: dict[str, str] = {}
    if http_headers:
        headers.update(http_headers)
    if api_key and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {api_key}"

    logging.debug("Voxtral WebSocket URL: %s (headers: %s)", ws_url, list(headers.keys()))

    websocket = await connect(
        ws_url,
        additional_headers=headers,
        open_timeout=timeout,
    )

    try:
        # vLLM/voxtral servers require the client to send an initial
        # session.update before they reply with session.created.
        session_update: dict = {
            "type": "session.update",
            "model": model,
            "session": {},
        }
        if target_streaming_delay_ms is not None:
            session_update["session"]["target_streaming_delay_ms"] = (
                target_streaming_delay_ms
            )
        await websocket.send(json.dumps(session_update))

        logging.debug("Sent session.update, waiting for session.created …")

        # Wait for the server to reply with session.created.
        timeout_ms = int(timeout * 1000)
        session, initial_events = await _recv_handshake(
            websocket, timeout_ms=timeout_ms
        )

        return RealtimeConnection(
            websocket=websocket,
            session=session,
            initial_events=initial_events,
        )
    except Exception:
        await websocket.close()
        raise


class MistralSttAgent(BaseSttAgent):
    def __init__(self, config: MistralConfig):
        super().__init__(config)

        stt_kwargs = config.to_stt_kwargs()

        if config.custom_endpoint:
            from mistralai.client import Mistral

            custom_client = Mistral(
                api_key=config.api_key,
                server_url=config.custom_endpoint,
            )
            # api_key is already embedded in the custom client; remove it
            # from stt_kwargs to avoid passing it twice.
            stt_kwargs.pop("api_key", None)
            stt_kwargs["client"] = custom_client

        self.stt = MistralSTT(**stt_kwargs)

        # For custom (vLLM) endpoints the WebSocket path differs from the
        # official Mistral API.  Replace the connection-pool callback on this
        # *instance* so only this agent is affected.
        if config.custom_endpoint and hasattr(self.stt, "_pool"):
            stt_instance = self.stt  # capture for the closure
            endpoint = config.custom_endpoint
            api_key = config.api_key

            async def _connect_ws_vllm(timeout: float):
                # Extract any headers the Mistral client may have set.
                http_headers: dict[str, str] | None = None
                cfg = custom_client.sdk_configuration
                client_headers = getattr(
                    cfg.async_client, "headers", None
                ) or getattr(cfg.client, "headers", None)
                if client_headers:
                    http_headers = dict(client_headers)

                return await _connect_vllm_ws(
                    server_url=endpoint,
                    model=stt_instance._opts.model,
                    api_key=api_key,
                    http_headers=http_headers,
                    target_streaming_delay_ms=stt_instance._opts.target_streaming_delay_ms,
                    timeout=timeout,
                )

            self.stt._pool._connect_cb = _connect_ws_vllm

    def _create_stt_stream(self, locale: str) -> stt.SpeechStream:
        return self.stt.stream(language=locale)

    def _update_stream_locale(self, user_id: str, locale: str):
        sanitized_locale = self._sanitize_locale(locale)
        self.stt.update_options(language=sanitized_locale)

    def _should_emit(self, event: stt.SpeechEvent) -> bool:
        if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
            min_confidence = self.config.min_confidence_final
        elif event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
            min_confidence = self.config.min_confidence_interim
        else:
            return True

        for alt in event.alternatives:
            if alt.confidence < min_confidence:
                logging.debug(
                    f"Discarding transcript: low confidence "
                    f"({alt.confidence} < {min_confidence})."
                )
                return False

        return True
