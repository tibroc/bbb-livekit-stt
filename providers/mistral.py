import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import List
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

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


def _make_vllm_realtime_transcription(sdk_configuration):
    """Create a RealtimeTranscription that connects to vLLM's /v1/realtime.

    The official Mistral SDK hardcodes the WebSocket path as
    ``/v1/audio/transcriptions/realtime``.  vLLM (and compatible servers)
    expose the endpoint at ``/v1/realtime`` instead.  This helper returns a
    ``RealtimeTranscription`` subclass whose ``_build_url`` uses the correct
    path and sends an initial session.update message to trigger the handshake.
    """
    from mistralai.client import utils as mistral_utils
    from mistralai.client.utils import generate_url
    from mistralai.extra.realtime import RealtimeTranscription
    from mistralai.extra.realtime.connection import RealtimeConnection
    from mistralai.extra.realtime.exceptions import RealtimeTranscriptionException
    from mistralai.extra.realtime.transcription import _recv_handshake, _extract_error_message
    from websockets.asyncio.client import connect
    from mistralai.client.models import AudioFormat
    from mistralai.client.utils import get_security, get_security_from_env

    class _VLLMRealtimeTranscription(RealtimeTranscription):
        def _build_url(self, model, *, server_url, query_params):
            if server_url is not None:
                base_url = mistral_utils.remove_suffix(server_url, "/")
            else:
                base_url, _ = self._sdk_config.get_server_details()

            url = generate_url(base_url, _VLLM_REALTIME_PATH, None)

            parsed = urlparse(url)
            merged = dict(parse_qsl(parsed.query, keep_blank_values=True))
            merged["model"] = model
            merged.update(dict(query_params))
            result_url = urlunparse(parsed._replace(query=urlencode(merged)))

            # Log the WebSocket URL for debugging
            ws_url = result_url.replace("https://", "wss://").replace("http://", "ws://")
            logging.debug(f"Voxtral WebSocket URL: {ws_url}")

            return result_url

        async def connect(
            self,
            model: str,
            audio_format=None,
            target_streaming_delay_ms=None,
            server_url=None,
            timeout_ms=None,
            http_headers=None,
        ):
            """Override connect to send session.update before handshake for vLLM/voxtral servers.
            
            vLLM servers require the client to send an initial session.update message
            before sending the session.created handshake response.
            """
            if timeout_ms is None:
                timeout_ms = self._sdk_config.timeout_ms

            security = self._sdk_config.security
            if security is not None and callable(security):
                security = security()

            resolved_security = get_security_from_env(security, None)
            
            headers: dict[str, str] = {}
            query_params: dict[str, str] = {}

            if resolved_security is not None:
                security_headers, security_query = get_security(resolved_security)
                headers |= security_headers
                for key, values in security_query.items():
                    if values:
                        query_params[key] = values[-1]

            if http_headers is not None:
                headers |= dict(http_headers)

            url = self._build_url(model, server_url=server_url, query_params=query_params)

            parsed = urlparse(url)
            if parsed.scheme == "https":
                parsed = parsed._replace(scheme="wss")
            elif parsed.scheme == "http":
                parsed = parsed._replace(scheme="ws")
            ws_url = urlunparse(parsed)
            open_timeout = None if timeout_ms is None else timeout_ms / 1000.0
            user_agent = self._sdk_config.user_agent

            websocket = None
            try:
                websocket = await connect(
                    ws_url,
                    additional_headers=dict(headers),
                    open_timeout=open_timeout,
                    user_agent_header=user_agent,
                )

                # Send initial session.update to trigger server handshake response.
                # This is required for vLLM/voxtral servers which won't send
                # session.created until they receive this message.
                session_update_payload = {}
                if audio_format is not None:
                    session_update_payload["audio_format"] = audio_format
                if target_streaming_delay_ms is not None:
                    session_update_payload["target_streaming_delay_ms"] = target_streaming_delay_ms
                
                # Send session.update message
                session_update_message = {
                    "type": "session.update",
                    "session": session_update_payload if session_update_payload else {}
                }
                await websocket.send(json.dumps(session_update_message))
                
                # Now receive the handshake (session.created)
                session, initial_events = await _recv_handshake(
                    websocket, timeout_ms=timeout_ms
                )
                connection = RealtimeConnection(
                    websocket=websocket,
                    session=session,
                    initial_events=initial_events,
                )

                # If audio_format or target_streaming_delay_ms was provided, send update again
                # (this is the standard behavior for Mistral's official API)
                if audio_format is not None or target_streaming_delay_ms is not None:
                    await connection.update_session(
                        audio_format,
                        target_streaming_delay_ms=target_streaming_delay_ms,
                    )

                return connection

            except RealtimeTranscriptionException:
                if websocket is not None:
                    await websocket.close()
                raise
            except Exception as exc:
                if websocket is not None:
                    await websocket.close()
                raise RealtimeTranscriptionException(f"Failed to connect: {exc}") from exc

    return _VLLMRealtimeTranscription(sdk_configuration)


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
            vllm_rt = _make_vllm_realtime_transcription(
                custom_client.sdk_configuration
            )
            stt_instance = self.stt  # capture for the closure

            async def _connect_ws_vllm(timeout: float):
                http_headers = None
                cfg = custom_client.sdk_configuration
                client_headers = getattr(
                    cfg.async_client, "headers", None
                ) or getattr(cfg.client, "headers", None)
                if client_headers:
                    http_headers = dict(client_headers)
                
                # Ensure Authorization header is included for vLLM/voxtral endpoints
                # The Mistral SDK may not include it in the client headers for custom endpoints
                if config.api_key and not (http_headers and "Authorization" in http_headers):
                    if http_headers is None:
                        http_headers = {}
                    http_headers["Authorization"] = f"Bearer {config.api_key}"
                
                logging.debug(
                    f"Connecting to voxtral WebSocket with headers: {list(http_headers.keys()) if http_headers else 'None'}"
                )
                
                return await asyncio.wait_for(
                    vllm_rt.connect(
                        model=stt_instance._opts.model,
                        target_streaming_delay_ms=stt_instance._opts.target_streaming_delay_ms,
                        http_headers=http_headers,
                    ),
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
