import asyncio
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
    path.
    """
    from mistralai.client import utils as mistral_utils
    from mistralai.client.utils import generate_url
    from mistralai.extra.realtime import RealtimeTranscription

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
            return urlunparse(parsed._replace(query=urlencode(merged)))

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
