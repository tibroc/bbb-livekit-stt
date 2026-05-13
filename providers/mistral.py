import logging
import os
from dataclasses import dataclass, field
from typing import List

from livekit.agents import stt
from livekit.plugins.mistralai import STT as MistralSTT

from config import _get_bool_env, _get_float_env, _get_json_env
from providers.base import BaseSttAgent, BaseSttConfig


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
