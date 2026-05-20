import logging
import os
from dataclasses import dataclass, field

from livekit.agents import stt

from providers.base import BaseSttAgent, BaseSttConfig


@dataclass
class OpenAiRealtimeConfig(BaseSttConfig):
    """Configuration for OpenAI STT provider."""

    api_key: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_REALTIME_API_KEY")
    )
    model: str = field(
        default_factory=lambda: os.getenv(
            "OPENAI_REALTIME_MODEL", "gpt-4o-mini-transcribe"
        )
    )
    base_url: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_REALTIME_BASE_URL", None)
    )
    # STT-specific options
    language: str = field(
        default_factory=lambda: os.getenv("OPENAI_REALTIME_LANGUAGE", "en")
    )
    detect_language: bool | None = field(
        default_factory=lambda: (
            os.getenv("OPENAI_REALTIME_DETECT_LANGUAGE", "False").lower() == "true"
            if os.getenv("OPENAI_REALTIME_DETECT_LANGUAGE")
            else None
        )
    )
    prompt: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_REALTIME_PROMPT", None)
    )
    turn_detection: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_REALTIME_TURN_DETECTION", None)
    )
    noise_reduction_type: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_REALTIME_NOISE_REDUCTION_TYPE", None)
    )


openai_realtime_config = OpenAiRealtimeConfig()


class OpenAiRealtimeSttAgent(BaseSttAgent):
    """OpenAI STT provider using LiveKit's OpenAI plugin.

    This provider uses the OpenAI STT API (gpt-4o-mini-transcribe) for speech
    transcription. It supports both REST and streaming modes.

    Supports:
    - Official OpenAI API (default)
    - Custom OpenAI-compatible servers (e.g., Azure OpenAI)
    - Language detection
    - Custom prompts

    Configuration via environment variables:
    - OPENAI_REALTIME_API_KEY: API key for authentication
    - OPENAI_REALTIME_MODEL: Model name (default: gpt-4o-mini-transcribe)
    - OPENAI_REALTIME_BASE_URL: Custom server URL (e.g., https://your-azure-endpoint.openai.azure.com/)
    - OPENAI_REALTIME_LANGUAGE: Language code (default: en)
    - OPENAI_REALTIME_DETECT_LANGUAGE: Enable language detection (default: False)
    - OPENAI_REALTIME_PROMPT: Prompt for transcription
    - OPENAI_REALTIME_TURN_DETECTION: Turn detection mode
    - OPENAI_REALTIME_NOISE_REDUCTION_TYPE: Noise reduction type (near_field/far_field)
    """

    def __init__(self, config: OpenAiRealtimeConfig):
        super().__init__(config)
        self._stt_plugin = None

    def _get_stt_plugin(self) -> stt.STT:
        """Get or create the LiveKit OpenAI STT plugin instance."""
        if self._stt_plugin is None:
            from livekit.plugins import openai

            # Build kwargs for STT
            kwargs: dict = {}

            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            if self.config.language:
                kwargs["language"] = self.config.language
            if self.config.detect_language:
                kwargs["detect_language"] = self.config.detect_language
            if self.config.prompt:
                kwargs["prompt"] = self.config.prompt
            if self.config.turn_detection:
                kwargs["turn_detection"] = self.config.turn_detection
            if self.config.noise_reduction_type:
                kwargs["noise_reduction_type"] = self.config.noise_reduction_type

            self._stt_plugin = openai.STT(
                model=self.config.model, api_key=self.config.api_key, **kwargs
            )
            logging.info(f"Created OpenAI STT plugin with model {self.config.model}")
        return self._stt_plugin

    def _create_stt_stream(self, locale: str) -> stt.SpeechStream:
        """Create an STT stream for the given locale."""
        stt_plugin = self._get_stt_plugin()
        stream = stt_plugin.stream(language=locale)
        logging.debug(
            f"Created STT stream for locale '{locale}' with model '{self.config.model}'"
        )
        return stream

    def _update_stream_locale(self, user_id: str, locale: str):
        """Update the locale for an active stream."""
        logging.info(f"Updating locale to '{locale}' for user {user_id}.")

        if user_id in self.processing_info:
            info = self.processing_info[user_id]
            stream = info.get("stream")

            if stream:
                # The LiveKit STT stream doesn't have a direct locale update method
                # We need to restart the stream with the new locale
                self.stop_transcription_for_user(user_id)
                self.start_transcription_for_user(user_id, locale, "openai-realtime")
            else:
                logging.warning(
                    f"Cannot update locale, no active stream for user {user_id}."
                )
        else:
            logging.warning(
                f"Won't update locale, no active transcription for user {user_id}."
            )

    # --- BaseSttAgent abstract method implementations ---

    @property
    def translation_lang_map(self) -> dict:
        """Map provider language codes to BBB locales.

        OpenAI Realtime uses ISO 639-1 codes (e.g., "en", "de").
        BBB uses <ISO 639-1>-<ISO 3166-1> format (e.g., "en-US", "de-DE").
        """
        return {}

    def _should_emit(self, event: stt.SpeechEvent) -> bool:
        """Filter events before emission.

        OpenAI Realtime may return empty transcripts during silence periods.
        Filter these out to avoid noise in the output.
        """
        if not event.alternatives:
            return False

        text = event.alternatives[0].text
        if not text or not text.strip():
            return False

        return True
