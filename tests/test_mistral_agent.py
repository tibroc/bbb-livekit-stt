import os
from unittest.mock import MagicMock, patch

import pytest
from livekit import rtc
from livekit.agents import stt
from livekit.plugins.mistralai import STT as MistralSTT

from providers.mistral import MistralConfig, MistralSttAgent


def _make_agent(**kwargs):
    """Create a MistralSttAgent with the given config options."""
    config = MistralConfig(api_key="fake-key", **kwargs)
    with patch("providers.mistral.MistralSTT", spec=MistralSTT):
        agent = MistralSttAgent(config)
    return agent


def _make_track_subscribed_args(source=rtc.TrackSource.SOURCE_MICROPHONE):
    mock_track = MagicMock()
    mock_publication = MagicMock()
    mock_publication.source = source
    mock_participant = MagicMock()
    return mock_track, mock_publication, mock_participant


def _make_agent_with_room(participants=None, **kwargs):
    """Create an agent with a mocked room containing the given participants."""
    agent = _make_agent(**kwargs)
    mock_room = MagicMock()
    participants = participants or {}
    mock_room.remote_participants = participants
    agent.room = mock_room
    return agent


def _make_participant(identity, audio_track=None):
    """Create a mock RemoteParticipant with an optional audio track."""
    participant = MagicMock(spec=rtc.RemoteParticipant)
    participant.identity = identity
    pubs = {}
    if audio_track:
        pub = MagicMock()
        pub.track = audio_track
        pub.track.kind = rtc.TrackKind.KIND_AUDIO
        pubs["audio"] = pub
    participant.track_publications = pubs
    return participant


class TestSanitizeLocale:
    def test_strips_region_from_bcp47_locale(self):
        agent = _make_agent()
        assert agent._sanitize_locale("en-US") == "en"
        assert agent._sanitize_locale("pt-BR") == "pt"
        assert agent._sanitize_locale("zh-CN") == "zh"
        assert agent._sanitize_locale("fr-FR") == "fr"

    def test_returns_language_code_unchanged_when_no_region(self):
        agent = _make_agent()
        assert agent._sanitize_locale("en") == "en"
        assert agent._sanitize_locale("de") == "de"

    def test_lowercases_language_code(self):
        agent = _make_agent()
        assert agent._sanitize_locale("EN-US") == "en"
        assert agent._sanitize_locale("PT") == "pt"


class TestMistralConfigDefaults:
    @pytest.fixture(autouse=True)
    def _clean_mistral_env(self, monkeypatch):
        for key in list(os.environ):
            if key.startswith("MISTRAL_"):
                monkeypatch.delenv(key, raising=False)

    def test_model_defaults_to_voxtral_mini_latest(self):
        assert MistralConfig().model == "voxtral-mini-latest"

    def test_api_key_defaults_to_none(self):
        assert MistralConfig().api_key is None

    def test_realtime_defaults_to_false(self):
        assert MistralConfig().realtime is False

    def test_vad_enabled_defaults_to_true(self):
        assert MistralConfig().vad_enabled is True

    def test_vad_aggressiveness_defaults_to_2(self):
        assert MistralConfig().vad_aggressiveness == 2

    def test_min_confidence_interim_defaults_to_0(self):
        assert MistralConfig().min_confidence_interim == 0.0

    def test_min_confidence_final_defaults_to_0(self):
        assert MistralConfig().min_confidence_final == 0.0


class TestMistralConfigFromEnvironment:
    def test_model_from_environment(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_MODEL", "voxtral-small-latest")
        config = MistralConfig()
        assert config.model == "voxtral-small-latest"

    def test_api_key_from_environment(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "env-api-key")
        config = MistralConfig()
        assert config.api_key == "env-api-key"

    def test_realtime_from_environment(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_REALTIME", "true")
        config = MistralConfig()
        assert config.realtime is True

    def test_vad_enabled_from_environment(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_VAD_ENABLED", "false")
        config = MistralConfig()
        assert config.vad_enabled is False

    def test_vad_aggressiveness_from_environment(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_VAD_AGGRESSIVENESS", "3")
        config = MistralConfig()
        assert config.vad_aggressiveness == 3

    def test_streaming_delay_from_environment(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_STREAMING_DELAY", "500")
        config = MistralConfig()
        assert config.streaming_delay_ms == 500

    def test_context_bias_from_environment(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_CONTEXT_BIAS", '["term1", "term2"]')
        config = MistralConfig()
        assert config.context_bias == ["term1", "term2"]

    def test_min_confidence_interim_from_environment(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_MIN_CONFIDENCE_INTERIM", "0.3")
        config = MistralConfig()
        assert config.min_confidence_interim == 0.3

    def test_min_confidence_final_from_environment(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_MIN_CONFIDENCE_FINAL", "0.5")
        config = MistralConfig()
        assert config.min_confidence_final == 0.5


class TestToSttKwargs:
    def test_includes_api_key(self):
        config = MistralConfig(api_key="test-key")
        kwargs = config.to_stt_kwargs()
        assert kwargs["api_key"] == "test-key"

    def test_includes_model(self):
        config = MistralConfig(model="voxtral-small-latest")
        kwargs = config.to_stt_kwargs()
        assert kwargs["model"] == "voxtral-small-latest"

    def test_includes_language(self):
        config = MistralConfig(language="en")
        kwargs = config.to_stt_kwargs()
        assert kwargs["language"] == "en"

    def test_includes_context_bias(self):
        config = MistralConfig(context_bias=["term1", "term2"])
        kwargs = config.to_stt_kwargs()
        assert kwargs["context_bias"] == ["term1", "term2"]

    def test_includes_streaming_delay_as_target_streaming_delay_ms(self):
        config = MistralConfig(streaming_delay_ms=500)
        kwargs = config.to_stt_kwargs()
        assert kwargs["target_streaming_delay_ms"] == 500

    def test_excludes_none_values(self):
        config = MistralConfig()
        kwargs = config.to_stt_kwargs()
        assert "api_key" not in kwargs
        # model has a default factory, so it's always present
        assert "model" in kwargs
        assert "language" not in kwargs

    def test_includes_vad_when_realtime_and_vad_enabled(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_REALTIME", "true")
        monkeypatch.setenv("MISTRAL_VAD_ENABLED", "true")
        # Mock the SileroVAD import at the source since it's imported inside to_stt_kwargs
        # We need to patch the import system since the import happens inside the function
        with patch.dict(
            "sys.modules",
            {"livekit.plugins.silero": MagicMock(VAD=MagicMock(load=MagicMock()))},
        ):
            import sys

            config = MistralConfig()
            kwargs = config.to_stt_kwargs()
            assert "vad" in kwargs
            # VAD should be loaded with correct parameters
            assert kwargs["vad"] is not None
            sys.modules["livekit.plugins.silero"].VAD.load.assert_called_once()

    def test_excludes_vad_when_realtime_but_vad_disabled(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_REALTIME", "true")
        monkeypatch.setenv("MISTRAL_VAD_ENABLED", "false")
        config = MistralConfig()
        kwargs = config.to_stt_kwargs()
        assert "vad" not in kwargs

    def test_excludes_vad_when_not_realtime(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_REALTIME", "false")
        monkeypatch.setenv("MISTRAL_VAD_ENABLED", "true")
        config = MistralConfig()
        kwargs = config.to_stt_kwargs()
        assert "vad" not in kwargs


class TestCreateSttStream:
    def test_creates_stream_with_locale(self):
        agent = _make_agent()
        mock_stt = MagicMock()
        agent.stt = mock_stt

        agent._create_stt_stream("en")

        mock_stt.stream.assert_called_once_with(language="en")

    def test_passes_locale_directly_to_stream(self):
        agent = _make_agent()
        mock_stt = MagicMock()
        agent.stt = mock_stt

        agent._create_stt_stream("en-US")

        mock_stt.stream.assert_called_once_with(language="en-US")


class TestUpdateStreamLocale:
    def test_sanitizes_locale_and_updates_stream(self):
        agent = _make_agent()
        mock_stream = MagicMock()
        agent.stt = MagicMock()
        agent.stt.update_options = mock_stream.update_options

        agent._update_stream_locale("user_1", "en-US")

        mock_stream.update_options.assert_called_once_with(language="en")

    def test_sanitizes_locale_to_language_code(self):
        agent = _make_agent()
        mock_stream = MagicMock()
        agent.stt = MagicMock()
        agent.stt.update_options = mock_stream.update_options

        agent._update_stream_locale("user_1", "pt-BR")

        mock_stream.update_options.assert_called_once_with(language="pt")


class TestShouldEmit:
    def test_allows_final_above_threshold(self):
        agent = _make_agent(min_confidence_final=0.5)
        event = MagicMock()
        event.type = stt.SpeechEventType.FINAL_TRANSCRIPT
        alt = MagicMock()
        alt.confidence = 0.8
        event.alternatives = [alt]

        assert agent._should_emit(event) is True

    def test_blocks_final_below_threshold(self):
        agent = _make_agent(min_confidence_final=0.5)
        event = MagicMock()
        event.type = stt.SpeechEventType.FINAL_TRANSCRIPT
        alt = MagicMock()
        alt.confidence = 0.3
        event.alternatives = [alt]

        assert agent._should_emit(event) is False

    def test_allows_interim_above_threshold(self):
        agent = _make_agent(min_confidence_interim=0.2)
        event = MagicMock()
        event.type = stt.SpeechEventType.INTERIM_TRANSCRIPT
        alt = MagicMock()
        alt.confidence = 0.5
        event.alternatives = [alt]

        assert agent._should_emit(event) is True

    def test_blocks_interim_below_threshold(self):
        agent = _make_agent(min_confidence_interim=0.5)
        event = MagicMock()
        event.type = stt.SpeechEventType.INTERIM_TRANSCRIPT
        alt = MagicMock()
        alt.confidence = 0.1
        event.alternatives = [alt]

        assert agent._should_emit(event) is False

    def test_allows_unknown_event_types(self):
        agent = _make_agent()
        event = MagicMock()
        event.type = stt.SpeechEventType.END_OF_SPEECH
        alt = MagicMock()
        alt.confidence = 0.1
        event.alternatives = [alt]

        assert agent._should_emit(event) is True

    def test_blocks_when_any_alternative_below_threshold(self):
        """_should_emit blocks if any alternative is below threshold."""
        agent = _make_agent(min_confidence_final=0.5)
        event = MagicMock()
        event.type = stt.SpeechEventType.FINAL_TRANSCRIPT
        alt1 = MagicMock()
        alt1.confidence = 0.3
        alt2 = MagicMock()
        alt2.confidence = 0.8
        event.alternatives = [alt1, alt2]

        assert agent._should_emit(event) is False

    def test_blocks_when_all_alternatives_below_threshold(self):
        agent = _make_agent(min_confidence_final=0.5)
        event = MagicMock()
        event.type = stt.SpeechEventType.FINAL_TRANSCRIPT
        alt1 = MagicMock()
        alt1.confidence = 0.3
        alt2 = MagicMock()
        alt2.confidence = 0.4
        event.alternatives = [alt1, alt2]

        assert agent._should_emit(event) is False


class TestVadLoading:
    @pytest.mark.skip(reason="Silero VAD not available in test environment")
    def test_vad_parameters_based_on_aggressiveness(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_VAD_AGGRESSIVENESS", "1")
        config = MistralConfig()
        kwargs = config.to_stt_kwargs()
        vad = kwargs["vad"]
        # Aggressiveness 1 -> activation_threshold = 0.25 + (1 * 0.1) = 0.35
        assert vad._activation_threshold == 0.35

        monkeypatch.setenv("MISTRAL_VAD_AGGRESSIVENESS", "3")
        config = MistralConfig()
        kwargs = config.to_stt_kwargs()
        vad = kwargs["vad"]
        # Aggressiveness 3 -> activation_threshold = 0.25 + (3 * 0.1) = 0.55
        assert vad._activation_threshold == 0.55
