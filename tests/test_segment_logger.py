import json
from unittest.mock import MagicMock

from livekit import rtc
from livekit.agents import stt

from segment_logger import make_segment_logger


def _agent(settings):
    agent = MagicMock()
    agent.participant_settings = settings
    return agent


def _participant(identity):
    p = MagicMock(spec=rtc.RemoteParticipant)
    p.identity = identity
    return p


def _final_event(alternatives):
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT, alternatives=alternatives
    )


def test_returns_none_when_env_unset(monkeypatch):
    monkeypatch.delenv("VOXTRAL_SEGMENT_LOG", raising=False)
    assert make_segment_logger(_agent({})) is None


async def test_logs_only_original_language_segment(monkeypatch, tmp_path):
    path = tmp_path / "segments.jsonl"
    monkeypatch.setenv("VOXTRAL_SEGMENT_LOG", str(path))

    agent = _agent({"u1": {"locale": "de-DE"}})
    log = make_segment_logger(agent)
    assert log is not None

    # One original (de) + one translation (en) alternative for the same segment.
    event = _final_event(
        [
            stt.SpeechData(
                text="Guten Tag", language="de", start_time=1.0, end_time=2.0
            ),
            stt.SpeechData(
                text="Good day", language="en", start_time=1.0, end_time=2.0
            ),
        ]
    )
    await log(participant=_participant("u1"), event=event, open_time=1000.0)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1, "only the original-language segment is logged"
    rec = json.loads(lines[0])
    assert rec["text"] == "Guten Tag"
    assert rec["src"] == "de"
    assert rec["locale"] == "de-DE"
    assert rec["start"] == 1.0 and rec["end"] == 2.0


async def test_skips_empty_text_and_missing_locale(monkeypatch, tmp_path):
    path = tmp_path / "segments.jsonl"
    monkeypatch.setenv("VOXTRAL_SEGMENT_LOG", str(path))

    # Participant with no locale → nothing logged.
    log = make_segment_logger(_agent({"u1": {}}))
    await log(
        participant=_participant("u1"),
        event=_final_event(
            [stt.SpeechData(text="hi", language="de", start_time=0.0, end_time=1.0)]
        ),
        open_time=0.0,
    )
    # Known locale but empty/whitespace text → nothing logged.
    agent2 = _agent({"u2": {"locale": "en-US"}})
    log2 = make_segment_logger(agent2)
    await log2(
        participant=_participant("u2"),
        event=_final_event(
            [stt.SpeechData(text="   ", language="en", start_time=0.0, end_time=1.0)]
        ),
        open_time=0.0,
    )

    assert not path.exists() or path.read_text() == ""


async def test_appends_across_calls(monkeypatch, tmp_path):
    path = tmp_path / "segments.jsonl"
    monkeypatch.setenv("VOXTRAL_SEGMENT_LOG", str(path))
    agent = _agent({"u1": {"locale": "de-DE"}})
    log = make_segment_logger(agent)

    for word in ("eins", "zwei", "drei"):
        await log(
            participant=_participant("u1"),
            event=_final_event(
                [stt.SpeechData(text=word, language="de", start_time=0.0, end_time=1.0)]
            ),
            open_time=0.0,
        )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["text"] for x in lines] == ["eins", "zwei", "drei"]
    assert [json.loads(x)["id"] for x in lines] == ["seg-1", "seg-2", "seg-3"]
