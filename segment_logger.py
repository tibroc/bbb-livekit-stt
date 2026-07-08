"""Optional JSONL logger for finalized original-language transcript segments.

Set ``VOXTRAL_SEGMENT_LOG`` to a file path to append one JSON object per
finalized segment — used to collect a golden set of real segments (with speaker
locale) for the translation service's test fixtures. When the env var is unset
the factory returns ``None`` and nothing is registered, so it is safe to leave
wired in production.

Only original-language finals are logged (``language == speaker locale prefix``);
translation finals are skipped, so the file is a clean set of MT *inputs*.

Privacy: the file contains meeting speech content. It carries no participant
identity, but scrub/anonymize before sharing outside the deployment if needed.
"""

import asyncio
import itertools
import json
import logging
import os

from livekit import rtc
from livekit.agents import stt


def _append_line(path: str, line: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def make_segment_logger(agent):
    """Return a `final_transcript` listener that appends segments to
    ``VOXTRAL_SEGMENT_LOG`` as JSONL, or ``None`` if the var is unset."""
    path = os.getenv("VOXTRAL_SEGMENT_LOG")
    if not path:
        return None

    counter = itertools.count(1)

    async def _log_segment(
        participant: rtc.RemoteParticipant,
        event: stt.SpeechEvent,
        open_time: float = 0.0,
        **_,
    ):
        settings = agent.participant_settings.get(participant.identity, {})
        locale = settings.get("locale")
        if not locale:
            return
        src = locale.split("-")[0].lower()

        for alt in event.alternatives:
            # Skip translations: keep only the original-language segment.
            if (alt.language or src).lower() != src:
                continue
            text = (alt.text or "").strip()
            if not text:
                continue
            record = {
                "id": f"seg-{next(counter)}",
                "text": text,
                "src": src,
                "locale": locale,
                "start": alt.start_time,
                "end": alt.end_time,
            }
            try:
                # File I/O must never run on the event loop: a slow write
                # (bind mount, network storage) stalls the loop, LiveKit
                # frames back up and are burst-processed, and Silero's
                # START/END timing degrades — cutting the very utterances
                # this logger is trying to collect.
                await asyncio.to_thread(
                    _append_line, path, json.dumps(record, ensure_ascii=False) + "\n"
                )
            except OSError as e:
                logging.warning(f"Segment log write to {path} failed: {e}")

    logging.info(f"Segment logging enabled → {path}")
    return _log_segment
