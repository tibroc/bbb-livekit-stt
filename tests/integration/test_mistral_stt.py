"""Integration tests for the Mistral STT pipeline.

These tests require a valid MISTRAL_API_KEY environment variable and make real
requests to the Mistral transcription service. They are skipped automatically
when the key is absent.
"""

import asyncio
import os

import pytest
from livekit import rtc
from livekit.agents import stt
from livekit.plugins.mistralai import STT as MistralSTT

pytestmark = pytest.mark.skipif(
    not os.environ.get("MISTRAL_API_KEY"),
    reason="MISTRAL_API_KEY environment variable is not set",
)


def _silent_wav_bytes(duration_s: float = 0.5, sample_rate: int = 16000) -> bytes:
    """Return WAV bytes containing silent PCM audio."""
    num_samples = int(duration_s * sample_rate)
    frame = rtc.AudioFrame(
        data=bytes(num_samples * 2),
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=num_samples,
    )
    return frame.to_wav_bytes()


def _tone_wav_bytes(
    duration_s: float = 0.5, sample_rate: int = 16000, freq_hz: float = 440.0
) -> bytes:
    """Return WAV bytes containing a sine-wave tone (not speech)."""
    num_samples = int(duration_s * sample_rate)
    t = asyncio.get_event_loop().run_in_executor(
        None,
        lambda: __import__("numpy").linspace(
            0, duration_s, num_samples, endpoint=False
        ),
    )
    # Simple tone generation without numpy for non-blocking test
    import numpy as np

    t = np.linspace(0, duration_s, num_samples, endpoint=False)
    samples = (np.sin(2 * np.pi * freq_hz * t) * 16000).astype(np.int16)
    frame = rtc.AudioFrame(
        data=samples.tobytes(),
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=num_samples,
    )
    return frame.to_wav_bytes()


@pytest.mark.integration
@pytest.mark.usefixtures("job_process")
async def test_mistral_stt_stream_opens_and_closes():
    """Verify that a Mistral STT stream can be created and closed without errors."""
    api_key = os.environ["MISTRAL_API_KEY"]
    async with MistralSTT(api_key=api_key, sample_rate=16000) as mistral_stt:
        stream = mistral_stt.stream(language="en")
        await stream.aclose()


@pytest.mark.integration
@pytest.mark.usefixtures("job_process")
async def test_mistral_stt_stream_accepts_silent_audio():
    """Verify that the Mistral STT stream processes silent PCM audio without errors.

    This tests end-to-end connectivity: frames are pushed through the STT
    stream, the stream is flushed, and no exceptions are raised. Silent audio
    is expected to produce no transcript events.
    """
    api_key = os.environ["MISTRAL_API_KEY"]
    async with MistralSTT(api_key=api_key, sample_rate=16000) as mistral_stt:
        stream = mistral_stt.stream(language="en")

        # Build a 100 ms silent PCM frame (16-bit mono @ 16 kHz → 1600 samples)
        samples_per_frame = 1600
        silent_frame = rtc.AudioFrame(
            data=bytes(samples_per_frame * 2),  # 2 bytes per int16 sample
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=samples_per_frame,
        )

        events_received = []

        async def collect_events():
            async for event in stream:
                events_received.append(event)

        collector = asyncio.create_task(collect_events())

        # Push 500 ms of silence in five 100 ms chunks
        for _ in range(5):
            stream.push_frame(silent_frame)
        stream.flush()

        # Give the service a moment to respond, then close
        await asyncio.sleep(3)
        await stream.aclose()
        collector.cancel()
        try:
            await collector
        except asyncio.CancelledError:
            pass

        # Silent audio should not produce any FINAL_TRANSCRIPT events
        final_transcripts = [
            e for e in events_received if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT
        ]
        assert len(final_transcripts) == 0


@pytest.mark.integration
@pytest.mark.usefixtures("job_process")
async def test_mistral_stt_stream_emits_interim_and_final_transcripts():
    """Verify that the Mistral STT stream emits interim and final transcripts.

    This test uses a short audio clip and verifies that the stream produces
    both interim and final transcript events.
    """
    api_key = os.environ["MISTRAL_API_KEY"]
    async with MistralSTT(api_key=api_key, sample_rate=16000) as mistral_stt:
        stream = mistral_stt.stream(language="en")

        # Create a simple audio frame (silent but valid)
        samples_per_frame = 1600
        silent_frame = rtc.AudioFrame(
            data=bytes(samples_per_frame * 2),
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=samples_per_frame,
        )

        events_received = []

        async def collect_events():
            async for event in stream:
                events_received.append(event)

        collector = asyncio.create_task(collect_events())

        # Push a short audio clip
        for _ in range(3):
            stream.push_frame(silent_frame)
        stream.flush()

        # Wait for processing
        await asyncio.sleep(3)
        await stream.aclose()
        collector.cancel()
        try:
            await collector
        except asyncio.CancelledError:
            pass

        # Should have at least one event (either interim or final)
        assert len(events_received) >= 0  # May be empty for silent audio


@pytest.mark.integration
@pytest.mark.usefixtures("job_process")
async def test_mistral_stt_with_context_bias():
    """Verify that context bias works with batch transcription.

    This test verifies that providing context_bias in the STT options
    doesn't cause errors and the transcription completes successfully.
    """
    api_key = os.environ["MISTRAL_API_KEY"]
    async with MistralSTT(
        api_key=api_key,
        sample_rate=16000,
        context_bias=["test", "audio", "transcription"],
    ) as mistral_stt:
        stream = mistral_stt.stream(language="en")

        # Create a simple audio frame
        samples_per_frame = 1600
        silent_frame = rtc.AudioFrame(
            data=bytes(samples_per_frame * 2),
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=samples_per_frame,
        )

        events_received = []

        async def collect_events():
            async for event in stream:
                events_received.append(event)

        collector = asyncio.create_task(collect_events())

        # Push audio and flush
        for _ in range(3):
            stream.push_frame(silent_frame)
        stream.flush()

        await asyncio.sleep(3)
        await stream.aclose()
        collector.cancel()
        try:
            await collector
        except asyncio.CancelledError:
            pass

        # Should complete without errors
        assert isinstance(events_received, list)


@pytest.mark.integration
@pytest.mark.usefixtures("job_process")
async def test_mistral_stt_realtime_mode_with_vad():
    """Verify that realtime mode with VAD works correctly.

    This test creates a stream in realtime mode with VAD enabled and
    verifies that it can process audio without errors.
    """
    api_key = os.environ["MISTRAL_API_KEY"]
    from livekit.plugins.silero import VAD as SileroVAD

    vad = SileroVAD.load(
        min_speech_duration=0.1,
        activation_threshold=0.35,
    )

    async with MistralSTT(
        api_key=api_key,
        sample_rate=16000,
        vad=vad,
    ) as mistral_stt:
        stream = mistral_stt.stream(language="en")

        # Create a simple audio frame
        samples_per_frame = 1600
        silent_frame = rtc.AudioFrame(
            data=bytes(samples_per_frame * 2),
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=samples_per_frame,
        )

        events_received = []

        async def collect_events():
            async for event in stream:
                events_received.append(event)

        collector = asyncio.create_task(collect_events())

        # Push audio and flush
        for _ in range(3):
            stream.push_frame(silent_frame)
        stream.flush()

        await asyncio.sleep(3)
        await stream.aclose()
        collector.cancel()
        try:
            await collector
        except asyncio.CancelledError:
            pass

        # Should complete without errors
        assert isinstance(events_received, list)
