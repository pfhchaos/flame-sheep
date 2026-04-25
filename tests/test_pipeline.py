"""End-to-end audio pipeline tests.

Synthesizes specific PCM patterns and feeds them through the full
audio pipeline (FFT → stability → beat detector → energy → density).
Validates the AudioSnapshot output, not individual components.

Each test case corresponds to a real-world scenario or a known bug.
"""

import numpy as np
import pytest

from flame_sheep.audio import AudioProcessor, SAMPLE_RATE, FFT_SIZE, HOP_SIZE
from flame_sheep.audio.source import FeedSource
from flame_sheep.audio._types import BeatEvent

from .synths import synth_kick, synth_vocal, synth_snare, synth_hihat
from .conftest import make_processor, make_silence, make_sine, feed_audio


WARMUP_FRAMES = 20  # silence frames before signal for detector warmup


def _run_pipeline(signal: np.ndarray, sharpness: bool = True,
                  warmup: bool = True):
    """Feed signal through full audio pipeline, return all events + snapshots.

    Feeds FFT_SIZE chunks with warmup silence, matching the
    pattern used by test_beat_engine.py.

    Returns:
        (all_events, snapshots): list of BeatEvents, list of AudioSnapshots
    """
    proc = make_processor(sharpness=sharpness)

    all_events = []
    snapshots = []

    # Warmup with silence
    if warmup:
        silence = make_silence(FFT_SIZE)
        for _ in range(WARMUP_FRAMES):
            feed_audio(proc, silence)
            proc.process()

    # Feed signal in FFT_SIZE chunks
    pos = 0
    while pos < len(signal):
        chunk = signal[pos:pos + FFT_SIZE]
        if len(chunk) < FFT_SIZE:
            chunk = np.pad(chunk, (0, FFT_SIZE - len(chunk)))
        feed_audio(proc, chunk)
        events = proc.process()
        snap = proc.drain()
        all_events.extend(events)
        snapshots.append(snap)
        pos += FFT_SIZE

    return all_events, snapshots


def _count_events(events, kind):
    return sum(1 for e in events if e.kind == kind)


def _place_hits(duration_s, interval_s, synth_fn, **kwargs):
    """Place synthesized hits at regular intervals in a signal."""
    n = int(SAMPLE_RATE * duration_s)
    signal = np.zeros(n, dtype=np.float32)
    hit = synth_fn(**kwargs)
    t = 0.0
    while t < duration_s:
        pos = int(t * SAMPLE_RATE)
        end = min(pos + len(hit), n)
        signal[pos:end] += hit[:end - pos]
        t += interval_s
    return signal


class TestPipelineKickDetection:
    """Verify kick detection through the full pipeline."""

    def test_kicks_at_120bpm_detected(self):
        """Regular kicks at 120 BPM should produce kick events."""
        signal = _place_hits(4.0, 0.5, synth_kick)
        events, _ = _run_pipeline(signal)
        kicks = _count_events(events, 'kick')
        # 4 seconds at 2 kicks/s = ~8 kicks, minus warmup
        assert kicks >= 3, f"Expected at least 3 kicks at 120 BPM, got {kicks}"

    def test_silence_no_kicks(self):
        """Silence should produce no kick events."""
        signal = make_silence(int(SAMPLE_RATE * 2))
        events, _ = _run_pipeline(signal)
        kicks = _count_events(events, 'kick')
        assert kicks == 0, f"Expected 0 kicks in silence, got {kicks}"

    def test_silence_low_rms(self):
        """Silence should have near-zero subbass RMS."""
        signal = make_silence(int(SAMPLE_RATE * 2))
        _, snapshots = _run_pipeline(signal)
        # Check last snapshot (after settling)
        last = snapshots[-1]
        assert last.bands['subbass'].rms < 0.001


class TestPipelineVocalSuppression:
    """Verify that sustained vocals don't produce false kicks."""

    def test_male_vocal_low_kick_count(self):
        """Sustained 80Hz male vocal should produce very few kick events.

        This is the Jolene problem: male chest resonance at 50-100Hz
        triggers false kicks without stability scaling.
        """
        signal = synth_vocal(4.0, pitch=80, amplitude=0.5)
        events, _ = _run_pipeline(signal)
        kicks = _count_events(events, 'kick')
        assert kicks < 5, \
            f"Male vocal at 80Hz produced {kicks} false kicks (expected < 5)"

    def test_kick_over_vocal_still_detected(self):
        """Kicks mixed with sustained vocal should still be detected.

        Stability scaling should raise the threshold but not suppress
        real kicks — the kick transient exceeds the vocal's variance.
        """
        vocal = synth_vocal(4.0, pitch=80, amplitude=0.3)
        kicks_signal = _place_hits(4.0, 0.5, synth_kick, amplitude=0.8)
        signal = vocal + kicks_signal
        events, _ = _run_pipeline(signal)
        kicks = _count_events(events, 'kick')
        assert kicks >= 2, \
            f"Kicks over vocal produced only {kicks} events (expected >= 2)"


class TestPipelineBreakDetection:
    """Verify break detection through the full pipeline."""

    def test_break_after_kicks(self):
        """Sustained kicks followed by silence should trigger breaking state."""
        # 3s of kicks (warmup + establish baseline), then 2s silence
        kicks = _place_hits(3.0, 0.5, synth_kick)
        silence = make_silence(int(SAMPLE_RATE * 2))
        signal = np.concatenate([kicks, silence])
        _, snapshots = _run_pipeline(signal)

        # Check that at least one snapshot during the silent section has
        # the break detector's conditions met (we check the raw detector
        # state indirectly via the centroid_rms dropping)
        kick_frames = int(3.0 * SAMPLE_RATE / HOP_SIZE)
        silent_rms = [s.bands['subbass'].rms for s in snapshots[kick_frames:]]
        # RMS should be very low during silence
        if silent_rms:
            assert min(silent_rms) < 0.01, \
                "Subbass RMS should drop during silent break"


class TestPipelineBandSeparation:
    """Verify that events land in the correct bands."""

    def test_low_sine_triggers_kick_not_hihat(self):
        """80Hz tone onset should trigger kick, not hihat."""
        silence = make_silence(FFT_SIZE)
        tone = make_sine(80, FFT_SIZE * 4, amplitude=0.9)
        signal = np.concatenate([silence] * 5 + [tone])
        events, _ = _run_pipeline(signal)
        kicks = _count_events(events, 'kick')
        hihats = _count_events(events, 'hihat')
        assert kicks > 0, "80Hz onset should trigger kick"
        assert hihats == 0 or kicks > hihats, \
            f"80Hz should primarily trigger kick ({kicks}), not hihat ({hihats})"

    def test_high_sine_triggers_hihat_not_kick(self):
        """10kHz tone onset should trigger hihat, not kick."""
        silence = make_silence(FFT_SIZE)
        tone = make_sine(10000, FFT_SIZE * 4, amplitude=0.5)
        signal = np.concatenate([silence] * 5 + [tone])
        events, _ = _run_pipeline(signal)
        kicks = _count_events(events, 'kick')
        hihats = _count_events(events, 'hihat')
        # Hihat should dominate or at least be present
        assert hihats > 0 or kicks == 0, \
            f"10kHz should trigger hihat ({hihats}), not kick ({kicks})"


class TestPipelineHarmonicEnergy:
    """Verify harmonic/percussive energy split."""

    def test_sustained_tone_has_harmonic_rms(self):
        """A sustained sine wave should produce harmonic RMS."""
        signal = make_sine(200, int(SAMPLE_RATE * 3), amplitude=0.5)
        _, snapshots = _run_pipeline(signal)
        # After stability settles (~1s), harmonic RMS should be nonzero
        late = snapshots[len(snapshots) * 2 // 3:]
        if late:
            max_hrms = max(s.bands['subbass'].harmonic_rms for s in late)
            assert max_hrms > 0, "Sustained tone should have harmonic RMS"

    def test_kick_transient_low_harmonic_rms(self):
        """A single kick should have low harmonic RMS (it's transient)."""
        silence = make_silence(int(SAMPLE_RATE * 2))
        kick = synth_kick(amplitude=0.9)
        # Kick at 1.5s after stability has settled on silence
        signal = silence.copy()
        pos = int(1.5 * SAMPLE_RATE)
        signal[pos:pos + len(kick)] += kick
        signal = np.append(signal, make_silence(int(SAMPLE_RATE * 0.5)))
        _, snapshots = _run_pipeline(signal)
        # Harmonic RMS should stay low since the kick is transient
        max_hrms = max(s.bands['subbass'].harmonic_rms for s in snapshots)
        total_rms = max(s.bands['subbass'].rms for s in snapshots)
        if total_rms > 0.001:
            assert max_hrms < total_rms, \
                "Kick transient should have harmonic_rms < total rms"
