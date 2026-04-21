"""
Tests for tempo detection and rhythm coherence gating.
"""

import numpy as np
import pytest
from flame_sheep.tempo import (
    TempoTracker, TempoState,
    MIN_BPM, MAX_BPM, GATE_THRESHOLD,
)


def simulate_beats(tracker: TempoTracker, bpm: float, count: int,
                   start: float = 0.0) -> list[bool]:
    """Simulate beats by generating a click track and feeding it frame-by-frame.

    Generates continuous audio with short sine pulses at the given BPM,
    then chops it into frames for the tracker. This avoids frame-boundary
    artifacts that plague simpler approaches.
    """
    interval = 60.0 / bpm
    sr = tracker._sr
    hop = tracker._hop
    duration = count * interval + 0.5  # a little extra
    total_samples = int(duration * sr)

    # Generate click track: short sine bursts at each beat
    audio = np.zeros(total_samples, dtype=np.float32)
    pulse_len = int(sr * 0.005)  # 5ms pulse
    pulse = np.sin(2 * np.pi * 1000 * np.arange(pulse_len) / sr).astype(np.float32)
    pulse *= np.hanning(pulse_len).astype(np.float32)
    for i in range(count):
        sample = int((start + i * interval) * sr)
        end = min(sample + pulse_len, total_samples)
        if sample >= 0 and sample < total_samples:
            audio[sample:end] = pulse[:end - sample]

    # Feed frame by frame
    results = []
    onset_idx = 0
    for pos in range(0, total_samples - hop, hop):
        frame = audio[pos:pos + hop]
        tracker.feed_audio(frame)

        frame_time = pos / sr
        next_time = (pos + hop) / sr

        # Check if any beats fall in this frame
        while onset_idx < count:
            beat_time = start + onset_idx * interval
            if beat_time >= next_time:
                break
            if beat_time >= frame_time:
                passed = tracker.process_onset('kick', beat_time)
                results.append(passed)
            onset_idx += 1

    return results


def simulate_random_onsets(tracker: TempoTracker, count: int,
                           min_gap: float = 0.2, max_gap: float = 0.8,
                           seed: int = 42, start: float = 0.0) -> list[bool]:
    """Simulate random-interval onsets with random flux."""
    import random
    random.seed(seed)
    rng = np.random.default_rng(seed)
    frame_duration = tracker._hop / tracker._sr

    results = []
    t = start
    for _ in range(count):
        gap = random.uniform(min_gap, max_gap)
        onset_time = t + gap

        # Feed random flux frames
        while t < onset_time:
            tracker.feed_audio(rng.uniform(-0.01, 0.01, size=tracker._hop).astype(np.float32))
            t += frame_duration

        noise = rng.uniform(-0.3, 0.3, size=tracker._hop).astype(np.float32); tracker.feed_audio(noise)
        t += frame_duration
        passed = tracker.process_onset('kick', onset_time)
        results.append(passed)

    return results


class TestTempoDetection:
    """Tests for tempo estimation accuracy."""

    def test_detects_120_bpm(self):
        tracker = TempoTracker()
        simulate_beats(tracker, 120, 30)
        # Allow octave equivalents
        assert _bpm_close(tracker.bpm, 120), f"Expected ~120 BPM, got {tracker.bpm}"

    def test_detects_90_bpm(self):
        tracker = TempoTracker()
        simulate_beats(tracker, 90, 30)
        assert _bpm_close(tracker.bpm, 90), f"Expected ~90 BPM, got {tracker.bpm}"

    def test_detects_150_bpm(self):
        tracker = TempoTracker()
        simulate_beats(tracker, 150, 30)
        assert _bpm_close(tracker.bpm, 150), f"Expected ~150 BPM, got {tracker.bpm}"


class TestConfidenceAndLocking:
    """Tests for confidence and lock/unlock behavior."""

    def test_regular_beats_lock(self):
        tracker = TempoTracker()
        simulate_beats(tracker, 120, 40)
        assert tracker.locked, \
            f"Should lock on regular beats, confidence={tracker.confidence}"

    def test_random_intervals_dont_lock(self):
        tracker = TempoTracker()
        simulate_random_onsets(tracker, 60)
        assert not tracker.locked, "Should not lock on random intervals"

    def test_unlocks_on_garbage(self):
        tracker = TempoTracker()
        simulate_beats(tracker, 120, 40)
        assert tracker.locked
        simulate_random_onsets(tracker, 80, start=100.0)
        assert tracker.confidence < GATE_THRESHOLD or not tracker.locked


class TestGating:
    """Tests for onset gating (pass/block behavior)."""

    def test_passes_during_learning(self):
        """All events pass while building initial hypothesis."""
        tracker = TempoTracker()
        results = simulate_beats(tracker, 120, 5)
        assert all(results), f"Should pass during learning, got {results}"

    def test_gates_off_grid_kicks_when_locked(self):
        tracker = TempoTracker()
        simulate_beats(tracker, 120, 40)
        assert tracker.locked

        # Off-grid kick between subdivisions
        period = tracker._beat_period
        last = tracker._last_beat_time
        off_grid_time = last + period * 0.375  # between 1/4 and 1/2
        passed = tracker.process_onset('kick', off_grid_time)
        assert not passed, "Off-grid kick should be blocked"

    def test_passes_on_beat_kicks_when_locked(self):
        tracker = TempoTracker()
        simulate_beats(tracker, 120, 40)
        assert tracker.locked

        last = tracker._last_beat_time
        on_beat = last + tracker._beat_period
        passed = tracker.process_onset('kick', on_beat)
        assert passed, "On-beat kick should pass"

    def test_passes_subdivision_kicks_when_locked(self):
        tracker = TempoTracker()
        simulate_beats(tracker, 120, 40)
        assert tracker.locked

        last = tracker._last_beat_time
        half_beat = last + tracker._beat_period / 2
        passed = tracker.process_onset('kick', half_beat)
        assert passed, "Half-beat kick should pass"

    def test_snare_hihat_always_pass(self):
        tracker = TempoTracker()
        simulate_beats(tracker, 120, 40)
        assert tracker.locked

        last = tracker._last_beat_time
        off_grid = last + tracker._beat_period * 0.375
        assert tracker.process_onset('snare', off_grid), "Snare should always pass"
        assert tracker.process_onset('hihat', off_grid), "Hihat should always pass"


class TestTrustedSource:

    def test_song_started_enables_trusted(self):
        tracker = TempoTracker()
        tracker.song_started()
        passed = tracker.process_onset('kick', 0.0)
        assert passed

    def test_tempo_hint_sets_hypothesis(self):
        tracker = TempoTracker()
        tracker.hint_tempo(120)
        assert tracker._has_hypothesis
        assert _bpm_close(tracker.bpm, 120)

    def test_trusted_bypasses_learning(self):
        tracker = TempoTracker()
        tracker.song_started()
        results = simulate_beats(tracker, 120, 5)
        assert all(results)

    def test_hint_locks_faster(self):
        tracker = TempoTracker()
        tracker.hint_tempo(120)
        simulate_beats(tracker, 120, 20)
        assert tracker.locked, "Should lock quickly with correct hint"

    def test_reset_clears_trusted(self):
        tracker = TempoTracker()
        tracker.song_started()
        assert tracker._trusted_source
        tracker.reset()
        assert not tracker._trusted_source


class TestState:

    def test_state_reflects_tracker(self):
        tracker = TempoTracker()
        simulate_beats(tracker, 120, 40)
        state = tracker.state
        assert isinstance(state, TempoState)
        assert state.bpm == tracker.bpm
        assert state.confidence == tracker.confidence
        assert state.locked == tracker.locked

    def test_phase_in_range(self):
        tracker = TempoTracker()
        simulate_beats(tracker, 120, 40)
        state = tracker.state
        assert 0.0 <= state.phase <= 1.0


class TestEdgeCases:

    def test_empty_tracker(self):
        tracker = TempoTracker()
        assert tracker.bpm == 0.0
        assert tracker.confidence == 0.0
        assert not tracker.locked

    def test_single_onset(self):
        tracker = TempoTracker()
        click = np.zeros(tracker._hop, dtype=np.float32); click[:100] = 1.0; tracker.feed_audio(click)
        tracker.process_onset('kick', 0.0)
        assert tracker.confidence == 0.0

    def test_reset_clears_all(self):
        tracker = TempoTracker()
        simulate_beats(tracker, 120, 40)
        assert tracker.locked
        tracker.reset()
        assert tracker.bpm == 0.0
        assert tracker.confidence == 0.0
        assert not tracker.locked


def _bpm_close(actual: float, expected: float, tolerance: float = 8.0) -> bool:
    """Check if BPM is close, accounting for octave equivalents."""
    for ratio in [0.5, 1.0, 2.0]:
        if abs(actual - expected * ratio) < tolerance:
            return True
    return False
