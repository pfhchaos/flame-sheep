"""
Tests for tempo detection.
"""

import numpy as np
import pytest
from flame_sheep.tempo import (
    TempoTracker, TempoState,
    MIN_BPM, MAX_BPM, LOCK_THRESHOLD,
)


def simulate_beats(tracker: TempoTracker, bpm: float, count: int,
                   start: float = 0.0):
    """Simulate regular kick onsets at given BPM."""
    interval = 60.0 / bpm
    for i in range(count):
        t = start + i * interval
        tracker.process_onset('kick', t)


def simulate_random_onsets(tracker: TempoTracker, count: int,
                           min_gap: float = 0.2, max_gap: float = 0.8,
                           seed: int = 42, start: float = 0.0):
    """Simulate random-interval kick onsets."""
    import random
    random.seed(seed)

    t = start
    for _ in range(count):
        gap = random.uniform(min_gap, max_gap)
        t += gap
        tracker.process_onset('kick', t)


class TestTempoDetection:
    """Tests for tempo estimation accuracy."""

    @pytest.mark.parametrize('bpm', [90, 120, 150, 174])
    def test_detects_bpm(self, bpm):
        tracker = TempoTracker()
        simulate_beats(tracker, bpm, 30)
        assert _bpm_close(tracker.bpm, bpm), f"Expected ~{bpm} BPM, got {tracker.bpm}"


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
        simulate_random_onsets(tracker, 120, start=100.0)
        assert tracker.confidence < LOCK_THRESHOLD or not tracker.locked


class TestTrustedSource:

    def test_song_started_resets(self):
        tracker = TempoTracker()
        simulate_beats(tracker, 120, 40)
        assert tracker.locked
        tracker.song_started()
        assert not tracker.locked
        assert tracker.bpm == 0.0

    def test_tempo_hint_sets_hypothesis(self):
        tracker = TempoTracker()
        tracker.hint_tempo(120)
        assert tracker.bpm > 0, "hint should produce a nonzero bpm"
        assert _bpm_close(tracker.bpm, 120)

    def test_hint_locks_faster(self):
        tracker = TempoTracker()
        tracker.hint_tempo(120)
        simulate_beats(tracker, 120, 20)
        assert tracker.locked, "Should lock quickly with correct hint"


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


class TestHighTempo:
    """Tests for high BPM and octave error resistance."""

    def test_detects_200_bpm(self):
        tracker = TempoTracker()
        simulate_beats(tracker, 200, 40)
        assert _bpm_close(tracker.bpm, 200), f"Expected ~200 BPM, got {tracker.bpm}"

    def test_eighth_note_kicks_no_double_time(self):
        """Eighth-note kick pattern should still detect quarter-note tempo."""
        tracker = TempoTracker()
        # 174 BPM eighth notes = onsets every 172ms
        eighth_interval = 60.0 / 174 / 2
        for i in range(60):
            tracker.process_onset('kick', i * eighth_interval)
        assert _bpm_close(tracker.bpm, 174), \
            f"Expected ~174 BPM, got {tracker.bpm} (octave error?)"

    def test_dnb_kick_ghost_pattern(self):
        """DnB kick + ghost kick pattern should detect correct BPM."""
        tracker = TempoTracker()
        beat_interval = 60.0 / 160
        for bar in range(10):
            base = bar * 4 * beat_interval
            # Kick on beat 1, ghost on 'and' of 1
            tracker.process_onset('kick', base)
            tracker.process_onset('kick', base + beat_interval * 0.5)
            # Kick on beat 3, ghost on 'and' of 3
            tracker.process_onset('kick', base + 2 * beat_interval)
            tracker.process_onset('kick', base + 2.5 * beat_interval)
        assert tracker.bpm > 0, "Should estimate BPM from DnB pattern"
        assert _bpm_close(tracker.bpm, 160, tolerance=15), \
            f"Expected ~160 BPM, got {tracker.bpm}"

    def test_accelerando_follows(self):
        """Tracker should follow a tempo ramp without permanently breaking."""
        tracker = TempoTracker()
        time = 0.0
        bpm = 140.0
        for _ in range(200):
            tracker.process_onset('kick', time)
            time += 60.0 / bpm
            bpm = min(200, 140 + (time / 20.0) * 60)
        # After ramp stabilizes at 200, should eventually lock
        for i in range(40):
            tracker.process_onset('kick', time)
            time += 60.0 / 200
        assert _bpm_close(tracker.bpm, 200, tolerance=15), \
            f"Expected ~200 BPM after ramp, got {tracker.bpm}"


class TestEdgeCases:

    def test_empty_tracker(self):
        tracker = TempoTracker()
        assert tracker.bpm == 0.0
        assert tracker.confidence == 0.0
        assert not tracker.locked

    def test_single_onset(self):
        tracker = TempoTracker()
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


def _bpm_close(actual: float, expected: float, tolerance: float = 8.0,
               allow_octave: bool = False) -> bool:
    """Check if BPM is close to expected."""
    if allow_octave:
        for ratio in [0.5, 1.0, 2.0]:
            if abs(actual - expected * ratio) < tolerance:
                return True
        return False
    return abs(actual - expected) < tolerance
