"""
Tests for tempo detection and rhythm coherence gating.
"""

import time
import pytest
from flame_sheep.tempo import (
    TempoTracker, TempoState,
    MIN_BPM, MAX_BPM, GATE_THRESHOLD,
)


def simulate_beats(tracker: TempoTracker, bpm: float, count: int, start: float = 0.0) -> list[bool]:
    """Simulate beats at a given BPM, return list of pass/block results."""
    interval = 60.0 / bpm
    results = []
    for i in range(count):
        t = start + i * interval
        passed = tracker.process_onset('kick', t)
        results.append(passed)
    return results


def simulate_random_onsets(tracker: TempoTracker, count: int,
                           min_gap: float = 0.2, max_gap: float = 0.8,
                           seed: int = 42, start: float = 0.0) -> list[bool]:
    """Simulate random-interval onsets (like speech), return pass/block results."""
    import random
    random.seed(seed)
    results = []
    t = start
    for _ in range(count):
        t += random.uniform(min_gap, max_gap)
        passed = tracker.process_onset('kick', t)
        results.append(passed)
    return results


class TestTempoDetection:
    """Tests for tempo estimation accuracy."""

    def test_detects_120_bpm(self):
        tracker = TempoTracker()
        simulate_beats(tracker, 120, 30)
        assert 115 <= tracker.bpm <= 125, f"Expected ~120 BPM, got {tracker.bpm}"

    def test_detects_90_bpm(self):
        tracker = TempoTracker()
        simulate_beats(tracker, 90, 20)
        assert 85 <= tracker.bpm <= 95, f"Expected ~90 BPM, got {tracker.bpm}"

    def test_detects_150_bpm(self):
        tracker = TempoTracker()
        simulate_beats(tracker, 150, 20)
        assert 145 <= tracker.bpm <= 155, f"Expected ~150 BPM, got {tracker.bpm}"

    def test_handles_jittery_timing(self):
        """Should still detect tempo with human-like timing variation."""
        import random
        random.seed(123)
        tracker = TempoTracker()
        interval = 0.5  # 120 BPM
        t = 0.0
        for _ in range(25):
            jitter = random.uniform(-0.02, 0.02)
            tracker.process_onset('kick', t + jitter)
            t += interval
        assert 115 <= tracker.bpm <= 125, f"Expected ~120 BPM with jitter, got {tracker.bpm}"


class TestConfidenceAndLocking:
    """Tests for confidence and lock/unlock behavior."""

    def test_regular_beats_build_confidence(self):
        tracker = TempoTracker()
        simulate_beats(tracker, 120, 25)
        assert tracker.confidence >= GATE_THRESHOLD, \
            f"Regular beats should build confidence above gate threshold, got {tracker.confidence}"

    def test_random_intervals_stay_low(self):
        tracker = TempoTracker()
        simulate_random_onsets(tracker, 60)
        assert not tracker.locked, "Should not lock on random intervals"

    def test_locks_on_regular_beats(self):
        tracker = TempoTracker()
        simulate_beats(tracker, 120, 30)
        assert tracker.locked, "Should be locked after 20 regular beats"

    def test_unlocks_on_garbage(self):
        tracker = TempoTracker()
        simulate_beats(tracker, 120, 30)
        assert tracker.locked
        # Feed random garbage — confidence should drop
        simulate_random_onsets(tracker, 80, start=100.0)
        assert tracker.confidence < GATE_THRESHOLD or not tracker.locked


class TestGating:
    """Tests for onset gating (pass/block behavior)."""

    def test_passes_during_learning(self):
        """All events pass while building initial hypothesis."""
        tracker = TempoTracker()
        results = simulate_beats(tracker, 120, 8)
        # Should pass everything during learning phase
        assert all(results), f"Should pass during learning, got {results}"

    def test_gates_off_grid_kicks_when_locked(self):
        """Once locked, kicks not on any grid subdivision should be blocked."""
        tracker = TempoTracker()
        simulate_beats(tracker, 120, 30)
        assert tracker.locked

        # Off-grid kick (between 4x subdivision lines)
        # At 120 BPM, period=0.5s, subdivisions at 0, 0.125, 0.25, 0.375, 0.5
        # Put it right between 0.125 and 0.25
        last_beat = 19 * 0.5
        off_grid_time = last_beat + 0.1875  # halfway between subdivisions
        passed = tracker.process_onset('kick', off_grid_time)
        assert not passed, "Off-grid kick should be blocked when locked"

    def test_passes_on_beat_kicks_when_locked(self):
        tracker = TempoTracker()
        simulate_beats(tracker, 120, 30)
        assert tracker.locked

        last_beat = 19 * 0.5
        on_beat_time = last_beat + 0.5
        passed = tracker.process_onset('kick', on_beat_time)
        assert passed, "On-beat kick should pass when locked"

    def test_passes_subdivision_kicks_when_locked(self):
        """Kicks on grid subdivisions (2x) should pass."""
        tracker = TempoTracker()
        simulate_beats(tracker, 120, 30)
        assert tracker.locked

        # Use the tracker's own period for accurate subdivision
        half_period = tracker._beat_period / 2
        last = tracker._last_beat_time
        passed = tracker.process_onset('kick', last + half_period)
        assert passed, "Half-beat kick should pass"

    def test_snare_hihat_always_pass(self):
        """Snare and hihat are never gated — they're used for estimation."""
        tracker = TempoTracker()
        simulate_beats(tracker, 120, 30)
        assert tracker.locked

        last_beat = 19 * 0.5
        off_beat = last_beat + 0.25
        assert tracker.process_onset('snare', off_beat), "Snare should always pass"
        assert tracker.process_onset('hihat', off_beat), "Hihat should always pass"


class TestTrustedSource:
    """Tests for trusted source mode (music player integration)."""

    def test_song_started_enables_trusted(self):
        tracker = TempoTracker()
        tracker.song_started()
        passed = tracker.process_onset('kick', 0.0)
        assert passed, "Should pass immediately in trusted mode"

    def test_tempo_hint_sets_hypothesis(self):
        tracker = TempoTracker()
        tracker.hint_tempo(120)
        assert 115 <= tracker.bpm <= 125
        assert tracker._has_hypothesis

    def test_trusted_bypasses_learning(self):
        tracker = TempoTracker()
        tracker.song_started()
        results = simulate_beats(tracker, 120, 8)
        assert all(results), f"Should pass all in trusted mode, got {results}"

    def test_hint_locks_faster(self):
        """Tempo hint should lead to faster locking."""
        tracker = TempoTracker()
        tracker.hint_tempo(120)
        simulate_beats(tracker, 120, 15)
        assert tracker.locked, "Should lock quickly with correct hint"

    def test_reset_clears_trusted(self):
        tracker = TempoTracker()
        tracker.song_started()
        assert tracker._trusted_source
        tracker.reset()
        assert not tracker._trusted_source


class TestState:
    """Tests for TempoState dataclass."""

    def test_state_reflects_tracker(self):
        tracker = TempoTracker()
        simulate_beats(tracker, 120, 30)
        state = tracker.state
        assert isinstance(state, TempoState)
        assert state.bpm == tracker.bpm
        assert state.confidence == tracker.confidence
        assert state.locked == tracker.locked

    def test_phase_in_range(self):
        tracker = TempoTracker()
        simulate_beats(tracker, 120, 30)
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
        tracker.process_onset('kick', 0.0)
        assert tracker.confidence == 0.0

    def test_reset_clears_all(self):
        tracker = TempoTracker()
        simulate_beats(tracker, 120, 30)
        assert tracker.locked
        tracker.reset()
        assert tracker.bpm == 0.0
        assert tracker.confidence == 0.0
        assert not tracker.locked
        assert len(tracker._ioi_history) == 0
