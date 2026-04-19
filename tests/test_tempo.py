"""
Tests for tempo detection and rhythm coherence gating.
"""

import time
import pytest
from flame_sheep.tempo import (
    TempoTracker, TempoState,
    MIN_BPM, MAX_BPM, LOCK_THRESHOLD, LOCK_STABILITY
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
        simulate_beats(tracker, 120, 20)
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
            jitter = random.uniform(-0.02, 0.02)  # +/- 20ms
            tracker.process_onset('kick', t + jitter)
            t += interval
        # Should be close to 120 BPM despite jitter
        assert 115 <= tracker.bpm <= 125, f"Expected ~120 BPM with jitter, got {tracker.bpm}"


class TestConfidenceAndLocking:
    """Tests for confidence calculation and lock/unlock behavior."""

    def test_regular_beats_high_confidence(self):
        tracker = TempoTracker()
        simulate_beats(tracker, 120, 15)
        assert tracker.confidence >= LOCK_THRESHOLD, \
            f"Regular beats should have high confidence, got {tracker.confidence}"

    def test_random_intervals_low_confidence(self):
        tracker = TempoTracker()
        simulate_random_onsets(tracker, 30)
        assert tracker.confidence < LOCK_THRESHOLD, \
            f"Random intervals should have low confidence, got {tracker.confidence}"

    def test_locks_after_stable_confidence(self):
        tracker = TempoTracker()
        simulate_beats(tracker, 120, 20)
        assert tracker.locked, "Should be locked after 20 regular beats"

    def test_never_locks_on_random(self):
        tracker = TempoTracker()
        simulate_random_onsets(tracker, 50)
        assert not tracker.locked, "Should not lock on random intervals"

    def test_requires_stability_to_lock(self):
        """Must have LOCK_STABILITY consecutive high-confidence onsets to lock."""
        tracker = TempoTracker()
        # Not enough beats to build stability
        simulate_beats(tracker, 120, 8 + LOCK_STABILITY - 2)
        assert not tracker.locked, "Should not lock without enough stable onsets"
        # A few more should lock it
        simulate_beats(tracker, 120, 5, start=100.0)
        assert tracker.locked, "Should lock after enough stable onsets"

    def test_unlocks_when_confidence_drops(self):
        tracker = TempoTracker()
        # Lock on regular beats
        simulate_beats(tracker, 120, 20)
        assert tracker.locked
        # Feed random garbage
        simulate_random_onsets(tracker, 50, start=100.0)
        # May or may not unlock depending on how much history persists
        # At minimum, confidence should drop
        assert tracker.confidence < LOCK_THRESHOLD or not tracker.locked


class TestGating:
    """Tests for onset gating (pass/block behavior)."""

    def test_blocks_during_warmup(self):
        """Without trusted source, should block until confident."""
        tracker = TempoTracker()
        results = simulate_beats(tracker, 120, 8)
        # First 8 beats should be blocked (building history)
        assert not any(results), f"Should block during warmup, got {results}"

    def test_passes_after_confidence_builds(self):
        tracker = TempoTracker()
        results = simulate_beats(tracker, 120, 15)
        # After warmup, should start passing
        assert any(results[8:]), f"Should pass after confidence builds"

    def test_blocks_random_intervals(self):
        tracker = TempoTracker()
        results = simulate_random_onsets(tracker, 30)
        # Most/all should be blocked due to low confidence
        pass_count = sum(results)
        assert pass_count < 10, f"Should block most random onsets, passed {pass_count}/30"

    def test_filters_off_beat_when_locked(self):
        """Once locked, should only pass on-beat events."""
        tracker = TempoTracker()
        # Lock on 120 BPM
        simulate_beats(tracker, 120, 20)
        assert tracker.locked
        
        # Now send an off-beat onset (halfway between beats)
        last_beat = 19 * 0.5  # last beat time
        off_beat_time = last_beat + 0.25  # halfway to next beat
        passed = tracker.process_onset('kick', off_beat_time)
        assert not passed, "Off-beat onset should be blocked when locked"
        
        # On-beat should pass
        on_beat_time = last_beat + 0.5  # next beat
        passed = tracker.process_onset('kick', on_beat_time)
        assert passed, "On-beat onset should pass when locked"


class TestTrustedSource:
    """Tests for trusted source mode (music player integration)."""

    def test_song_started_enables_trusted(self):
        tracker = TempoTracker()
        tracker.song_started()
        # Should pass events immediately
        passed = tracker.process_onset('kick', 0.0)
        assert passed, "Should pass immediately in trusted mode"

    def test_tempo_hint_enables_trusted(self):
        tracker = TempoTracker()
        tracker.hint_tempo(120)
        # Should pass events immediately
        passed = tracker.process_onset('kick', 0.0)
        assert passed, "Should pass immediately after tempo hint"

    def test_trusted_bypasses_warmup(self):
        tracker = TempoTracker()
        tracker.song_started()
        results = simulate_beats(tracker, 120, 8)
        # All should pass in trusted mode
        assert all(results), f"Should pass all in trusted mode, got {results}"

    def test_tempo_hint_biases_detection(self):
        """Tempo hint should help lock faster."""
        tracker = TempoTracker()
        tracker.hint_tempo(120)
        simulate_beats(tracker, 120, 15)
        assert 115 <= tracker.bpm <= 125, f"Should detect hinted tempo"

    def test_reset_clears_trusted(self):
        tracker = TempoTracker()
        tracker.song_started()
        assert tracker._trusted_source
        tracker.reset()
        assert not tracker._trusted_source, "Reset should clear trusted mode"


class TestState:
    """Tests for TempoState dataclass."""

    def test_state_reflects_tracker(self):
        tracker = TempoTracker()
        simulate_beats(tracker, 120, 15)
        state = tracker.state
        assert isinstance(state, TempoState)
        assert state.bpm == tracker.bpm
        assert state.confidence == tracker.confidence
        assert state.locked == tracker.locked

    def test_phase_in_range(self):
        tracker = TempoTracker()
        simulate_beats(tracker, 120, 15)
        state = tracker.state
        assert 0.0 <= state.phase <= 1.0, f"Phase should be 0..1, got {state.phase}"


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_tracker(self):
        tracker = TempoTracker()
        assert tracker.bpm == 0.0
        assert tracker.confidence == 0.0
        assert not tracker.locked

    def test_single_onset(self):
        tracker = TempoTracker()
        tracker.process_onset('kick', 0.0)
        # Should not crash, should have zero confidence
        assert tracker.confidence == 0.0

    def test_respects_bpm_bounds(self):
        """Intervals outside MIN_BPM..MAX_BPM range should be ignored."""
        tracker = TempoTracker()
        # Very fast (> MAX_BPM) - 0.2s = 300 BPM
        for i in range(20):
            tracker.process_onset('kick', i * 0.2)
        # Should not detect a valid tempo since intervals are out of range
        # (or should clamp to MAX_BPM)
        assert tracker.bpm <= MAX_BPM or tracker.confidence < 0.5

    def test_reset_clears_all(self):
        tracker = TempoTracker()
        simulate_beats(tracker, 120, 20)
        assert tracker.locked
        tracker.reset()
        assert tracker.bpm == 0.0
        assert tracker.confidence == 0.0
        assert not tracker.locked
        assert len(tracker._ioi_history) == 0
