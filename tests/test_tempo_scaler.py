"""Tests for the sigmoid tempo scaler.

Covers sigmoid shape, scale() blending, beats_to_frames conversion,
and alpha_for_beats EMA calculation.
"""

import math

import pytest

from flame_sheep.audio.tempo_scaler import TempoScaler, _FRAMES_PER_SECOND


class TestSigmoid:
    """Verify sigmoid shape properties."""

    def test_midpoint_is_half(self):
        s = TempoScaler(midpoint=120)
        assert abs(s.sigmoid(120) - 0.5) < 1e-6

    def test_low_bpm_near_zero(self):
        s = TempoScaler(midpoint=120, steepness=0.03)
        assert s.sigmoid(0) < 0.05

    def test_high_bpm_near_one(self):
        s = TempoScaler(midpoint=120, steepness=0.03)
        assert s.sigmoid(300) > 0.95

    def test_monotonically_increasing(self):
        s = TempoScaler()
        vals = [s.sigmoid(bpm) for bpm in range(40, 300, 10)]
        for i in range(1, len(vals)):
            assert vals[i] > vals[i-1]

    def test_steepness_controls_sharpness(self):
        gentle = TempoScaler(steepness=0.01)
        sharp = TempoScaler(steepness=0.1)
        # At 60 BPM (below midpoint), sharper steepness = lower value
        assert sharp.sigmoid(60) < gentle.sigmoid(60)
        # At 180 BPM (above midpoint), sharper steepness = higher value
        assert sharp.sigmoid(180) > gentle.sigmoid(180)


class TestScale:
    """Test slow/fast value blending."""

    def test_midpoint_is_average(self):
        s = TempoScaler(midpoint=120)
        result = s.scale(120, slow_val=10.0, fast_val=20.0)
        assert abs(result - 15.0) < 0.01

    def test_slow_tempo_returns_slow_val(self):
        s = TempoScaler(midpoint=120, steepness=0.03)
        result = s.scale(0, slow_val=10.0, fast_val=20.0)
        assert abs(result - 10.0) < 0.5

    def test_fast_tempo_returns_fast_val(self):
        s = TempoScaler(midpoint=120, steepness=0.03)
        result = s.scale(300, slow_val=10.0, fast_val=20.0)
        assert abs(result - 20.0) < 0.5

    def test_steepness_override(self):
        s = TempoScaler(midpoint=120, steepness=0.03)
        # Very steep: 60 BPM should be near slow_val
        result = s.scale(60, slow_val=0.0, fast_val=1.0, steepness=0.1)
        assert result < 0.01

    def test_inverted_range(self):
        """slow_val > fast_val should work (e.g. cooldown shrinks with tempo)."""
        s = TempoScaler(midpoint=120)
        result = s.scale(120, slow_val=20.0, fast_val=4.0)
        assert abs(result - 12.0) < 0.1


class TestBeatsToFrames:
    """Test beat-duration to frame conversion."""

    def test_120bpm_one_beat(self):
        s = TempoScaler()
        frames = s.beats_to_frames(120, beats=1.0)
        # 1 beat at 120 BPM = 0.5s
        expected = int(0.5 * _FRAMES_PER_SECOND)
        assert frames == expected

    def test_60bpm_one_beat(self):
        s = TempoScaler()
        frames = s.beats_to_frames(60, beats=1.0)
        expected = int(1.0 * _FRAMES_PER_SECOND)
        assert frames == expected

    def test_faster_tempo_fewer_frames(self):
        s = TempoScaler()
        slow = s.beats_to_frames(60, beats=1.0)
        fast = s.beats_to_frames(240, beats=1.0)
        assert fast < slow

    def test_zero_bpm_uses_midpoint(self):
        s = TempoScaler(midpoint=120)
        result = s.beats_to_frames(0, beats=1.0)
        expected = s.beats_to_frames(120, beats=1.0)
        assert result == expected

    def test_minimum_one_frame(self):
        s = TempoScaler()
        # Extremely fast tempo, tiny beat fraction
        assert s.beats_to_frames(9999, beats=0.001) >= 1


class TestAlphaForBeats:
    """Test EMA alpha calculation from beat duration."""

    def test_range_bounds(self):
        s = TempoScaler()
        for bpm in [30, 60, 120, 240, 400]:
            alpha = s.alpha_for_beats(bpm, beats=2.0)
            assert 0.5 <= alpha <= 0.999

    def test_faster_tempo_lower_alpha(self):
        """Faster tempo = shorter beats = fewer frames = lower alpha (faster decay)."""
        s = TempoScaler()
        slow_alpha = s.alpha_for_beats(60, beats=2.0)
        fast_alpha = s.alpha_for_beats(240, beats=2.0)
        assert fast_alpha < slow_alpha

    def test_more_beats_higher_alpha(self):
        """More beats of memory = more frames = higher alpha (slower decay)."""
        s = TempoScaler()
        short = s.alpha_for_beats(120, beats=1.0)
        long = s.alpha_for_beats(120, beats=4.0)
        assert long > short

    def test_alpha_clamped_at_extremes(self):
        s = TempoScaler()
        # Very fast, very short — should hit floor
        alpha = s.alpha_for_beats(400, beats=0.1)
        assert alpha >= 0.5
        # Very slow, very long — should hit ceiling
        alpha = s.alpha_for_beats(30, beats=16.0)
        assert alpha <= 0.999
