"""Tests for PaletteAxis — snare-driven palette morphing.

Covers morph progression, snare event handling, density scaling,
and palette selection behavior.
"""

import numpy as np
import pytest

from flame_sheep.audio._types import BeatEvent, BandState, AudioState
from flame_sheep.axes.palette_axis import PaletteAxis
from flame_sheep.genome import _random_palette


def _make_palette(seed=0):
    return _random_palette(np.random.default_rng(seed))


def _audio(events=None, snare_density=0.0, **kw):
    bands = {name: BandState() for name in
             ('subbass', 'kick', 'snare', 'hihat')}
    bands['snare'] = BandState(onset_density=snare_density)
    return AudioState(events=events or [], bands=bands, **kw)


class TestPaletteMorph:
    """Verify basic palette interpolation."""

    def test_initial_state(self):
        p = _make_palette()
        axis = PaletteAxis(p)
        assert axis.palette_t == 0.0
        np.testing.assert_array_equal(axis.palette_current, p)

    def test_morph_advances(self):
        axis = PaletteAxis(_make_palette())
        audio = _audio()
        for _ in range(10):
            axis.tick(audio, dt=1/60, clock=0.0)
        assert axis.palette_t > 0

    def test_morph_wraps_at_1(self):
        """palette_t should reset to 0 after reaching 1.0."""
        axis = PaletteAxis(_make_palette())
        axis.palette_speed = 0.5  # fast enough to complete quickly
        audio = _audio()
        for _ in range(100):
            axis.tick(audio, dt=1/60, clock=0.0)
        # Should have wrapped at least once — t is back near 0
        assert axis.palette_t < 1.0

    def test_contribute_produces_interpolated_palette(self):
        p1 = _make_palette(0)
        p2 = _make_palette(1)
        axis = PaletteAxis(p1)
        axis.palette_target = p2
        axis.palette_t = 0.5

        class Frame:
            palette = None
        frame = Frame()
        axis.contribute(frame)
        assert frame.palette is not None
        assert frame.palette.shape == p1.shape
        # Should be between p1 and p2 (not exactly either)
        assert not np.array_equal(frame.palette, p1)
        assert not np.array_equal(frame.palette, p2)


class TestSnareEvents:
    """Verify snare events trigger palette changes."""

    def test_snare_resets_morph(self):
        axis = PaletteAxis(_make_palette())
        # Advance morph a bit
        axis.palette_t = 0.5
        axis.palette_speed = 0.001

        snare = BeatEvent(kind='snare', energy=0.7)
        audio = _audio(events=[snare])
        axis.tick(audio, dt=1/60, clock=0.0)

        # Snare should reset t to 0 and boost speed
        assert axis.palette_t == 0.0
        assert axis.palette_speed > 0.001

    def test_snare_speed_scales_with_energy(self):
        low_energy = BeatEvent(kind='snare', energy=0.2)
        high_energy = BeatEvent(kind='snare', energy=0.9)

        axis_low = PaletteAxis(_make_palette())
        axis_low.tick(_audio(events=[low_energy]), dt=1/60, clock=0.0)

        axis_high = PaletteAxis(_make_palette())
        axis_high.tick(_audio(events=[high_energy]), dt=1/60, clock=0.0)

        assert axis_high.palette_speed > axis_low.palette_speed

    def test_non_snare_events_ignored(self):
        axis = PaletteAxis(_make_palette())
        axis.palette_t = 0.3
        original_t = axis.palette_t

        kick = BeatEvent(kind='kick', energy=0.9)
        audio = _audio(events=[kick])
        axis.tick(audio, dt=1/60, clock=0.0)

        # Kick should not reset palette — only snare does
        assert axis.palette_t > 0  # advanced slightly, but not reset

    def test_speed_decays_toward_drift(self):
        axis = PaletteAxis(_make_palette())
        axis.palette_speed = 0.1  # boosted from a snare
        audio = _audio()
        speeds = []
        for _ in range(50):
            axis.tick(audio, dt=1/60, clock=0.0)
            speeds.append(axis.palette_speed)
        # Speed should decrease over time
        assert speeds[-1] < speeds[0]


class TestDensityScaling:
    """Verify onset density dampens snare response."""

    def test_high_density_reduces_speed(self):
        snare = BeatEvent(kind='snare', energy=0.7)

        axis_low_density = PaletteAxis(_make_palette())
        axis_low_density.tick(
            _audio(events=[snare], snare_density=0.0),
            dt=1/60, clock=0.0)
        speed_low = axis_low_density.palette_speed

        axis_high_density = PaletteAxis(_make_palette())
        axis_high_density.tick(
            _audio(events=[snare], snare_density=10.0),
            dt=1/60, clock=0.0)
        speed_high = axis_high_density.palette_speed

        assert speed_high < speed_low


class TestPaletteHistory:
    """Verify palette ID history tracking."""

    def test_history_starts_empty(self):
        axis = PaletteAxis(_make_palette())
        assert axis.palette_history == []

    def test_history_max_size(self):
        axis = PaletteAxis(_make_palette())
        for i in range(20):
            axis._track_palette(i)
        assert len(axis.palette_history) <= PaletteAxis.PALETTE_HISTORY_SIZE

    def test_no_consecutive_duplicates(self):
        axis = PaletteAxis(_make_palette())
        axis._track_palette(1)
        axis._track_palette(1)
        axis._track_palette(1)
        assert axis.palette_history == [1]

    def test_non_consecutive_duplicates_allowed(self):
        axis = PaletteAxis(_make_palette())
        axis._track_palette(1)
        axis._track_palette(2)
        axis._track_palette(1)
        assert axis.palette_history == [1, 2, 1]
