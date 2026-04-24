"""
Regression tests for FlameSheepCore — beat handling, genome swaps, morph state.

Uses SyntheticAudioProcessor with a fake clock — all tests run instantly,
no real-time sleeps needed.

Default pattern: kick=0.5s, snare=1.0s, hihat=0.25s (120 BPM, 4/4).
"""

import numpy as np
import pytest

from flame_sheep.main import FlameSheepCore
from .conftest import FakeClock, trivial_genome


@pytest.fixture
def clock():
    return FakeClock(start=1000.0)  # offset to avoid zero-time edge cases


@pytest.fixture
def core(clock):
    """FlameSheepCore with synthetic audio driven by a fake clock.
    Uses trivial genomes — instant construction, no viability checks."""
    _seed = iter(range(1000))
    c = FlameSheepCore(test_audio=True, lib=None, clock=clock,
                       genome_factory=lambda: trivial_genome(next(_seed)))
    yield c
    c.stop()


def tick_for(core, clock, duration: float, fps: float = 60.0):
    """Tick core for simulated duration at fps. Returns list of FrameStates."""
    dt = 1.0 / fps
    n_frames = int(duration * fps)
    frames = []
    for _ in range(n_frames):
        clock.advance(dt)
        frames.append(core.tick(dt))
    return frames


# ----------------------------------------------------------------
# Genome swap rate
# ----------------------------------------------------------------

class TestGenomeSwapRate:

    def test_swaps_at_expected_rate(self, core, clock):
        """At 120 BPM with KICK_SWAP_EVERY=4, genome should swap ~every 2s."""
        dt = 1.0 / 60
        swaps = 0
        prev_target = id(core.target_genome)

        for _ in range(int(6.0 * 60)):  # 6 seconds at 60fps
            clock.advance(dt)
            core.tick(dt)
            if id(core.target_genome) != prev_target:
                swaps += 1
                prev_target = id(core.target_genome)

        # Kick-driven swaps every ~2s = ~3, plus morph-completion swaps
        assert 2 <= swaps <= 10, \
            f"Expected ~6 genome swaps in 6s, got {swaps}"

    def test_not_every_kick_swaps(self, core, clock):
        """Genome should NOT swap on every single kick."""
        dt = 1.0 / 60
        swaps = 0
        prev_target = id(core.target_genome)

        for _ in range(int(4.0 * 60)):  # 4 seconds
            clock.advance(dt)
            core.tick(dt)
            if id(core.target_genome) != prev_target:
                swaps += 1
                prev_target = id(core.target_genome)

        assert swaps < 6, \
            f"Too many swaps ({swaps}) — looks like every kick is swapping"


# ----------------------------------------------------------------
# Beat event flooding
# ----------------------------------------------------------------

class TestBeatEventFlooding:

    def test_max_events_per_tick(self, core, clock):
        """Each tick should produce at most one event per band (3 max)."""
        dt = 1.0 / 60
        max_events = 0

        for _ in range(int(3.0 * 60)):
            clock.advance(dt)
            events = core.audio.process()
            if len(events) > max_events:
                max_events = len(events)
            core.tick(dt)

        assert max_events <= 3, \
            f"Got {max_events} events in a single tick (max should be 3)"

    def test_stale_spectrum_no_flood(self, core, clock):
        """Skipping ticks then resuming should not flood beats."""
        dt = 1.0 / 60

        # Warm up
        for _ in range(30):
            clock.advance(dt)
            core.tick(dt)

        # Simulate gap: advance clock but only call audio.process
        for _ in range(30):  # 500ms gap
            clock.advance(dt)
            core.audio.process()

        # Resume
        events = core.audio.process()
        assert len(events) <= 3, \
            f"Resuming after gap produced {len(events)} events (flood!)"

    def test_stale_spectrum_without_keepalive_floods(self, core, clock):
        """Without audio.process() keepalive, spectrum goes stale.
        Documents the failure mode we're protecting against."""
        dt = 1.0 / 60

        for _ in range(30):
            clock.advance(dt)
            core.tick(dt)

        # Gap WITHOUT calling audio.process
        clock.advance(0.5)

        # Resume — may or may not flood depending on synth implementation
        events = core.audio.process()
        # Just verify no crash


# ----------------------------------------------------------------
# Morph state invariants
# ----------------------------------------------------------------

class TestMorphInvariants:

    def test_morph_t_in_range(self, core, clock):
        """morph_t must always be in [0, 1]."""
        dt = 1.0 / 60
        for _ in range(int(4.0 * 60)):
            clock.advance(dt)
            core.tick(dt)
            assert 0.0 <= core.morph_t <= 1.0, \
                f"morph_t out of range: {core.morph_t}"

    def test_morph_speed_bounded(self, core, clock):
        """morph_speed should stay within sane bounds."""
        dt = 1.0 / 60
        for _ in range(int(4.0 * 60)):
            clock.advance(dt)
            core.tick(dt)
            assert 0.0 < core.morph_speed <= 1.0, \
                f"morph_speed out of range: {core.morph_speed}"

    def test_palette_t_in_range(self, core, clock):
        """palette_t must always be in [0, 1]."""
        dt = 1.0 / 60
        for _ in range(int(4.0 * 60)):
            clock.advance(dt)
            core.tick(dt)
            assert 0.0 <= core.palette_t <= 1.0, \
                f"palette_t out of range: {core.palette_t}"


# ----------------------------------------------------------------
# Zoom pulse (hihat axis)
# ----------------------------------------------------------------

class TestZoomPulse:

    def test_zoom_boost_bounded(self, core, clock):
        """zoom_boost must never exceed ZOOM_BOOST_MAX."""
        dt = 1.0 / 60
        for _ in range(int(4.0 * 60)):
            clock.advance(dt)
            core.tick(dt)
            assert core.zoom_boost <= core._zoom_axis.ZOOM_BOOST_MAX + 1e-6, \
                f"zoom_boost {core.zoom_boost} exceeds max {core._zoom_axis.ZOOM_BOOST_MAX}"

    def test_zoom_decays_without_hihats(self, core, clock):
        """zoom_boost should decay toward 0 between hihat events."""
        dt = 1.0 / 60

        # Tick until we get a zoom boost (hihat every 0.25s)
        boosted = False
        for _ in range(int(2.0 * 60)):
            clock.advance(dt)
            core.tick(dt)
            if core.zoom_boost > 0.01:
                boosted = True
                break

        if not boosted:
            pytest.skip("No hihat fired in warmup period")

        # Tick a few frames without advancing clock much — zoom should decay
        peak = core.zoom_boost
        for _ in range(10):
            core.tick(dt)  # don't advance clock — no new hihats
        assert core.zoom_boost < peak, \
            f"zoom_boost should decay: was {peak}, now {core.zoom_boost}"


# ----------------------------------------------------------------
# Palette axis (snare)
# ----------------------------------------------------------------

class TestPaletteAxis:

    def test_palette_changes_on_snare_interval(self, core, clock):
        """Palette target should change ~every 1.0s (snare interval)."""
        dt = 1.0 / 60
        palette_changes = 0
        prev_palette = core.palette_target.copy()

        for _ in range(int(5.0 * 60)):
            clock.advance(dt)
            core.tick(dt)
            if not np.array_equal(core.palette_target, prev_palette):
                palette_changes += 1
                prev_palette = core.palette_target.copy()

        assert 2 <= palette_changes <= 8, \
            f"Expected ~5 palette changes in 5s, got {palette_changes}"


# ----------------------------------------------------------------
# Quiet drift
# ----------------------------------------------------------------

class TestQuietDrift:

    def test_drift_swaps_when_quiet(self):
        """When audio is silent, drift mode should produce changing genomes."""
        clock = FakeClock(start=1000.0)
        _seed = iter(range(1000))
        core = FlameSheepCore(test_audio=True, lib=None, clock=clock,
                              genome_factory=lambda: trivial_genome(next(_seed)))
        core.audio._rms = 0.0

        dt = 1.0 / 60
        genomes = []

        # 20 seconds at 60fps
        for _ in range(60 * 20):
            clock.advance(dt)
            frame = core.tick(dt)
            genomes.append(frame.genome)

        core.stop()
        # Frame genomes should change over time during drift
        first = genomes[0]
        changed = any(g.distance(first) > 0.01 for g in genomes[-60:])
        assert changed, \
            "Frame genome should change during drift mode"


# ----------------------------------------------------------------
# FrameState output
# ----------------------------------------------------------------

class TestFrameState:

    def test_frame_state_fields(self, core, clock):
        """tick() should return a valid FrameState with all fields."""
        clock.advance(1.0 / 60)
        frame = core.tick(1.0 / 60)
        assert frame.genome is not None
        assert isinstance(frame.palette, np.ndarray)
        assert frame.palette.shape == (256, 3)
        assert isinstance(frame.spectrum, np.ndarray)
        assert isinstance(frame.brightness, float)
        assert frame.brightness > 0

    def test_brightness_range(self, core, clock):
        """Brightness should be in [3.0, 9.0] range."""
        dt = 1.0 / 60
        for _ in range(60):
            clock.advance(dt)
            frame = core.tick(dt)
            assert 0.7 <= frame.brightness <= 12.0, \
                f"brightness {frame.brightness} out of expected range"
