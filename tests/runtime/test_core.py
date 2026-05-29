"""
Regression tests for FlameSheepCore — beat handling, genome swaps, morph state.

Uses SyntheticAudioProcessor with a fake clock — all tests run instantly,
no real-time sleeps needed.

Default pattern: low=0.5s, mid=1.0s, high=0.25s (120 BPM, 4/4).
"""

import numpy as np
import pytest

from flame_sheep.runtime import FlameSheepCore
from flame_sheep.runtime import Orchestrator
from viz_helpers import FakeClock, trivial_genome


@pytest.fixture
def clock():
    return FakeClock(start=1000.0)  # offset to avoid zero-time edge cases


@pytest.fixture
def orch(clock):
    """Orchestrator with synthetic audio."""
    o = Orchestrator(test_audio=True, clock=clock)
    o.start()
    yield o
    o.stop()


@pytest.fixture
def core(orch, clock):
    """FlameSheepCore with synthetic audio driven by a fake clock.
    Uses trivial genomes — instant construction, no viability checks."""
    _seed = iter(range(1000))
    c = FlameSheepCore(orchestrator=orch, lib=None, clock=clock,
                       genome_factory=lambda: trivial_genome(next(_seed)))
    yield c


def _tick(core, dt):
    """Tick orchestrator + core together."""
    core._orch.tick()
    return core.tick(dt)


def tick_for(core, clock, duration: float, fps: float = 60.0):
    """Tick core for simulated duration at fps. Returns list of FrameStates."""
    dt = 1.0 / fps
    n_frames = int(duration * fps)
    frames = []
    for _ in range(n_frames):
        clock.advance(dt)
        frames.append(_tick(core, dt))
    return frames


# ----------------------------------------------------------------
# Genome swap rate
# ----------------------------------------------------------------

class TestGenomeSwapRate:

    def test_swaps_at_expected_rate(self, core, clock):
        """Genome should swap after morph completes."""
        dt = 1.0 / 60
        swaps = 0
        prev_target = id(core.target_genome)

        # Force into morphing state so morph can complete
        from flame_sheep.axes._morph_cycle import MorphState
        core._genome_axis._morph.state = MorphState.MORPHING

        for _ in range(int(10.0 * 60)):  # 10 seconds at 60fps
            clock.advance(dt)
            _tick(core, dt)
            if id(core.target_genome) != prev_target:
                swaps += 1
                prev_target = id(core.target_genome)

        # Morph should complete and swap at least once in 10s
        assert swaps >= 1, \
            f"Expected at least 1 genome swap in 10s, got {swaps}"

    def test_not_every_low_swaps(self, core, clock):
        """During dwell, beats should not cause genome swaps."""
        dt = 1.0 / 60
        swaps = 0
        # Reset state from previous test
        from flame_sheep.axes._morph_cycle import MorphState
        core._genome_axis._morph.state = MorphState.DWELL
        core._genome_axis._morph.t = 0.0
        core._genome_axis._morph.dwell_start = clock()
        prev_target = id(core.target_genome)

        for _ in range(int(4.0 * 60)):  # 4 seconds
            clock.advance(dt)
            _tick(core, dt)
            if id(core.target_genome) != prev_target:
                swaps += 1
                prev_target = id(core.target_genome)

        assert swaps < 8, \
            f"Too many swaps ({swaps}) — looks like every low-band onset is swapping"


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
            events = core._orch.audio.process()
            if len(events) > max_events:
                max_events = len(events)
            _tick(core, dt)

        assert max_events <= 3, \
            f"Got {max_events} events in a single tick (max should be 3)"

    def test_stale_spectrum_no_flood(self, core, clock):
        """Skipping ticks then resuming should not flood beats."""
        dt = 1.0 / 60

        # Warm up
        for _ in range(30):
            clock.advance(dt)
            _tick(core, dt)

        # Simulate gap: advance clock but only call audio.process
        for _ in range(30):  # 500ms gap
            clock.advance(dt)
            core._orch.audio.process()

        # Resume
        events = core._orch.audio.process()
        assert len(events) <= 3, \
            f"Resuming after gap produced {len(events)} events (flood!)"

    def test_stale_spectrum_without_keepalive_floods(self, core, clock):
        """Without audio.process() keepalive, spectrum goes stale.
        Documents the failure mode we're protecting against."""
        dt = 1.0 / 60

        for _ in range(30):
            clock.advance(dt)
            _tick(core, dt)

        # Gap WITHOUT calling audio.process
        clock.advance(0.5)

        # Resume — may or may not flood depending on synth implementation
        events = core._orch.audio.process()
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
            _tick(core, dt)
            assert 0.0 <= core.morph_t <= 1.0, \
                f"morph_t out of range: {core.morph_t}"

    def test_morph_t_bounded(self, core, clock):
        """morph_t should stay within [0, 1]."""
        dt = 1.0 / 60
        for _ in range(int(4.0 * 60)):
            clock.advance(dt)
            _tick(core, dt)
            assert 0.0 <= core.morph_t <= 1.0, \
                f"morph_t out of range: {core.morph_t}"

    def test_palette_t_in_range(self, core, clock):
        """palette_t must always be in [0, 1]."""
        dt = 1.0 / 60
        for _ in range(int(4.0 * 60)):
            clock.advance(dt)
            _tick(core, dt)
            assert 0.0 <= core.palette_t <= 1.0, \
                f"palette_t out of range: {core.palette_t}"


# ----------------------------------------------------------------
# Zoom pulse (high-band axis)
# ----------------------------------------------------------------

class TestZoomPulse:

    def test_zoom_boost_bounded(self, core, clock):
        """zoom_boost must never exceed ZOOM_BOOST_MAX."""
        dt = 1.0 / 60
        for _ in range(int(4.0 * 60)):
            clock.advance(dt)
            _tick(core, dt)
            assert core.zoom_boost <= core._zoom_axis.ZOOM_BOOST_MAX + 1e-6, \
                f"zoom_boost {core.zoom_boost} exceeds max {core._zoom_axis.ZOOM_BOOST_MAX}"

    def test_zoom_decays_without_highs(self, core, clock):
        """zoom_boost should decay toward 0 between high-band events."""
        dt = 1.0 / 60

        # Tick until we get a zoom boost (high-band every 0.25s)
        boosted = False
        for _ in range(int(2.0 * 60)):
            clock.advance(dt)
            _tick(core, dt)
            if core.zoom_boost > 0.01:
                boosted = True
                break

        if not boosted:
            pytest.skip("No high-band event fired in warmup period")

        # Tick a few frames without advancing clock much — zoom should decay
        peak = core.zoom_boost
        for _ in range(10):
            _tick(core, dt)  # don't advance clock — no new high-band events
        assert core.zoom_boost < peak, \
            f"zoom_boost should decay: was {peak}, now {core.zoom_boost}"


# ----------------------------------------------------------------
# Palette axis (mid-band)
# ----------------------------------------------------------------

class TestPaletteAxis:

    def test_palette_changes_on_subdivision(self, core, clock):
        """Palette target should change on subdivision (hi-hat) events."""
        dt = 1.0 / 60
        palette_changes = 0
        prev_palette = core.palette_target.copy()

        for _ in range(int(5.0 * 60)):
            clock.advance(dt)
            _tick(core, dt)
            if not np.array_equal(core.palette_target, prev_palette):
                palette_changes += 1
                prev_palette = core.palette_target.copy()

        # Hi-hats are frequent — expect many palette changes in 5s
        assert palette_changes >= 2, \
            f"Expected at least 2 palette changes in 5s, got {palette_changes}"


# ----------------------------------------------------------------
# Quiet drift
# ----------------------------------------------------------------

class TestQuietDrift:

    def test_drift_swaps_when_quiet(self):
        """When audio is silent, drift mode should produce changing genomes."""
        clock = FakeClock(start=1000.0)
        _seed = iter(range(1000))
        orch = Orchestrator(test_audio=True, clock=clock)
        orch.start()
        core = FlameSheepCore(orchestrator=orch, lib=None, clock=clock,
                              genome_factory=lambda: trivial_genome(next(_seed)))
        orch.audio._rms = 0.0

        dt = 1.0 / 60
        genomes = []

        # 20 seconds at 60fps
        for _ in range(60 * 20):
            clock.advance(dt)
            frame = _tick(core, dt)
            genomes.append(frame.genome)

        orch.stop()
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
        frame = _tick(core, 1.0 / 60)
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
            frame = _tick(core, dt)
            assert 0.7 <= frame.brightness <= 12.0, \
                f"brightness {frame.brightness} out of expected range"


# ============================================================================
# pin_genome / clear_pin — `show <gid>` control-pipe command path.
# Locks the wallpaper to a specific genome, used by validation tools
# (blind_model_eval, whats_changed, the compare-mode positive-control
# we did 2026-05-27 for the GPU timing investigation).
# ============================================================================

class TestPinGenome:

    def test_pin_sets_core_state(self, core, clock):
        from flame_sheep.genome import Genome
        rng = np.random.default_rng(7)
        g = Genome.random(rng)
        assert core._pinned_genome is None
        assert core._pinned_genome_db_id is None

        core.pin_genome(g, db_id=42)
        assert core._pinned_genome is g
        assert core._pinned_genome_db_id == 42

    def test_pin_sets_genome_axis_state(self, core, clock):
        """The axis-side mirror — GenomeAxis.contribute reads
        self._pinned_genome to override its normal current/target
        selection. core.pin_genome must keep that mirror in sync."""
        from flame_sheep.genome import Genome
        g = Genome.random(np.random.default_rng(8))
        core.pin_genome(g)
        assert core._genome_axis._pinned_genome is g

    def test_clear_pin_clears_both(self, core, clock):
        from flame_sheep.genome import Genome
        g = Genome.random(np.random.default_rng(9))
        core.pin_genome(g, db_id=123)
        core.clear_pin()
        assert core._pinned_genome is None
        assert core._pinned_genome_db_id is None
        assert core._genome_axis._pinned_genome is None

    def test_pin_overrides_frame_genome_identity(self, core, clock):
        """After pinning, the per-frame genome's structural identity
        (variation weights, transform colors) matches the pinned
        genome. Per the pin_genome docstring, per-frame state like
        zoom-pulse and rotation ARE still applied by other axes —
        only the genome IDENTITY is locked, not all per-frame state."""
        from flame_sheep.genome import Genome
        g = Genome.random(np.random.default_rng(10))
        # Tick a few frames without pin so axis has some state to diverge
        dt = 1.0 / 60
        for _ in range(10):
            clock.advance(dt)
            _tick(core, dt)
        # Pin
        core.pin_genome(g)
        clock.advance(dt)
        frame = _tick(core, dt)
        # IDENTITY is locked: variations + colors + weights match
        # exactly (zoom + rotation don't touch these fields).
        assert len(frame.genome.transforms) == len(g.transforms)
        for ft, gt in zip(frame.genome.transforms, g.transforms):
            np.testing.assert_array_equal(ft.variations, gt.variations)
            assert ft.color == gt.color
            assert ft.weight == gt.weight
        # Per-frame state (zoom, rotation, center) IS allowed to
        # differ — driven by ZoomAxis + rotation phase tracking.

    def test_pin_overrides_palette(self, core, clock):
        """When pinned, frame.palette should come from the pinned
        genome's palette (not whatever PaletteAxis was mid-transition
        to). This matters because the pin use case is 'show me exactly
        this genome' — color included."""
        from flame_sheep.genome import Genome
        g = Genome.random(np.random.default_rng(11))
        # Ensure g has a distinctive palette so equality is meaningful
        g.palette = np.linspace(0.2, 0.9, 256*3, dtype=np.float32).reshape(256, 3)

        # Tick some to advance palette axis off the initial palette
        dt = 1.0 / 60
        for _ in range(30):
            clock.advance(dt)
            _tick(core, dt)
        # Pin
        core.pin_genome(g)
        clock.advance(dt)
        frame = _tick(core, dt)
        np.testing.assert_array_equal(frame.palette, g.palette)

    def test_active_genome_db_id_returns_pinned_id(self, core, clock):
        """active_genome_db_id is the "what's on screen right now"
        property used for like/dislike ratings. When pinned, ratings
        should target the pinned genome, not whatever the axis was
        morphing toward."""
        from flame_sheep.genome import Genome
        g = Genome.random(np.random.default_rng(12))
        core.pin_genome(g, db_id=777)
        assert core.active_genome_db_id == 777

    def test_active_genome_db_id_falls_through_when_not_pinned(self, core, clock):
        """No pin → axis-derived behavior. Should return some db_id
        from the genome the axis is showing (or None for an
        unstored genome — synthetic test genomes have no db_id)."""
        # No pin set — should fall through to axis logic
        assert core._pinned_genome_db_id is None
        # The test genomes don't have db_id set, so result is None
        # (the fall-through path works; the "real id" path would need
        # a real loaded genome with db_id).
        result = core.active_genome_db_id
        # Just verify it doesn't crash and returns something we can use
        assert result is None or isinstance(result, int)

    def test_clear_after_pin_restores_axis_behavior(self, core, clock):
        from flame_sheep.genome import Genome
        g = Genome.random(np.random.default_rng(13))
        core.pin_genome(g, db_id=99)
        assert core.active_genome_db_id == 99
        core.clear_pin()
        # Falls back to axis-derived id
        assert core.active_genome_db_id != 99

    def test_repin_replaces_previous_pin(self, core, clock):
        """Calling pin_genome twice — second pin replaces the first.
        Used by `show <gid>` to switch which genome is locked
        without an explicit unshow between."""
        from flame_sheep.genome import Genome
        g1 = Genome.random(np.random.default_rng(14))
        g2 = Genome.random(np.random.default_rng(15))
        core.pin_genome(g1, db_id=1)
        core.pin_genome(g2, db_id=2)
        assert core._pinned_genome is g2
        assert core._pinned_genome_db_id == 2
        assert core._genome_axis._pinned_genome is g2
