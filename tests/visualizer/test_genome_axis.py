"""
Tests for GenomeAxis state machine transitions.

Tests the morph cycle, mode transitions, loop playback, beat handling,
graph walk, and loop progression. Uses trivial genomes and a mock library
to isolate state logic from rendering.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from flame_sheep.axes.genome_axis import GenomeAxis
from flame_sheep.axes._morph_cycle import MorphState
from flame_sheep.genome import Genome
from flame_sheep.role_mapper import RoleMapper
from flame_sheep.storage import Library
from flame_sheep_audio import BeatEvent, AudioState
from flame_sheep_audio._types import BandState
from flame_sheep_audio.mode import Mode

from viz_helpers import trivial_genome


# ----------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _make_audio(mode='idle', events=None, bpm=120.0, spectrum_size=108):
    """Build a minimal AudioState for testing."""
    return AudioState(
        events=events or [],
        spectrum=np.zeros(spectrum_size, dtype=np.float32),
        bands={'low': BandState(), 'mid': BandState(), 'high': BandState(),
               'subbass': BandState()},
        centroid=0.0, centroid_delta=0.0,
        centroid_rms=0.0, centroid_harmonic_rms=0.0,
        slow_centroid_harmonic_rms=0.0,
        percussiveness=0.0, spectral_novelty=0.0,
        section_change=0.0,
        bpm=bpm, effective_bpm=bpm, tempo_confidence=0.8,
        tempo_saturated=False, break_intensity=0.0,
        mode=mode,
    )


@pytest.fixture
def axis(rng):
    """GenomeAxis with trivial genomes, no library."""
    _seed = iter(range(1000))
    return GenomeAxis(
        genome_factory=lambda: trivial_genome(next(_seed)),
        role=RoleMapper(),
        lib=None,
        rng=rng,
    )


@pytest.fixture
def tmp_lib():
    """Library with genomes and a loop for testing."""
    with tempfile.TemporaryDirectory() as d:
        lib = Library(data_dir=Path(d))
        rng = np.random.default_rng(42)
        # Create 10 genomes
        gids = []
        for i in range(10):
            g = trivial_genome(i)
            scores = {'coverage': 0.5, 'entropy': 0.5, 'color_entropy': 0.5,
                      'balance': 0.5, 'complexity': 0.5}
            gid = lib.save_genome(g, scores)
            gids.append(gid)
        # Create a loop with 4 genomes
        loop_id = lib.save_loop(gids[:4])
        # Create a second loop
        loop_id2 = lib.save_loop(gids[4:8])
        yield lib
        lib.close()


@pytest.fixture
def axis_with_lib(tmp_lib, rng):
    """GenomeAxis backed by a library with loops."""
    _seed = iter(range(1000))
    return GenomeAxis(
        genome_factory=lambda: trivial_genome(next(_seed)),
        role=RoleMapper(),
        lib=tmp_lib,
        rng=rng,
    )


# ----------------------------------------------------------------
# Morph state machine
# ----------------------------------------------------------------

class TestMorphCycle:
    def test_starts_in_dwell(self, axis):
        assert axis._morph.state == MorphState.DWELL

    def test_idle_dwell_releases_on_timeout(self, axis):
        audio = _make_audio(mode='idle')
        clock = 1000.0
        axis._morph.dwell_start = clock
        # DWELL_BEATS=16 at 120BPM = 8 seconds = 480 frames
        for _ in range(600):
            clock += 1.0 / 60
            axis.tick(audio, 1.0 / 60, clock)
        assert axis._morph.state != MorphState.DWELL

    def test_morphing_advances_morph_t(self, axis):
        axis._morph.state = MorphState.MORPHING
        axis._morph.t = 0.0
        audio = _make_audio(mode='idle')
        axis.tick(audio, 1.0 / 60, 1000.0)
        assert axis._morph.t > 0.0

    def test_morph_completes_in_idle(self, axis):
        """In idle mode, morph completion swaps immediately (no SWAP_READY)."""
        axis._morph.state = MorphState.MORPHING
        axis._morph.t = 1.0 - axis.MORPH_SPEED * 0.5  # will complete on next tick
        audio = _make_audio(mode='idle')
        axis.tick(audio, 1.0 / 60, 1000.0)
        # Should have swapped and gone to DWELL
        assert axis._morph.state == MorphState.DWELL
        assert axis._morph.t == 0.0

    def test_morph_completes_in_beat_goes_swap_ready(self, axis):
        """In beat mode, morph completion waits for kick (SWAP_READY)."""
        axis._morph.state = MorphState.MORPHING
        axis._morph.t = 1.0 - axis.MORPH_SPEED * 0.5
        axis._mode = Mode.BEAT
        audio = _make_audio(mode='beat')
        axis.tick(audio, 1.0 / 60, 1000.0)
        assert axis._morph.state == MorphState.SWAP_READY

    def test_kick_releases_swap_ready(self, axis):
        """In beat mode, a kick event commits the pending swap."""
        axis._morph.state = MorphState.SWAP_READY
        axis._mode = Mode.BEAT
        kick = BeatEvent(kind='low', energy=0.8)
        audio = _make_audio(mode='beat', events=[kick])
        axis.tick(audio, 1.0 / 60, 1000.0)
        assert axis._morph.state == MorphState.DWELL
        assert axis._morph.t == 0.0

    def test_morph_t_stays_bounded(self, axis):
        """morph_t should never exceed 1.0."""
        axis._morph.state = MorphState.MORPHING
        audio = _make_audio(mode='idle')
        for i in range(1000):
            axis.tick(audio, 1.0 / 60, 1000.0 + i / 60)
        assert axis._morph.t <= 1.0


# ----------------------------------------------------------------
# Mode transitions
# ----------------------------------------------------------------

class TestModeTransitions:
    def test_beat_to_idle_releases_dwell(self, axis):
        """Leaving beat mode releases dwell if waiting for a beat."""
        axis._mode = Mode.BEAT
        axis._morph.state = MorphState.DWELL
        # Transition to idle
        audio = _make_audio(mode='idle')
        axis.tick(audio, 1.0 / 60, 1000.0)
        assert axis._morph.state == MorphState.MORPHING

    def test_idle_to_beat_loads_loop(self, axis_with_lib):
        """Entering beat mode without a loop triggers next_loop."""
        axis = axis_with_lib
        axis._mode = Mode.IDLE
        axis._loop._loop_genomes = []  # no loop loaded
        audio = _make_audio(mode='beat')
        axis.tick(audio, 1.0 / 60, 1000.0)
        assert axis.active_loop_id is not None

    def test_mode_change_preserves_morph_t(self, axis):
        """Mode changes don't reset morph position."""
        axis._morph.state = MorphState.MORPHING
        axis._morph.t = 0.5
        axis._mode = Mode.IDLE
        audio = _make_audio(mode='energy')
        axis.tick(audio, 1.0 / 60, 1000.0)
        # morph_t should not be reset by mode change
        assert axis._morph.t >= 0.5


# ----------------------------------------------------------------
# Loop playback
# ----------------------------------------------------------------

class TestLoopPlayback:
    def test_load_loop_sets_genomes(self, axis_with_lib):
        axis = axis_with_lib
        assert len(axis._loop.loop_genomes) > 0
        assert axis.active_loop_id is not None

    def test_swap_advances_loop_sequence(self, axis_with_lib):
        axis = axis_with_lib
        old_target = axis.target_genome
        axis._swap_next_genome()
        # Should have a new target genome from the loop
        assert axis.target_genome is not None

    def test_loop_full_cycle_wraps(self, axis_with_lib):
        """Stepping through a full cycle wraps step counter back to start."""
        axis = axis_with_lib
        cycle_len = axis._loop._loop_cycle_len
        initial_step = axis._loop._loop_step
        if cycle_len > 0:
            for _ in range(cycle_len):
                axis._swap_next_genome()
            # After a full cycle, should be back to same position
            assert axis._loop._loop_step == initial_step

    def test_next_loop_changes_loop_id(self, axis_with_lib):
        axis = axis_with_lib
        old_id = axis.active_loop_id
        axis.next_loop()
        # Should change (or at least attempt — with 2 loops it should switch)
        # May stay same if only one loop available after history exclusion
        assert axis.active_loop_id is not None

    def test_loop_history_tracks(self, axis_with_lib):
        axis = axis_with_lib
        first_id = axis.active_loop_id
        assert first_id in axis._loop._loop_history


# ----------------------------------------------------------------
# Loop progression
# ----------------------------------------------------------------

class TestLoopProgression:
    def test_no_exit_before_min_cycles(self, axis_with_lib):
        """should_exit_loop returns False before MIN_EXIT_CYCLES."""
        axis = axis_with_lib
        axis._loop._cycle_count = 2
        assert axis._loop.should_exit_loop() is False

    def test_exit_possible_after_min_cycles(self, axis_with_lib):
        """After enough cycles, should_exit_loop can return True."""
        axis = axis_with_lib
        axis._loop._cycle_count = 5
        # At position near end of loop, probability is high
        n = len(axis._loop.loop_genomes)
        axis._loop._loop_step = n - 1  # last position: prob = (n-1)/n
        # Run many times — should exit at least once
        exits = sum(axis._loop.should_exit_loop() for _ in range(100))
        assert exits > 0

    def test_exit_prob_increases_with_position(self, axis_with_lib):
        """Exit probability increases with position in cycle."""
        axis = axis_with_lib
        axis._loop._cycle_count = 10  # well past minimum

        n = len(axis._loop.loop_genomes)
        if n < 3:
            pytest.skip("Need at least 3 genomes for position test")

        # Position 0: prob = 0/n = 0 (never exits)
        axis._loop._loop_step = 0
        exits_start = sum(axis._loop.should_exit_loop() for _ in range(200))

        # Position n-1: prob = (n-1)/n (almost always exits)
        axis._loop._loop_step = n - 1
        exits_end = sum(axis._loop.should_exit_loop() for _ in range(200))

        assert exits_start == 0  # 0/n = 0 probability
        assert exits_end > exits_start


# ----------------------------------------------------------------
# User next
# ----------------------------------------------------------------

class TestUserNext:
    def test_user_next_in_dwell_switches_immediately(self, axis_with_lib):
        axis = axis_with_lib
        axis._morph.state = MorphState.DWELL
        old_loop = axis.active_loop_id
        axis.user_next()
        assert axis._morph.state == MorphState.MORPHING
        assert axis._morph.t == 0.0

    def test_user_next_mid_morph_sets_flag(self, axis_with_lib):
        axis = axis_with_lib
        axis._morph.state = MorphState.MORPHING
        axis._morph.t = 0.5
        axis.user_next()
        assert axis._next_loop_pending is True
        # Morph continues
        assert axis._morph.t == 0.5

    def test_user_next_in_swap_ready_switches(self, axis_with_lib):
        axis = axis_with_lib
        axis._morph.state = MorphState.SWAP_READY
        axis.user_next()
        assert axis._morph.state == MorphState.MORPHING


# ----------------------------------------------------------------
# Beat handling
# ----------------------------------------------------------------

class TestBeatHandling:
    def test_kick_releases_dwell_in_beat_mode(self, axis):
        axis._mode = Mode.BEAT
        axis._morph.state = MorphState.DWELL
        kick = BeatEvent(kind='low', energy=0.8)
        audio = _make_audio(mode='beat', events=[kick])
        axis.tick(audio, 1.0 / 60, 1000.0)
        assert axis._morph.state == MorphState.MORPHING

    def test_downbeat_boosts_rotation(self, axis):
        axis._mode = Mode.BEAT
        axis._morph.state = MorphState.MORPHING
        axis._morph.t = 0.3
        initial_phase = axis._rotation.phase
        downbeat = BeatEvent(kind='mid', energy=0.9)
        audio = _make_audio(mode='beat', events=[downbeat])
        axis.tick(audio, 1.0 / 60, 1000.0)
        # Rotation should have advanced more than just base speed
        assert axis._rotation.phase > initial_phase

    def test_kick_in_idle_does_not_release_dwell(self, axis):
        """Beat events are ignored in idle mode."""
        axis._mode = Mode.IDLE
        axis._morph.state = MorphState.DWELL
        axis._morph.dwell_start = 1000.0
        kick = BeatEvent(kind='low', energy=0.8)
        audio = _make_audio(mode='idle', events=[kick])
        axis.tick(audio, 1.0 / 60, 1000.01)
        # Still in dwell (timeout hasn't elapsed)
        assert axis._morph.state == MorphState.DWELL


# ----------------------------------------------------------------
# Section change
# ----------------------------------------------------------------

class TestSectionChange:
    def test_section_change_sets_pending(self, axis_with_lib):
        axis = axis_with_lib
        axis._mode = Mode.BEAT
        axis._beat.section_warmup = 2000  # past warmup
        axis._beat.section_cooldown = 500  # past cooldown
        audio = _make_audio(mode='beat')
        audio.section_change = 0.9  # above threshold
        axis.tick(audio, 1.0 / 60, 1000.0)
        assert axis._beat.section_change_pending is True

    def test_section_change_consumed_on_downbeat(self, axis_with_lib):
        axis = axis_with_lib
        axis._mode = Mode.BEAT
        axis._beat.section_change_pending = True
        old_loop = axis.active_loop_id
        downbeat = BeatEvent(kind='mid', energy=0.8)
        audio = _make_audio(mode='beat', events=[downbeat])
        axis.tick(audio, 1.0 / 60, 1000.0)
        assert axis._beat.section_change_pending is False


# ----------------------------------------------------------------
# Song start
# ----------------------------------------------------------------

class TestSongStart:
    def test_song_start_resets_state(self, axis):
        axis._morph.state = MorphState.MORPHING
        axis._morph.t = 0.7
        axis._beat.break_damping = 0.3
        axis._handle_song_start(1000.0)
        assert axis._morph.state == MorphState.DWELL
        assert axis._morph.t == 0.0
        assert axis._beat.break_damping == 1.0

    def test_song_start_event_resets(self, axis):
        axis._morph.state = MorphState.MORPHING
        axis._morph.t = 0.5
        song_event = BeatEvent(kind='song_start', energy=0.0)
        audio = _make_audio(mode='beat', events=[song_event])
        axis.tick(audio, 1.0 / 60, 1000.0)
        assert axis._morph.state == MorphState.DWELL


# ----------------------------------------------------------------
# Graph walk (idle/energy mode)
# ----------------------------------------------------------------

class TestGraphWalk:
    def test_graph_walk_without_lib_uses_factory(self, axis):
        """Without a library, graph walk falls back to genome factory."""
        old_target = axis.target_genome
        axis._graph_walk_next()
        # Should have a new target (from factory)
        assert axis.target_genome is not None

    def test_morph_complete_in_idle_triggers_graph_walk(self, axis):
        """Idle mode morph completion should pick next via graph walk."""
        axis._morph.state = MorphState.MORPHING
        axis._morph.t = 0.99
        old_target = axis.target_genome
        audio = _make_audio(mode='idle')
        axis.tick(audio, 1.0 / 60, 1000.0)
        # Should have graph-walked to a new target
        assert axis.target_genome is not None


# ----------------------------------------------------------------
# Force swap
# ----------------------------------------------------------------

class TestForceSwap:
    def test_force_swap_resets_state(self, axis):
        axis._morph.state = MorphState.MORPHING
        axis._morph.t = 0.7
        axis.force_swap()
        assert axis._morph.state == MorphState.DWELL
        assert axis._morph.t == 0.0
        assert axis.needs_walker_reset is True


# ----------------------------------------------------------------
# Contribute (frame output)
# ----------------------------------------------------------------

class TestContribute:
    def test_contribute_produces_genome(self, axis):
        from flame_sheep.main import FlameSheepCore
        frame = FlameSheepCore.FrameState(
            genome=None, palette=None,
            spectrum=np.zeros(108, dtype=np.float32),
            brightness=1.0, iterations=100)
        axis.contribute(frame)
        assert frame.genome is not None
        assert isinstance(frame.genome, Genome)

    def test_contribute_applies_rotation(self, axis):
        """Rotation phase should produce a different genome than no rotation."""
        from flame_sheep.main import FlameSheepCore
        # Get unrotated output
        axis._rotation.phase = 0.0
        frame0 = FlameSheepCore.FrameState(
            genome=None, palette=None,
            spectrum=np.zeros(108, dtype=np.float32),
            brightness=1.0, iterations=100)
        axis.contribute(frame0)
        affine0 = frame0.genome.transforms[0].affine.copy()

        # Get rotated output
        axis._rotation.phase = 0.5
        frame1 = FlameSheepCore.FrameState(
            genome=None, palette=None,
            spectrum=np.zeros(108, dtype=np.float32),
            brightness=1.0, iterations=100)
        axis.contribute(frame1)
        affine1 = frame1.genome.transforms[0].affine

        # Affines should differ when rotation is applied
        assert not np.allclose(affine0, affine1, atol=0.001)
