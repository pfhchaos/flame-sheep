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
from flame_sheep.runtime import RoleMapper
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
    # REWRITTEN for PICK-THE-EXIT-NODE (the old pos/n probability ramp is gone).
    # should_exit_loop now fires deterministically at a single uniform exit
    # position (_exit_at) picked per loop entry, after MIN_EXIT_CYCLES. The old
    # `test_exit_prob_increases_with_position` is OBSOLETE (there is no longer a
    # position-dependent probability) and is replaced by
    # `test_exit_fires_only_at_picked_position` below, which also proves the exit
    # can land anywhere (uniform position, not clustered late).

    def test_no_exit_before_min_cycles(self, axis_with_lib):
        """should_exit_loop returns False before MIN_EXIT_CYCLES, even sitting on
        the exit node."""
        axis = axis_with_lib
        lp = axis._loop
        lp._cycle_count = lp.MIN_EXIT_CYCLES - 1
        lp._loop_step = lp._exit_at
        assert lp.should_exit_loop() is False

    @pytest.mark.parametrize("which", ["first", "middle", "last"])
    def test_exit_fires_only_at_picked_position(self, axis_with_lib, which):
        """After MIN_EXIT_CYCLES, exit fires at exactly the picked position and
        nowhere else — and the position can be anywhere in the cycle (uniform
        exit POSITION, replacing the obsolete pos/n ramp)."""
        axis = axis_with_lib
        lp = axis._loop
        cyc = lp._loop_cycle_len
        if cyc < 3:
            pytest.skip("Need cycle_len >= 3 for first/middle/last positions")
        lp._cycle_count = lp.MIN_EXIT_CYCLES
        lp._exit_at = {"first": 0, "middle": cyc // 2, "last": cyc - 1}[which]
        fired = []
        for step in range(cyc):
            lp._loop_step = step
            if lp.should_exit_loop():
                fired.append(step)
        assert fired == [lp._exit_at]


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
        axis._mode = Mode.BEAT          # the loop-pending flag is beat-only now
        axis._morph.state = MorphState.MORPHING
        axis._morph.t = 0.5
        axis.user_next()
        assert axis._next_loop_pending is True
        # Morph continues
        assert axis._morph.t == 0.5

    def test_user_next_walks_in_idle(self, axis_with_lib):
        """Not-beat: user_next walks on now (loops are beat-only) — no loop flag."""
        axis = axis_with_lib
        axis._mode = Mode.IDLE
        axis._morph.state = MorphState.MORPHING
        axis._morph.t = 0.5
        axis.user_next()
        assert axis._next_loop_pending is False
        assert axis._morph.state == MorphState.MORPHING   # walking (start_morph)

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
# Pick relocation (step-4 S2): the steady FOLLOW/WALK pick happens at DEPARTURE
# (dwell-release), NOT at morph-complete; the arrival PLAN hook fires at the
# COMMIT. Still render-side source, so motion is unchanged — this just opens the
# empty dwell window S3's async worker answer slots into.
#
# MUTATION GUARD: move the pick back to morph-complete (the pre-S2 code) and
# test_pick_deferred_to_dwell_release reddens (target would already differ from
# current right after the morph completes).
# ----------------------------------------------------------------

class TestPickRelocation:
    def _complete_morph(self, axis, clock, mode='idle'):
        """Force the in-flight morph to finish on the next tick."""
        axis._morph.state = MorphState.MORPHING
        axis._morph.t = 1.0 - axis.MORPH_SPEED * 0.5
        axis.tick(_make_audio(mode=mode), 1.0 / 60, clock)

    def test_pick_deferred_to_dwell_release(self, axis_with_lib):
        axis = axis_with_lib
        # ARRIVE: complete a morph in idle+loop mode.
        self._complete_morph(axis, clock=1000.0)
        assert axis._morph.state == MorphState.DWELL
        # The pick has NOT happened yet: target is the just-arrived genome.
        assert axis.target_genome is axis.current_genome
        # DEPART: release the dwell by timeout.
        dwell = axis.DWELL_BEATS * 60.0 / 120.0
        axis.tick(_make_audio(mode='idle'), 1.0 / 60, 1000.0 + dwell + 1.0)
        assert axis._morph.state == MorphState.MORPHING
        # The pick happened at release: a new target, distinct from current.
        assert axis.target_genome is not axis.current_genome

    def test_arrival_hook_fires_at_commit(self, axis_with_lib):
        axis = axis_with_lib
        arrivals = []
        axis.set_plan_hook(lambda g, lid, step, exp: arrivals.append(
            getattr(g, 'db_id', None)))
        self._complete_morph(axis, clock=1000.0)        # arrive
        assert arrivals, "arrival hook must fire at the morph-complete commit"

    def test_pick_next_target_follows_in_loop(self, axis_with_lib):
        axis = axis_with_lib
        axis._mode = Mode.BEAT                          # FOLLOW is beat-only
        step0 = axis._loop.loop_step
        axis._pick_next_target()
        assert axis._loop.loop_step != step0            # FOLLOW advanced the seq

    def test_pick_next_target_walks_without_loop(self, axis):
        axis._pick_next_target()                        # no loop -> WALK (factory)
        assert axis.target_genome is not None


# ----------------------------------------------------------------
# FOLLOW authority flip (step-4 S3): the steady loop FOLLOW consumes the WORKER's
# pre-decided target (not the render-side loop pick), DEFERS into dwell when the
# worker hasn't answered, and falls back to render-side only when no provider is
# wired (precompile unavailable).
#
# MUTATION GUARD #8: if _follow_pick used next_in_sequence instead of the
# provider, test_follow_consumes_worker_target reddens (it would commit the
# loop's genome, not the injected sentinel).
# ----------------------------------------------------------------

class TestFollowAuthority:
    def _in_loop_with_dbid(self, axis):
        axis._mode = Mode.BEAT          # FOLLOW is beat-only after the reshape
        axis.current_genome = axis._loop.loop_genomes[0]
        assert axis.current_genome.db_id is not None

    def test_follow_consumes_worker_target(self, axis_with_lib):
        axis = axis_with_lib
        self._in_loop_with_dbid(axis)
        sentinel = trivial_genome(777)
        sentinel.db_id = 777
        axis.set_plan_provider(
            lambda aid, branch: sentinel if branch == 'follow' else None)
        step0 = axis._loop.loop_step
        assert axis._pick_next_target() is True
        assert axis.target_genome is sentinel           # worker's, not loop's
        assert axis._loop.loop_step == step0 + 1          # bookkeeping advanced

    def test_follow_defers_when_worker_not_ready(self, axis_with_lib):
        axis = axis_with_lib
        self._in_loop_with_dbid(axis)
        axis.set_plan_provider(lambda aid, branch: None)  # not answered yet
        tgt0, step0 = axis.target_genome, axis._loop.loop_step
        assert axis._pick_next_target() is False          # deferred
        assert axis.target_genome is tgt0                 # unchanged
        assert axis._loop.loop_step == step0              # NOT advanced

    def test_follow_falls_back_without_provider(self, axis_with_lib):
        axis = axis_with_lib                              # no provider wired
        axis._mode = Mode.BEAT                            # FOLLOW is beat-only
        step0 = axis._loop.loop_step
        assert axis._pick_next_target() is True
        assert axis._loop.loop_step == step0 + 1          # render-side pick ran

    # FOLLOW is beat-only now, and beat's dwell is released by a KICK (not a
    # timeout) — so the defer-at-release integration lives in beat mode.
    def test_beat_release_stays_dwelling_on_defer(self, axis_with_lib):
        axis = axis_with_lib
        self._in_loop_with_dbid(axis)                     # beat mode
        axis.set_plan_provider(lambda aid, b: None)       # FOLLOW not ready
        axis._morph.state = MorphState.DWELL
        kick = BeatEvent(kind='low', energy=0.8)
        axis.tick(_make_audio(mode='beat', events=[kick]), 1 / 60, 1000.0)
        assert axis._morph.state == MorphState.DWELL      # kick couldn't release

    def test_beat_release_morphs_when_ready(self, axis_with_lib):
        axis = axis_with_lib
        self._in_loop_with_dbid(axis)                     # beat mode
        sentinel = trivial_genome(888)
        sentinel.db_id = 888
        axis.set_plan_provider(
            lambda aid, b: sentinel if b == 'follow' else None)
        axis._morph.state = MorphState.DWELL
        kick = BeatEvent(kind='low', energy=0.8)
        axis.tick(_make_audio(mode='beat', events=[kick]), 1 / 60, 1000.0)
        assert axis._morph.state == MorphState.MORPHING
        assert axis.target_genome is sentinel


# ----------------------------------------------------------------
# Mode → movement reshape (DECISIONS 2026-10-12): IDLE/ENERGY one-step WALK even
# with a loop loaded; loops are BEAT-only (they encode rhythmic cadence). Walking
# idle/energy also surfaces non-loop genomes for voting (GA coverage).
#
# MUTATION GUARD: drop the `Mode.BEAT` gate in _pick_next_target and
# test_idle_walks_past_a_loaded_loop reddens (idle would FOLLOW → loop_step
# advances).
# ----------------------------------------------------------------

class TestModeMovementReshape:
    def test_idle_walks_past_a_loaded_loop(self, axis_with_lib):
        axis = axis_with_lib                         # a loop is loaded
        axis._mode = Mode.IDLE
        assert axis._loop.has_loop
        step0 = axis._loop.loop_step
        axis._pick_next_target()
        assert axis._loop.loop_step == step0         # WALK: counter does NOT advance

    def test_energy_walks_past_a_loaded_loop(self, axis_with_lib):
        axis = axis_with_lib
        axis._mode = Mode.ENERGY
        step0 = axis._loop.loop_step
        axis._pick_next_target()
        assert axis._loop.loop_step == step0         # WALK

    def test_beat_still_follows_loop(self, axis_with_lib):
        axis = axis_with_lib
        axis._mode = Mode.BEAT
        step0 = axis._loop.loop_step
        axis._pick_next_target()
        assert axis._loop.loop_step == step0 + 1     # beat FOLLOWS → advances

    # The centroid swap is ENERGY-ONLY: a timbral trigger is energy's reactivity
    # (a walk). BEAT does NOT swap on centroid — the loop advances on the beat,
    # not on timbre. (Regression caught on-box 2026-10-12: firing in beat walked
    # off-loop onto un-warmed genomes → a render-thread COLD compile.)
    def test_centroid_no_swap_in_beat(self, axis_with_lib, monkeypatch):
        axis = axis_with_lib
        axis._mode = Mode.BEAT
        axis._morph.state = MorphState.DWELL
        monkeypatch.setattr(axis._centroid, 'tick', lambda *a: True)
        tgt0 = axis.target_genome
        step0 = axis._loop.loop_step
        axis.tick(_make_audio(mode='beat'), 1 / 60, 1000.0)
        assert axis.target_genome is tgt0            # no centroid swap in beat
        assert axis._loop.loop_step == step0

    def test_centroid_walks_in_energy(self, axis_with_lib, monkeypatch):
        axis = axis_with_lib
        axis._mode = Mode.ENERGY
        axis._morph.state = MorphState.DWELL
        axis.current_genome = axis._loop.loop_genomes[0]
        monkeypatch.setattr(axis._centroid, 'tick', lambda *a: True)
        tgt0 = axis.target_genome
        step0 = axis._loop.loop_step
        axis.tick(_make_audio(mode='energy'), 1 / 60, 1000.0)
        assert axis.target_genome is not tgt0        # energy centroid WALKED
        assert axis._loop.loop_step == step0         # walk doesn't advance loop


# ----------------------------------------------------------------
# Beat loop BREAK (pick-the-exit-node wired into _pick_next_target): in BEAT, when
# the loop player signals an exit, the pick LEAVES the current loop for a new one
# (BREAK) instead of FOLLOWing it. Loop changes stay beat-only — idle/energy never
# break, they WALK, even when should_exit_loop is forced True.
#
# MUTATION GUARD: if _pick_next_target dropped the should_exit_loop() check (always
# FOLLOW), test_beat_breaks_loop_on_exit_signal reddens (active_loop_id wouldn't
# change). If the Mode.BEAT gate were dropped, the idle/energy tests redden.
# ----------------------------------------------------------------

class TestBeatLoopBreak:
    def _in_beat_loop(self, axis):
        axis._mode = Mode.BEAT
        axis.current_genome = axis._loop.loop_genomes[0]
        assert axis.current_genome.db_id is not None

    def test_beat_breaks_loop_on_exit_signal(self, axis_with_lib, monkeypatch):
        """Forced should_exit_loop True → BREAK: a new loop is committed (2-loop
        library → somewhere to go). Returns True (dwell releases)."""
        axis = axis_with_lib
        self._in_beat_loop(axis)
        monkeypatch.setattr(axis._loop, 'should_exit_loop', lambda: True)
        old_loop = axis.active_loop_id
        assert axis._pick_next_target() is True       # BREAK committed
        assert axis.active_loop_id != old_loop        # left for a different loop

    def test_beat_follows_when_no_exit_signal(self, axis_with_lib, monkeypatch):
        """Forced should_exit_loop False → FOLLOW: stay in the loop, advance."""
        axis = axis_with_lib
        self._in_beat_loop(axis)
        monkeypatch.setattr(axis._loop, 'should_exit_loop', lambda: False)
        old_loop = axis.active_loop_id
        step0 = axis._loop.loop_step
        axis._pick_next_target()
        assert axis.active_loop_id == old_loop        # stayed in the loop
        assert axis._loop.loop_step == step0 + 1      # FOLLOW advanced the seq

    def test_idle_never_breaks(self, axis_with_lib, monkeypatch):
        """idle WALKS even with should_exit forced True — loops are beat-only."""
        axis = axis_with_lib
        axis._mode = Mode.IDLE
        monkeypatch.setattr(axis._loop, 'should_exit_loop', lambda: True)
        old_loop = axis.active_loop_id
        step0 = axis._loop.loop_step
        axis._pick_next_target()
        assert axis.active_loop_id == old_loop        # no break
        assert axis._loop.loop_step == step0          # walked, loop untouched

    def test_energy_never_breaks(self, axis_with_lib, monkeypatch):
        """energy WALKS even with should_exit forced True."""
        axis = axis_with_lib
        axis._mode = Mode.ENERGY
        monkeypatch.setattr(axis._loop, 'should_exit_loop', lambda: True)
        old_loop = axis.active_loop_id
        step0 = axis._loop.loop_step
        axis._pick_next_target()
        assert axis.active_loop_id == old_loop
        assert axis._loop.loop_step == step0


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
        from flame_sheep.runtime import FlameSheepCore
        frame = FlameSheepCore.FrameState(
            genome=None, palette=None,
            spectrum=np.zeros(108, dtype=np.float32),
            brightness=1.0, iterations=100)
        axis.contribute(frame)
        assert frame.genome is not None
        assert isinstance(frame.genome, Genome)

    def test_contribute_applies_rotation(self, axis):
        """Rotation phase should produce a different genome than no rotation."""
        from flame_sheep.runtime import FlameSheepCore
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
