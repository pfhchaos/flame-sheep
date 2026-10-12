r"""Parameterized movement matrix over synthetic library (bootstrap) states.

WHAT THIS PINS
--------------
This file is the executable spec for the mode->movement reshape (commits
c627360 "idle/energy WALK, loops become beat-only" and dd801b1). It drives a
real ``GenomeAxis`` built on a synthetic ``FakeLibrary`` through the departure
pick (``_pick_next_target``) and the per-tick centroid path, and asserts the
MOVEMENT each ``(library_state, mode)`` pair actually takes.

The library states double as the BOOTSTRAP TIMELINE — they are the catalog the
axis sees at successive stages of the GA's life:

  stage 0  EMPTY            no genomes at all (cold boot, nothing bred yet)
  stage 1  GRAPH_NO_LOOPS   GA-bred genomes + a transition graph, no loops yet
  edge     DISCONNECTED     current genome has NO outgoing transitions
  stage 2  ONE_LOOP         first loop found (+ non-loop genomes/transitions)
  stage 3  MULTI_LOOP       several loops available

MOVEMENT RULES (confirmed by reading the source, not this summary):

  ``GenomeAxis._pick_next_target`` (genome_axis.py):
      if mode is BEAT and self._loop.has_loop:  FOLLOW  (_follow_pick)
      else:                                     WALK    (_graph_walk_next)

  * FOLLOW (render-side, no plan provider wired) -> ``_swap_next_genome`` ->
    ``LoopPlayer.next_in_sequence`` -> ``_loop_step`` ADVANCES by one position.
  * WALK -> ``_graph_walk_next`` -> ``LoopPlayer.graph_walk`` -> a graph
    NEIGHBOR when the current genome has transitions, else the genome FACTORY
    (fresh random). ``_loop_step`` does NOT change.

  The CENTROID swap in ``tick`` fires ONLY in ``Mode.ENERGY`` (a WALK). BEAT and
  IDLE never centroid-swap (beat's loop advances on the kick, not on timbre;
  firing it in beat walked off-loop onto un-warmed genomes -> cold compile,
  regression caught on-box 2026-10-12).

So the matrix is:

  state \ mode     IDLE     ENERGY   BEAT
  EMPTY            WALK     WALK     WALK*   (* no loop -> walk, not follow)
  GRAPH_NO_LOOPS   WALK     WALK     WALK*
  DISCONNECTED     WALK     WALK     WALK*   (walk falls back to factory)
  ONE_LOOP         WALK     WALK     FOLLOW
  MULTI_LOOP       WALK     WALK     FOLLOW

The IDLE/ENERGY columns for ONE_LOOP and MULTI_LOOP are the whole point of the
reshape: even WITH a loop loaded, idle/energy WALK (loop_step frozen) — loops
are a beat-only construct.

A NOTE ON "beat + no loop": ``_pick_next_target`` gates FOLLOW on
``self._loop.has_loop``, so beat with no loop loaded WALKS. These tests set
``axis._mode`` directly, so the ``_on_mode_change`` hook (which, on a live
IDLE->BEAT transition, calls ``next_loop()`` to pull in a loop if any exist)
does NOT fire. For EMPTY/GRAPH_NO_LOOPS there are no loops in the library at
all, so even that hook would find nothing — beat genuinely walks there. The
beat-no-loop row therefore models bootstrap stage 0/1, before any loop exists.

MUTATION GUARDS (provable-where-it-matters):
  * Drop the ``Mode.BEAT`` gate in ``_pick_next_target`` -> the idle/energy rows
    for ONE_LOOP/MULTI_LOOP redden (idle/energy would FOLLOW, advancing
    loop_step). Spot-checked 2026-10-12 by temporarily removing the gate; the
    four parametrized ``movement=FOLLOW`` expectations for idle/energy fired red,
    then the source was reverted.
  * Drop the ``has_loop`` half of the gate -> the beat rows for EMPTY/GRAPH/
    DISCONNECTED redden (``_follow_pick`` would run with an empty loop).
  * Move the centroid block out of the ``Mode.ENERGY`` guard -> the
    beat-centroid NO-swap case reddens.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from flame_sheep.axes.genome_axis import GenomeAxis
from flame_sheep.axes._morph_cycle import MorphState
from flame_sheep.runtime import RoleMapper
from flame_sheep_audio import AudioState, BeatEvent
from flame_sheep_audio._types import BandState
from flame_sheep_audio.mode import Mode

from viz_helpers import trivial_genome

# FakeLibrary lives in the scheduler test tree (reused as shared infra).
from tests.scheduler.fake_library import FakeLibrary


# ----------------------------------------------------------------
# Audio helper (local copy of test_genome_axis._make_audio — the matrix file is
# self-contained so it can be read as the spec without chasing imports).
# ----------------------------------------------------------------

def _make_audio(mode='idle', events=None, bpm=120.0, spectrum_size=108):
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
def rng():
    return np.random.default_rng(42)


# ----------------------------------------------------------------
# Library-state builders — the bootstrap timeline, as FakeLibrary states.
# Each returns a freshly populated FakeLibrary holding REAL trivial genomes
# (so GenomeAxis's distance/lerp math works; FakeGenome stubs would not).
# ----------------------------------------------------------------

def build_empty():
    """Stage 0: no genomes, no transitions, no loops."""
    return FakeLibrary()


def build_graph_no_loops():
    """Stage 1: GA-bred genomes with a transition graph, but NO loops yet.
    Chain 0-1-2-3 so a walk from genome 0 has real neighbors to pick."""
    lib = FakeLibrary()
    for i in range(4):
        lib.add_genome(i, trivial_genome(i), scores={'coverage': 0.5})
    lib.store_transition(0, 1, 0.5)
    lib.store_transition(1, 2, 0.5)
    lib.store_transition(2, 3, 0.5)
    return lib


def build_disconnected():
    """Edge case: a genome (id 99) that exists but has NO outgoing transitions.
    A walk from it must fall back to the factory, not crash. No loops."""
    lib = FakeLibrary()
    lib.add_genome(0, trivial_genome(0), scores={'coverage': 0.5})
    lib.add_genome(1, trivial_genome(1), scores={'coverage': 0.5})
    lib.store_transition(0, 1, 0.5)          # 0 and 1 are connected...
    lib.add_genome(99, trivial_genome(99), scores={'coverage': 0.5})  # ...99 is not
    return lib


def build_one_loop():
    """Stage 2: one loop (genomes 0-3) plus non-loop genomes (4,5) and
    transitions linking them."""
    lib = FakeLibrary()
    for i in range(6):
        lib.add_genome(i, trivial_genome(i), scores={'coverage': 0.5})
    lib.add_loop(0, [0, 1, 2, 3], structure='cyclic', fitness=1.0)
    lib.store_transition(4, 5, 0.4)          # non-loop pair
    lib.store_transition(0, 4, 0.6)          # loop <-> non-loop link
    return lib


def build_multi_loop():
    """Stage 3: two loops with a bridging transition."""
    lib = FakeLibrary()
    for i in range(8):
        lib.add_genome(i, trivial_genome(i), scores={'coverage': 0.5})
    lib.add_loop(0, [0, 1, 2, 3], structure='cyclic', fitness=1.0)
    lib.add_loop(1, [4, 5, 6, 7], structure='cyclic', fitness=0.8)
    lib.store_transition(0, 4, 0.5)
    return lib


_BUILDERS = {
    'EMPTY': build_empty,
    'GRAPH_NO_LOOPS': build_graph_no_loops,
    'DISCONNECTED': build_disconnected,
    'ONE_LOOP': build_one_loop,
    'MULTI_LOOP': build_multi_loop,
}

# Per-state current-genome override (db id) so a WALK exercises a REAL graph
# neighbor where one should exist. GRAPH_NO_LOOPS: sit on genome 0 (has edges).
# DISCONNECTED: sit on genome 99 (no edges -> factory fallback path).
# The loop states leave current at the boot factory genome (db_id None); the
# idle/energy assertion there is only that loop_step is FROZEN, which the walk
# never touches regardless of which branch it takes.
_CURRENT_OVERRIDE = {
    'GRAPH_NO_LOOPS': 0,
    'DISCONNECTED': 99,
}


def _make_axis(state_name, mode, rng):
    """Build a GenomeAxis on the named library state, in the given mode.
    Mirrors the axis_with_lib fixture pattern, parameterized over state."""
    lib = _BUILDERS[state_name]()
    seeds = itertools.count(9000)
    axis = GenomeAxis(
        genome_factory=lambda: trivial_genome(next(seeds)),
        role=RoleMapper(),
        lib=lib,
        rng=rng,
    )
    axis._mode = mode
    override = _CURRENT_OVERRIDE.get(state_name)
    if override is not None:
        axis.current_genome = lib.load_genome(override)
    return axis, lib


# ----------------------------------------------------------------
# The matrix: (state, mode) -> expected movement.
# ----------------------------------------------------------------

WALK = 'WALK'
FOLLOW = 'FOLLOW'

_MATRIX = [
    # EMPTY — nothing bred: every mode walks (no loop to follow).
    ('EMPTY', Mode.IDLE, WALK),
    ('EMPTY', Mode.ENERGY, WALK),
    ('EMPTY', Mode.BEAT, WALK),
    # GRAPH_NO_LOOPS — genomes + graph, no loops: every mode walks.
    ('GRAPH_NO_LOOPS', Mode.IDLE, WALK),
    ('GRAPH_NO_LOOPS', Mode.ENERGY, WALK),
    ('GRAPH_NO_LOOPS', Mode.BEAT, WALK),
    # DISCONNECTED — current has no edges: walk falls back to factory, no crash.
    ('DISCONNECTED', Mode.IDLE, WALK),
    ('DISCONNECTED', Mode.ENERGY, WALK),
    ('DISCONNECTED', Mode.BEAT, WALK),
    # ONE_LOOP — the reshape: idle/energy WALK past the loaded loop; beat FOLLOWS.
    ('ONE_LOOP', Mode.IDLE, WALK),
    ('ONE_LOOP', Mode.ENERGY, WALK),
    ('ONE_LOOP', Mode.BEAT, FOLLOW),
    # MULTI_LOOP — same gate with more loops available.
    ('MULTI_LOOP', Mode.IDLE, WALK),
    ('MULTI_LOOP', Mode.ENERGY, WALK),
    ('MULTI_LOOP', Mode.BEAT, FOLLOW),
]


@pytest.mark.parametrize(
    'state_name, mode, movement',
    _MATRIX,
    ids=[f'{s}-{m.value}-{mv}' for s, m, mv in _MATRIX],
)
def test_departure_movement(state_name, mode, movement, rng):
    """At DEPARTURE (``_pick_next_target``), the axis takes the expected move
    for its (library_state, mode). FOLLOW advances the loop step by one
    position; WALK freezes it and still produces a target without crashing."""
    axis, _lib = _make_axis(state_name, mode, rng)

    step0 = axis._loop.loop_step
    cycle0 = axis._loop.cycle_count

    # No plan provider wired -> FOLLOW is render-side, always resolves (True).
    moved = axis._pick_next_target()
    assert moved is True, 'render-side pick never defers (no worker provider)'
    assert axis.target_genome is not None, 'a target must always be produced'

    if movement == FOLLOW:
        assert axis._loop.has_loop, 'FOLLOW requires a loaded loop'
        cycle_len = axis._loop._loop_cycle_len
        assert cycle_len > 0
        expected = (step0 + 1) % cycle_len
        advanced = (axis._loop.loop_step == expected
                    and axis._loop.loop_step != step0) or (
                        # wrap: step returned to a smaller value, cycle ticked
                        axis._loop.loop_step == 0
                        and step0 == cycle_len - 1
                        and axis._loop.cycle_count == cycle0 + 1)
        assert advanced, (
            f'FOLLOW must advance one loop position: '
            f'step {step0} -> {axis._loop.loop_step} (cycle_len={cycle_len})')
    else:  # WALK
        assert axis._loop.loop_step == step0, (
            'WALK must NOT advance the loop step (the mode gate)')
        assert axis._loop.cycle_count == cycle0, 'WALK must not tick a cycle'


# ----------------------------------------------------------------
# Disconnected-walk specifics: a walk from a genome with no transitions must
# produce a FRESH factory genome (not one of the library's known ids, not None)
# and must not raise.
# ----------------------------------------------------------------

def test_disconnected_walk_falls_back_to_factory(rng):
    axis, lib = _make_axis('DISCONNECTED', Mode.IDLE, rng)
    assert axis.current_genome.db_id == 99          # on the isolated node
    assert lib.nearest_transitions(99) == []        # genuinely disconnected
    step0 = axis._loop.loop_step

    axis._pick_next_target()                        # must not raise

    tgt = axis.target_genome
    assert tgt is not None
    # Factory genomes are not registered in the library -> db_id is None (or at
    # least not one of the library's real ids). trivial_genome() leaves db_id
    # unset, so the factory fallback target carries no db id.
    assert getattr(tgt, 'db_id', None) is None, (
        'walk from a disconnected node must come from the factory, not the lib')
    assert axis._loop.loop_step == step0            # no loop, nothing advanced


def test_graph_walk_returns_a_neighbor_when_connected(rng):
    """Contrast to the disconnected case: a walk from a CONNECTED node returns
    one of its graph neighbors (a real library genome), not a factory genome."""
    axis, lib = _make_axis('GRAPH_NO_LOOPS', Mode.IDLE, rng)
    assert axis.current_genome.db_id == 0
    neighbor_ids = {b for b, _ in lib.nearest_transitions(0)}
    assert neighbor_ids, 'precondition: genome 0 has neighbors'

    axis._pick_next_target()

    assert axis.target_genome.db_id in neighbor_ids, (
        'walk from a connected node should land on a graph neighbor')


# ----------------------------------------------------------------
# Centroid swap: ENERGY-only. A forced centroid trigger WALKS in energy and does
# NOTHING in beat. Driven through ``tick`` (the centroid block lives there),
# monkeypatching the centroid detector to fire deterministically.
# ----------------------------------------------------------------

def test_energy_forced_centroid_walks(rng, monkeypatch):
    axis, _lib = _make_axis('ONE_LOOP', Mode.ENERGY, rng)
    axis._morph.state = MorphState.DWELL
    # Sit on a real loop genome so the walk has a db_id to traverse from.
    axis.current_genome = axis._loop.loop_genomes[0]
    monkeypatch.setattr(axis._centroid, 'tick', lambda *a: True)

    tgt0 = axis.target_genome
    step0 = axis._loop.loop_step
    axis.tick(_make_audio(mode='energy'), 1.0 / 60, 1000.0)

    assert axis.target_genome is not tgt0, 'energy centroid must WALK (new target)'
    assert axis._loop.loop_step == step0, 'a walk must not advance the loop step'


def test_beat_forced_centroid_does_not_swap(rng, monkeypatch):
    axis, _lib = _make_axis('ONE_LOOP', Mode.BEAT, rng)
    axis._morph.state = MorphState.DWELL
    monkeypatch.setattr(axis._centroid, 'tick', lambda *a: True)

    tgt0 = axis.target_genome
    step0 = axis._loop.loop_step
    # No kick events -> beat's dwell stays put; the centroid block is skipped
    # entirely because it is gated to Mode.ENERGY.
    axis.tick(_make_audio(mode='beat'), 1.0 / 60, 1000.0)

    assert axis.target_genome is tgt0, 'beat must NOT centroid-swap'
    assert axis._loop.loop_step == step0


# ----------------------------------------------------------------
# Bootstrap-stage sanity: the library states do model the timeline they claim.
# (Guards the fixtures themselves — a mis-built state would silently weaken the
# matrix above.)
# ----------------------------------------------------------------

@pytest.mark.parametrize('state_name, has_loop', [
    ('EMPTY', False),
    ('GRAPH_NO_LOOPS', False),
    ('DISCONNECTED', False),
    ('ONE_LOOP', True),
    ('MULTI_LOOP', True),
])
def test_boot_loads_loop_iff_loops_exist(state_name, has_loop, rng):
    """GenomeAxis boot loads a loop exactly when the library has one — the
    stage-0/1 states come up loopless, stage-2/3 come up on a loop."""
    axis, lib = _make_axis(state_name, Mode.IDLE, rng)
    assert axis._loop.has_loop is has_loop
    assert (axis.active_loop_id is not None) is has_loop
    assert (lib.loop_count() > 0) is has_loop
