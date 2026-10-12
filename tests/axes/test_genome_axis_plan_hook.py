"""GenomeAxis step-3 PLAN/A-B arrival hook (shader-warming.md step 3, render
side).

BEHAVIOR SPEC. When a plan hook is wired via ``set_plan_hook`` and the axis makes
a loop-mode swap (``_swap_next_genome``), the axis fires
``on_arrival(arrival_genome, loop_id, loop_step, expected_follow)`` EXACTLY ONCE,
BEFORE the sequence advances, where:
  * arrival_genome is the current (just-arrived) genome,
  * loop_step is the position arrived at (pre-advance),
  * expected_follow is the render thread's OWN upcoming FOLLOW
    (``_loop.peek_next_in_sequence`` at that step).
It is purely observational: the REAL swap still comes from ``_loop`` (authority
stays render-side until step 4), and the hook must not change which target the
axis commits. It does NOT fire for a lerp-intermediate arrival (db_id None) or
outside loop mode.

Decomposition: unit — white-box, with the LoopPlayer's loop state populated
directly (no Library, no GL). Mutation-provable: if the emit moved AFTER
swap_next, loop_step would come back advanced and the assert would go red.
"""
from __future__ import annotations

import numpy as np
import pytest

from flame_sheep.axes.genome_axis import GenomeAxis
from flame_sheep.axes._loop_player import follow_after
from flame_sheep.loops import loop_sequence
from flame_sheep.runtime import RoleMapper

from viz_helpers import trivial_genome


@pytest.fixture
def axis():
    _seed = iter(range(1000))
    return GenomeAxis(genome_factory=lambda: trivial_genome(next(_seed)),
                      role=RoleMapper(), lib=None,
                      rng=np.random.default_rng(42))


def _load_loop_state(axis, genomes, structure='cyclic', step=1, loop_id=10):
    """Populate the axis's LoopPlayer with a live loop at a given position,
    advancing the generator to match ``step`` (so next_in_sequence agrees with
    the step counter)."""
    lp = axis._loop
    lp._loop_genomes = list(genomes)
    lp._loop_structure = structure
    seq = loop_sequence(genomes, structure)
    for _ in range(step):
        next(seq)
    lp._loop_sequence = seq
    lp._loop_cycle_len = len(genomes)
    lp._loop_step = step
    lp.active_loop_id = loop_id


def test_arrival_hook_fires_with_preadvance_state(axis):
    g0, g1, g2 = (trivial_genome(i) for i in range(3))
    for i, g in enumerate((g0, g1, g2)):
        g.db_id = i
    _load_loop_state(axis, [g0, g1, g2], step=1, loop_id=10)
    axis.current_genome = g0

    calls = []
    axis.set_plan_hook(lambda *a: calls.append(a))

    axis._swap_next_genome()

    # Fired exactly once, with the PRE-advance step and the render-side peek.
    assert len(calls) == 1
    arrival, loop_id, loop_step, expected_follow = calls[0]
    assert arrival is g0
    assert loop_id == 10
    assert loop_step == 1                                  # pre-advance
    assert expected_follow is follow_after([g0, g1, g2], 'cyclic', 1)
    assert expected_follow is g1

    # Authority UNCHANGED: the real swap still came from _loop (the sequence
    # advanced and target is the loop's pick, not anything the hook supplied).
    assert axis._loop._loop_step == 2
    assert axis.target_genome is g1


def test_no_hook_for_db_id_none_arrival(axis):
    """A lerp-intermediate arrival (db_id None) must NOT fire the hook — the
    planner keys off a real arrival id."""
    g0, g1 = trivial_genome(0), trivial_genome(1)
    g1.db_id = 1
    _load_loop_state(axis, [g0, g1], step=0)
    axis.current_genome = g0            # db_id stays None
    calls = []
    axis.set_plan_hook(lambda *a: calls.append(a))
    axis._swap_next_genome()
    assert calls == []


def test_no_hook_without_loop(axis):
    """Outside loop mode (empty loop), no FOLLOW to compare → no hook fire."""
    axis._loop._loop_genomes = []
    g = trivial_genome(0)
    g.db_id = 5
    axis.current_genome = g
    calls = []
    axis.set_plan_hook(lambda *a: calls.append(a))
    axis._swap_next_genome()
    assert calls == []


def test_hook_exception_does_not_break_swap(axis):
    """A throwing hook must not break the real swap (observational only)."""
    g0, g1, g2 = (trivial_genome(i) for i in range(3))
    g0.db_id, g1.db_id, g2.db_id = 0, 1, 2
    _load_loop_state(axis, [g0, g1, g2], step=1, loop_id=3)
    axis.current_genome = g0

    def _boom(*a):
        raise RuntimeError('hook blew up')
    axis.set_plan_hook(_boom)

    axis._swap_next_genome()            # must not raise
    assert axis._loop._loop_step == 2   # swap still happened
