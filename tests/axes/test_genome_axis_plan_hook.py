"""GenomeAxis PLAN arrival hook (shader-warming.md; updated for step-4 S2 and the
S5 A/B retirement).

BEHAVIOR SPEC. When a plan hook is wired via ``set_plan_hook``, the axis fires
``on_arrival(arrival_genome, loop_id, loop_step)`` at each loop-mode ARRIVAL —
via ``_emit_arrival``, called when a swap commits (morph-complete / the
commit-kick), BEFORE the next pick advances the sequence — where:
  * arrival_genome is the current (just-arrived) genome,
  * loop_step is the position arrived at (the next pick will advance it).
It is purely observational: ``_emit_arrival`` is a PURE PEEK (it must not advance
the sequence or change which target the axis commits). It does NOT fire for a
lerp-intermediate arrival (db_id None) or outside loop mode.

NOTE (S2): the hook used to be fired from inside ``_swap_next_genome``; the pick
moved to dwell-release, so ``_emit_arrival`` now fires at the arrival/commit and
``_swap_next_genome`` is just the pick. These tests target ``_emit_arrival``
directly (the emitter unit); that it fires at the morph-complete commit is
covered by ``test_genome_axis.py::TestPickRelocation``.

NOTE (S5): the hook no longer carries ``expected_follow`` — that fed the retired
render-side A/B compare; the worker's targets are now consumed live, so there is
no second opinion to ship.

Decomposition: unit — white-box, LoopPlayer state populated directly (no Library,
no GL). Mutation-provable: if ``_emit_arrival`` advanced the sequence, loop_step
would come back changed and the asserts would go red.
"""
from __future__ import annotations

import numpy as np
import pytest

from flame_sheep.axes.genome_axis import GenomeAxis
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


def test_arrival_hook_fires_with_arrival_state(axis):
    g0, g1, g2 = (trivial_genome(i) for i in range(3))
    for i, g in enumerate((g0, g1, g2)):
        g.db_id = i
    _load_loop_state(axis, [g0, g1, g2], step=1, loop_id=10)
    axis.current_genome = g0

    calls = []
    axis.set_plan_hook(lambda *a: calls.append(a))

    axis._emit_arrival()

    # Fired exactly once, with the arrived position.
    assert len(calls) == 1
    arrival, loop_id, loop_step = calls[0]
    assert arrival is g0
    assert loop_id == 10
    assert loop_step == 1
    # PURE PEEK: _emit_arrival must NOT advance the live sequence.
    assert axis._loop._loop_step == 1


def test_no_hook_for_db_id_none_arrival(axis):
    """A lerp-intermediate arrival (db_id None) must NOT fire the hook — the
    planner keys off a real arrival id."""
    g0, g1 = trivial_genome(0), trivial_genome(1)
    g1.db_id = 1
    _load_loop_state(axis, [g0, g1], step=0)
    axis.current_genome = g0            # db_id stays None
    calls = []
    axis.set_plan_hook(lambda *a: calls.append(a))
    axis._emit_arrival()
    assert calls == []


def test_hook_fires_without_loop(axis):
    """S4a BEHAVIOR CHANGE: a NO-LOOP arrival with a real db_id now FIRES the hook
    (it used not to), so the worker gets an id to plan WALK from — closing the
    populated-but-no-loop bootstrap gap where idle/energy would otherwise DB on the
    render thread. loop_id is None (no active loop), loop_step 0."""
    axis._loop._loop_genomes = []
    g = trivial_genome(0)
    g.db_id = 5
    axis.current_genome = g
    calls = []
    axis.set_plan_hook(lambda *a: calls.append(a))
    axis._emit_arrival()
    assert len(calls) == 1
    arrival, loop_id, loop_step = calls[0]
    assert arrival is g
    assert loop_id is None
    assert loop_step == 0


def test_hook_exception_does_not_propagate(axis):
    """A throwing hook must not escape _emit_arrival (observational only) nor
    disturb the sequence."""
    g0, g1, g2 = (trivial_genome(i) for i in range(3))
    g0.db_id, g1.db_id, g2.db_id = 0, 1, 2
    _load_loop_state(axis, [g0, g1, g2], step=1, loop_id=3)
    axis.current_genome = g0

    def _boom(*a):
        raise RuntimeError('hook blew up')
    axis.set_plan_hook(_boom)

    axis._emit_arrival()                 # must not raise
    assert axis._loop._loop_step == 1    # pure peek, unchanged
