"""Worker/planner component (shader-warming.md map #1, #4): the planner that
computes the 3 next-move targets in the worker process (Option A, step 3).

BEHAVIOR SPEC. ``WorkerPlanner.plan(arrival_id, loop_id, loop_step)`` returns the
three ``target_to_wire`` dicts under keys follow/break/walk:
  * FOLLOW = next-in-sequence of the active loop — DETERMINISTIC given
    (loop_id, loop_step); must equal the shared ``follow_after`` helper's result
    (the exact same code the render thread peeks with), and carry the target's
    keep_vars. ``None`` when loop_id is None.
  * WALK = ``graph_walk`` of G — stochastic (worker's own RNG), so the SPEC is
    structural: the pick is one of G's nearest transitions, and the shipped
    ``neighborhood`` is that neighbor id set.
  * BREAK = ``next_loop`` of G — stochastic; structural spec: the pick is a loop
    entry genome reachable from G (``nearest_loop_transitions``), with the entry
    set shipped as ``neighborhood``; ``None`` when no loop is reachable.
  * plan() NEVER raises — a bad arrival id yields all-None.

The planner REUSES the production ``LoopPlayer.graph_walk`` / ``next_loop`` (no
parallel re-impl) + its own RNG. Decomposition: unit — in-memory ``FakeLibrary``
graph + seeded RNG; genomes are lightweight REAL ``Genome`` objects (so
``distance``/``_describe_genome``/``to_dict`` work) with pinned keep_vars, not
``Genome.random`` (no 14s fixture). The real worker SUBPROCESS + its DB are a
GPU/process-gated skip — notated at the bottom.
"""
from __future__ import annotations

import numpy as np

from flame_sheep.genome import Genome, Transform
from flame_sheep.axes._loop_player import follow_after
from flame_sheep.scheduler.planner import WorkerPlanner
from flame_sheep.scheduler.plan_protocol import target_from_wire

from tests.scheduler.fake_library import FakeLibrary


def _mk_genome(var_ids, n_transforms: int | None = None) -> Genome:
    """A lightweight REAL genome whose keep_vars == ``var_ids`` (each listed
    variation index gets weight on the first transform). Instant — no viability
    loop, no scoring."""
    ids = sorted(int(v) for v in var_ids)
    n = n_transforms if n_transforms is not None else max(1, len(ids))
    trs = []
    for i in range(n):
        t = Transform()
        t.variations = np.zeros(t.variations.shape[0], dtype=np.float32)
        t.weight = 1.0 + 0.1 * i
        trs.append(t)
    for v in ids:
        trs[0].variations[v] = 0.5
    return Genome(transforms=trs)


def _graph() -> FakeLibrary:
    """Catalog + graph:
      genomes 0..4 (distinct keep_vars), G=0.
      transitions 0-1(.5) 0-2(.7) 0-3(.6)  (so 0's nearest = [1,3,2] by dist).
      loop #10=[0,1,2] cyclic  (G's current loop),
      loop #20=[3,4] cyclic    (reachable from 0 via 0->3, entry genome 3)."""
    lib = FakeLibrary()
    lib.add_genome(0, _mk_genome({0, 1}), scores={'coverage': 0.5})
    lib.add_genome(1, _mk_genome({2, 3}), scores={'coverage': 0.6})
    lib.add_genome(2, _mk_genome({4}),    scores={'coverage': 0.9})
    lib.add_genome(3, _mk_genome({5, 6}), scores={'coverage': 0.7})
    lib.add_genome(4, _mk_genome({7}),    scores={'coverage': 0.8})
    lib.store_transition(0, 1, 0.5)
    lib.store_transition(0, 2, 0.7)
    lib.store_transition(0, 3, 0.6)
    lib.add_loop(10, [0, 1, 2], 'cyclic', fitness=1.0)
    lib.add_loop(20, [3, 4], 'cyclic', fitness=2.0)
    return lib


def _planner(seed=0):
    return WorkerPlanner(_graph(), rng=np.random.default_rng(seed))


# ================================================================
# 1. FOLLOW — deterministic, matches the shared follow_after
# ----------------------------------------------------------------
def test_follow_matches_shared_helper_every_step():
    """MUTATION GUARD on the step arithmetic: the planner's FOLLOW must equal
    follow_after(loop_genomes, structure, step) for each step. If the planner
    off-by-ones the step, some step's assert goes red."""
    lib = _graph()
    genomes = [g for _, g, _ in lib.load_loop(10)]
    wp = WorkerPlanner(lib, rng=np.random.default_rng(0))
    for step in range(5):                       # wraps past the 3-cycle
        follow = target_from_wire(wp._follow(10, step))
        expect = follow_after(genomes, 'cyclic', step)
        assert follow.id == expect.db_id, f'step {step}'
        # keep_vars travels too (catches a correct-id/wrong-genome bug).
        from flame_sheep.scheduler.precompile_warm import _genome_tuple
        assert follow.keep_vars == _genome_tuple(expect)[2], f'step {step}'


def test_follow_none_without_loop():
    wp = _planner()
    assert wp._follow(None, 0) is None


# ================================================================
# 2. WALK — structural: valid candidate from the right neighborhood
# ----------------------------------------------------------------
def test_walk_picks_from_nearest_transitions():
    """The walk target is one of G's graph neighbors, and the shipped
    neighborhood is exactly that neighbor id set (what the render-side structural
    check validates against)."""
    wp = _planner(seed=1)
    walk = target_from_wire(wp._walk(wp._lib.load_genome(0)))
    neigh = {b for b, _ in wp._lib.nearest_transitions(0, n=10)}
    assert neigh == {1, 2, 3}
    assert walk.id in neigh
    assert set(walk.neighborhood) == neigh


def test_walk_single_neighbor_is_forced():
    """MUTATION GUARD: with exactly one neighbor the walk MUST pick it — proves
    the planner queries G's neighborhood (not some other node / empty set)."""
    lib = FakeLibrary()
    lib.add_genome(0, _mk_genome({0}), scores={'coverage': 0.5})
    lib.add_genome(9, _mk_genome({9}), scores={'coverage': 0.5})
    lib.store_transition(0, 9, 0.5)
    wp = WorkerPlanner(lib, rng=np.random.default_rng(3))
    walk = target_from_wire(wp._walk(lib.load_genome(0)))
    assert walk.id == 9


# ================================================================
# 3. BREAK — structural: valid loop entry reachable from G
# ----------------------------------------------------------------
def test_break_picks_reachable_loop_entry():
    wp = _planner(seed=2)
    brk = target_from_wire(wp._break(wp._lib.load_genome(0), loop_id=10))
    # Only loop #20 is reachable from 0 (via edge 0->3), entry genome 3.
    entries = {e for _l, _f, _d, e in
               wp._lib.nearest_loop_transitions(0, exclude_loop=10, n=3)}
    assert entries == {3}
    assert brk.id in entries
    assert set(brk.neighborhood) == entries


def test_break_none_when_no_loops_at_all():
    """With zero loops, ``next_loop`` returns None up front → BREAK is None.
    (NB: with a single loop that IS the current one, production ``next_loop``
    FALLS BACK to re-entering it rather than returning None — the planner
    faithfully reproduces that, so "no loops at all" is the real None trigger.)"""
    lib = FakeLibrary()
    lib.add_genome(0, _mk_genome({0}), scores={'coverage': 0.5})
    lib.add_genome(1, _mk_genome({1}), scores={'coverage': 0.5})
    lib.store_transition(0, 1, 0.5)
    wp = WorkerPlanner(lib, rng=np.random.default_rng(0))
    assert wp._break(lib.load_genome(0), loop_id=None) is None


# ================================================================
# 4. plan() isolates failures
# ----------------------------------------------------------------
def test_plan_bad_arrival_id_all_none_no_raise():
    wp = _planner()
    moves = wp.plan(arrival_id=999, loop_id=10, loop_step=0)
    assert moves == {'follow': None, 'break': None, 'walk': None}


def test_plan_full_shape():
    wp = _planner(seed=5)
    moves = wp.plan(arrival_id=0, loop_id=10, loop_step=1)
    assert set(moves) == {'follow', 'break', 'walk'}
    # follow at step 1 of [0,1,2] cyclic == genome 1.
    assert target_from_wire(moves['follow']).id == 1


# ================================================================
# NOT TESTED HERE (notated skips, shader-warming.md "Test method"):
#   * The real worker SUBPROCESS answering a PLAN line end-to-end over real
#     pipes with a real Library + GL context (map #5 "worker actually compiles"
#     is GPU-gated; the planner thread + real DB is process/GPU-gated). Manual /
#     on-box verification. The planner LOGIC is fully covered above against the
#     FakeLibrary; the subprocess plumbing (dispatch routing to the plan queue,
#     stdout response framing, driver request serialization) is covered in
#     test_plan_protocol.py without a real process.
# ================================================================
