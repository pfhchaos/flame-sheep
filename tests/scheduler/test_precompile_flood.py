"""Step-2 shader-warming redesign — the flood path + the loop-change wiring.

Step 1 moved near_tuples OFF the render thread INTO the worker, but the
worker still STREAMED the flood tuples back for the driver to enqueue
(FLOOD/FLOOD_DONE). **Step 2 removes that round-trip**: the worker computes
the flood and SELF-SCHEDULES the tuples into its own in-process queue. So
this file now covers:
  * compute_flood           — precompile_worker_gl  (section 1; was
                              handle_flood_request, minus the stdout stream)
  * _enqueue_loop_change_warm — runtime.wallpaper   (section 2; unchanged)
  * the driver shim's request serialization — precompile_driver (section 3;
    was the FLOOD/FLOOD_DONE recv assembly, now deleted plumbing)

The worker's in-process CompileQueue + dispatch routing (the scheduling that
MOVED into the worker) are covered in test_precompile_worker_queue.py.

Retrofitted to the component-spec / test-decomposition standard
(shader-warming.md -> "Test method & component map"): each touched component
carries a written BEHAVIOR SPEC, then the decomposition into unit (mock
deps) / simple-integration (real seam) / e2e (GL mocked). Coverage gaps are
NOTATED in-file with their reason (see "NOT TESTED HERE" at the bottom).

The flood tests run against the in-memory ``FakeLibrary`` (fake_library.py),
not a real ``Library(data_dir=...)``: the real one costs ~14s to seed
(Genome.random + scoring) and buys these tests nothing. The fake gives an
EXACT graph, so the assertions are exact. The one test that DOES touch a
real Library is the fake-fidelity proof, marked ``slow``.
"""
from __future__ import annotations

import numpy as np
import pytest

from flame_sheep.genome import Genome
from flame_sheep.storage.library import Library
from flame_sheep.scheduler.precompile_warm import (
    near_tuples, _genome_tuple,
)
from flame_sheep.scheduler.precompile_worker_gl import compute_flood

from tests.scheduler.fake_library import FakeLibrary, FakeGenome


# A small, EXACT catalog + transition graph shared by the fast flood tests.
# Graph edges: 0-1 (0.5), 0-2 (0.7), 1-3 (0.6). Var sets chosen so the flood
# produces a non-trivial, hand-verifiable tuple set (endpoints + the two
# distinct transition-union shaders, with the union==endpoint collapses
# exercised on the 0-2 and 2-0 edges).
def _make_graph() -> tuple[FakeLibrary, list[int]]:
    lib = FakeLibrary()
    lib.add_genome(0, FakeGenome({0, 1}, n_transforms=2))
    lib.add_genome(1, FakeGenome({2, 3}, n_transforms=2))
    lib.add_genome(2, FakeGenome({0},    n_transforms=1))
    lib.add_genome(3, FakeGenome({4},    n_transforms=1))
    lib.store_transition(0, 1, 0.5)
    lib.store_transition(0, 2, 0.7)
    lib.store_transition(1, 3, 0.6)
    return lib, [0, 1, 2, 3]


# The exact flood from seed [0], 2 hops, fan 5 over _make_graph(). Derived by
# hand from near_tuples' BFS + union rules (see module/graph comment above).
_EXPECTED_FLOOD = {
    (2, 0, frozenset({0, 1})),        # endpoint g0
    (2, 0, frozenset({2, 3})),        # endpoint g1
    (1, 0, frozenset({0})),           # endpoint g2
    (1, 0, frozenset({4})),           # endpoint g3
    (0, 0, frozenset({0, 1, 2, 3})),  # union g0|g1 (edge 0-1)
    (0, 0, frozenset({2, 3, 4})),     # union g1|g3 (edge 1-3)
}


# ================================================================
# 1. compute_flood  [home: precompile_worker_gl]
# ----------------------------------------------------------------
# BEHAVIOR SPEC. Given the worker's own Library and a flood spec
# {seed_ids, hops, fan_per_node}, compute_flood MUST:
#   (a) return EXACTLY the pure near_tuples flood (same DB, same math) —
#       it is the render thread's old inline work, relocated to the process,
#       and must not alter the result;
#   (b) return an empty set on empty seeds (nothing to flood from);
#   (c) swallow a MALFORMED spec and return an empty set (the task is dropped,
#       the worker loop keeps running) — mutation-provable. NB: per-genome DB
#       errors are already swallowed INSIDE near_tuples, so compute_flood's own
#       guard is what protects against bad spec FIELDS (non-int seed_ids, etc.)
#       that raise before near_tuples is reached;
#   (d) never touch GL (pure DB+math) — which is why it is unit-testable here.
# Step 2: there is NO stdout stream anymore — the caller (worker main loop)
# self-schedules the returned set. Decomposition: all unit-level (dep =
# FakeLibrary). The real-subprocess variant is a notated skip (bottom).
# ================================================================

def test_compute_flood_returns_exact_flood():
    lib, _ids = _make_graph()
    tuples = compute_flood(lib, {'seed_ids': [0], 'hops': 2,
                                 'fan_per_node': 5})
    # (a) exact flood, AND identical to the pure near_tuples on the same lib
    # (proves compute_flood is a pass-through to near_tuples, not a reimpl).
    assert tuples == _EXPECTED_FLOOD
    assert tuples == near_tuples(lib, [0], hops=2, fan_per_node=5)


def test_compute_flood_empty_seeds():
    # (b) nothing to flood from -> empty set, no DB work.
    assert compute_flood(FakeLibrary(), {'seed_ids': []}) == set()
    assert compute_flood(FakeLibrary(), {}) == set()


def test_compute_flood_swallows_malformed_spec():
    """(c) MUTATION GUARD — a malformed spec must not crash the worker's
    compile loop; compute_flood returns an empty set and the task is dropped.
    Delete the try/except in compute_flood and this goes red: the non-int
    seed_id raises ValueError out of the list comprehension, before
    near_tuples is ever called. (A raising Library would NOT prove this guard —
    near_tuples swallows per-genome DB errors itself — so the mutation target
    is deliberately a bad spec field.)"""
    assert compute_flood(FakeLibrary(), {'seed_ids': ['not-an-int']}) == set()


# ================================================================
# 2. _enqueue_loop_change_warm  [home: runtime.wallpaper]
# ----------------------------------------------------------------
# BEHAVIOR SPEC. On an active-loop change, running on the RENDER THREAD, it
# MUST:
#   (a) enqueue the loop members' own pipeline tuples at MED (priority 0) —
#       cheap pure _genome_tuple math on already-loaded genomes;
#   (b) request the 2-hop flood FROM THE WORKER (request_flood) for the loop
#       members that have a db_id, and NEVER compute near_tuples itself (DB on
#       the render thread is the bug the redesign removes) — mutation-provable;
#   (c) when no loop member has a db_id (factory/ephemeral), skip the flood
#       request entirely but still warm the members.
# Decomposition: unit-level with a mock driver (_FakeDriver captures the two
# seams: enqueue + request_flood). The "no DB on the render thread" guarantee
# is the durable mutation guard (test (b)). Unchanged by step 2 — the render
# side still calls the same two shim methods; only the worker internals moved.
# ================================================================

class _FakeDriver:
    def __init__(self):
        self.enqueued: list[tuple[int, list]] = []
        self.floods: list[tuple[list, int, int]] = []

    def enqueue(self, tuples, priority=0):
        tuples = list(tuples)
        self.enqueued.append((priority, tuples))
        return len(tuples)

    def request_flood(self, seed_ids, hops=2, fan_per_node=5):
        self.floods.append((list(seed_ids), hops, fan_per_node))


def test_loop_change_requests_flood_not_inline(monkeypatch):
    """(b) MUTATION GUARD — the durable "no DB on the render thread" proof.

    Revert _enqueue_loop_change_warm to call near_tuples inline and this test
    goes red on BOTH asserts: the _boom patch fires (near_tuples ran on the
    render thread) AND no flood is requested from the worker."""
    from flame_sheep.runtime import wallpaper as wp
    import flame_sheep.scheduler.precompile_warm as pw

    def _boom(*a, **k):
        raise AssertionError(
            'near_tuples ran on the render thread — redesign regression')
    monkeypatch.setattr(pw, 'near_tuples', _boom)

    rng = np.random.default_rng(3)
    g1 = Genome.random(rng); g1.db_id = 11
    g2 = Genome.random(rng); g2.db_id = 12

    drv = _FakeDriver()
    n_med, n_seeds = wp._enqueue_loop_change_warm(drv, [g1, g2])

    # Flood requested from the worker with the loop members' db ids.
    assert drv.floods == [([11, 12], 2, 5)]
    # (a) Loop members still enqueued at MED (priority 0) on the render thread.
    assert n_med == 2
    assert drv.enqueued and drv.enqueued[0][0] == 0
    assert n_seeds == 2


def test_loop_change_skips_flood_when_no_db_ids():
    """(c) Genomes without db_id (factory/ephemeral) -> no flood request, but
    loop members still warm at MED."""
    from flame_sheep.runtime import wallpaper as wp

    rng = np.random.default_rng(4)
    g = Genome.random(rng)  # db_id stays None
    drv = _FakeDriver()
    n_med, n_seeds = wp._enqueue_loop_change_warm(drv, [g])

    assert drv.floods == []
    assert n_seeds == 0
    assert n_med == 1


# ================================================================
# 3. driver shim request serialization  [home: precompile_driver]
# ----------------------------------------------------------------
# BEHAVIOR SPEC (step 2). The driver no longer owns a queue; enqueue /
# request_flood just serialize ONE newline-framed JSON request to the
# worker's stdin, and the worker queues/dedups/schedules internally:
#   enqueue(tuples, p)       -> {"enqueue": {"priority": p,
#                                 "tuples": [[n, hf, [sorted vars]], ...]}}
#                               and empty tuples -> no write, returns 0.
#   request_flood(seeds,...) -> {"flood": {"seed_ids", "hops",
#                                 "fan_per_node", "priority"}}
#                               and empty seeds -> no write (no-op).
# Decomposition: simple-integration on the real PrecompileDriver with NO
# subprocess started — a FakeProc (fake_pipe.py) captures stdin. GPU-free.
# (The old FLOOD/FLOOD_DONE recv assembly this section used to test is DELETED
# plumbing; its coverage is retired with it.)
# ================================================================

def _make_shim():
    from flame_sheep.scheduler.precompile_driver import PrecompileDriver
    from tests.scheduler.fake_pipe import FakeProc
    drv = PrecompileDriver(canvas_w=0, canvas_h=0, policy=object(),
                           worker_module='flame_sheep.scheduler.'
                                         'precompile_worker_gl')
    proc = FakeProc()
    drv._proc = proc  # no start(); no real subprocess / GL
    return drv, proc


def test_enqueue_serializes_batch_request():
    drv, proc = _make_shim()
    try:
        t_a = (3, 1, frozenset({1, 2}))
        t_b = (0, 0, frozenset({5, 1, 2}))
        n = drv.enqueue([t_a, t_b], priority=-100)
    finally:
        drv._flag.close()

    assert n == 2  # requested count (dedup now happens in the worker)
    objs = proc.objects_written()
    assert len(objs) == 1
    assert objs[0] == {'enqueue': {
        'priority': -100,
        # has_final preserved as 0/1; vars SORTED on the wire.
        'tuples': [[3, 1, [1, 2]], [0, 0, [1, 2, 5]]],
    }}


def test_enqueue_empty_is_noop():
    drv, proc = _make_shim()
    try:
        assert drv.enqueue([], priority=0) == 0
    finally:
        drv._flag.close()
    assert proc.lines_written() == []  # nothing written


def test_request_flood_serializes_request():
    drv, proc = _make_shim()
    try:
        drv.request_flood([11, 12], hops=2, fan_per_node=5, priority=0)
        # Empty seeds -> no-op (nothing to flood from).
        drv.request_flood([])
    finally:
        drv._flag.close()

    objs = proc.objects_written()
    assert objs == [{'flood': {'seed_ids': [11, 12], 'hops': 2,
                               'fan_per_node': 5, 'priority': 0}}]


# ================================================================
# FAKE-LIBRARY FIDELITY  [infra verification]
# ----------------------------------------------------------------
# Proves FakeLibrary.load_genome / nearest_transitions behave like the real
# Library on a tiny graph, so the fast tests above can trust the fake. This is
# the ONLY test here that touches a real Library(data_dir=...), so it is marked
# `slow` (Genome.random cost) to keep the default/fast suite DB-free. Scoring
# is skipped by passing an explicit scores dict to save_genome.
# ================================================================

@pytest.mark.slow
def test_fake_library_matches_real(tmp_path):
    rng = np.random.default_rng(7)
    genomes = [Genome.random(rng) for _ in range(3)]
    dummy_scores = dict(coverage=0.2, entropy=0.5, color_entropy=0.5,
                        balance=0.5, complexity=0.5, edge_sharpness=0.1,
                        contour_coherence=0.1, centroid_offset_x=0.0,
                        centroid_offset_y=0.0)

    real = Library(data_dir=tmp_path)
    fake = FakeLibrary()
    try:
        ids = []
        for g in genomes:
            gid = real.save_genome(g, scores=dummy_scores)
            fake.add_genome(gid, g)
            ids.append(gid)
        # Distinct distances so ORDER BY has no tie ambiguity across backends.
        real.store_transition(ids[0], ids[1], 0.5)
        real.store_transition(ids[0], ids[2], 0.7)
        real.store_transition(ids[1], ids[2], 0.6)
        fake.store_transition(ids[0], ids[1], 0.5)
        fake.store_transition(ids[0], ids[2], 0.7)
        fake.store_transition(ids[1], ids[2], 0.6)

        # nearest_transitions: identical (genome_b, distance) ordering + limit.
        for gid in ids:
            for n in (1, 2, 3, 10):
                assert (fake.nearest_transitions(gid, n)
                        == real.nearest_transitions(gid, n)), (gid, n)

        # load_genome: db_id stamped, and the pipeline tuple round-trips
        # identically through both backends.
        for gid in ids:
            assert fake.load_genome(gid).db_id == gid
            assert real.load_genome(gid).db_id == gid
            assert _genome_tuple(fake.load_genome(gid)) == \
                   _genome_tuple(real.load_genome(gid))

        # KeyError on an unknown id, same as the real one.
        missing = max(ids) + 999
        with pytest.raises(KeyError):
            real.load_genome(missing)
        with pytest.raises(KeyError):
            fake.load_genome(missing)
    finally:
        real.close()
        fake.close()


# ================================================================
# NOT TESTED HERE — deliberate coverage gaps, with reasons
# (shader-warming.md standard: gaps are notated in-file, never silent).
# ----------------------------------------------------------------
# (1) REAL-SUBPROCESS END-TO-END (render thread -> real worker stdin ->
#     dispatch_line -> CompileQueue -> compute_flood -> self-schedule ->
#     real GL compile) is NOT tested — GPU-GATED: the real worker
#     (precompile_worker_gl.main) creates a standalone moderngl context +
#     FlameRenderer at startup before it ever reads stdin, so it can't run in
#     the headless/CI fast suite. Every piece either side of the GL boundary
#     IS covered in isolation: compute_flood here (section 1, fake lib), the
#     CompileQueue + dispatch routing in test_precompile_worker_queue.py, and
#     the driver shim serialization here (section 3).
#
# (2) THE ACTUAL GL SHADER COMPILE (_get_chaos_shader_for_keep_vars warming
#     Mesa's disk cache) is GPU-gated — real GL context + driver required.
#     Verified manually / on-box, not in this suite. compute_flood and the
#     queue are explicitly the GPU-FREE halves, which is why they ARE unit
#     tested; the compile half is unreachable without a GPU.
