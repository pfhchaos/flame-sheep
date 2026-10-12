"""Step-1 shader-warming redesign — the loop-change flood's near_tuples
computation moves OFF the render thread INTO the precompile worker process.

Retrofitted to the component-spec / test-decomposition standard
(shader-warming.md -> "Test method & component map"). Each touched component
below carries a written BEHAVIOR SPEC, then the decomposition into unit (mock
deps) / simple-integration (real seam) / e2e (GL mocked). Coverage gaps are
NOTATED in-file with their reason (see "NOT TESTED HERE" at the bottom) —
never silent.

The flood-computation tests run against the in-memory ``FakeLibrary``
(tests/scheduler/fake_library.py), not a real ``Library(data_dir=...)``: the
real one costs ~14s to seed (Genome.random + scoring) and buys these tests
nothing. The fake gives an EXACT graph, so the assertions are exact, not
"== whatever near_tuples returned". The one test that DOES touch a real
Library is the fake-fidelity proof, and it is marked ``slow`` so the default
(fast) suite stays DB-free.

Touched components (home -> spec is at the head of its section):
  * handle_flood_request  — precompile_worker_gl  (section 1)
  * _enqueue_loop_change_warm — runtime.wallpaper  (section 2)
  * request_flood + the FLOOD/FLOOD_DONE recv handling — precompile_driver
    (section 3)
"""
from __future__ import annotations

import io
import types

import numpy as np
import pytest

from flame_sheep.genome import Genome
from flame_sheep.storage.library import Library
from flame_sheep.scheduler.precompile_warm import (
    near_tuples, tuple_to_json_line, json_line_to_tuple, _genome_tuple,
)
from flame_sheep.scheduler.precompile_worker_gl import handle_flood_request

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
# 1. handle_flood_request  [home: precompile_worker_gl]
# ----------------------------------------------------------------
# BEHAVIOR SPEC. Given the worker's own Library and a flood spec
# {seed_ids, hops, fan_per_node}, handle_flood_request MUST:
#   (a) compute EXACTLY the pure near_tuples flood (same DB, same math) —
#       it is the render thread's old inline work, relocated to the process,
#       and must not alter the result;
#   (b) stream one ``FLOOD <json>`` line per tuple then ``FLOOD_DONE <count>``;
#   (c) on empty seeds, compute nothing and emit just ``FLOOD_DONE 0``;
#   (d) never touch GL (pure DB+math) — which is why it is unit-testable here.
# Decomposition: all unit-level (deps = FakeLibrary, a StringIO for `out`).
# The real-subprocess variant is a notated skip (see bottom of file).
# ================================================================

def test_handle_flood_request_computes_exact_flood():
    lib, ids = _make_graph()
    buf = io.StringIO()

    tuples = handle_flood_request(lib, {'seed_ids': [0], 'hops': 2,
                                        'fan_per_node': 5}, buf)

    # (a) exact flood, AND identical to the pure near_tuples on the same lib
    # (proves the handler is a pass-through to near_tuples, not a reimpl).
    assert tuples == _EXPECTED_FLOOD
    assert tuples == near_tuples(lib, [0], hops=2, fan_per_node=5)


def test_handle_flood_request_protocol_roundtrips():
    lib, ids = _make_graph()
    buf = io.StringIO()

    tuples = handle_flood_request(lib, {'seed_ids': [0], 'hops': 2,
                                        'fan_per_node': 5}, buf)

    # (b) one FLOOD line per tuple, terminal FLOOD_DONE <count>, and the
    # streamed lines parse back to exactly the computed set.
    lines = buf.getvalue().strip().splitlines()
    assert lines[-1] == f'FLOOD_DONE {len(tuples)}'
    parsed = {
        json_line_to_tuple(ln[len('FLOOD '):])
        for ln in lines if ln.startswith('FLOOD ')
    }
    assert parsed == tuples == _EXPECTED_FLOOD


def test_handle_flood_request_empty_seeds():
    lib = FakeLibrary()
    buf = io.StringIO()

    tuples = handle_flood_request(lib, {'seed_ids': []}, buf)

    # (c) nothing computed, just the terminal marker (releases the driver's
    # send thread even on an empty/failed flood).
    assert tuples == set()
    assert buf.getvalue().strip() == 'FLOOD_DONE 0'


# ================================================================
# 2. _enqueue_loop_change_warm  [home: runtime.wallpaper]
# ----------------------------------------------------------------
# BEHAVIOR SPEC. On an active-loop change, running on the RENDER THREAD, it
# MUST:
#   (a) enqueue the loop members' own pipeline tuples at MED (priority 0) —
#       cheap pure _genome_tuple math on already-loaded genomes;
#   (b) request the 2-hop flood FROM THE WORKER (request_flood) for the loop
#       members that have a db_id, and NEVER compute near_tuples itself (DB on
#       the render thread is the bug step 1 removes) — mutation-provable;
#   (c) when no loop member has a db_id (factory/ephemeral), skip the flood
#       request entirely but still warm the members.
# Decomposition: unit-level with a mock driver (_FakeDriver captures the two
# seams: enqueue + request_flood). The "no DB on the render thread" guarantee
# is the durable mutation guard (test (b)).
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
    render thread) AND no flood is requested from the worker. (Re-verified red
    under that mutation on 2026-10-11.)"""
    from flame_sheep.runtime import wallpaper as wp
    import flame_sheep.scheduler.precompile_warm as pw

    def _boom(*a, **k):
        raise AssertionError(
            'near_tuples ran on the render thread — step-1 regression')
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
# 3. request_flood + FLOOD/FLOOD_DONE recv  [home: precompile_driver]
# ----------------------------------------------------------------
# BEHAVIOR SPEC.
#   request_flood MUST enqueue a single 'flood' control payload (seed_ids +
#   params) onto the SAME priority queue as compiles — so the worker computes
#   it in priority order, not on the render thread — and MUST no-op on empty
#   seed_ids.
#   The recv side MUST buffer each streamed ``FLOOD <json>`` tuple and, on
#   ``FLOOD_DONE``, enqueue the whole buffer at LOW (priority 10) IN ORDER,
#   then reset the buffer for the next flood.
# Decomposition: simple-integration on the real PrecompileDriver object with
# NO subprocess started (construction + these two methods are GPU-free); the
# IPC/queue plumbing is an INTERIM seam (step 2 collapses it into the worker),
# so a lighter test is acceptable per the standard.
# ================================================================

def test_request_flood_enqueues_control_payload():
    from flame_sheep.scheduler.precompile_driver import PrecompileDriver

    drv = PrecompileDriver(canvas_w=0, canvas_h=0, policy=object(),
                           worker_module='flame_sheep.scheduler.'
                                         'precompile_worker_gl')
    drv.request_flood([11, 12], hops=2, fan_per_node=5)

    # One item on the queue: (priority, seq, payload-dict). No subprocess ran.
    assert drv._q.qsize() == 1
    priority, _seq, payload = drv._q.get_nowait()
    assert payload == {'type': 'flood', 'seed_ids': [11, 12],
                       'hops': 2, 'fan_per_node': 5}

    # Empty seeds -> no-op (nothing to flood from).
    drv.request_flood([])
    assert drv._q.qsize() == 0
    drv._flag.close()


def test_recv_loop_enqueues_flood_tuples_at_low(monkeypatch):
    from flame_sheep.scheduler.precompile_driver import PrecompileDriver

    drv = PrecompileDriver(canvas_w=0, canvas_h=0, policy=object(),
                           worker_module='flame_sheep.scheduler.'
                                         'precompile_worker_gl')
    captured: list[tuple[int, list]] = []
    monkeypatch.setattr(
        drv, 'enqueue',
        lambda tuples, priority=0: (
            captured.append((priority, list(tuples))) or len(list(tuples))))

    t_a = (3, 0, frozenset({1, 2}))
    t_b = (0, 0, frozenset({1, 2, 5}))
    lines = iter([
        'FLOOD ' + tuple_to_json_line(t_a),
        'FLOOD ' + tuple_to_json_line(t_b),
        'FLOOD_DONE 2',
        '',  # EOF -> recv loop exits
    ])

    class _FakeStdout:
        def readline(self):
            return next(lines, '')

    drv._proc = types.SimpleNamespace(stdout=_FakeStdout())
    try:
        drv._recv_loop()
    finally:
        drv._flag.close()

    # FLOOD_DONE flushed the buffer to enqueue at LOW (priority 10), in order.
    assert captured == [(10, [t_a, t_b])]
    # Buffer reset for the next flood.
    assert drv._flood_buf == []


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
# (1) REAL-SUBPROCESS END-TO-END round-trip (render thread -> real worker
#     stdin -> handle_flood_request -> stdout -> driver recv -> enqueue) is
#     NOT tested. Two reasons, either sufficient:
#       - INTERIM SEAM: this split (driver owns the queue; worker just ships
#         tuples home) is step-1 scaffolding that step 2 collapses — the
#         planner + queue move INTO the worker — so an e2e lock-in test here
#         would test plumbing we are about to delete.
#       - GPU-GATED: the real worker (precompile_worker_gl.main) creates a
#         standalone moderngl context + FlameRenderer at startup before it ever
#         reads stdin, so it cannot run in the headless/CI fast suite. The
#         pieces either side of the GL boundary ARE covered in isolation:
#         handle_flood_request (section 1, fake lib) and the driver recv
#         (section 3, fake stdout).
#
# (2) THE ACTUAL GL SHADER COMPILE (_get_chaos_shader_for_keep_vars warming
#     Mesa's disk cache) is GPU-gated — it needs a real GL context + driver.
#     Verified manually / on-box, not in this suite. handle_flood_request is
#     explicitly the GPU-FREE half (pure DB+math), which is why it IS unit-
#     tested; the compile half is not reachable without a GPU.
#
# (3) THE WORKER'S STDIN DISPATCH (routing a {"flood": ...} line to
#     handle_flood_request vs. treating a line as a compile spec) is NOT
#     unit-tested. DECISION: notated GPU-gated skip, not a unit. The routing
#     lives INLINE in precompile_worker_gl.main()'s `for line in sys.stdin`
#     loop, which runs only AFTER main() has created the moderngl context +
#     FlameRenderer + Library. It is not factored into a separately-callable,
#     GL-free function, so it cannot be exercised without initializing GL, and
#     extracting it would mean editing the step-1 implementation (out of scope
#     for this retrofit). handle_flood_request — the branch target — is fully
#     covered in section 1. If step 2 factors the stdin loop out of main()
#     (likely, since the queue moves into the worker), add the mutation-
#     provable routing unit then.
