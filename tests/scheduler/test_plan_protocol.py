"""PLAN protocol (shader-warming.md step 3): the render ↔ worker wire format,
the render-side A/B comparison, and the worker/driver plumbing that carries it —
WITHOUT a real subprocess or GL.

Covers (component map #6/#7-ish for the render side, #13 IPC framing):
  * wire framing round-trip (request / target / response), incl. a serialized
    Genome inside a target;
  * the A/B comparison logic — FOLLOW exact-match, WALK/BREAK structural —
    which handles the worker-vs-render RNG divergence (the critical subtlety);
  * ``dispatch_line`` routing a PLAN to the PREEMPT lane (plan_queue), never the
    compile queue (resolved decision #3);
  * ``PrecompileDriver.request_plan`` serialization + the stdout reader
    dispatching a response to ``on_plan_response``.

All GPU-free / process-free: ``CompileQueue``/``dispatch_line`` touch no GL; the
driver tests use ``FakeProc`` (fake_pipe.py) instead of a real subprocess. The
real worker subprocess answering over real pipes is the notated skip (bottom).
"""
from __future__ import annotations

import json

import numpy as np

from flame_sheep.genome import Genome, Transform
from flame_sheep.scheduler import plan_protocol as pp
from flame_sheep.scheduler.plan_protocol import (
    make_plan_request, parse_plan_request,
    make_plan_response, parse_plan_response,
    target_to_wire, target_from_wire, PlanTarget,
    compare_follow, check_walk_structure, check_break_structure,
    MATCH, MISMATCH, STRUCTURAL_OK, STRUCTURAL_FAIL, EMPTY,
)
from flame_sheep.scheduler.precompile_worker_gl import dispatch_line

from tests.scheduler.fake_pipe import FakeProc


def _mk_genome(var_ids, db_id=None) -> Genome:
    g = Genome(transforms=[Transform()])
    g.transforms[0].variations = np.zeros(
        g.transforms[0].variations.shape[0], dtype=np.float32)
    for v in var_ids:
        g.transforms[0].variations[v] = 0.5
    g.db_id = db_id
    return g


# ================================================================
# 1. Wire framing round-trips
# ================================================================
def test_request_roundtrip():
    obj = make_plan_request(req=7, arrival_id=3, loop_id=10, loop_step=2)
    assert obj == {'plan': {'req': 7, 'arrival_id': 3,
                            'loop_id': 10, 'loop_step': 2}}
    got = parse_plan_request(json.loads(json.dumps(obj))['plan'])
    assert got == {'req': 7, 'arrival_id': 3, 'loop_id': 10, 'loop_step': 2}


def test_request_loop_id_none():
    obj = make_plan_request(1, 5, None, 0)
    assert obj['plan']['loop_id'] is None
    assert parse_plan_request(obj['plan'])['loop_id'] is None


def test_target_roundtrip_carries_genome_and_keepvars():
    g = _mk_genome({2, 5}, db_id=9)
    g.transforms[0].color_speed = 0.73        # non-default, must travel
    wire = json.loads(json.dumps(target_to_wire(g, neighborhood=[1, 2, 3])))
    t = target_from_wire(wire)
    assert t.id == 9
    assert t.keep_vars == frozenset({2, 5})
    assert t.neighborhood == [1, 2, 3]
    # The serialized genome round-trips (so the render thread needs no DB load),
    # including color_speed (the field the core serializer added).
    assert t.genome.db_id == 9
    assert abs(t.genome.transforms[0].color_speed - 0.73) < 1e-6
    assert np.array_equal(t.genome.transforms[0].variations,
                          g.transforms[0].variations)


def test_target_db_id_is_separate_wire_field_not_in_blob():
    """db_id rides as the top-level ``id`` field; the core genome blob excludes
    it. ``target_from_wire`` re-stamps it so the genome is still a faithful
    DB-load stand-in — but it is recovered from ``id``, not from the blob."""
    g = _mk_genome({1}, db_id=42)
    wire = target_to_wire(g)
    assert wire['id'] == 42
    assert 'db_id' not in wire['genome']        # not inside the serialized genome
    # If the separate id is lost, the recovered genome has no identity.
    wire_no_id = dict(wire, id=None)
    assert target_from_wire(wire_no_id).genome.db_id is None
    # With the id present it is faithfully recovered.
    assert target_from_wire(wire).genome.db_id == 42


def test_target_none_is_none():
    assert target_to_wire(None) is None
    assert target_from_wire(None) is None


def test_response_roundtrip():
    follow = target_to_wire(_mk_genome({0}, db_id=1))
    walk = target_to_wire(_mk_genome({3}, db_id=2), neighborhood=[2, 4])
    obj = make_plan_response(req=5, arrival_id=1,
                             follow=follow, break_=None, walk=walk)
    resp = parse_plan_response(json.loads(json.dumps(obj))['plan'])
    assert resp.req == 5 and resp.arrival_id == 1
    assert resp.follow.id == 1
    assert resp.break_ is None
    assert resp.walk.id == 2 and resp.walk.neighborhood == [2, 4]


# ================================================================
# 2. A/B comparison — the RNG-divergence-aware verdicts
# ----------------------------------------------------------------
# FOLLOW is deterministic -> exact match. WALK/BREAK are independent-RNG picks
# -> validate STRUCTURE (neighborhood membership), never the exact pick.
# ================================================================
def _target(db_id, keep_vars=frozenset()):
    return PlanTarget(id=db_id, keep_vars=keep_vars,
                      genome=Genome(), neighborhood=None)


def test_compare_follow_exact_match_and_mismatch():
    w = _target(5, frozenset({0, 1}))
    assert compare_follow(w, 5, frozenset({0, 1})) == MATCH
    # MUTATION GUARD: wrong id OR wrong keep_vars is a mismatch (a real planner
    # bug — FOLLOW is supposed to be deterministic).
    assert compare_follow(w, 6, frozenset({0, 1})) == MISMATCH
    assert compare_follow(w, 5, frozenset({0, 2})) == MISMATCH


def test_compare_follow_empty_and_half_present():
    assert compare_follow(None, None, None) == EMPTY
    assert compare_follow(None, 5, frozenset()) == MISMATCH
    assert compare_follow(_target(5), None, None) == MISMATCH


def test_compare_follow_ignores_keepvars_when_not_supplied():
    # When the render side didn't stash keep_vars, id agreement alone matches.
    assert compare_follow(_target(5, frozenset({9})), 5, None) == MATCH


def test_check_walk_structure():
    assert check_walk_structure(_target(2), [1, 2, 3]) == STRUCTURAL_OK
    # MUTATION GUARD: a pick outside G's neighborhood is a structural failure
    # (the worker planner wandered off the right graph node).
    assert check_walk_structure(_target(9), [1, 2, 3]) == STRUCTURAL_FAIL
    assert check_walk_structure(None, [1, 2, 3]) == EMPTY
    assert check_walk_structure(_target(2), []) == EMPTY


def test_check_break_structure():
    assert check_break_structure(_target(3), [3, 7]) == STRUCTURAL_OK
    assert check_break_structure(_target(5), [3, 7]) == STRUCTURAL_FAIL
    assert check_break_structure(None, [3]) == EMPTY
    assert check_break_structure(_target(3), []) == EMPTY


# ================================================================
# 3. dispatch_line — PLAN goes to the preempt lane, not the compile queue
# ================================================================
class _RecordingCompileQueue:
    def __init__(self):
        self.enqueued = []
        self.floods = []

    def enqueue(self, tuples, priority=0):
        self.enqueued.append((list(tuples), priority))
        return len(list(tuples))

    def enqueue_flood(self, spec):
        self.floods.append(spec)


class _RecordingPlanQueue:
    def __init__(self):
        self.items = []

    def put(self, spec):
        self.items.append(spec)


def test_dispatch_routes_plan_to_plan_queue():
    """MUTATION GUARD on the preempt path: a PLAN must land on the plan queue and
    NEVER on the compile queue — that separation is what stops it sitting behind
    a cold compile. Swap the branch and these asserts flip."""
    cq = _RecordingCompileQueue()
    pq = _RecordingPlanQueue()
    line = json.dumps(make_plan_request(3, 7, 10, 2))
    tag = dispatch_line(line, cq, pq)
    assert tag == 'plan'
    assert pq.items == [{'req': 3, 'arrival_id': 7, 'loop_id': 10, 'loop_step': 2}]
    assert cq.enqueued == [] and cq.floods == []    # never touched the compiler


def test_dispatch_plan_without_queue_is_unknown():
    """Back-compat: an older dispatch_line call (no plan_queue) treats a plan
    line as an unknown control line, not a crash."""
    cq = _RecordingCompileQueue()
    assert dispatch_line(json.dumps(make_plan_request(1, 1, 1, 0)), cq) == 'unknown'


# ================================================================
# 4. Driver — request serialization + stdout response dispatch
# ----------------------------------------------------------------
# Decomposition: simple-integration on the real PrecompileDriver with NO
# subprocess — a FakeProc captures stdin / scripts stdout. GPU-free.
# ================================================================
def _driver(on_plan_response=None):
    from flame_sheep.scheduler.precompile_driver import PrecompileDriver
    return PrecompileDriver(canvas_w=0, canvas_h=0, policy=object(),
                            worker_module='flame_sheep.scheduler.precompile_worker_gl',
                            on_plan_response=on_plan_response)


def test_request_plan_serializes_one_line():
    drv = _driver()
    proc = FakeProc()
    drv._proc = proc
    drv.request_plan(req=7, arrival_id=3, loop_id=10, loop_step=2)
    assert proc.objects_written() == [
        {'plan': {'req': 7, 'arrival_id': 3, 'loop_id': 10, 'loop_step': 2}}]


def test_stdout_loop_dispatches_plan_response():
    """The driver's stdout reader parses a worker PLAN response line and hands
    the INNER plan dict to on_plan_response. Non-JSON noise is skipped."""
    received = []
    drv = _driver(on_plan_response=received.append)
    follow = target_to_wire(_mk_genome({0}, db_id=1))
    resp_line = json.dumps(make_plan_response(5, 1, follow, None, None))
    drv._proc = FakeProc(stdout_lines=[
        'some non-json log noise\n',   # ignored
        resp_line + '\n',
        '',                            # EOF
    ])
    drv._stdout_loop()
    assert len(received) == 1
    parsed = parse_plan_response(received[0])
    assert parsed.req == 5 and parsed.follow.id == 1


# ================================================================
# 5. PlanCache — the render-side target slot (step 4, stage S1)
# ================================================================
#
# Spec: the worker answers PLANs async on the reader thread; the move path reads
# a branch's pre-decided target by the arrival genome it is moving away from. A
# single latest-slot (one arrival in flight per dwell). Serves a target ONLY on
# an arrival_id match; any mismatch (not landed yet, or superseded) -> None,
# which the move path reads as "defer". MUTATION GUARD: drop the arrival_id check
# in get_target and test_get_target_rejects_stale_arrival reddens; break the
# branch map and test_get_target_branch_mapping reddens.
#
# NOT tested here (notated): true concurrent reader/render access -- the lock is
# exercised structurally (every path takes it) but a race is not deterministically
# reproducible in a unit test; the single-slot design is what removes the hazard.

def _resp(arrival_id, follow_id=None, break_id=None, walk_id=None):
    """Build a parsed PlanResponse via the real wire path (round-tripped), so the
    cache sees exactly what the reader thread would hand it."""
    def tw(gid):
        return None if gid is None else target_to_wire(_mk_genome({0}, db_id=gid))
    wire = make_plan_response(1, arrival_id,
                              tw(follow_id), tw(break_id), tw(walk_id))
    return parse_plan_response(json.loads(json.dumps(wire))['plan'])


def test_get_target_before_ingest_is_none():
    assert pp.PlanCache().get_target(7, 'follow') is None


def test_get_target_matching_arrival_returns_branch():
    c = pp.PlanCache()
    c.ingest(_resp(7, follow_id=70, break_id=71, walk_id=72))
    assert c.get_target(7, 'follow').id == 70
    assert c.get_target(7, 'break').id == 71
    assert c.get_target(7, 'walk').id == 72


def test_get_target_rejects_stale_arrival():
    """A plan is served only for the arrival it was computed from."""
    c = pp.PlanCache()
    c.ingest(_resp(7, follow_id=70))
    assert c.get_target(8, 'follow') is None   # different arrival -> defer
    assert c.get_target(7, 'follow').id == 70   # same arrival -> served


def test_get_target_branch_mapping():
    """'break' maps to the break_ field, not follow/walk."""
    c = pp.PlanCache()
    c.ingest(_resp(7, follow_id=70, break_id=71, walk_id=72))
    # each token resolves to its OWN target, proving no cross-wiring
    assert (c.get_target(7, 'follow').id,
            c.get_target(7, 'break').id,
            c.get_target(7, 'walk').id) == (70, 71, 72)


def test_latest_ingest_supersedes():
    c = pp.PlanCache()
    c.ingest(_resp(7, follow_id=70))
    c.ingest(_resp(8, follow_id=80))
    assert c.get_target(7, 'follow') is None    # superseded
    assert c.get_target(8, 'follow').id == 80


def test_get_target_none_branch_on_match():
    """A branch the worker couldn't plan (e.g. no loops -> break None) returns
    None even on an arrival match."""
    c = pp.PlanCache()
    c.ingest(_resp(7, follow_id=70, break_id=None))
    assert c.get_target(7, 'break') is None
    assert c.get_target(7, 'follow').id == 70


def test_get_target_none_arrival_is_none():
    c = pp.PlanCache()
    c.ingest(_resp(7, follow_id=70))
    assert c.get_target(None, 'follow') is None


def test_get_target_unknown_branch_raises():
    import pytest
    c = pp.PlanCache()
    c.ingest(_resp(7, follow_id=70))
    with pytest.raises(ValueError):
        c.get_target(7, 'sideways')


# ================================================================
# NOT TESTED HERE (notated skips):
#   * The real worker SUBPROCESS: stdin line -> planner thread -> real Library
#     plan -> stdout response, answered CONCURRENTLY with an in-flight cold
#     compile (the true cross-thread preempt timing). Process+GPU-gated; manual
#     / on-box. The preempt DESIGN is unit-proven here structurally: PLAN is
#     routed to a separate queue drained by a separate thread, so it provably
#     does not enter the compile main thread's path.
# ================================================================
