"""PlanCoordinator — render-side PLAN coordination (component map #6).

UNIT spec for `scheduler/plan_coordinator.py`, the glue extracted from the
`_run_wallpaper` closures in S2-0b (previously untested — only exercised on-box).
Isolated with a fake driver + a stub A/B library; the compare FUNCTIONS
themselves are already covered in test_plan_protocol, so here we test the
COORDINATOR's wiring: request firing + stash, cache ingest, stash pop, and that
the A/B lib is consulted and the verdict logged.

MUTATION GUARDS:
  * on_arrival not stashing  -> handle_response finds no stash -> NO [plan-ab]
    log -> test_roundtrip_logs_follow_match reddens.
  * handle_response not ingesting -> cache stays empty ->
    test_handle_response_populates_cache reddens.

NOT tested here (notated): the real driver subprocess / real Library (covered
elsewhere / GPU-gated); the compare math (test_plan_protocol); true concurrent
thread interleave (the lock is exercised, a race isn't deterministic in a unit).
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from flame_sheep.genome import Genome, Transform
from flame_sheep.scheduler.plan_coordinator import PlanCoordinator
from flame_sheep.scheduler.plan_protocol import make_plan_response, target_to_wire


def _mk_genome(var_ids, db_id=None) -> Genome:
    g = Genome(transforms=[Transform()])
    g.transforms[0].variations = np.zeros(
        g.transforms[0].variations.shape[0], dtype=np.float32)
    for v in var_ids:
        g.transforms[0].variations[v] = 0.5
    g.db_id = db_id
    return g


class _FakeDriver:
    def __init__(self):
        self.requests = []

    def request_plan(self, req, arrival_id, loop_id, loop_step):
        self.requests.append((req, arrival_id, loop_id, loop_step))


class _StubABLib:
    """Controls the WALK/BREAK neighborhoods so verdicts are deterministic."""
    def __init__(self, neighbors=(), entries=()):
        self.neighbors, self.entries = list(neighbors), list(entries)
        self.calls = []

    def nearest_transitions(self, gid, n=10):
        self.calls.append(('nt', gid))
        return [(b, 1.0) for b in self.neighbors]

    def nearest_loop_transitions(self, gid, exclude_loop=None, n=3):
        self.calls.append(('nlt', gid))
        return [(99, 1.0, 1.0, e) for e in self.entries]


def _inner(req, arrival_id, follow_id=None, break_id=None, walk_id=None,
           walk_neigh=None, break_neigh=None):
    def tw(gid, neigh):
        return None if gid is None else target_to_wire(
            _mk_genome({0}, db_id=gid), neighborhood=neigh)
    wire = make_plan_response(req, arrival_id,
                              tw(follow_id, None), tw(break_id, break_neigh),
                              tw(walk_id, walk_neigh))
    return json.loads(json.dumps(wire))['plan']


# --- on_arrival ------------------------------------------------------------

def test_on_arrival_fires_request_and_increments():
    drv = _FakeDriver()
    c = PlanCoordinator(logger=logging.getLogger('t'))
    c.driver = drv
    c.on_arrival(_mk_genome({0}, db_id=5), loop_id=7, loop_step=2,
                 expected_follow=None)
    c.on_arrival(_mk_genome({0}, db_id=6), loop_id=7, loop_step=3,
                 expected_follow=None)
    assert drv.requests == [(1, 5, 7, 2), (2, 6, 7, 3)]   # req id increments


def test_on_arrival_noop_without_driver():
    c = PlanCoordinator()  # .driver is None
    c.on_arrival(_mk_genome({0}, db_id=5), 7, 2, None)   # must not raise


def test_on_arrival_skips_when_no_db_id():
    drv = _FakeDriver()
    c = PlanCoordinator()
    c.driver = drv
    c.on_arrival(_mk_genome({0}, db_id=None), 7, 2, None)
    assert drv.requests == []


# --- handle_response: cache ------------------------------------------------

def test_handle_response_populates_cache():
    """The move path (S3+) reads the cache — handle_response must ingest."""
    c = PlanCoordinator()
    c.handle_response(_inner(1, arrival_id=5, follow_id=50, walk_id=51))
    assert c.cache.get_target(5, 'follow').id == 50
    assert c.cache.get_target(5, 'walk').id == 51


def test_handle_response_unknown_req_still_caches():
    """No prior on_arrival (no stash): must still cache, not crash, not log."""
    c = PlanCoordinator(logger=logging.getLogger('t'))
    c.handle_response(_inner(99, arrival_id=5, follow_id=50))  # no stash for 99
    assert c.cache.get_target(5, 'follow').id == 50


# --- handle_response: A/B round-trip + logging -----------------------------

def test_roundtrip_logs_follow_match(caplog):
    drv = _FakeDriver()
    c = PlanCoordinator(logger=logging.getLogger('t'))
    c.driver = drv
    # arrival at #5, worker's FOLLOW will be #50; expected_follow also #50 -> match
    c.on_arrival(_mk_genome({0}, db_id=5), loop_id=7, loop_step=2,
                 expected_follow=_mk_genome({0}, db_id=50))
    req = drv.requests[0][0]
    with caplog.at_level(logging.INFO, logger='t'):
        c.handle_response(_inner(req, arrival_id=5, follow_id=50))
    assert any('[plan-ab]' in r.message and 'FOLLOW=match' in r.message
               for r in caplog.records)


def test_roundtrip_follow_mismatch_warns(caplog):
    drv = _FakeDriver()
    c = PlanCoordinator(logger=logging.getLogger('t'))
    c.driver = drv
    c.on_arrival(_mk_genome({0}, db_id=5), 7, 2,
                 expected_follow=_mk_genome({0}, db_id=50))
    req = drv.requests[0][0]
    with caplog.at_level(logging.INFO, logger='t'):
        c.handle_response(_inner(req, arrival_id=5, follow_id=999))  # != expected
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING
                and 'FOLLOW=mismatch' in r.message]
    assert warnings


def test_ab_lib_consulted_for_walk_break(caplog):
    drv = _FakeDriver()
    ab = _StubABLib(neighbors=[50], entries=[60])  # walk #50 ok, break #60 ok
    c = PlanCoordinator(ab_lib_factory=lambda: ab, logger=logging.getLogger('t'))
    c.driver = drv
    c.on_arrival(_mk_genome({0}, db_id=5), 7, 2, expected_follow=None)
    req = drv.requests[0][0]
    with caplog.at_level(logging.INFO, logger='t'):
        c.handle_response(_inner(req, arrival_id=5,
                                 walk_id=50, break_id=60))
    assert ('nt', 5) in ab.calls and ('nlt', 5) in ab.calls
    line = next(r.message for r in caplog.records if '[plan-ab]' in r.message)
    assert 'WALK=structural-ok' in line and 'BREAK=structural-ok' in line


def test_no_ab_lib_reports_no_lib(caplog):
    drv = _FakeDriver()
    c = PlanCoordinator(ab_lib_factory=None, logger=logging.getLogger('t'))
    c.driver = drv
    c.on_arrival(_mk_genome({0}, db_id=5), 7, 2, expected_follow=None)
    req = drv.requests[0][0]
    with caplog.at_level(logging.INFO, logger='t'):
        c.handle_response(_inner(req, arrival_id=5, walk_id=50))
    line = next(r.message for r in caplog.records if '[plan-ab]' in r.message)
    assert 'WALK=no-lib' in line and 'BREAK=no-lib' in line
