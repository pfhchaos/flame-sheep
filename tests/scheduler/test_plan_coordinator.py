"""PlanCoordinator — render-side PLAN coordination (component map #6).

UNIT spec for `scheduler/plan_coordinator.py`, the glue extracted from the
`_run_wallpaper` closures in S2-0b (previously untested — only exercised on-box).
Isolated with a fake driver; here we test the COORDINATOR's wiring: request
firing (+ request-id increment) and cache ingest (what the move path reads).

MUTATION GUARDS:
  * on_arrival not firing / not incrementing the request id ->
    test_on_arrival_fires_request_and_increments reddens.
  * handle_response not ingesting -> cache stays empty ->
    test_handle_response_populates_cache reddens.

NOT tested here (notated): the real driver subprocess / real Library (covered
elsewhere / GPU-gated); true concurrent thread interleave (the reader thread only
writes PlanCache, which has its own lock; a race isn't deterministic in a unit).
"""
from __future__ import annotations

import json
import logging

import numpy as np

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
    c.on_arrival(_mk_genome({0}, db_id=5), loop_id=7, loop_step=2)
    c.on_arrival(_mk_genome({0}, db_id=6), loop_id=7, loop_step=3)
    assert drv.requests == [(1, 5, 7, 2), (2, 6, 7, 3)]   # req id increments


def test_on_arrival_noop_without_driver():
    c = PlanCoordinator()  # .driver is None
    c.on_arrival(_mk_genome({0}, db_id=5), 7, 2)   # must not raise


def test_on_arrival_skips_when_no_db_id():
    drv = _FakeDriver()
    c = PlanCoordinator()
    c.driver = drv
    c.on_arrival(_mk_genome({0}, db_id=None), 7, 2)
    assert drv.requests == []


# --- handle_response: cache ------------------------------------------------

def test_handle_response_populates_cache():
    """The move path reads the cache — handle_response must ingest."""
    c = PlanCoordinator()
    c.handle_response(_inner(1, arrival_id=5, follow_id=50, walk_id=51))
    assert c.cache.get_target(5, 'follow').id == 50
    assert c.cache.get_target(5, 'walk').id == 51


def test_handle_response_caches_without_prior_arrival():
    """No prior on_arrival: handle_response must still cache (and not crash)."""
    c = PlanCoordinator(logger=logging.getLogger('t'))
    c.handle_response(_inner(99, arrival_id=5, follow_id=50))
    assert c.cache.get_target(5, 'follow').id == 50


def test_plan_target_returns_cached_genome():
    """plan_target unwraps the cached PlanTarget to its genome for the move path;
    an un-cached arrival/branch defers (None)."""
    c = PlanCoordinator()
    c.handle_response(_inner(1, arrival_id=5, follow_id=50))
    g = c.plan_target(5, 'follow')
    assert g is not None and g.db_id == 50
    assert c.plan_target(5, 'walk') is None      # no walk in this response
    assert c.plan_target(6, 'follow') is None    # different arrival -> defer


def test_handle_response_malformed_does_not_raise():
    c = PlanCoordinator(logger=logging.getLogger('t'))
    c.handle_response({'not': 'a plan'})         # must not raise
    assert c.cache.current_arrival() is None
