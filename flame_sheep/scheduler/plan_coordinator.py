"""PlanCoordinator — render-side coordination of the worker PLAN protocol.

Extracted (shader-warming.md step 4, stage S2-0b) from the `_run_wallpaper`
closures so the render-process PLAN integration (component map #6) is a unit —
testable with a fake driver + fake A/B library — instead of GL-bound glue only
exercised on-box. Behavior is faithful to the former closures; nothing here
changes what the wallpaper does (that starts at S3, when the move path reads
`self.cache`).

Responsibilities:
  * `on_arrival` (RENDER thread): stamp a request id, stash the render thread's
    own expected-FOLLOW, fire `driver.request_plan`.
  * `handle_response` (driver STDOUT-READER thread): parse, `cache.ingest` the
    targets (what S3 will consume), then the step-3 A/B compare + `[plan-ab]` log.
  * `cache` (`PlanCache`): the hand-off slot the move path reads.

Thread model: `on_arrival` and `handle_response` run on different threads; the
lock guards the shared counter + pending stash. `PlanCache` has its own lock.
The A/B compare (and its reader-thread `Library`) is step-3 scaffolding that S5
retires per decision (i); until then it is preserved verbatim.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from .plan_protocol import PlanCache

log = logging.getLogger(__name__)

_PENDING_BOUND = 64  # drop stale unanswered requests past this


class PlanCoordinator:
    def __init__(self, ab_lib_factory: Callable[[], Any] | None = None,
                 logger: logging.Logger | None = None) -> None:
        # `driver` is set after construction: the driver is built with
        # `handle_response` as its callback, so the coordinator must exist first
        # (same ordering the old closures had via late binding). `on_arrival`
        # no-ops until it is wired.
        self.driver = None
        self._ab_lib_factory = ab_lib_factory
        self._log = logger or log
        self._counter = 0
        self._pending: dict[int, tuple] = {}
        self._lock = threading.Lock()
        self.cache = PlanCache()
        self._ab_lib_box: dict = {'lib': None, 'tried': False}

    # -- A/B reader-thread Library (lazy, one per reader thread) -------------
    def _ab_lib(self):
        if self._ab_lib_factory is None:
            return None
        if self._ab_lib_box['lib'] is None and not self._ab_lib_box['tried']:
            self._ab_lib_box['tried'] = True
            try:
                self._ab_lib_box['lib'] = self._ab_lib_factory()
            except Exception:
                self._log.exception('[plan-ab] reader-thread Library init failed')
        return self._ab_lib_box['lib']

    # -- render thread -------------------------------------------------------
    def on_arrival(self, arrival_genome, loop_id, loop_step, expected_follow) -> None:
        """At each loop-mode arrival: fire a PLAN request + stash the render
        thread's own expected FOLLOW for the A/B compare. No-op without a
        driver or a db_id."""
        if self.driver is None:
            return
        try:
            from .precompile_warm import _genome_tuple
            arrival_id = getattr(arrival_genome, 'db_id', None)
            if arrival_id is None:
                return
            exp_id = getattr(expected_follow, 'db_id', None) \
                if expected_follow is not None else None
            exp_kv = (_genome_tuple(expected_follow)[2]
                      if expected_follow is not None else None)
            with self._lock:
                self._counter += 1
                req = self._counter
                if len(self._pending) > _PENDING_BOUND:
                    self._pending.clear()
                self._pending[req] = (arrival_id, loop_id, exp_id, exp_kv)
            self.driver.request_plan(req, arrival_id, loop_id, loop_step)
        except Exception:
            self._log.exception('[plan-ab] arrival hook failed')

    # -- reader thread -------------------------------------------------------
    def handle_response(self, inner: dict) -> None:
        """Parse a worker PLAN response, cache its targets (for the move path),
        then run the step-3 A/B compare + log."""
        from .plan_protocol import (
            parse_plan_response, compare_follow,
            check_walk_structure, check_break_structure)
        try:
            resp = parse_plan_response(inner)
        except Exception:
            self._log.exception('[plan-ab] malformed plan response: %r', inner)
            return
        # cache the worker's targets for the (S3+) move path
        self.cache.ingest(resp)
        with self._lock:
            stash = self._pending.pop(resp.req, None)
        if stash is None:
            return  # unknown/expired request id
        arrival_id, loop_id, exp_follow_id, exp_follow_kv = stash

        follow_v = compare_follow(resp.follow, exp_follow_id, exp_follow_kv)

        ab = self._ab_lib()
        walk_v = break_v = 'no-lib'
        if ab is not None:
            try:
                neigh = [b for b, _ in ab.nearest_transitions(arrival_id, n=10)]
            except Exception:
                neigh = []
            walk_v = check_walk_structure(resp.walk, neigh)
            try:
                entries = [e for _l, _f, _d, e in ab.nearest_loop_transitions(
                    arrival_id, exclude_loop=loop_id, n=3)]
            except Exception:
                entries = []
            break_v = check_break_structure(resp.break_, entries)

        w_follow = resp.follow.id if resp.follow else None
        w_walk = resp.walk.id if resp.walk else None
        w_break = resp.break_.id if resp.break_ else None
        line = (f'[plan-ab] arrival=#{arrival_id} loop={loop_id} '
                f'FOLLOW={follow_v} (worker=#{w_follow} expect=#{exp_follow_id}) '
                f'WALK={walk_v} (worker=#{w_walk}) '
                f'BREAK={break_v} (worker=#{w_break})')
        if follow_v == 'mismatch' or walk_v == 'structural-fail' \
                or break_v == 'structural-fail':
            self._log.warning(line)
        else:
            self._log.info(line)
