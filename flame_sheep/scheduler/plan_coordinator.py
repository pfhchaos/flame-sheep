"""PlanCoordinator — render-side coordination of the worker PLAN protocol.

Extracted (shader-warming.md step 4, stage S2-0b) from the `_run_wallpaper`
closures so the render-process PLAN integration (component map #6) is a unit —
testable with a fake driver — instead of GL-bound glue only exercised on-box.

Responsibilities:
  * `on_arrival` (RENDER thread): stamp a request id, fire `driver.request_plan`
    so the worker computes this arrival's FOLLOW/BREAK/WALK targets.
  * `handle_response` (driver STDOUT-READER thread): parse the response and
    `cache.ingest` the targets (what the move path consumes).
  * `cache` (`PlanCache`): the hand-off slot the move path reads.

Thread model: `on_arrival` (render thread) stamps the request id; the counter is
only touched there. `handle_response` runs on the driver's stdout-reader thread
and only writes `PlanCache`, which has its own lock.
"""
from __future__ import annotations

import logging

from .plan_protocol import PlanCache

log = logging.getLogger(__name__)


class PlanCoordinator:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        # `driver` is set after construction: the driver is built with
        # `handle_response` as its callback, so the coordinator must exist first
        # (same ordering the old closures had via late binding). `on_arrival`
        # no-ops until it is wired.
        self.driver = None
        self._log = logger or log
        self._counter = 0
        self.cache = PlanCache()

    # -- render thread -------------------------------------------------------
    def on_arrival(self, arrival_genome, loop_id, loop_step) -> None:
        """At each loop-mode arrival: fire a PLAN request so the worker computes
        this arrival's targets. No-op without a driver or a db_id."""
        if self.driver is None:
            return
        try:
            arrival_id = getattr(arrival_genome, 'db_id', None)
            if arrival_id is None:
                return
            self._counter += 1
            req = self._counter
            self.driver.request_plan(req, arrival_id, loop_id, loop_step)
        except Exception:
            self._log.exception('[plan] arrival hook failed')

    # -- render thread (move-path consumption, step 4 S3+) -------------------
    def plan_target(self, arrival_id, branch: str):
        """Return the worker's pre-decided target GENOME for ``branch`` planned
        FROM ``arrival_id``, or None if no matching plan is cached yet (the move
        path reads None as 'defer'). ``branch`` is 'follow'/'break'/'walk'."""
        t = self.cache.get_target(arrival_id, branch)
        return None if t is None else t.genome

    # -- reader thread -------------------------------------------------------
    def handle_response(self, inner: dict) -> None:
        """Parse a worker PLAN response and cache its targets for the move path."""
        from .plan_protocol import parse_plan_response
        try:
            resp = parse_plan_response(inner)
        except Exception:
            self._log.exception('[plan] malformed plan response: %r', inner)
            return
        self.cache.ingest(resp)
