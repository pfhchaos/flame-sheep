"""Worker-side movement planner (shader-warming.md step 3, Option A).

The planner computes the 3 reachable next-move targets from a just-arrived
genome G, so the worker can (step 4+) pre-warm their shaders and the render
thread can reuse the pre-decided targets. In step 3 it runs in PARALLEL to the
render thread's own ``_loop.*`` selection purely so the two can be A/B-compared
before step 4 flips authority.

The three moves (same three ``GenomeAxis`` makes):
  * FOLLOW = ``next_in_sequence`` of the active loop — DETERMINISTIC given
             (loop_id, loop_step). Recomputed via the shared ``follow_after``
             helper, the SAME code the render thread's peek uses, so the FOLLOW
             A/B check is exact.
  * BREAK  = ``next_loop`` — stochastic (softmax over nearest loop transitions).
  * WALK   = ``graph_walk`` — stochastic (softmax over nearest transitions).

It REUSES the production ``LoopPlayer.graph_walk`` / ``LoopPlayer.next_loop``
rather than re-porting the softmax (avoids the parallel-impl drift the user
flags): it builds a throwaway ``LoopPlayer`` bound to the worker's own Library +
its OWN RNG. Owning the RNG is the whole point (shader-warming.md "owns the walk
RNG and pre-rolls") — there is exactly one roll, here, and it is an independent
stream from the render thread's, which is why the stochastic A/B checks validate
STRUCTURE not an exact pick.

INVARIANT: the planner is DB-ONLY — it touches ``Library`` (SQLite) and the RNG,
never a GL context. It therefore runs safely on its own worker thread (the
dedicated planner thread in ``precompile_worker_gl``), which both satisfies "no
GL context on a thread" and gives the PLAN request its preempt path (it never
waits behind an in-flight compile on the main thread). The ``Library`` it holds
is created on, and only touched from, that one thread, so its SQLite connection
stays thread-local.
"""
from __future__ import annotations

import logging

import numpy as np

from flame_sheep.axes._loop_player import LoopPlayer, follow_after
from flame_sheep.scheduler.plan_protocol import target_to_wire

log = logging.getLogger(__name__)


class WorkerPlanner:
    """Computes FOLLOW / BREAK / WALK targets from an arrival genome."""

    def __init__(self, lib, rng: np.random.Generator | None = None):
        self._lib = lib
        # The worker's OWN RNG stream (independent of the render thread's).
        self._rng = rng if rng is not None else np.random.default_rng()

    def plan(self, arrival_id: int, loop_id: int | None,
             loop_step: int) -> dict:
        """Compute the 3 next-move targets from genome ``arrival_id``.

        Returns the three ``target_to_wire`` dicts (or ``None`` per move when
        the move is unreachable) under keys ``follow``/``break``/``walk``, ready
        for :func:`plan_protocol.make_plan_response`. Never raises: each move is
        isolated so one failing (bad id, empty graph) still ships the others.
        """
        try:
            G = self._lib.load_genome(arrival_id)
        except Exception:
            log.exception('[planner] load arrival genome #%s failed', arrival_id)
            return {'follow': None, 'break': None, 'walk': None}
        return {
            'follow': self._follow(loop_id, loop_step),
            'break': self._break(G, loop_id),
            'walk': self._walk(G),
        }

    # -- individual moves --

    def _follow(self, loop_id: int | None, loop_step: int) -> dict | None:
        """Deterministic next-in-sequence of the active loop. Shares
        ``follow_after`` with the render thread's peek, so it matches exactly."""
        if loop_id is None:
            return None
        try:
            items = self._lib.load_loop(loop_id)
            genomes = [g for _, g, _ in items]
            structure = self._lib.loop_type(loop_id)
            follow = follow_after(genomes, structure, loop_step)
            return target_to_wire(follow)  # deterministic → no neighborhood
        except Exception:
            log.exception('[planner] FOLLOW failed (loop=%s step=%s)',
                          loop_id, loop_step)
            return None

    def _walk(self, G) -> dict | None:
        """Stochastic graph walk. Reuses ``LoopPlayer.graph_walk`` (rolls the
        worker RNG). Carries back the neighbor id set it drew from for the
        render-side structural check."""
        try:
            lp = self._loop_player(G)
            target = lp.graph_walk(G)
            neighborhood = None
            if G.db_id is not None:
                neighborhood = [b for b, _ in
                                self._lib.nearest_transitions(G.db_id, n=10)]
            return target_to_wire(target, neighborhood)
        except Exception:
            log.exception('[planner] WALK failed')
            return None

    def _break(self, G, loop_id: int | None) -> dict | None:
        """Stochastic loop break. Reuses ``LoopPlayer.next_loop`` (rolls the
        worker RNG, excludes the current loop). Carries back the reachable
        loop-entry id set for the render-side structural check."""
        try:
            lp = self._loop_player(G)
            lp.active_loop_id = loop_id
            target = lp.next_loop(G)
            if target is None:
                return None
            neighborhood = None
            if G.db_id is not None:
                neighborhood = [entry for _lid, _f, _d, entry in
                                self._lib.nearest_loop_transitions(
                                    G.db_id, exclude_loop=loop_id, n=3)]
            return target_to_wire(target, neighborhood)
        except Exception:
            log.exception('[planner] BREAK failed')
            return None

    def _loop_player(self, G) -> LoopPlayer:
        """A throwaway ``LoopPlayer`` bound to the worker Library + the shared
        RNG. The genome factory is only hit on the graph-walk fallback (no
        neighbors); returning ``G`` avoids a costly ``Genome.random`` there."""
        return LoopPlayer(self._lib, lambda: G, G, self._rng)
