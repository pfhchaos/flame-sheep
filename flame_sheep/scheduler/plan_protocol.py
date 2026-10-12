"""PLAN protocol — the render thread ↔ worker planner wire format + the
render-side A/B comparison (shader-warming.md step 3).

Step 3 stands up the planner in the worker and ships its 3 next-move targets
(FOLLOW / BREAK / WALK) back to the render thread, which LOGS/COMPARES them
against its own ``_loop.*`` picks WITHOUT consuming them for the actual swap
(authority stays render-side until step 4). This module is the shared seam so
the two ends can't drift — mirrors how ``precompile_warm.tuple_to_wire`` /
``tuple_from_wire`` pair the compile-request format.

Wire shapes (one JSON object per line, like the enqueue/flood requests):

  request   (render → worker, over stdin):
    {"plan": {"req": R, "arrival_id": G, "loop_id": L, "loop_step": S}}

  response  (worker → render, over stdout):
    {"plan": {"req": R, "arrival_id": G,
              "follow": TARGET | null,
              "break":  TARGET | null,
              "walk":   TARGET | null}}

  TARGET = {"id": db_id|null,
            "keep_vars": [sorted var indices],
            "genome": <Genome.to_dict()>,
            "neighborhood": [candidate ids] | null}

``follow`` is deterministic (no neighborhood). ``break`` / ``walk`` are the
stochastic picks — the worker rolls its OWN RNG, so they will NOT match the
render thread's pick exactly; ``neighborhood`` is the candidate id set the
worker drew from, carried back as telemetry. The render side validates STRUCTURE
for those (membership in its own independently-queried neighborhood), not an
exact pick — see :func:`check_walk_structure` / :func:`check_break_structure`.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from flame_sheep.genome import Genome
from flame_sheep.scheduler.precompile_warm import _genome_tuple


# --- request framing ------------------------------------------------------

def make_plan_request(req: int, arrival_id: int, loop_id: int | None,
                      loop_step: int) -> dict:
    """Build the ``{"plan": {...}}`` request object the driver serializes to the
    worker's stdin."""
    return {'plan': {
        'req': int(req),
        'arrival_id': int(arrival_id),
        'loop_id': None if loop_id is None else int(loop_id),
        'loop_step': int(loop_step),
    }}


def parse_plan_request(spec: dict) -> dict:
    """Parse the inner ``plan`` dict (already unwrapped from ``{"plan": ...}``)
    the stdin reader hands the planner thread."""
    return {
        'req': int(spec['req']),
        'arrival_id': int(spec['arrival_id']),
        'loop_id': None if spec.get('loop_id') is None else int(spec['loop_id']),
        'loop_step': int(spec.get('loop_step', 0)),
    }


# --- target framing -------------------------------------------------------

def target_to_wire(genome: Genome | None,
                   neighborhood: list[int] | None = None) -> dict | None:
    """Serialize ONE planner target. ``None`` genome → ``None`` (the move was
    not reachable, e.g. no loops for BREAK). ``keep_vars`` is derived from the
    genome so the render side never recomputes it from the heavy GPU arrays.

    ``db_id`` rides as the SEPARATE ``id`` field, NOT inside ``genome`` — the
    core ``Genome.to_dict`` serializes only intrinsic genome data (identity is
    not genome data). ``target_from_wire`` re-stamps ``id`` onto the recovered
    genome's ``db_id`` so it is still a faithful stand-in for a DB-loaded one."""
    if genome is None:
        return None
    _, _, keep_vars = _genome_tuple(genome)
    return {
        'id': genome.db_id,
        'keep_vars': sorted(int(v) for v in keep_vars),
        'genome': genome.to_dict(),
        'neighborhood': (None if neighborhood is None
                         else [int(x) for x in neighborhood]),
    }


@dataclass
class PlanTarget:
    """A parsed planner target on the render side."""
    id: int | None
    keep_vars: frozenset[int]
    genome: Genome
    neighborhood: list[int] | None


def target_from_wire(d: dict | None) -> PlanTarget | None:
    if d is None:
        return None
    db_id = d.get('id')
    genome = Genome.from_dict(d['genome'])
    # db_id travels as a separate wire field (not in the core blob); re-stamp it
    # so the deserialized genome matches what a DB load would have produced.
    genome.db_id = db_id
    return PlanTarget(
        id=db_id,
        keep_vars=frozenset(int(v) for v in d.get('keep_vars', [])),
        genome=genome,
        neighborhood=(None if d.get('neighborhood') is None
                      else [int(x) for x in d['neighborhood']]),
    )


# --- response framing -----------------------------------------------------

def make_plan_response(req: int, arrival_id: int,
                       follow: dict | None, break_: dict | None,
                       walk: dict | None) -> dict:
    """Assemble the ``{"plan": {...}}`` response the worker writes to stdout.
    ``follow``/``break_``/``walk`` are already ``target_to_wire`` dicts."""
    return {'plan': {
        'req': int(req),
        'arrival_id': int(arrival_id),
        'follow': follow,
        'break': break_,
        'walk': walk,
    }}


@dataclass
class PlanResponse:
    req: int
    arrival_id: int
    follow: PlanTarget | None
    break_: PlanTarget | None
    walk: PlanTarget | None


def parse_plan_response(spec: dict) -> PlanResponse:
    """Parse the inner ``plan`` dict of a response (unwrapped from
    ``{"plan": ...}``)."""
    return PlanResponse(
        req=int(spec['req']),
        arrival_id=int(spec['arrival_id']),
        follow=target_from_wire(spec.get('follow')),
        break_=target_from_wire(spec.get('break')),
        walk=target_from_wire(spec.get('walk')),
    )


# --- render-side target cache (shader-warming.md step 4, stage S1) ---------
#
# The worker answers PLAN requests ASYNCHRONOUSLY on the driver's stdout reader
# thread; the render thread reads the pre-decided targets later, at move time.
# This is the thread-safe hand-off slot between them. Step 4 stages flip the
# move path to read it; S1 only populates it (render still uses `_loop.*`).

# Branch token -> PlanResponse field.
_BRANCH_FIELD = {'follow': 'follow', 'break': 'break_', 'walk': 'walk'}


class PlanCache:
    """Thread-safe latest-plan slot. Only the most recent arrival's targets are
    kept -- one arrival is in flight per dwell, so a single slot (not a growing
    map) matches the cadence and can't leak. A target is served only when its
    ``arrival_id`` matches the genome the render thread is moving away from; a
    mismatch (the plan for THIS arrival hasn't landed yet, or a newer arrival
    superseded it) returns ``None``, which the move path reads as "defer"."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._resp: PlanResponse | None = None

    def ingest(self, resp: PlanResponse) -> None:
        """Store the latest response (reader-thread side). Replaces any prior."""
        with self._lock:
            self._resp = resp

    def get_target(self, arrival_id: int | None, branch: str) -> PlanTarget | None:
        """Return the ``branch`` target the worker planned FROM ``arrival_id``,
        or ``None`` if no matching plan is cached yet (→ the caller defers).
        ``branch`` is one of 'follow' / 'break' / 'walk'."""
        field = _BRANCH_FIELD.get(branch)
        if field is None:
            raise ValueError(f'unknown branch {branch!r}')
        if arrival_id is None:
            return None
        with self._lock:
            r = self._resp
        if r is None or r.arrival_id != arrival_id:
            return None
        return getattr(r, field)

    def current_arrival(self) -> int | None:
        """The arrival_id of the cached plan, or None. (Diagnostics/tests.)"""
        with self._lock:
            return self._resp.arrival_id if self._resp is not None else None


# --- A/B comparison (render side, step 3) ---------------------------------
#
# CRITICAL subtlety (shader-warming.md): the worker's walk/break RNG is an
# INDEPENDENT stream from the render thread's, so the STOCHASTIC picks (WALK,
# BREAK) will NOT exactly match the render thread's pick. So:
#   FOLLOW -> exact match (deterministic given loop+position).
#   WALK/BREAK -> STRUCTURE only: the worker chose a candidate from the correct
#                 graph neighborhood (nearest_transitions / nearest_loop_
#                 transitions of G). Never assert equality of the pick itself.
# Each returns a short verdict tag for logging + mutation-provable tests.

# Verdict tags.
MATCH = 'match'                 # FOLLOW: exact pick agreement
MISMATCH = 'mismatch'           # FOLLOW: pick disagreement (a real planner bug)
STRUCTURAL_OK = 'structural-ok'  # WALK/BREAK: valid candidate from neighborhood
STRUCTURAL_FAIL = 'structural-fail'  # WALK/BREAK: candidate outside neighborhood
EMPTY = 'empty'                 # move not available on either side (no-op agree)


def compare_follow(worker: PlanTarget | None,
                   expected_id: int | None,
                   expected_keep_vars: frozenset[int] | None) -> str:
    """FOLLOW is deterministic given (loop, position): the worker's pick MUST
    equal what the render thread's own sequence peek produced. Compares both the
    db_id AND the keep_vars (the latter catches a correct-id/wrong-genome
    serialization bug). ``EMPTY`` if neither side has a follow (no loop)."""
    if worker is None and expected_id is None:
        return EMPTY
    if worker is None or expected_id is None:
        return MISMATCH
    if worker.id != expected_id:
        return MISMATCH
    if expected_keep_vars is not None and worker.keep_vars != expected_keep_vars:
        return MISMATCH
    return MATCH


def check_walk_structure(worker: PlanTarget | None,
                         neighbor_ids: list[int] | set[int] | None) -> str:
    """WALK: the worker rolled its own RNG, so don't compare the pick. Validate
    it chose one of the current genome's graph neighbors
    (``nearest_transitions`` of G), which the render side queries independently.
    ``neighbor_ids`` None/empty → no neighborhood to validate against (the walk
    fell back to a random genome) → ``EMPTY``."""
    if worker is None:
        return EMPTY
    neighbors = set(neighbor_ids or ())
    if not neighbors:
        return EMPTY
    return STRUCTURAL_OK if worker.id in neighbors else STRUCTURAL_FAIL


def check_break_structure(worker: PlanTarget | None,
                          entry_ids: list[int] | set[int] | None) -> str:
    """BREAK: validate the worker's loop-entry genome is one of the entries the
    current genome can reach (``nearest_loop_transitions`` of G, entry-genome
    ids), queried independently on the render side. ``None``/empty worker target
    means no loops were reachable → ``EMPTY``."""
    if worker is None:
        return EMPTY
    entries = set(entry_ids or ())
    if not entries:
        return EMPTY
    return STRUCTURAL_OK if worker.id in entries else STRUCTURAL_FAIL
