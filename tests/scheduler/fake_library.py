"""Foundational test infra: an in-memory fake ``Library``.

WHY THIS EXISTS
---------------
The real ``Library(data_dir=tmp_path)`` is expensive to seed for a test:
``Genome.random`` is ~0.8s each and ``save_genome``'s default scoring
(``aesthetic_score``) is ~0.9s each, so the old flood fixture cost ~14s to
stand up four genomes. None of that cost buys the shader-warming / planner
tests anything — they only need a genome catalog with an EXACT variation set
and an EXACT transition graph. This fake serves exactly the methods those
paths call, over a graph the test hands it, with zero SQLite and zero scoring,
so the tests are fast AND deterministic (no RANDOM()/float-tie ambiguity).

FIDELITY
--------
The methods below mirror the real ``Library``'s signatures, return shapes, and
observable semantics (see ``test_fake_library_matches_real`` in
``test_precompile_flood.py`` for the behavior-equivalence proof on a tiny real
graph):
  * ``load_genome(id)``          -> the stored genome, with ``.db_id`` stamped;
                                    raises ``KeyError`` for an unknown id.
  * ``nearest_transitions(id,n)``-> ``[(genome_b, distance), ...]`` sorted by
                                    distance ascending, truncated to ``n``.
  * ``store_transition(a,b,d)``  -> bidirectional edge; skips self/zero edges
                                    and REPLACES a prior (a,b) distance, same
                                    as the real INSERT OR REPLACE.

The fake is agnostic about WHAT genome objects it holds. Fast tests hand it
lightweight ``FakeGenome`` stubs (exact ``keep_vars`` with no ``Genome.random``
cost); the fidelity test hands it the same real ``Genome`` objects it saved to
a real ``Library`` so the two can be compared tuple-for-tuple.

EXTENSION POINTS (later planner steps — build when the step needs them)
----------------------------------------------------------------------
The planner (shader-warming.md, Option A) will also call
``nearest_loop_transitions`` and ``genome_scores``. They are deliberately NOT
built now (don't gold-plate a fake for code that doesn't exist yet). The
scaffolding to add them is already here:
  * ``add_genome(id, genome, scores=...)`` already accepts and stores a scores
    dict, so ``genome_scores`` becomes a one-line dict lookup.
  * a loop table (``add_loop``) + reusing ``_transitions`` is all
    ``nearest_loop_transitions`` needs.
Stubs that raise ``NotImplementedError`` mark the exact spots below.
"""
from __future__ import annotations

import numpy as np

# Buffer geometry that _genome_tuple (precompile_warm) reshapes to: it reads
# arrays['active_vars'].reshape(7, 8, 10) and takes [..., 0] as the var ids.
_N_SLOTS, _MAX_ACTIVE, _SLOT = 7, 8, 10


def _pack_vars(var_ids) -> np.ndarray:
    """Build an ``active_vars``-shaped (7, 80) array whose [..., 0] lanes hold
    exactly ``var_ids`` (one per slot), everything else -1 (empty). Lets a test
    pin a genome's ``keep_vars`` without constructing a real ``Genome``."""
    a = np.full((_N_SLOTS, _MAX_ACTIVE, _SLOT), -1.0, dtype=np.float32)
    ids = [int(v) for v in var_ids]
    if len(ids) > _N_SLOTS * _MAX_ACTIVE:
        raise ValueError(f'too many vars for the stub buffer: {len(ids)}')
    for k, v in enumerate(ids):
        slot, lane = divmod(k, _MAX_ACTIVE)
        a[slot, lane, 0] = v
    return a.reshape(_N_SLOTS, _MAX_ACTIVE * _SLOT)


class FakeGenome:
    """A minimal stand-in for ``Genome`` that satisfies ``_genome_tuple``.

    ``_genome_tuple`` only touches ``.transforms`` (for the count) and
    ``.to_gpu_arrays()`` (for the var set + has_final flag), so that's all this
    implements. Pin ``n_transforms`` to control the tuple's first field
    exactly; it defaults to the var count (at least 1)."""

    def __init__(self, var_ids, *, n_transforms: int | None = None,
                 has_final: bool = False, db_id: int | None = None):
        self._vars = frozenset(int(v) for v in var_ids)
        n = n_transforms if n_transforms is not None else max(1, len(self._vars))
        self.transforms = [None] * n
        self._has_final = has_final
        self.db_id = db_id

    def to_gpu_arrays(self) -> dict:
        empty = np.full((_N_SLOTS, _MAX_ACTIVE * _SLOT), -1.0, dtype=np.float32)
        return {
            'active_vars': _pack_vars(self._vars),
            'pre_active_vars': empty,
            'has_final_xform': self._has_final,
        }


class FakeLibrary:
    """In-memory stand-in for ``Library`` serving the flood/planner read path.

    Construct empty, then populate with ``add_genome`` + ``store_transition``.
    No SQLite, no scoring, no ``Genome.random``.
    """

    def __init__(self):
        self._genomes: dict[int, object] = {}
        self._scores: dict[int, dict] = {}          # extension-point storage
        # a -> {b: distance}; dict-of-dict gives the real table's
        # INSERT-OR-REPLACE semantics (a later identical edge overwrites).
        self._transitions: dict[int, dict[int, float]] = {}

    # -- population helpers (test-facing; not part of the real interface) --

    def add_genome(self, genome_id: int, genome,
                   scores: dict | None = None) -> int:
        """Register a genome under an id. ``scores`` is stored for the future
        ``genome_scores`` extension point (see module docstring)."""
        genome.db_id = genome_id
        self._genomes[genome_id] = genome
        if scores is not None:
            self._scores[genome_id] = dict(scores)
        return genome_id

    def store_transition(self, genome_a: int, genome_b: int,
                         distance: float) -> None:
        """Mirror of Library.store_transition: bidirectional, skips
        self/zero-distance edges, REPLACES a prior (a,b) distance."""
        if distance <= 0 or genome_a == genome_b:
            return
        self._transitions.setdefault(genome_a, {})[genome_b] = distance
        self._transitions.setdefault(genome_b, {})[genome_a] = distance

    # -- the real Library read interface the flood uses --

    def load_genome(self, genome_id: int):
        """Return the stored genome with ``.db_id`` stamped; KeyError if
        absent (same message shape as the real one)."""
        try:
            genome = self._genomes[genome_id]
        except KeyError:
            raise KeyError(f'No genome with id {genome_id}')
        genome.db_id = genome_id
        return genome

    def nearest_transitions(self, genome_id: int,
                            n: int = 10) -> list[tuple[int, float]]:
        """Return the n nearest (genome_b, distance) sorted by distance
        ascending — same ordering/limit as the SQL query."""
        edges = self._transitions.get(genome_id, {})
        ordered = sorted(edges.items(), key=lambda kv: kv[1])
        return [(b, d) for b, d in ordered[:n]]

    def close(self) -> None:  # parity with Library; nothing to release.
        pass

    # -- EXTENSION POINTS: planner steps 2+ (shader-warming.md Option A) --
    # Not built now; the storage above (self._scores, self._transitions) makes
    # each a few lines. Wire them up in the step that first needs them.

    def genome_scores(self, genome_id: int) -> dict[str, float]:
        raise NotImplementedError(
            'FakeLibrary.genome_scores: extension point — return '
            'self._scores[genome_id] when a planner test needs scores')

    def nearest_loop_transitions(self, genome_id: int, **kwargs):
        raise NotImplementedError(
            'FakeLibrary.nearest_loop_transitions: extension point — add a '
            'loop table + reuse self._transitions when a planner test '
            'needs loop reachability')
