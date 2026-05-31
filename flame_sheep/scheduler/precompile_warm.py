"""Enumerate the genome catalog into the unique pipeline-cache keys
the precompile worker will warm.

A "tuple" here is the cache key flame_chaos.comp+spec-const+var-trim
uses: (n_transforms, has_final_xform, frozenset(variation_indices)).
The variation set is the union of var_idx >= 0 across the genome's
active_vars + pre_vars buffers — same set ChaosGame.set_genome
extracts at render time. Two genomes with identical tuples share a
compiled pipeline, so deduping here matters: the handoff measured
~2206 distinct tuples across 3796 non-archived genomes (~1.72x reuse).
"""
from __future__ import annotations

import logging

import numpy as np

from typing import TYPE_CHECKING

from flame_sheep.storage.library import Library

if TYPE_CHECKING:
    from flame_sheep.genome import Genome

log = logging.getLogger(__name__)


def _genome_tuple(genome) -> tuple[int, int, frozenset[int]]:
    """Compute the pipeline cache key for a Genome (same shape
    ChaosGame.set_genome derives from the upload buffers)."""
    arrays = genome.to_gpu_arrays()
    # Same reshape as wallpaper_vk._genome_to_chaos_kwargs
    av = arrays['active_vars'].reshape(7, 8, 10)
    pv = arrays['pre_active_vars'].reshape(7, 8, 10)
    var_ids = np.concatenate([av[..., 0].ravel(), pv[..., 0].ravel()])
    keep_vars = frozenset(int(v) for v in var_ids if v >= 0)
    n_tx = len(genome.transforms)
    has_final = 1 if arrays['has_final_xform'] else 0
    return (n_tx, has_final, keep_vars)


def enumerate_catalog_tuples(lib: Library | None = None
                              ) -> set[tuple[int, int, frozenset[int]]]:
    """Walk all non-archived genomes in the catalog, return the
    deduped set of pipeline cache keys.

    DEPRECATED: this dumps the entire catalog (~2200 unique tuples)
    onto the precompile queue at wallpaper startup, which keeps the
    precompile worker busy for ~20 minutes consuming GPU time the
    wallpaper would otherwise have. Use near_tuples() instead — it
    enqueues only the genomes likely to actually play this session
    (current loop members + a graph-flood-fill of neighbors).

    Kept around for tools that genuinely want to warm the entire
    catalog (e.g. cold-start the Mesa shader cache for a benchmark).
    """
    own_lib = lib is None
    if own_lib:
        lib = Library()
    try:
        cur = lib.conn.execute(
            'SELECT id FROM genomes WHERE archived = 0')
        all_ids = [row[0] for row in cur.fetchall()]
        tuples: set[tuple[int, int, frozenset[int]]] = set()
        for n, gid in enumerate(all_ids):
            try:
                g = lib.load_genome(gid)
            except Exception as e:
                log.debug(f'skip genome {gid}: {e}')
                continue
            tuples.add(_genome_tuple(g))
            if (n + 1) % 500 == 0:
                log.debug(f'scanned {n+1}/{len(all_ids)} genomes, '
                          f'{len(tuples)} unique tuples so far')
    finally:
        if own_lib:
            lib.close()

    log.info(f'enumerated {len(all_ids)} genomes → {len(tuples)} '
             f'unique pipeline tuples '
             f'({len(all_ids)/max(1,len(tuples)):.2f}× reuse)')
    return tuples


def near_tuples(lib: Library,
                 seed_genome_ids: list[int],
                 hops: int = 2,
                 fan_per_node: int = 5,
                 ) -> set[tuple[int, int, frozenset[int]]]:
    """BFS flood-fill in the genome transition graph from `seed_genome_ids`,
    return the deduped pipeline cache keys for everything within
    `hops` graph distance — PLUS the transition-union keys for every
    graph edge we'd cross.

    Why include transitions: the wallpaper renders
    frame.genome = current.lerp(target, t) during morph. The lerp'd
    intermediate has keep_vars = UNION of both endpoints'
    variations — a third shader, not just one of the two endpoints.
    Pre-warming only the genome endpoints leaves the morph itself
    cold, causing a ~500-800ms inline compile stall on the render
    thread at morph start. Including the union covers the actual
    workload the wallpaper produces.

    Order emitted (for caller's priority-ordered enqueue):
      pass 1: the seeds themselves + their immediate transition
              unions to nearest neighbors (= the very next morph
              the wallpaper will perform)
      pass 2: those neighbors + transitions from them
      ... etc

    Bounded growth: still O(hops * fan_per_node ** hops) on genome
    count; transition unions are at most one per BFS edge, so
    bounded by the same product.
    """
    visited: set[int] = set(seed_genome_ids)
    # (parent_id, child_id) edges to materialize union tuples for.
    edges: list[tuple[int, int]] = []
    frontier = list(seed_genome_ids)
    for _ in range(hops):
        next_frontier: list[int] = []
        for gid in frontier:
            try:
                neighbors = lib.nearest_transitions(gid, n=fan_per_node)
            except Exception as e:
                log.debug(f'nearest_transitions({gid}) failed: {e}')
                continue
            for neighbor_id, _dist in neighbors:
                # Edge for transition pre-warm regardless of whether
                # neighbor is already visited (we want the union shader
                # for this pair compiled even if neighbor came in via
                # another path).
                edges.append((gid, neighbor_id))
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    next_frontier.append(neighbor_id)
        frontier = next_frontier
        if not frontier:
            break

    # Load all visited genomes once for keep_vars extraction.
    genome_by_id: dict[int, 'Genome'] = {}
    for gid in visited:
        try:
            genome_by_id[gid] = lib.load_genome(gid)
        except Exception as e:
            log.debug(f'skip genome {gid}: {e}')

    tuples: set[tuple[int, int, frozenset[int]]] = set()
    # Genome-endpoint tuples (used when current_genome == target_genome
    # or when caller wants the steady-state shader).
    for g in genome_by_id.values():
        tuples.add(_genome_tuple(g))

    # Transition-union tuples. Synthetic (n_transforms=0, has_final=0)
    # because the precompile worker only consults the `vars` list for
    # GL; Vk uses n_transforms/has_final but transition unions aren't
    # a meaningful Vk concept (Vk-side warm-gate at chaos_game level
    # catches them differently).
    union_count = 0
    for parent_id, child_id in edges:
        parent = genome_by_id.get(parent_id)
        child = genome_by_id.get(child_id)
        if parent is None or child is None:
            continue
        _, _, parent_vars = _genome_tuple(parent)
        _, _, child_vars = _genome_tuple(child)
        union_vars = parent_vars | child_vars
        # Skip if it equals one of the endpoints (no extra shader needed).
        if union_vars == parent_vars or union_vars == child_vars:
            continue
        before = len(tuples)
        tuples.add((0, 0, union_vars))
        if len(tuples) > before:
            union_count += 1
    log.debug(f'flood from {len(seed_genome_ids)} seeds, {hops}-hop, '
              f'fan={fan_per_node} → {len(visited)} genomes / '
              f'{len(tuples)} unique tuples ({union_count} transition unions)')
    return tuples


def _genome_pair_to_union_tuple(a: 'Genome', b: 'Genome'
                                  ) -> tuple[int, int, frozenset[int]]:
    """Synthetic precompile tuple for the lerp(a, b, t) intermediate
    shader. Used by callers that already know the specific transition
    they're about to start (orchestrator-side morph-start hook), as a
    complement to the near_tuples flood-fill which anticipates many
    pairs at once."""
    _, _, va = _genome_tuple(a)
    _, _, vb = _genome_tuple(b)
    return (0, 0, va | vb)


def tuple_to_json_line(t: tuple[int, int, frozenset[int]]) -> str:
    """Serialize for the precompile_worker's stdin protocol."""
    import json
    n_tx, has_final, keep_vars = t
    return json.dumps({
        'n_transforms': n_tx,
        'has_final_xform': bool(has_final),
        'vars': sorted(keep_vars),
    })
