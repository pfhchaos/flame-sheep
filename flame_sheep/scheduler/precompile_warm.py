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

from flame_sheep.storage.library import Library

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
    `hops` graph distance.

    Replaces the catalog-dump model: instead of compiling every
    shader in the library at startup, we only compile what's likely
    to actually play soon (current loop members + their N-hop graph
    neighborhood via the genome_transitions table). Re-run on each
    loop change to flood from the new location.

    Bounded growth: at most hops * fan_per_node ** hops genomes
    visited per call. With defaults (2 hops × 5 fan) that's ≤25
    genomes added per seed before dedup; in practice much less due
    to graph overlap. Total per loop change is typically 30-80
    genomes vs the 2200 from enumerate_catalog_tuples.
    """
    visited: set[int] = set(seed_genome_ids)
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
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    next_frontier.append(neighbor_id)
        frontier = next_frontier
        if not frontier:
            break

    tuples: set[tuple[int, int, frozenset[int]]] = set()
    for gid in visited:
        try:
            g = lib.load_genome(gid)
        except Exception as e:
            log.debug(f'skip genome {gid}: {e}')
            continue
        tuples.add(_genome_tuple(g))
    log.debug(f'flood from {len(seed_genome_ids)} seeds, {hops}-hop, '
              f'fan={fan_per_node} → {len(visited)} genomes / '
              f'{len(tuples)} unique tuples')
    return tuples


def tuple_to_json_line(t: tuple[int, int, frozenset[int]]) -> str:
    """Serialize for the precompile_worker's stdin protocol."""
    import json
    n_tx, has_final, keep_vars = t
    return json.dumps({
        'n_transforms': n_tx,
        'has_final_xform': bool(has_final),
        'vars': sorted(keep_vars),
    })
