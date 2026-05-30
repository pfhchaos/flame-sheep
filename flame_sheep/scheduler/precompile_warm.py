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

    Slow-ish (loads every genome's params blob) but only runs once
    at wallpaper startup. Bounded — catalog is currently ~4k
    genomes, this takes O(seconds) on local SSD.
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


def tuple_to_json_line(t: tuple[int, int, frozenset[int]]) -> str:
    """Serialize for the precompile_worker's stdin protocol."""
    import json
    n_tx, has_final, keep_vars = t
    return json.dumps({
        'n_transforms': n_tx,
        'has_final_xform': bool(has_final),
        'vars': sorted(keep_vars),
    })
