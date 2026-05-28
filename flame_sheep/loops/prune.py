"""Loop pruning — remove duplicate rotations and low-fitness loops
whose genomes are already covered by other surviving loops."""

from __future__ import annotations

import logging

from ..storage import Library
from .candidate import _canonical_rotation

log = logging.getLogger(__name__)


def prune_duplicate_loops(lib: Library) -> int:
    """Remove loops that are rotations of each other, keeping the highest-fitness one.

    Also removes loops with identical genome sets but different structures,
    keeping only the best-scoring structure variant.
    """
    all_loops = lib.conn.execute(
        'SELECT id, fitness FROM loops ORDER BY fitness DESC'
    ).fetchall()

    seen: dict[tuple[int, ...], int] = {}  # canonical genome set → best loop ID
    to_delete = []

    for lid, fitness in all_loops:
        gids = lib.loop_genome_ids(lid)
        if not gids:
            to_delete.append(lid)
            continue
        key = _canonical_rotation(gids)
        # Also check reverse rotation as a separate canonical form
        # (we keep forward and reverse as distinct, but not rotations)
        if key in seen:
            to_delete.append(lid)
        else:
            # Check if same genome SET exists with different ordering/structure
            frozen = frozenset(gids)
            # For set-dedup, use sorted tuple as key
            set_key = tuple(sorted(gids))
            if set_key in seen:
                to_delete.append(lid)
            else:
                seen[key] = lid
                seen[set_key] = lid  # also register the set key

    if to_delete:
        n = lib.delete_loops(to_delete)
        log.info('Pruned %d duplicate/rotation loops', n)
        return n
    return 0


def prune_low_fitness_loops(lib: Library, min_fitness: float | None = None) -> int:
    """Remove low-fitness loops that don't provide unique genome coverage.

    A loop is safe to delete only if ALL its genomes appear in at least
    one other surviving loop. Never deletes user-rated loops.
    """
    all_loops = lib.conn.execute(
        'SELECT id, fitness FROM loops ORDER BY fitness DESC'
    ).fetchall()

    # Never delete loops with user votes
    protected: set[int] = set()
    for lid, _ in all_loops:
        if lib.net_rating('loop', lid) != 0:
            protected.add(lid)

    # Build genome → loop membership (only surviving loops)
    genome_loops: dict[int, set[int]] = {}
    loop_genomes: dict[int, list[int]] = {}
    for lid, _ in all_loops:
        gids = lib.loop_genome_ids(lid)
        loop_genomes[lid] = gids
        for gid in gids:
            genome_loops.setdefault(gid, set()).add(lid)

    # Walk worst-to-best: delete if all genomes are covered by other loops
    to_delete = []
    for lid, fitness in reversed(all_loops):
        if lid in protected:
            continue
        if min_fitness is not None and (fitness or 0) >= min_fitness:
            continue

        gids = loop_genomes.get(lid, [])
        if not gids:
            to_delete.append(lid)
            continue

        # Check if every genome in this loop has at least one other loop
        all_covered = all(
            len(genome_loops.get(gid, set()) - {lid} - set(to_delete)) >= 1
            for gid in gids
        )
        if all_covered:
            to_delete.append(lid)

    if to_delete:
        n = lib.delete_loops(to_delete)
        log.info('Pruned %d low-fitness loops (kept %d)', n, len(all_loops) - n)
        return n
    return 0


def prune_loops(lib: Library, min_fitness: float | None = None) -> int:
    """Run all pruning passes: duplicates first, then coverage-safe fitness prune."""
    n = prune_duplicate_loops(lib)
    n += prune_low_fitness_loops(lib, min_fitness=min_fitness)
    return n
