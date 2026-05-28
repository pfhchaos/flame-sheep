"""Loop composition — two strategies:

  - `compose_loops`: motion-field-based, computes motion fields between
    candidate genome pairs from the top of the library and builds loops
    by greedy coherence-following extension.
  - `compose_loops_graph`: graph-based, uses cached transition distances
    from `genome_transitions` table; faster and works once the transition
    graph is populated.

Both return ranked `LoopCandidate` lists; `save_best_loops` commits the
top N to the library.
"""

from __future__ import annotations

import logging

import numpy as np

from ..genome import Genome
from ..storage import (
    Library, compute_motion_field, motion_field_coherence,
)
from .candidate import LoopCandidate, _dedup_rotations

log = logging.getLogger(__name__)


def compose_loops(
    lib: Library,
    pool_size: int = 40,
    loop_length: int = 6,
    n_attempts: int = 50,
    min_distance: float = 0.15,
    max_distance: float = 0.5,
    min_coherence: float = -0.2,
) -> list[LoopCandidate]:
    """
    Generate candidate loops from the library's top genomes.

    Parameters
    ----------
    lib : Library
        Database containing scored genomes.
    pool_size : int
        Number of top genomes to draw from.
    loop_length : int
        Target number of genomes per loop (4-8 is typical).
    n_attempts : int
        Number of random starting points to try.
    min_distance : float
        Minimum perceptual distance between consecutive genomes in a loop.
        Prevents near-duplicate transitions that look like stutters.
    min_coherence : float
        Minimum coherence for a transition to be considered.
        Negative values allow mild direction changes.

    Returns
    -------
    list[LoopCandidate]
        Candidate loops sorted by mean coherence (best first).
        Typically 0-n_attempts results; bad candidates are filtered out.
    """
    top = lib.top_genomes(n=pool_size)
    if len(top) < loop_length:
        log.warning('Not enough genomes in library (%d < %d)', len(top), loop_length)
        return []

    ids = [t[0] for t in top]
    genomes = {gid: lib.load_genome(gid) for gid in ids}

    # Precompute motion fields for all directed pairs
    log.info('Precomputing motion fields for %d genomes...', len(ids))
    motion_cache: dict[tuple[int, int], np.ndarray] = {}
    for i, a_id in enumerate(ids):
        for b_id in ids:
            if a_id != b_id:
                motion_cache[(a_id, b_id)] = compute_motion_field(
                    genomes[a_id], genomes[b_id]
                )
    log.info('Motion field cache: %d entries', len(motion_cache))

    # Precompute distances for the min_distance filter
    distance_cache: dict[tuple[int, int], float] = {}
    for a_id in ids:
        for b_id in ids:
            if a_id != b_id:
                distance_cache[(a_id, b_id)] = genomes[a_id].distance(genomes[b_id])

    rng = np.random.default_rng()
    candidates: list[LoopCandidate] = []

    for attempt in range(n_attempts):
        loop = _build_one_loop(
            ids, genomes, motion_cache, distance_cache, rng,
            loop_length, min_distance, max_distance, min_coherence,
        )
        if loop is not None:
            candidates.append(loop)

    candidates = _dedup_rotations(candidates)
    candidates.sort(key=lambda c: c.mean_coherence, reverse=True)
    log.info('Generated %d unique loops from %d attempts', len(candidates), n_attempts)
    return candidates


# ----------------------------------------------------------------
# Graph-based loop composition (uses transition cache + CNN scores)
# ----------------------------------------------------------------

def compose_loops_graph(
    lib: Library,
    loop_length: int = 6,
    n_attempts: int = 50,
    min_distance: float = 0.15,
    cnn_floor: float | None = None,
) -> list[LoopCandidate]:
    """Generate candidate loops by traversing the genome transition graph.

    Uses cached transition distances instead of computing motion fields.
    Seeds weighted by CNN score and coverage bonus (genomes not yet in
    many loops get priority).

    Parameters
    ----------
    lib : Library
        Database with populated genome_transitions table.
    loop_length : int
        Target genomes per loop.
    n_attempts : int
        Number of random seeds to try.
    min_distance : float
        Minimum normalized distance between consecutive genomes.
        Prevents near-duplicate transitions.
    cnn_floor : float or None
        Minimum CNN score for genomes to be included. None = no filter.

    Returns
    -------
    list[LoopCandidate]
        Candidate loops sorted by composite score (best first).
    """
    # Build adjacency list from transition cache
    rows = lib.conn.execute(
        '''SELECT genome_a, genome_b, distance
           FROM genome_transitions
           WHERE genome_a != genome_b'''
    ).fetchall()
    if not rows:
        log.warning('No transition data — cannot compose graph loops')
        return []

    adj: dict[int, list[tuple[int, float]]] = {}
    for ga, gb, dist in rows:
        adj.setdefault(ga, []).append((gb, dist))

    # Filter by CNN score floor
    node_scores: dict[int, float] = {}
    if cnn_floor is not None:
        score_rows = lib.conn.execute(
            'SELECT id, cnn_score FROM genomes WHERE cnn_score IS NOT NULL AND cnn_score >= ? AND COALESCE(archived, 0) = 0',
            (cnn_floor,),
        ).fetchall()
    else:
        score_rows = lib.conn.execute(
            'SELECT id, COALESCE(cnn_score, 0) FROM genomes WHERE COALESCE(archived, 0) = 0 AND id IN (SELECT DISTINCT genome_a FROM genome_transitions WHERE genome_a != genome_b)'
        ).fetchall()
    node_scores = {r[0]: r[1] for r in score_rows}

    # Only keep nodes that have transition edges AND pass CNN filter
    valid_nodes = set(node_scores.keys()) & set(adj.keys())
    if len(valid_nodes) < loop_length:
        log.warning('Not enough connected genomes with scores (%d < %d)',
                    len(valid_nodes), loop_length)
        return []

    # Coverage bonus: prefer genomes not yet in many loops
    loop_counts = lib.genome_loop_counts()

    # Build seed weights: CNN score * coverage bonus
    node_list = list(valid_nodes)
    seed_weights = np.array([
        max(node_scores.get(gid, 0), 0.01) * (1.0 / (1.0 + loop_counts.get(gid, 0)))
        for gid in node_list
    ])
    seed_weights /= seed_weights.sum()

    rng = np.random.default_rng()
    candidates: list[LoopCandidate] = []

    for _ in range(n_attempts):
        loop = _build_graph_loop(
            node_list, adj, node_scores, loop_counts, seed_weights,
            rng, loop_length, min_distance, valid_nodes,
        )
        if loop is not None:
            candidates.append(loop)

    # Score and sort: CNN quality * transition smoothness
    def _loop_score(c: LoopCandidate) -> float:
        # Use genome IDs stored on the candidate to look up CNN scores
        cnn_scores = [node_scores.get(gid, 0) for gid in c.genome_ids]
        avg_cnn = np.mean(cnn_scores) if cnn_scores else 0
        # Coherences hold transition distances (repurposed); lower = better
        avg_dist = np.mean(c.coherences) if c.coherences else 1.0
        smoothness = 1.0 / (1.0 + avg_dist)
        coverage = np.mean([
            1.0 / (1.0 + loop_counts.get(gid, 0)) for gid in c.genome_ids
        ])
        return avg_cnn * smoothness * (0.5 + 0.5 * coverage)

    candidates = _dedup_rotations(candidates)
    candidates.sort(key=_loop_score, reverse=True)
    log.info('Graph composition: %d unique loops from %d attempts (%d nodes)',
             len(candidates), n_attempts, len(valid_nodes))
    return candidates


def _build_graph_loop(
    node_list: list[int],
    adj: dict[int, list[tuple[int, float]]],
    node_scores: dict[int, float],
    loop_counts: dict[int, int],
    seed_weights: np.ndarray,
    rng: np.random.Generator,
    target_length: int,
    min_distance: float,
    valid_nodes: set[int],
) -> LoopCandidate | None:
    """Build one loop by greedy graph traversal."""
    # Pick seed
    seed_idx = rng.choice(len(node_list), p=seed_weights)
    seed = node_list[seed_idx]

    sequence = [seed]
    distances = []

    for _ in range(target_length - 1):
        current = sequence[-1]
        neighbors = adj.get(current, [])
        if not neighbors:
            break

        used = set(sequence)
        # Score neighbors: CNN quality * proximity * coverage
        scored: list[tuple[int, float, float]] = []
        for nb, dist in neighbors:
            if nb in used or nb not in valid_nodes:
                continue
            norm_dist = dist / (dist + 1.0)
            if norm_dist < min_distance:
                continue
            cnn = max(node_scores.get(nb, 0), 0.01)
            coverage = 1.0 / (1.0 + loop_counts.get(nb, 0))
            score = cnn * (1.0 / (1.0 + dist)) * (0.5 + 0.5 * coverage)
            scored.append((nb, dist, score))

        if not scored:
            break

        # Weighted random selection (not pure greedy — preserves diversity)
        scores = np.array([s[2] for s in scored])
        weights = scores / scores.sum()
        idx = rng.choice(len(scored), p=weights)
        chosen, chosen_dist, _ = scored[idx]

        sequence.append(chosen)
        distances.append(chosen_dist)

    if len(sequence) < 3:
        return None

    # Check closing edge (last → seed)
    closing_dist = None
    for nb, dist in adj.get(sequence[-1], []):
        if nb == seed:
            closing_dist = dist
            break

    if closing_dist is None:
        return None

    distances.append(closing_dist)

    # Build LoopCandidate — store transition distances in coherences field
    # (repurposed: coherences normally hold motion field coherence values,
    # but graph-composed loops don't compute motion fields)
    return LoopCandidate(
        genome_ids=sequence,
        motion_fields=[],  # no motion fields computed
        coherences=distances,  # transition distances (lower = better)
    )


def _build_one_loop(
    pool_ids: list[int],
    genomes: dict[int, Genome],
    motion_cache: dict[tuple[int, int], np.ndarray],
    distance_cache: dict[tuple[int, int], float],
    rng: np.random.Generator,
    target_length: int,
    min_distance: float,
    max_distance: float,
    min_coherence: float,
) -> LoopCandidate | None:
    """
    Attempt to build one loop by greedy extension.

    1. Pick a random starting genome.
    2. Pick a random second genome (with sufficient distance).
    3. From there, greedily pick the next genome that:
       - isn't already in the loop
       - is far enough from the previous genome
       - has the highest coherence with the current motion direction
    4. Score the closing transition and overall loop quality.
    """
    start = int(rng.choice(pool_ids))
    available = [g for g in pool_ids if g != start
                 and min_distance <= distance_cache.get((start, g), 0) <= max_distance]
    if not available:
        return None

    # Pick second genome randomly to seed variety
    second = int(rng.choice(available))

    sequence = [start, second]
    motion_fields = [motion_cache[(start, second)]]

    for step in range(target_length - 2):
        prev_id = sequence[-1]
        prev_mf = motion_fields[-1]
        used = set(sequence)

        # Score all candidates
        best_id = None
        best_score = -2.0
        for cand_id in pool_ids:
            if cand_id in used:
                continue
            d = distance_cache.get((prev_id, cand_id), 0)
            if d < min_distance or d > max_distance:
                continue
            mf = motion_cache[(prev_id, cand_id)]
            coh = motion_field_coherence(prev_mf, mf)
            if coh > best_score:
                best_score = coh
                best_id = cand_id

        if best_id is None or best_score < min_coherence:
            return None  # couldn't extend

        sequence.append(best_id)
        motion_fields.append(motion_cache[(prev_id, best_id)])

    # Closing transition: last → first
    close_mf = motion_cache[(sequence[-1], sequence[0])]
    motion_fields.append(close_mf)

    # Compute coherences between consecutive motion fields
    # (including the wrap: last_mf → close_mf)
    coherences = []
    for i in range(len(motion_fields)):
        mf_a = motion_fields[i]
        mf_b = motion_fields[(i + 1) % len(motion_fields)]
        coherences.append(motion_field_coherence(mf_a, mf_b))

    loop = LoopCandidate(
        genome_ids=sequence,
        motion_fields=motion_fields,
        coherences=coherences,
    )

    # Reject loops with any badly incoherent transition
    if loop.min_coherence < min_coherence:
        return None

    return loop


def _too_similar(lib: Library, child_ids: list[int],
                 existing_loop_ids: list[int],
                 max_overlap: float = 0.5) -> bool:
    """Check if a child loop overlaps too much with any existing loop."""
    child_set = set(child_ids)
    for lid in existing_loop_ids:
        other = lib.loop_genome_ids(lid)
        other_set = set(other)
        if not other_set:
            continue
        shared = len(child_set & other_set)
        if shared / min(len(child_set), len(other_set)) > max_overlap:
            return True
    return False


def save_best_loops(
    lib: Library,
    candidates: list[LoopCandidate],
    n_keep: int = 5,
) -> list[int]:
    """
    Save the top N candidate loops to the library.

    Returns list of loop IDs.
    """
    loop_ids = []
    for cand in candidates[:n_keep]:
        name = f'auto-{cand.length}g-coh{cand.mean_coherence:.2f}'
        loop_id = lib.save_loop(
            cand.genome_ids,
            motion_fields=cand.motion_fields,
            name=name,
        )
        loop_ids.append(loop_id)
        log.info('Saved loop %d: %s (min_coh=%.2f)', loop_id, name, cand.min_coherence)
    return loop_ids
