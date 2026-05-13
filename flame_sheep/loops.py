"""
Loop composition — assemble genomes into coherent visual cycles.

A loop is an ordered sequence of genomes where morphing from each to the
next produces a consistent directional flow, and the last genome wraps
back to the first.  The 3x3 motion field between adjacent genomes
characterises the visual motion of each transition; good loops have high
coherence between consecutive motion fields (the flow keeps going the
same way rather than jerking back and forth).

Composition strategy:
  1. Start from a pool of scored genomes.
  2. Precompute motion fields for candidate pairs.
  3. Build loops by greedy extension: at each step pick the next genome
     whose incoming transition continues the current flow.
  4. Score the closing transition (last → first) for cycle quality.
  5. Run many attempts from different seeds, keep the best loops.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .genome import Genome
from .storage import (
    Library, compute_motion_field, motion_field_coherence,
    MOTION_GRID,
)

log = logging.getLogger(__name__)


@dataclass
class LoopCandidate:
    """A candidate loop before it's committed to the library."""
    genome_ids: list[int]
    motion_fields: list[np.ndarray]   # one per transition (including wrap)
    coherences: list[float]           # coherence between consecutive motion fields
    structure: str = 'cyclic'         # 'cyclic' | 'palindrome' | 'rondo'

    @property
    def mean_coherence(self) -> float:
        return float(np.mean(self.coherences)) if self.coherences else 0.0

    @property
    def min_coherence(self) -> float:
        return float(np.min(self.coherences)) if self.coherences else 0.0

    @property
    def length(self) -> int:
        return len(self.genome_ids)


def _canonical_rotation(ids: list[int]) -> tuple[int, ...]:
    """Canonical form of a cyclic genome sequence (smallest rotation)."""
    n = len(ids)
    if n == 0:
        return ()
    doubled = ids + ids
    best = tuple(ids)
    for start in range(1, n):
        candidate = tuple(doubled[start:start + n])
        if candidate < best:
            best = candidate
    return best


def _dedup_rotations(candidates: list[LoopCandidate]) -> list[LoopCandidate]:
    """Remove loops that are rotations of each other, keeping the first seen."""
    seen: set[tuple[int, ...]] = set()
    result = []
    for c in candidates:
        key = _canonical_rotation(c.genome_ids)
        if key not in seen:
            seen.add(key)
            result.append(c)
    return result


# ----------------------------------------------------------------
# Loop pruning
# ----------------------------------------------------------------

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


# ----------------------------------------------------------------
# Loop structure playback
# ----------------------------------------------------------------

STRUCTURES = ('cyclic', 'palindrome', 'rondo')


def loop_sequence(items: list, structure: str = 'cyclic'):
    """Yield items in playback order for the given structure, repeating forever.

    Works with any list type — genomes, genome IDs, indices, etc.

    Cyclic:      A B C D  A B C D  A B C D ...
    Palindrome:  A B C D  C B  A B C D  C B ...  (no doubled endpoints)
    Rondo:       A B A C A D  A B A C A D ...    (home=first, episodes=rest)
    """
    if not items:
        return

    if structure == 'palindrome':
        forward = items
        backward = items[-2:0:-1]  # exclude first and last
        while True:
            yield from forward
            yield from backward
    elif structure == 'rondo':
        home = items[0]
        episodes = items[1:]
        if not episodes:
            # Degenerate: single genome, just repeat
            while True:
                yield home
        while True:
            for ep in episodes:
                yield home
                yield ep
    else:  # cyclic (default)
        while True:
            yield from items


def cycle_length(n_items: int, structure: str = 'cyclic') -> int:
    """Number of steps in one full cycle of a loop structure.

    Cyclic:      n
    Palindrome:  2*(n-1)  (forward + backward, no doubled endpoints)
    Rondo:       2*(n-1)  (home + episode for each of n-1 episodes)
    """
    if n_items <= 1:
        return max(n_items, 1)
    if structure == 'palindrome':
        return 2 * (n_items - 1)
    elif structure == 'rondo':
        return 2 * (n_items - 1)
    else:
        return n_items


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
            'SELECT id, cnn_score FROM genomes WHERE cnn_score IS NOT NULL AND cnn_score >= ?',
            (cnn_floor,),
        ).fetchall()
    else:
        score_rows = lib.conn.execute(
            'SELECT id, COALESCE(cnn_score, 0) FROM genomes WHERE id IN (SELECT DISTINCT genome_a FROM genome_transitions WHERE genome_a != genome_b)'
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


# ------------------------------------------------------------------
# Loop breeding
# ------------------------------------------------------------------

def crossover(
    lib: Library,
    loop_a_id: int,
    loop_b_id: int,
    rng: np.random.Generator | None = None,
) -> int | None:
    """
    Breed two loops by splicing a subsequence from one into the other.

    Picks a random crossover point and length, takes that slice from loop_b
    and inserts it into loop_a (replacing the same-length section), then
    saves the result if it's viable.

    Returns the new loop ID, or None if the result is degenerate.
    """
    if rng is None:
        rng = np.random.default_rng()

    ids_a = lib.loop_genome_ids(loop_a_id)
    ids_b = lib.loop_genome_ids(loop_b_id)

    if len(ids_a) < 3 or len(ids_b) < 3:
        return None

    # Pick a slice length (1 to half the loop)
    max_slice = max(1, len(ids_a) // 2)
    slice_len = int(rng.integers(1, max_slice + 1))

    # Pick crossover points in each loop
    point_a = int(rng.integers(0, len(ids_a)))
    point_b = int(rng.integers(0, len(ids_b)))

    # Extract slice from B
    donor = []
    for i in range(slice_len):
        donor.append(ids_b[(point_b + i) % len(ids_b)])

    # Build child: A with the slice at point_a replaced by donor
    child_ids = list(ids_a)
    for i in range(slice_len):
        idx = (point_a + i) % len(child_ids)
        child_ids[idx] = donor[i]

    # Deduplicate — if crossover introduced repeats, bail
    if len(set(child_ids)) < len(child_ids):
        return None

    # Inherit structure from fitter parent (loop_a is typically the fitter one)
    child_structure = lib.loop_type(loop_a_id)
    name = f'breed-{loop_a_id}x{loop_b_id}'
    loop_id = lib.save_loop(child_ids, name=name, loop_type=child_structure,
                            parent_a=loop_a_id, parent_b=loop_b_id)
    log.debug('Bred loop %d (%s) from %d x %d',
              loop_id, child_structure, loop_a_id, loop_b_id)
    return loop_id


def mutate_loop(
    lib: Library,
    loop_id: int,
    rng: np.random.Generator | None = None,
) -> int | None:
    """
    Mutate a loop by replacing one genome with a nearby variant.

    Picks a random position, finds a genome in the library that's close
    to the current one (distance 0.1-0.4) but not identical, and swaps it in.

    Returns the new loop ID, or None if no suitable replacement found.
    """
    if rng is None:
        rng = np.random.default_rng()

    ids = lib.loop_genome_ids(loop_id)
    if not ids:
        return None

    # Pick a position to mutate
    pos = int(rng.integers(0, len(ids)))
    target_id = ids[pos]
    target = lib.load_genome(target_id)

    # Find a replacement from the transition cache
    neighbors = lib.nearest_transitions(target_id, n=30)
    id_set = set(ids)
    replacements = []
    for gid, dist in neighbors:
        if gid in id_set:
            continue
        norm_dist = dist / (dist + 1.0)
        if 0.1 <= norm_dist <= 0.4:
            replacements.append((gid, dist))

    # Fall back to brute-force scan if transition cache is empty
    if not replacements:
        candidates = lib.top_genomes(n=50)
        for cid, _ in candidates:
            if cid == target_id or cid in id_set:
                continue
            d = target.distance(lib.load_genome(cid))
            if 0.1 <= d <= 0.4:
                replacements.append((cid, d))

    if not replacements:
        return None

    # Pick randomly from viable replacements (weighted toward closer)
    rep_ids = [r[0] for r in replacements]
    rep_dists = np.array([r[1] for r in replacements])
    weights = 1.0 / (rep_dists + 0.01)
    weights /= weights.sum()
    replacement = int(rng.choice(rep_ids, p=weights))

    child_ids = list(ids)
    child_ids[pos] = replacement

    # Inherit parent's structure
    parent_structure = lib.loop_type(loop_id)
    name = f'mutate-{loop_id}-pos{pos}'
    new_id = lib.save_loop(child_ids, name=name, loop_type=parent_structure,
                           parent_a=loop_id)
    log.debug('Mutated loop %d -> %d (pos %d: %d -> %d)',
              loop_id, new_id, pos, target_id, replacement)
    return new_id


def refine_loop(
    lib: Library,
    loop_id: int,
    rng: np.random.Generator | None = None,
) -> int | None:
    """
    Replace the weakest genome in a loop with a crossover of its neighbors.

    1. Find the genome with lowest fitness in the loop.
    2. Crossover its two neighbors (wrapping for cyclic loops).
    3. Mutate the offspring slightly.
    4. If the offspring has higher fitness, create a new loop with it.

    This preserves the loop's character while improving its weakest link.
    The offspring is naturally "between" its neighbors in parameter space,
    so morph transitions should be smooth.

    Returns the new loop ID, or None if refinement failed.
    """
    from .genome import Genome

    if rng is None:
        rng = np.random.default_rng()

    ids = lib.loop_genome_ids(loop_id)
    if not ids or len(ids) < 3:
        return None

    # Find weakest genome by fitness
    fitnesses = [(i, lib.genome_fitness(gid)) for i, gid in enumerate(ids)]
    worst_pos, worst_fitness = min(fitnesses, key=lambda x: x[1])
    worst_id = ids[worst_pos]

    # Get neighbors (wrapping for cyclic)
    n = len(ids)
    prev_pos = (worst_pos - 1) % n
    next_pos = (worst_pos + 1) % n
    parent_a = lib.load_genome(ids[prev_pos])
    parent_b = lib.load_genome(ids[next_pos])

    # Crossover: interpolate between neighbors
    # Blend factor biased toward center (0.3-0.7) for smooth transitions
    blend = float(rng.uniform(0.3, 0.7))
    child = parent_a.lerp(parent_b, blend)

    # Light mutation: jitter the child's parameters
    child = child.jitter(rng, scale=0.05)

    # Score the child
    from .genome import _score_from_histogram
    from .variations import apply_variations_cpu

    grid_size = 64
    n_iter = 20_000
    fuse = 20
    bound = 4.0

    hit_grid = np.zeros((grid_size, grid_size), dtype=np.float64)
    color_grid = np.zeros((grid_size, grid_size), dtype=np.float64)

    weights = np.array([tr.weight for tr in child.transforms], dtype=np.float64)
    if weights.sum() == 0:
        return None
    weights /= weights.sum()
    cumw = np.cumsum(weights)

    x, y, c = 0.0, 0.0, 0.5
    for i in range(fuse + n_iter):
        r = rng.random()
        tidx = min(int(np.searchsorted(cumw, r)), len(child.transforms) - 1)
        tr = child.transforms[tidx]
        a, b, cc, d, e, f = tr.affine
        nx = a * x + b * y + cc
        ny = d * x + e * y + f
        nx, ny = apply_variations_cpu(tr.variations, nx, ny, tr.affine)
        x, y = nx, ny
        c = (c + tr.color) * 0.5
        if not (np.isfinite(x) and np.isfinite(y)):
            return None  # child diverged
        if i >= fuse and abs(x) < bound and abs(y) < bound:
            gx = max(0, min(grid_size - 1, int((x + bound) / (2 * bound) * grid_size)))
            gy = max(0, min(grid_size - 1, int((y + bound) / (2 * bound) * grid_size)))
            hit_grid[gy, gx] += 1.0
            color_grid[gy, gx] += c

    scores = _score_from_histogram(hit_grid, color_grid)
    child_fitness = (
        scores.get('coverage', 0) + scores.get('entropy', 0)
        + scores.get('edge_sharpness', 0) + scores.get('contour_coherence', 0)
        + scores.get('complexity', 0)
    )

    # Only replace if child is better than the worst
    if child_fitness <= worst_fitness:
        log.debug('Refine loop %d: child fitness %.3f <= worst %.3f, skipped',
                  loop_id, child_fitness, worst_fitness)
        return None

    # Frame and save child genome
    child.survey_and_correct()
    child_gid = lib.save_genome(child)
    child_ids = list(ids)
    child_ids[worst_pos] = child_gid

    parent_structure = lib.loop_type(loop_id)
    name = f'refine-{loop_id}-pos{worst_pos}'
    new_id = lib.save_loop(child_ids, name=name, loop_type=parent_structure,
                           parent_a=loop_id)
    log.info('Refined loop %d -> %d (pos %d: fitness %.3f -> %.3f)',
             loop_id, new_id, worst_pos, worst_fitness, child_fitness)
    return new_id


def insert_genome(
    lib: Library,
    loop_id: int,
    rng: np.random.Generator | None = None,
    max_length: int = 10,
) -> int | None:
    """
    Insert a genome at the roughest transition to smooth it out.

    Finds the transition with the highest genome distance, then picks
    a genome from the library that's between the two endpoints
    (distance to each < distance between them).

    Returns new loop ID, or None if loop is already at max length or
    no suitable bridging genome found.
    """
    if rng is None:
        rng = np.random.default_rng()

    ids = lib.loop_genome_ids(loop_id)
    if not ids or len(ids) >= max_length:
        return None

    # Find transition distances
    genomes = [lib.load_genome(gid) for gid in ids]
    n = len(genomes)
    dists = []
    for i in range(n):
        dists.append(genomes[i].distance(genomes[(i + 1) % n]))

    # Pick the roughest transition
    worst_idx = int(np.argmax(dists))
    g_before = genomes[worst_idx]
    g_after = genomes[(worst_idx + 1) % n]
    gap_dist = dists[worst_idx]

    # Find a bridging genome via transition cache intersection
    id_set = set(ids)
    before_id = ids[worst_idx]
    after_id = ids[(worst_idx + 1) % n]

    # Get neighbors of both endpoints
    before_neighbors = {gid: dist for gid, dist in lib.nearest_transitions(before_id, n=30)}
    after_neighbors = {gid: dist for gid, dist in lib.nearest_transitions(after_id, n=30)}

    # Intersect: genomes near both endpoints are natural bridges
    bridges = []
    common = set(before_neighbors) & set(after_neighbors) - id_set
    for cid in common:
        d_before = before_neighbors[cid]
        d_after = after_neighbors[cid]
        score = d_before + d_after
        bridges.append((cid, score))

    # Fall back to brute-force if no cached bridges
    if not bridges:
        candidates = lib.top_genomes(n=50)
        for cid, _ in candidates:
            if cid in id_set:
                continue
            cg = lib.load_genome(cid)
            d_before = g_before.distance(cg)
            d_after = cg.distance(g_after)
            if d_before < gap_dist * 0.8 and d_after < gap_dist * 0.8:
                score = d_before + d_after
                bridges.append((cid, score))

    if not bridges:
        return None

    # Pick from best bridges (weighted toward lower total distance)
    bridges.sort(key=lambda x: x[1])
    top_bridges = bridges[:min(5, len(bridges))]
    scores = np.array([b[1] for b in top_bridges])
    weights = 1.0 / (scores + 0.01)
    weights /= weights.sum()
    chosen = top_bridges[int(rng.choice(len(top_bridges), p=weights))][0]

    # Insert after worst_idx
    child_ids = list(ids)
    insert_pos = worst_idx + 1
    child_ids.insert(insert_pos, chosen)

    parent_structure = lib.loop_type(loop_id)
    name = f'insert-{loop_id}-pos{insert_pos}'
    new_id = lib.save_loop(child_ids, name=name, loop_type=parent_structure,
                           parent_a=loop_id)
    log.debug('Inserted into loop %d -> %d (pos %d, gap=%.3f)',
              loop_id, new_id, insert_pos, gap_dist)
    return new_id


def delete_genome(
    lib: Library,
    loop_id: int,
    rng: np.random.Generator | None = None,
    min_length: int = 3,
) -> int | None:
    """
    Delete a genome at the smoothest transition to tighten the loop.

    Finds the transition with the lowest genome distance (most similar
    neighbors), removes the second genome of that pair.

    Returns new loop ID, or None if loop is already at min length.
    """
    if rng is None:
        rng = np.random.default_rng()

    ids = lib.loop_genome_ids(loop_id)
    if not ids or len(ids) <= min_length:
        return None

    # Find transition distances
    genomes = [lib.load_genome(gid) for gid in ids]
    n = len(genomes)
    dists = []
    for i in range(n):
        dists.append(genomes[i].distance(genomes[(i + 1) % n]))

    # Pick the smoothest transition — remove the second genome
    smoothest_idx = int(np.argmin(dists))
    remove_pos = (smoothest_idx + 1) % n

    child_ids = list(ids)
    child_ids.pop(remove_pos)

    parent_structure = lib.loop_type(loop_id)
    name = f'delete-{loop_id}-pos{remove_pos}'
    new_id = lib.save_loop(child_ids, name=name, loop_type=parent_structure,
                           parent_a=loop_id)
    log.debug('Deleted from loop %d -> %d (pos %d, smoothest=%.3f)',
              loop_id, new_id, remove_pos, dists[smoothest_idx])
    return new_id


def jitter_loop(
    lib: Library,
    loop_id: int,
    rng: np.random.Generator | None = None,
    scale: float = 0.1,
) -> int | None:
    """
    Mutate a loop by jittering variation parameters of one genome.

    Picks a random position and creates a jittered copy of that genome.
    The new genome is saved to the library, and a new loop is created
    with the jittered genome in place of the original.

    Returns the new loop ID, or None if the genome has no params to jitter.
    """
    if rng is None:
        rng = np.random.default_rng()

    ids = lib.loop_genome_ids(loop_id)
    if not ids:
        return None

    pos = int(rng.integers(0, len(ids)))
    genome = lib.load_genome(ids[pos])

    # Check if any transform has params to jitter
    has_params = any(tr.var_params for tr in genome.transforms)
    if not has_params:
        return None

    jittered = genome.jitter(rng, scale=scale)
    if not jittered.is_viable():
        return None
    jittered.survey_and_correct()

    new_gid = lib.save_genome(jittered)
    child_ids = list(ids)
    child_ids[pos] = new_gid

    parent_structure = lib.loop_type(loop_id)
    name = f'jitter-{loop_id}-pos{pos}'
    new_id = lib.save_loop(child_ids, name=name, loop_type=parent_structure,
                           parent_a=loop_id)
    log.debug('Jittered loop %d -> %d (pos %d, scale %.2f)',
              loop_id, new_id, pos, scale)
    return new_id


def mutate_structure(
    lib: Library,
    loop_id: int,
    rng: np.random.Generator | None = None,
) -> int | None:
    """
    Mutate a loop's structure type.

    Mutations:
    - cyclic ↔ palindrome (natural flip, just changes playback)
    - cyclic/palindrome → rondo (if loop has ≥3 genomes)
    - rondo → palindrome (demotion)

    Returns new loop ID with the same genomes but different structure,
    or None if no valid mutation.
    """
    if rng is None:
        rng = np.random.default_rng()

    current = lib.loop_type(loop_id)
    ids = lib.loop_genome_ids(loop_id)
    if not ids:
        return None

    # Choose a new structure
    if current == 'cyclic':
        new_structure = rng.choice(['palindrome', 'rondo']) if len(ids) >= 3 else 'palindrome'
    elif current == 'palindrome':
        new_structure = rng.choice(['cyclic', 'rondo']) if len(ids) >= 3 else 'cyclic'
    elif current == 'rondo':
        new_structure = rng.choice(['cyclic', 'palindrome'])
    else:
        new_structure = 'cyclic'

    name = f'structure-{loop_id}-{current}-to-{new_structure}'
    new_id = lib.save_loop(ids, name=name, loop_type=new_structure, parent_a=loop_id)
    log.debug('Structure mutation %d (%s -> %s) -> %d',
              loop_id, current, new_structure, new_id)
    return new_id


def explore_breed(
    lib: Library,
    rng: np.random.Generator | None = None,
    n_children: int = 3,
    jitter_scale: float = 0.1,
) -> list[int]:
    """Breed new genomes near high-CNN genomes that have few transition neighbors.

    These genomes are quality isolates — good fractals that can't participate
    in loops because they have no smooth transitions to anything. Breeding
    nearby creates neighbors, growing the transition graph over time.

    The new genomes are saved to the library. Background workers will
    score them (CNN) and compute transition distances automatically.

    Returns list of new genome IDs.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Find genomes with CNN scores but few transition neighbors
    transition_counts = lib.genome_transition_counts()
    rows = lib.conn.execute(
        '''SELECT id, cnn_score FROM genomes
           WHERE cnn_score IS NOT NULL
           ORDER BY cnn_score DESC
           LIMIT 200'''
    ).fetchall()

    if not rows:
        return []

    # Score by CNN quality * isolation (fewer neighbors = higher priority)
    scored = []
    for gid, cnn in rows:
        n_neighbors = transition_counts.get(gid, 0)
        # Priority = CNN score * isolation factor
        # Genomes with 0 neighbors get max isolation bonus
        isolation = 1.0 / (1.0 + n_neighbors)
        scored.append((gid, cnn * isolation))

    scored.sort(key=lambda x: x[1], reverse=True)

    new_ids = []
    for gid, _ in scored[:n_children * 2]:  # try more than needed
        if len(new_ids) >= n_children:
            break
        genome = lib.load_genome(gid)
        child = genome.jitter(rng, scale=jitter_scale)
        if not child.is_viable():
            continue
        child.survey_and_correct()
        child_id = lib.save_genome(child)
        new_ids.append(child_id)
        log.debug('Explore breed: genome #%d -> #%d (jitter %.2f)',
                  gid, child_id, jitter_scale)

    if new_ids:
        log.info('Explore breed: %d new genomes near %d quality isolates',
                 len(new_ids), min(len(scored), n_children * 2))
    return new_ids


def evolve_loops(
    lib: Library,
    n_generations: int = 5,
    n_offspring: int = 10,
    n_survivors: int = 5,
    fresh_blood_ratio: float = 0.3,
    max_overlap: float = 0.5,
    pool_size: int = 30,
    loop_length: int = 6,
) -> list[int]:
    """
    Run a simple evolutionary loop on the library's loops.

    Each generation:
      1. Take top loops by fitness (including user ratings).
      2. Breed pairs and mutate individuals to produce offspring.
      3. Inject fresh randomly-composed loops to maintain diversity.
      4. Reject offspring that overlap too much with existing loops.
      5. Keep the best survivors for the next round.

    Parameters
    ----------
    fresh_blood_ratio : float
        Fraction of offspring slots reserved for randomly composed loops
        (not bred from existing parents). Prevents population collapse.
    max_overlap : float
        Maximum genome overlap allowed between a new loop and any existing
        loop. Offspring exceeding this are discarded.
    pool_size : int
        Genome pool size for fresh blood composition.
    loop_length : int
        Target length for fresh blood loops.

    Returns IDs of the final top loops.
    """
    rng = np.random.default_rng()

    n_fresh = max(1, int(n_offspring * fresh_blood_ratio))
    n_bred = n_offspring - n_fresh

    for gen in range(n_generations):
        top = lib.top_loops(n=n_survivors * 2)
        if len(top) < 2:
            log.warning('Not enough loops to evolve (gen %d)', gen)
            break

        top_ids = [t[0] for t in top]
        all_existing = [t[0] for t in lib.top_loops(n=100)]
        new_ids = []

        # Bred offspring: crossover + mutation
        n_crossover = n_bred // 2
        n_mutation = n_bred - n_crossover

        for _ in range(n_crossover):
            a, b = rng.choice(top_ids, size=2, replace=False)
            child = crossover(lib, int(a), int(b), rng)
            if child is not None:
                child_ids = lib.loop_genome_ids(child)
                if _too_similar(lib, child_ids, all_existing, max_overlap):
                    log.debug('Discarded crossover %d (too similar)', child)
                else:
                    new_ids.append(child)
                    all_existing.append(child)

        # Mutations: 40% genome swap, 25% param jitter, 15% length, 20% structure
        n_structure_mut = max(1, n_mutation // 5)
        n_length_mut = max(1, n_mutation * 3 // 20)
        n_jitter_mut = max(1, n_mutation // 4)
        n_genome_mut = n_mutation - n_structure_mut - n_length_mut - n_jitter_mut

        for _ in range(n_genome_mut):
            parent = int(rng.choice(top_ids))
            child = mutate_loop(lib, parent, rng)
            if child is not None:
                child_ids = lib.loop_genome_ids(child)
                if _too_similar(lib, child_ids, all_existing, max_overlap):
                    log.debug('Discarded mutation %d (too similar)', child)
                else:
                    new_ids.append(child)
                    all_existing.append(child)

        for _ in range(n_length_mut):
            parent = int(rng.choice(top_ids))
            if rng.random() < 0.5:
                child = insert_genome(lib, parent, rng)
            else:
                child = delete_genome(lib, parent, rng)
            if child is not None:
                child_ids = lib.loop_genome_ids(child)
                if _too_similar(lib, child_ids, all_existing, max_overlap):
                    log.debug('Discarded length mutation %d (too similar)', child)
                else:
                    new_ids.append(child)
                    all_existing.append(child)

        for _ in range(n_jitter_mut):
            parent = int(rng.choice(top_ids))
            child = jitter_loop(lib, parent, rng)
            if child is not None:
                child_ids = lib.loop_genome_ids(child)
                if _too_similar(lib, child_ids, all_existing, max_overlap):
                    log.debug('Discarded jitter mutation %d (too similar)', child)
                else:
                    new_ids.append(child)
                    all_existing.append(child)

        # Refinement: replace weakest genome in top loops
        n_refine = max(1, n_mutation // 5)
        for _ in range(n_refine):
            parent = int(rng.choice(top_ids))
            child = refine_loop(lib, parent, rng)
            if child is not None:
                child_ids = lib.loop_genome_ids(child)
                if _too_similar(lib, child_ids, all_existing, max_overlap):
                    log.debug('Discarded refinement %d (too similar)', child)
                else:
                    new_ids.append(child)
                    all_existing.append(child)

        for _ in range(n_structure_mut):
            parent = int(rng.choice(top_ids))
            child = mutate_structure(lib, parent, rng)
            if child is not None:
                child_ids = lib.loop_genome_ids(child)
                if _too_similar(lib, child_ids, all_existing, max_overlap):
                    log.debug('Discarded structure mutation %d (too similar)', child)
                else:
                    new_ids.append(child)
                    all_existing.append(child)

        # Fresh blood: graph-based composition (requires transition data)
        # Sparse graphs need many attempts to find closeable cycles
        fresh_candidates = compose_loops_graph(
            lib, loop_length=loop_length, n_attempts=max(n_fresh * 3, 200),
        )
        n_added_fresh = 0
        for cand in fresh_candidates:
            if n_added_fresh >= n_fresh:
                break
            if _too_similar(lib, cand.genome_ids, all_existing, max_overlap):
                continue
            fresh_id = lib.save_loop(cand.genome_ids,
                                     motion_fields=cand.motion_fields,
                                     name=f'fresh-gen{gen}')
            new_ids.append(fresh_id)
            all_existing.append(fresh_id)
            n_added_fresh += 1

        # Explore breeding: grow transition graph near quality isolates
        explore_breed(lib, rng, n_children=2)

        # Update fitness for all loops (incorporates user ratings)
        for lid in top_ids + new_ids:
            lib.update_loop_fitness(lid)

        # Prune duplicates and low-fitness loops
        prune_loops(lib)

        log.info('Generation %d: %d new (%d fresh), %d total loops',
                 gen, len(new_ids), n_added_fresh, lib.loop_count())

    return [t[0] for t in lib.top_loops(n=n_survivors)]
