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

    @property
    def mean_coherence(self) -> float:
        return float(np.mean(self.coherences)) if self.coherences else 0.0

    @property
    def min_coherence(self) -> float:
        return float(np.min(self.coherences)) if self.coherences else 0.0

    @property
    def length(self) -> int:
        return len(self.genome_ids)


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

    # Sort by mean coherence descending
    candidates.sort(key=lambda c: c.mean_coherence, reverse=True)
    log.info('Generated %d valid loops from %d attempts', len(candidates), n_attempts)
    return candidates


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


def loop_overlap(lib: Library, loop_ids_a: list[int], loop_ids_b: list[int]) -> float:
    """
    Fraction of genomes shared between two loops.

    Returns 0.0 (completely different) to 1.0 (identical genome sets).
    Based on set intersection over the smaller loop's size.
    """
    set_a = set(loop_ids_a)
    set_b = set(loop_ids_b)
    if not set_a or not set_b:
        return 0.0
    shared = len(set_a & set_b)
    return shared / min(len(set_a), len(set_b))


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

    name = f'breed-{loop_a_id}x{loop_b_id}'
    loop_id = lib.save_loop(child_ids, name=name,
                            parent_a=loop_a_id, parent_b=loop_b_id)
    log.info('Bred loop %d from %d x %d', loop_id, loop_a_id, loop_b_id)
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

    # Find a replacement from the library that's nearby but different
    candidates = lib.top_genomes(n=50)
    replacements = []
    for cid, _ in candidates:
        if cid == target_id or cid in ids:
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

    name = f'mutate-{loop_id}-pos{pos}'
    new_id = lib.save_loop(child_ids, name=name, parent_a=loop_id)
    log.info('Mutated loop %d -> %d (pos %d: %d -> %d)',
             loop_id, new_id, pos, target_id, replacement)
    return new_id


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

        for _ in range(n_mutation):
            parent = int(rng.choice(top_ids))
            child = mutate_loop(lib, parent, rng)
            if child is not None:
                child_ids = lib.loop_genome_ids(child)
                if _too_similar(lib, child_ids, all_existing, max_overlap):
                    log.debug('Discarded mutation %d (too similar)', child)
                else:
                    new_ids.append(child)
                    all_existing.append(child)

        # Fresh blood: randomly composed loops from the genome pool
        fresh_candidates = compose_loops(
            lib, pool_size=pool_size, loop_length=loop_length,
            n_attempts=n_fresh * 3, min_coherence=-0.5,
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

        # Update fitness for all loops (incorporates user ratings)
        for lid in top_ids + new_ids:
            lib.update_loop_fitness(lid)

        log.info('Generation %d: %d new (%d fresh)', gen, len(new_ids), n_added_fresh)

    return [t[0] for t in lib.top_loops(n=n_survivors)]
