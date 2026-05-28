"""Loop evolution — crossover, mutate, refine, insert/delete genomes,
jitter, structure mutation, and the top-level `evolve_loops` driver that
runs a generational loop.

Also `explore_breed`: a side-channel that breeds new genomes near
high-CNN isolates (genomes with few transition neighbors), growing the
transition graph so future loop composition has more material to work
with.
"""

from __future__ import annotations

import logging

import numpy as np

from ..storage import Library
from .compose import _too_similar, compose_loops_graph
from .prune import prune_loops

log = logging.getLogger(__name__)


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
    from ..genome import Genome

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
    from ..genome import _score_from_histogram
    from ..variations import apply_variations_cpu

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
           WHERE cnn_score IS NOT NULL AND COALESCE(archived, 0) = 0
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
    fresh_blood_ratio: float = 0.5,
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
    n_offspring : int
        Base offspring count. Scaled inversely with current genome
        coverage per generation: effective n_offspring = n_offspring +
        n_offspring/max(coverage, 0.01). At 25% coverage you get 5x the
        base rate (50 if base is 10), tapering to 2x at full coverage —
        push hard when uncovered genomes are abundant, settle to a
        maintenance rate once we've saturated.
    fresh_blood_ratio : float
        Fraction of offspring slots reserved for randomly composed loops
        (not bred from existing parents). Prevents population collapse.
        Default 0.5 — fresh-blood loops are the primary source of new
        genome recruitment, while bred loops mostly recombine existing
        parents. Coverage is built by fresh blood, maintained by both.
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
    base_n_offspring = n_offspring

    for gen in range(n_generations):
        # Recompute per-generation: coverage changes as loops are added
        # during the run, so n_offspring tapers naturally as we approach
        # full coverage. Coverage = distinct genomes appearing in any
        # surviving loop, over active (un-archived) genomes total.
        n_active = lib.conn.execute(
            "SELECT COUNT(*) FROM genomes WHERE COALESCE(archived, 0) = 0"
        ).fetchone()[0]
        n_covered = lib.conn.execute(
            """SELECT COUNT(DISTINCT li.genome_id) FROM loop_items li
                 JOIN genomes g ON g.id = li.genome_id
                WHERE COALESCE(g.archived, 0) = 0"""
        ).fetchone()[0]
        coverage = (n_covered / n_active) if n_active else 0.0
        # 1% floor caps the divide-by-zero / cold-start blowup at 100x the base.
        n_offspring = base_n_offspring + int(base_n_offspring
                                              / max(coverage, 0.01))
        n_fresh = max(1, int(n_offspring * fresh_blood_ratio))
        n_bred = n_offspring - n_fresh
        log.info('[evolve_loops gen %d] coverage=%.1f%% (%d/%d) → '
                 'n_offspring=%d (fresh=%d, bred=%d)',
                 gen, coverage * 100, n_covered, n_active,
                 n_offspring, n_fresh, n_bred)
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
