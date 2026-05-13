"""Transition quality scoring between genome pairs.

Measures how smoothly the renderer can interpolate from one genome to
another. Used for loop composition (smooth internal transitions) and
loop switching (pick the best entry point into a new loop).

Two-level comparison:
  1. Structural filter: variation signature distance (cheap bag comparison)
  2. Parameter distance: permutation-matched transform comparison

Lower distance = smoother transition.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .genome import Genome, Transform, MAX_TRANSFORMS

# Threshold for considering a variation "active" in a transform
ACTIVE_THRESHOLD = 0.01

# Maximum structural (signature) distance before skipping full comparison
MAX_SIGNATURE_DISTANCE = 6

# Penalty per unmatched transform
UNMATCHED_PENALTY = 0.5

# Distance returned for pairs that fail the structural filter
FILTERED_DISTANCE = 10.0

# Normalization ranges for parameter distance
_AFFINE_RANGE = 4.0      # affine coefficients roughly in [-2, 2]
_ZOOM_RANGE = 3.5         # zoom roughly in [0.5, 4.0]
_ROTATION_RANGE = 2 * np.pi
_CENTER_RANGE = 8.0       # center roughly in [-4, 4]

# Maximum distance to cache in genome_transitions table
CACHE_THRESHOLD = 2.0


def _active_variations(transform: Transform) -> set[int]:
    """Active variation indices for a single transform."""
    return {int(i) for i in np.where(transform.variations > ACTIVE_THRESHOLD)[0]}


def variation_signature(genome: Genome) -> str:
    """Sorted bag of active variation indices across all transforms.

    Example: transforms using [26], [26], [18,23] → "18,23,26,26"
    Genomes with final_xform get "+F" suffix with the final xform's variations.
    """
    indices: list[int] = []
    for tr in genome.transforms:
        indices.extend(sorted(_active_variations(tr)))
    indices.sort()
    sig = ','.join(str(i) for i in indices)
    if genome.final_xform is not None:
        f_indices = sorted(_active_variations(genome.final_xform))
        f_sig = ','.join(str(i) for i in f_indices)
        sig = f'{sig}+F{f_sig}' if f_sig else f'{sig}+F'
    return sig


def signature_distance(sig_a: str, sig_b: str) -> int:
    """Manhattan distance between two variation signature bags.

    Counts how many variation slots differ between the two signatures.
    """
    if not sig_a and not sig_b:
        return 0
    bag_a = Counter(sig_a.split(',')) if sig_a else Counter()
    bag_b = Counter(sig_b.split(',')) if sig_b else Counter()
    all_keys = set(bag_a) | set(bag_b)
    return sum(abs(bag_a.get(k, 0) - bag_b.get(k, 0)) for k in all_keys)


def match_transforms(
    a: list[Transform], b: list[Transform],
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match transforms between two genomes by variation overlap.

    Uses greedy matching on Jaccard similarity of active variation sets.

    Returns:
        (matched_pairs, unmatched_a, unmatched_b)
        where matched_pairs is [(a_idx, b_idx), ...]
    """
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return [], list(range(n_a)), list(range(n_b))

    # Compute active variation sets
    sets_a = [_active_variations(t) for t in a]
    sets_b = [_active_variations(t) for t in b]

    # Jaccard similarity matrix
    scores = np.zeros((n_a, n_b), dtype=np.float64)
    for i in range(n_a):
        for j in range(n_b):
            union = len(sets_a[i] | sets_b[j])
            if union == 0:
                scores[i, j] = 1.0  # both empty = perfect match
            else:
                scores[i, j] = len(sets_a[i] & sets_b[j]) / union

    # Greedy matching: pick best pair, remove both, repeat
    matched = []
    used_a: set[int] = set()
    used_b: set[int] = set()

    for _ in range(min(n_a, n_b)):
        best_score = -1.0
        best_i, best_j = -1, -1
        for i in range(n_a):
            if i in used_a:
                continue
            for j in range(n_b):
                if j in used_b:
                    continue
                if scores[i, j] > best_score:
                    best_score = scores[i, j]
                    best_i, best_j = i, j
        if best_i < 0:
            break
        matched.append((best_i, best_j))
        used_a.add(best_i)
        used_b.add(best_j)

    unmatched_a = [i for i in range(n_a) if i not in used_a]
    unmatched_b = [j for j in range(n_b) if j not in used_b]
    return matched, unmatched_a, unmatched_b


def _transform_distance(a: Transform, b: Transform) -> float:
    """Normalized distance between two matched transforms."""
    # Affine distance (6 coefficients)
    affine_diff = (a.affine - b.affine) / _AFFINE_RANGE
    affine_dist = float(np.linalg.norm(affine_diff)) / np.sqrt(6)

    # Active variation weight distance (only shared + unique variations)
    active = _active_variations(a) | _active_variations(b)
    if active:
        idx = sorted(active)
        var_diff = a.variations[idx] - b.variations[idx]
        var_dist = float(np.linalg.norm(var_diff)) / np.sqrt(len(idx))
    else:
        var_dist = 0.0

    # Color and weight (already in [0, 1] range)
    color_dist = abs(a.color - b.color)
    weight_a = a.weight
    weight_b = b.weight
    # Normalize weight by max of the pair to get relative difference
    w_max = max(weight_a, weight_b, 1e-10)
    weight_dist = abs(weight_a - weight_b) / w_max

    # Post-affine distance
    if a.post_affine is not None and b.post_affine is not None:
        pa_diff = (a.post_affine - b.post_affine) / _AFFINE_RANGE
        post_affine_dist = float(np.linalg.norm(pa_diff)) / np.sqrt(6)
    elif a.post_affine is not None or b.post_affine is not None:
        post_affine_dist = 0.3  # structural mismatch penalty
    else:
        post_affine_dist = 0.0

    return float(affine_dist + var_dist + color_dist * 0.3 + weight_dist * 0.3
                 + post_affine_dist)


def _global_distance(a: Genome, b: Genome) -> float:
    """Normalized distance between global parameters."""
    zoom_dist = abs(a.zoom - b.zoom) / _ZOOM_RANGE
    rot_diff = abs(a.rotation - b.rotation)
    # Handle rotation wrap-around
    rot_dist = min(rot_diff, 2 * np.pi - rot_diff) / np.pi
    center_dist = float(np.linalg.norm(a.center - b.center)) / _CENTER_RANGE
    return float(zoom_dist + rot_dist + center_dist)


def compute_transition_distance(
    genome_a: Genome,
    genome_b: Genome,
    sig_a: str | None = None,
    sig_b: str | None = None,
) -> float:
    """Compute transition quality distance between two genomes.

    Lower = smoother transition. Returns FILTERED_DISTANCE for
    structurally incompatible pairs.

    Args:
        genome_a, genome_b: Genomes to compare
        sig_a, sig_b: Pre-computed variation signatures (optional,
            computed if not provided)
    """
    # Level 1: structural filter
    if sig_a is None:
        sig_a = variation_signature(genome_a)
    if sig_b is None:
        sig_b = variation_signature(genome_b)
    sig_dist = signature_distance(sig_a, sig_b)
    if sig_dist > MAX_SIGNATURE_DISTANCE:
        return FILTERED_DISTANCE

    # Level 2: parameter distance
    matched, unmatched_a, unmatched_b = match_transforms(
        genome_a.transforms, genome_b.transforms)

    # Matched transform distance
    if matched:
        matched_dist = sum(
            _transform_distance(genome_a.transforms[i], genome_b.transforms[j])
            for i, j in matched
        ) / len(matched)
    else:
        matched_dist = 0.0

    # Unmatched penalty
    n_unmatched = len(unmatched_a) + len(unmatched_b)
    unmatched_dist = n_unmatched * UNMATCHED_PENALTY

    # Global parameter distance
    global_dist = _global_distance(genome_a, genome_b)

    # Final xform distance
    fa = genome_a.final_xform
    fb = genome_b.final_xform
    if fa is not None and fb is not None:
        final_dist = _transform_distance(fa, fb)
    elif fa is not None or fb is not None:
        final_dist = 0.5  # structural mismatch penalty
    else:
        final_dist = 0.0

    return float(matched_dist + unmatched_dist + global_dist + final_dist)
