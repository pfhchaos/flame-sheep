"""Variation signature — cheap structural fingerprint of a genome.

The signature is a sorted bag of active variation indices across all
transforms (with an "+F" suffix for the final xform's variations).
Signature distance is the bag manhattan distance — a fast pre-filter
before paying for the full transform-match parameter distance.

Used by transitions.distance.compute_transition_distance and elsewhere
as a coarse genome similarity proxy.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..genome import Genome, Transform

# Threshold for considering a variation "active" in a transform
ACTIVE_THRESHOLD = 0.01

# Maximum structural (signature) distance before skipping full comparison
MAX_SIGNATURE_DISTANCE = 6


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
