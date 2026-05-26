"""Transition quality scoring between genome pairs.

Submodules:
  - signature.py: variation_signature, signature_distance — cheap structural
    fingerprint and bag-distance pre-filter
  - distance.py: compute_transition_distance — full permutation-matched
    parameter distance with the signature pre-filter
  - worker.py: BackgroundTransitionScorer — load-aware daemon that fills
    the genome_transitions cache table

This __init__.py re-exports the public API per the convention used by
storage/, runtime/, rendering/, etc. External code should import from
`flame_sheep.transitions`, not from submodules directly — submodule
layout is an implementation detail of Stage 4a of docs/reorg_plan.md.
"""

from .signature import (
    variation_signature,
    signature_distance,
    ACTIVE_THRESHOLD,
    MAX_SIGNATURE_DISTANCE,
    _active_variations,
)
from .distance import (
    match_transforms,
    compute_transition_distance,
    UNMATCHED_PENALTY,
    FILTERED_DISTANCE,
    CACHE_THRESHOLD,
)
from .worker import BackgroundTransitionScorer

__all__ = [
    # signature
    'variation_signature',
    'signature_distance',
    'ACTIVE_THRESHOLD',
    'MAX_SIGNATURE_DISTANCE',
    # distance
    'match_transforms',
    'compute_transition_distance',
    'UNMATCHED_PENALTY',
    'FILTERED_DISTANCE',
    'CACHE_THRESHOLD',
    # worker
    'BackgroundTransitionScorer',
]
