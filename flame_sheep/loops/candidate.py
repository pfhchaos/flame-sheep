"""LoopCandidate — pre-commit loop representation, plus canonical-form
helpers used by composition and pruning to deduplicate rotations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
