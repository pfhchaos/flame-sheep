"""Symmetry metrics from a hit-count histogram.

Separate from histogram.py because symmetry detection is tier 2
(moderate cost) — not run during fast genome generation, only during
GPU scoring or idle background passes. Wraps the actual symmetry
implementation in genome.symmetry (the math, separately).
"""

from __future__ import annotations

import numpy as np


def _score_symmetry(hit_grid: np.ndarray) -> dict[str, float]:
    """Compute symmetry metrics from a hit-count histogram."""
    from ..symmetry import symmetry_scores
    sym = symmetry_scores(hit_grid)
    return dict(
        symmetry_max=sym['symmetry_max'],
        rotational=sym['rotational_best'],
        reflective=sym['reflective_best'],
        radial=sym['radial'],
        periodic=sym['periodic'],
        fractal_dim=sym['fractal_dim'],
        self_similarity=sym['self_similarity'],
    )
