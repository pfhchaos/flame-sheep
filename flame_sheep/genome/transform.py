"""Transform — one IFS function within a flame fractal genome.

A flame fractal is a weighted blend of N transforms; each transform is
an affine map composed with a sum of nonlinear "variations" (sinusoidal,
spherical, julian, etc.) and an optional post-affine.

The Genome class (in genome/genome.py) holds a list of these plus the
global parameters (palette, zoom, rotation, center).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..variations import (
    NUM_VARIATIONS,
    random_var_params,
)


@dataclass
class Transform:
    """One IFS function: affine transform + variation blend + color.

    Full pipeline per transform:
        pre_variations → affine → variations → post_affine

    Pre-variations and post-affine are optional (None = identity/skip).
    """
    # Affine coefficients: x' = a*x + b*y + c, y' = d*x + e*y + f
    affine: np.ndarray = field(default_factory=lambda: np.array([1,0,0,0,1,0], dtype=np.float32))
    # Post-affine: applied after variations (None = identity)
    post_affine: np.ndarray | None = None
    # Variation weights — how much of each variation to blend
    variations: np.ndarray = field(default_factory=lambda: np.zeros(NUM_VARIATIONS, dtype=np.float32))
    # Pre-affine variation weights (None = no pre-variations)
    pre_variations: np.ndarray | None = None
    # Color index blended during chaos game
    color: float = 0.0
    # Color blend speed: color = speed * xform.color + (1-speed) * prev_color
    color_speed: float = 0.5
    # Probability weight for this transform being chosen
    weight: float = 1.0
    # Per-variation parameters (e.g. julian_power, splits_x)
    var_params: dict[str, float] = field(default_factory=dict)

    @classmethod
    def random(cls, rng: np.random.Generator) -> 'Transform':
        t = cls()
        # Random affine — keep it contractive (det < 1) to ensure attractor exists
        while True:
            a, b, d, e = rng.uniform(-1, 1, 4)
            M = np.array([[a, b], [d, e]])
            # contractivity: all singular values must be < 1
            # determinant < 1 is necessary but not sufficient (shear can still stretch)
            if np.max(np.linalg.svd(M, compute_uv=False)) < 0.9:
                break
        c, f = rng.uniform(-1, 1, 2)
        t.affine = np.array([a, b, c, d, e, f], dtype=np.float32)

        # Pick 1-2 variations with random weights
        n_vars = rng.integers(1, 3)
        chosen = rng.choice(NUM_VARIATIONS, n_vars, replace=False)
        weights = rng.uniform(0.3, 1.0, n_vars)
        weights /= weights.sum()
        t.variations[chosen] = weights

        # Initialize params for parametric variations
        for v in chosen:
            params = random_var_params(int(v), rng)
            t.var_params.update(params)

        t.color = float(rng.uniform(0, 1))
        t.color_speed = float(rng.uniform(0.0, 1.0))
        t.weight = float(rng.uniform(0.5, 2.0))

        # ~25% chance of post_affine (near-identity, contractive)
        if rng.random() < 0.25:
            while True:
                pa = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
                pa += rng.uniform(-0.3, 0.3, 6).astype(np.float32)
                M = np.array([[pa[0], pa[1]], [pa[3], pa[4]]])
                if np.max(np.linalg.svd(M, compute_uv=False)) < 0.9:
                    break
            t.post_affine = pa

        return t
