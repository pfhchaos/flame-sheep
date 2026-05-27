"""Palette generation primitives.

Currently a single random-palette generator: 3-6 control points uniformly
distributed in RGB[0,1]^3, linearly interpolated to 256 samples with
uniform-spaced knots. See `project_palette_design.md` memory for the
deferred upgrade direction (OKLCh + cubic + free knot positions).
"""

from __future__ import annotations

import numpy as np


def _lerp_arr(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (a * (1 - t) + b * t).astype(a.dtype)


def _random_palette(rng: np.random.Generator) -> np.ndarray:
    """Generate a smooth random color palette by interpolating random control points."""
    n_points = rng.integers(3, 7)
    control = rng.uniform(0, 1, (n_points, 3)).astype(np.float32)
    palette = np.zeros((256, 3), dtype=np.float32)
    for i in range(256):
        t = i / 255.0 * (n_points - 1)
        lo, hi = int(t), min(int(t) + 1, n_points - 1)
        palette[i] = _lerp_arr(control[lo], control[hi], t - lo)
    return palette
