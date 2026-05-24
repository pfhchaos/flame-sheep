"""Motion-field geometry — temporary home (Stage 2).

These functions are geometric operations on genome pairs, not
persistence. Stage 4 of docs/reorg_plan.md renames this module to
`genome/motion_field.py`. Living at top level for now keeps the
storage/ package pure persistence and avoids creating an
import-time cycle with loops/.
"""

from __future__ import annotations

import numpy as np

from .genome import Genome

# Motion field resolution: 3x3 grid of 2D vectors
MOTION_GRID = 3


def compute_motion_field(genome_a: Genome, genome_b: Genome,
                         n_test: int = 5000, fuse: int = 20,
                         bound: float = 4.0) -> np.ndarray:
    """
    Compute a 3x3 grid of 2D motion vectors characterizing the visual
    transition from genome_a to genome_b.

    For each cell in the 3x3 grid, runs a chaos game for both genomes,
    computes the density centroid of points landing in that cell, and
    returns the displacement vector (centroid_b - centroid_a).

    Returns shape (3, 3, 2) float32.
    """
    grid = MOTION_GRID
    centroids_a = _grid_centroids(genome_a, grid, n_test, fuse, bound)
    centroids_b = _grid_centroids(genome_b, grid, n_test, fuse, bound)
    return (centroids_b - centroids_a).astype(np.float32)


def _grid_centroids(genome: Genome, grid: int, n_test: int,
                    fuse: int, bound: float) -> np.ndarray:
    """
    Run chaos game and compute density-weighted centroid for each cell
    in a grid×grid partition of the viewport.

    Returns shape (grid, grid, 2) — centroid (x, y) per cell.
    """
    from .genome import _apply_variation_cpu

    rng = np.random.default_rng()
    weights = np.array([tr.weight for tr in genome.transforms], dtype=np.float64)
    weights /= weights.sum()
    cumw = np.cumsum(weights)

    # Accumulators: sum of positions and count per cell
    cell_sum = np.zeros((grid, grid, 2), dtype=np.float64)
    cell_count = np.zeros((grid, grid), dtype=np.float64)

    x, y = 0.0, 0.0
    cell_size = (2 * bound) / grid

    for i in range(fuse + n_test):
        r = rng.random()
        tidx = min(int(np.searchsorted(cumw, r)), len(genome.transforms) - 1)
        tr = genome.transforms[tidx]
        a, b, c, d, e, f = tr.affine
        nx = a * x + b * y + c
        ny = d * x + e * y + f

        best_var = int(np.argmax(tr.variations))
        w = float(tr.variations[best_var])
        if w > 0.0:
            nx, ny = _apply_variation_cpu(best_var, nx, ny, w, tr.affine)

        x, y = nx, ny

        if not (np.isfinite(x) and np.isfinite(y)):
            x, y = 0.0, 0.0
            continue

        if i >= fuse and abs(x) < bound and abs(y) < bound:
            gx = min(int((x + bound) / cell_size), grid - 1)
            gy = min(int((y + bound) / cell_size), grid - 1)
            cell_sum[gy, gx, 0] += x
            cell_sum[gy, gx, 1] += y
            cell_count[gy, gx] += 1

    # Compute centroids; cells with no hits get center of cell
    centroids = np.zeros((grid, grid, 2), dtype=np.float64)
    for gy in range(grid):
        for gx in range(grid):
            if cell_count[gy, gx] > 0:
                centroids[gy, gx] = cell_sum[gy, gx] / cell_count[gy, gx]
            else:
                centroids[gy, gx, 0] = -bound + (gx + 0.5) * cell_size
                centroids[gy, gx, 1] = -bound + (gy + 0.5) * cell_size
    return centroids


def motion_field_coherence(field_a: np.ndarray, field_b: np.ndarray) -> float:
    """
    How well two motion fields flow in the same direction.

    Returns a value in [-1, 1]:
      +1 = identical direction everywhere
       0 = unrelated
      -1 = exactly opposite

    Uses cosine similarity averaged over the 3x3 grid, weighted by
    magnitude (cells with more motion matter more).
    """
    a = field_a.reshape(-1, 2)
    b = field_b.reshape(-1, 2)
    mag_a = np.linalg.norm(a, axis=1)
    mag_b = np.linalg.norm(b, axis=1)
    weights = mag_a * mag_b
    total_weight = weights.sum()
    if total_weight < 1e-10:
        return 0.0
    dots = np.sum(a * b, axis=1)
    return float(np.sum(dots) / (total_weight + 1e-10) * (weights.sum() / (weights.sum() + 1e-10)))


def motion_field_to_blob(field: np.ndarray) -> bytes:
    return field.astype(np.float32).tobytes()


def motion_field_from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).reshape(MOTION_GRID, MOTION_GRID, 2)
