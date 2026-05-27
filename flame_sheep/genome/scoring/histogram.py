"""Histogram-based aesthetic metrics.

Shared by both CPU (64×64 coarse grid) and GPU (full-res) scoring
paths. Operates on a (h, w) hit-count grid plus a same-shape color
accumulator.

Returns a metrics dict consumed by the persistence layer (column names
in genomes table mirror these keys) and by the genome aesthetic_score().
"""

from __future__ import annotations

import numpy as np


def _score_from_histogram(hit_grid: np.ndarray, color_grid: np.ndarray) -> dict[str, float]:
    """
    Compute aesthetic metrics from a hit-count histogram and color accumulator.

    Shared by both CPU (64x64 coarse grid) and GPU (full-res) scorers.

    Returns dict with keys:
      coverage      -- fraction of pixels that got hit (0=collapsed, 1=fills frame)
      entropy       -- normalized Shannon entropy of hit distribution (0=single point, 1=uniform)
      color_entropy -- palette utilization (0=monochrome, 1=full range)
      balance       -- how centered the attractor is (0=corner, 1=dead center)
      complexity    -- multi-scale density variation (0=flat, 1=rich structure)
    """
    h, w = hit_grid.shape
    total_hits = hit_grid.sum()

    if total_hits == 0:
        return dict(coverage=0.0, entropy=0.0, color_entropy=0.0,
                    balance=0.0, complexity=0.0)

    # -- coverage: fraction of cells with any hits
    coverage = float(np.count_nonzero(hit_grid)) / (h * w)

    # -- entropy: Shannon entropy of hit distribution, normalized to [0, 1]
    p = hit_grid.ravel() / total_hits
    p = p[p > 0]
    max_entropy = np.log(h * w)
    entropy = float(-np.sum(p * np.log(p)) / max_entropy) if max_entropy > 0 else 0.0

    # -- color_entropy: how many palette colors are used
    #    Quantize color indices to 32 bins and compute entropy over that
    n_color_bins = 32
    hit_mask = hit_grid > 0
    if hit_mask.any():
        avg_colors = color_grid[hit_mask]
        avg_colors = np.clip(avg_colors, 0.0, 1.0)
        color_bins = np.floor(avg_colors * (n_color_bins - 1)).astype(int)
        color_hist = np.bincount(color_bins, minlength=n_color_bins).astype(np.float64)
        color_total = color_hist.sum()
        if color_total > 0:
            cp = color_hist / color_total
            cp = cp[cp > 0]
            color_entropy = float(-np.sum(cp * np.log(cp)) / np.log(n_color_bins))
        else:
            color_entropy = 0.0
    else:
        color_entropy = 0.0

    # -- balance: 1 - normalized distance of weighted centroid from center
    ys, xs = np.mgrid[0:h, 0:w]
    cx = float(np.sum(xs * hit_grid) / total_hits)
    cy = float(np.sum(ys * hit_grid) / total_hits)
    center_x, center_y = w / 2.0, h / 2.0
    max_dist = np.sqrt(center_x**2 + center_y**2)
    dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
    balance = float(1.0 - dist / max_dist)

    # -- complexity: how much density variation exists within the attractor
    #    Compute coefficient of variation of log-density over *hit* cells only,
    #    then repeat at coarser scales and average.
    log_hits = np.log1p(hit_grid)
    scales = []
    current = log_hits
    for _ in range(3):
        if current.shape[0] < 4 or current.shape[1] < 4:
            break
        nonzero = current[current > 0]
        if len(nonzero) > 1:
            scales.append(float(nonzero.std() / nonzero.mean()))
        # 2x downsample by averaging 2x2 blocks
        ch = (current.shape[0] // 2) * 2
        cw = (current.shape[1] // 2) * 2
        current = current[:ch, :cw].reshape(current.shape[0] // 2, 2,
                                            current.shape[1] // 2, 2).mean(axis=(1, 3))
    complexity = float(np.mean(scales)) if scales else 0.0
    # Normalize — CoV of ~1.5 in hit cells is high complexity
    complexity = min(complexity / 1.5, 1.0)

    # Centroid offset from grid center, normalized to [-1, 1]
    centroid_offset_x = float((cx - center_x) / center_x) if center_x > 0 else 0.0
    centroid_offset_y = float((cy - center_y) / center_y) if center_y > 0 else 0.0

    # -- edge_sharpness: how crisp the filament structures are
    #    Gradient magnitude of log-density, averaged over hit pixels.
    #    Sharp filaments = high gradient, blobs = low gradient.
    if log_hits.shape[0] >= 3 and log_hits.shape[1] >= 3:
        # Sobel-like gradient via finite differences
        gy = log_hits[2:, 1:-1] - log_hits[:-2, 1:-1]
        gx = log_hits[1:-1, 2:] - log_hits[1:-1, :-2]
        grad_mag = np.sqrt(gx**2 + gy**2)
        # Average over non-zero gradient pixels
        grad_nonzero = grad_mag[grad_mag > 0]
        if len(grad_nonzero) > 0:
            edge_sharpness = float(grad_nonzero.mean())
            # Normalize — empirically, mean gradient ~1.5 is sharp
            edge_sharpness = min(edge_sharpness / 1.5, 1.0)
        else:
            edge_sharpness = 0.0
    else:
        edge_sharpness = 0.0

    # -- contour_coherence: do edges form continuous structures or noise?
    #    Threshold the gradient to find edge pixels, then count connected
    #    components. Few large components = structured, many small = noisy.
    if log_hits.shape[0] >= 3 and log_hits.shape[1] >= 3 and edge_sharpness > 0:
        from scipy.ndimage import label
        edge_mask = grad_mag > (grad_nonzero.mean() * 0.5 if len(grad_nonzero) > 0 else 0)
        labels, n_components = label(edge_mask)
        if n_components > 0:
            component_sizes = np.bincount(labels.ravel())[1:]  # skip background
            # Ratio of largest component to total edge pixels
            largest = component_sizes.max()
            total_edge = component_sizes.sum()
            # More coherent = largest component is bigger fraction
            contour_coherence = float(largest / total_edge)
        else:
            contour_coherence = 0.0
    else:
        contour_coherence = 0.0

    return dict(
        coverage=coverage,
        entropy=entropy,
        color_entropy=color_entropy,
        balance=balance,
        complexity=complexity,
        edge_sharpness=edge_sharpness,
        contour_coherence=contour_coherence,
        centroid_offset_x=centroid_offset_x,
        centroid_offset_y=centroid_offset_y,
    )
