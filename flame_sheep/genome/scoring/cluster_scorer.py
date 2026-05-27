"""Cluster-based genome scorer — segments by color index, scores per cluster.

The chaos game assigns each pixel a color index c (0-1) that maps into the
palette. Since the palette is a smooth gradient, similar c = similar colors.
Each transform naturally produces clusters in c-space.

This scorer:
1. Computes avg c per pixel from hit_grid/color_grid
2. Clusters into segments (histogram peak finding)
3. Runs metrics per cluster (coverage, edges, symmetry)
4. Aggregates into genome-level scores

The key insight: humans perceive symmetry and structure per colored region,
not globally. A symmetric blue spiral with random red noise reads as symmetric.
"""

from __future__ import annotations

import json
import logging

import numpy as np
from scipy.ndimage import label
from scipy.signal import find_peaks

log = logging.getLogger(__name__)


def _find_c_clusters(avg_c: np.ndarray, hit_mask: np.ndarray,
                     n_bins: int = 50, min_fraction: float = 0.05) -> list[dict]:
    """Find clusters in the color index distribution.

    Returns list of dicts: {center, lo, hi, mask} for each cluster.
    """
    c_values = avg_c[hit_mask]
    if len(c_values) < 100:
        return []

    # Histogram of c values
    hist, bin_edges = np.histogram(c_values, bins=n_bins, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # Smooth histogram for peak finding
    kernel = np.array([0.25, 0.5, 0.25])
    smoothed = np.convolve(hist.astype(float), kernel, mode='same')

    # Find peaks
    peaks, properties = find_peaks(smoothed, height=len(c_values) * min_fraction,
                                    distance=3)

    if len(peaks) == 0:
        # No clear peaks — treat entire range as one cluster
        return [{'center': float(c_values.mean()),
                 'lo': 0.0, 'hi': 1.0,
                 'mask': hit_mask}]

    # Assign each peak a range (midpoints between adjacent peaks)
    clusters = []
    for i, peak_idx in enumerate(peaks):
        center = float(bin_centers[peak_idx])

        # Boundaries: midpoint to neighboring peaks or 0/1
        if i == 0:
            lo = 0.0
        else:
            lo = (bin_centers[peaks[i - 1]] + center) / 2

        if i == len(peaks) - 1:
            hi = 1.0
        else:
            hi = (center + bin_centers[peaks[i + 1]]) / 2

        # Spatial mask: pixels in this c range
        cluster_mask = hit_mask & (avg_c >= lo) & (avg_c < hi)

        if cluster_mask.sum() < 50:
            continue

        clusters.append({
            'center': center,
            'lo': float(lo),
            'hi': float(hi),
            'mask': cluster_mask,
        })

    return clusters


def _sobel_magnitude(channel: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude on a 2D array."""
    if channel.shape[0] < 3 or channel.shape[1] < 3:
        return np.zeros_like(channel)
    gy = channel[2:, 1:-1] - channel[:-2, 1:-1]
    gx = channel[1:-1, 2:] - channel[1:-1, :-2]
    result = np.zeros_like(channel)
    result[1:-1, 1:-1] = np.sqrt(gx**2 + gy**2)
    return result


def _cluster_symmetry(mask: np.ndarray) -> dict[str, float]:
    """Rotational and reflective symmetry of a binary mask."""
    from scipy.ndimage import rotate

    h, w = mask.shape
    float_mask = mask.astype(np.float64)

    # Center on centroid
    ys, xs = np.where(mask)
    if len(ys) < 10:
        return {'rotational': 0.0, 'reflective': 0.0}
    cy, cx = ys.mean(), xs.mean()

    # Crop to bounding box with padding
    pad = 5
    y0 = max(0, int(cy - max(cy - ys.min(), ys.max() - cy) - pad))
    y1 = min(h, int(cy + max(cy - ys.min(), ys.max() - cy) + pad))
    x0 = max(0, int(cx - max(cx - xs.min(), xs.max() - cx) - pad))
    x1 = min(w, int(cx + max(cx - xs.min(), xs.max() - cx) + pad))
    cropped = float_mask[y0:y1, x0:x1]

    if cropped.size < 100:
        return {'rotational': 0.0, 'reflective': 0.0}

    norm = np.sqrt(np.sum(cropped**2))
    if norm < 1e-10:
        return {'rotational': 0.0, 'reflective': 0.0}

    # Rotational symmetry: best NCC for n-fold rotation (n=2,3,4,6)
    best_rot = 0.0
    for n in [2, 3, 4, 6]:
        angle = 360.0 / n
        rotated = rotate(cropped, angle, reshape=False, order=1)
        ncc = np.sum(cropped * rotated) / (norm * np.sqrt(np.sum(rotated**2)) + 1e-10)
        best_rot = max(best_rot, float(ncc))

    # Reflective symmetry: horizontal and vertical flip
    best_ref = 0.0
    for flipped in [cropped[::-1, :], cropped[:, ::-1]]:
        ncc = np.sum(cropped * flipped) / (norm * np.sqrt(np.sum(flipped**2)) + 1e-10)
        best_ref = max(best_ref, float(ncc))

    return {'rotational': best_rot, 'reflective': best_ref}


def _cluster_edges(avg_c: np.ndarray, mask: np.ndarray) -> float:
    """Edge sharpness of a cluster's color boundaries."""
    # Edge detection on avg_c within this cluster's spatial region
    c_masked = avg_c * mask.astype(np.float32)
    grad = _sobel_magnitude(c_masked)
    inner_mask = mask[1:-1, 1:-1]
    inner_grad = grad[1:-1, 1:-1]
    vals = inner_grad[inner_mask]
    if len(vals) < 10:
        return 0.0
    return float(vals.mean())


def score_from_clusters(hit_grid: np.ndarray, color_grid: np.ndarray,
                        store_detail: bool = False) -> dict[str, float]:
    """Score a genome by clustering its color index and analyzing per-cluster.

    Args:
        hit_grid: (H, W) float64 hit counts from chaos game
        color_grid: (H, W) float64 accumulated color indices
        store_detail: if True, include per-cluster detail in output

    Returns:
        Dict of aggregated scores + optionally 'cluster_detail' JSON string.
    """
    h, w = hit_grid.shape
    n_pixels = h * w
    hit_mask = hit_grid > 0

    # Average color index per pixel
    avg_c = np.zeros_like(hit_grid, dtype=np.float32)
    avg_c[hit_mask] = (color_grid[hit_mask] / hit_grid[hit_mask]).astype(np.float32)

    # Total coverage
    total_coverage = float(hit_mask.sum()) / n_pixels

    if total_coverage < 0.005:
        result = dict(
            cl_coverage=0.0, cl_edge_sharpness=0.0,
            cl_symmetry_best=0.0, cl_cluster_count=0,
            cl_dominance=0.0, cl_balance=0.0,
        )
        if store_detail:
            result['cluster_detail'] = '[]'
        return result

    # Find clusters
    clusters = _find_c_clusters(avg_c, hit_mask)
    n_clusters = len(clusters)

    if n_clusters == 0:
        result = dict(
            cl_coverage=total_coverage, cl_edge_sharpness=0.0,
            cl_symmetry_best=0.0, cl_cluster_count=0,
            cl_dominance=1.0, cl_balance=0.0,
        )
        if store_detail:
            result['cluster_detail'] = '[]'
        return result

    # Per-cluster metrics
    cluster_details = []
    symmetry_scores = []
    edge_scores = []
    cluster_sizes = []

    for cl in clusters:
        mask = cl['mask']
        cl_coverage = float(mask.sum()) / n_pixels
        cluster_sizes.append(cl_coverage)

        # Symmetry
        sym = _cluster_symmetry(mask)
        sym_best = max(sym['rotational'], sym['reflective'])
        symmetry_scores.append(sym_best)

        # Edge sharpness in c-space
        edges = _cluster_edges(avg_c, mask)
        edge_scores.append(edges)

        cluster_details.append({
            'c_center': cl['center'],
            'c_range': [cl['lo'], cl['hi']],
            'coverage': cl_coverage,
            'symmetry_rot': sym['rotational'],
            'symmetry_ref': sym['reflective'],
            'symmetry_best': sym_best,
            'edge_sharpness': edges,
        })

    # Aggregate
    cluster_sizes = np.array(cluster_sizes)
    size_weights = cluster_sizes / cluster_sizes.sum() if cluster_sizes.sum() > 0 else np.ones(len(cluster_sizes)) / len(cluster_sizes)

    # Best symmetry across clusters
    best_symmetry = max(symmetry_scores) if symmetry_scores else 0.0

    # Size-weighted mean edge sharpness
    weighted_edges = float(np.sum(np.array(edge_scores) * size_weights))
    # Normalize
    weighted_edges = min(weighted_edges / 0.05, 1.0)

    # Dominance: largest cluster / total coverage
    dominance = float(max(cluster_sizes) / total_coverage) if total_coverage > 0 else 0.0

    # Balance: luminance-weighted centroid distance from center
    ys, xs = np.mgrid[0:h, 0:w]
    log_density = np.log1p(hit_grid)
    total_weight = log_density.sum()
    if total_weight > 0:
        cx = float(np.sum(xs * log_density) / total_weight)
        cy = float(np.sum(ys * log_density) / total_weight)
        center_x, center_y = w / 2.0, h / 2.0
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
        balance = float(1.0 - dist / max_dist)
    else:
        balance = 0.0

    result = dict(
        cl_coverage=total_coverage,
        cl_edge_sharpness=weighted_edges,
        cl_symmetry_best=best_symmetry,
        cl_cluster_count=n_clusters,
        cl_dominance=dominance,
        cl_balance=balance,
    )

    if store_detail:
        result['cluster_detail'] = json.dumps(cluster_details)

    return result


def score_from_transform_hits(hit_grid: np.ndarray,
                              transform_hits: np.ndarray) -> dict[str, float]:
    """Score a genome by analyzing per-transform spatial regions.

    Each pixel knows how many times each transform hit it. We cluster by
    dominant transform — which transform contributed the most hits — and
    score each cluster's spatial properties.

    Args:
        hit_grid: (H, W) float64 hit counts.
        transform_hits: (H, W, N) uint32 per-transform hit counts.

    Returns:
        Dict of tf_* scores.
    """
    h, w = hit_grid.shape
    n_pixels = h * w
    n_transforms = transform_hits.shape[2]
    hit_mask = hit_grid > 0

    total_coverage = float(hit_mask.sum()) / n_pixels

    if total_coverage < 0.005 or n_transforms < 2:
        return dict(
            tf_coverage=total_coverage,
            tf_n_clusters=0,
            tf_avg_purity=0.0,
            tf_symmetry_best=0.0,
            tf_balance=0.0,
            tf_separation=0.0,
        )

    # Dominant transform per pixel
    dominant = np.argmax(transform_hits, axis=2)

    # Purity: fraction of hits from dominant transform
    total_per_pixel = transform_hits.sum(axis=2).astype(np.float64)
    dominant_counts = np.take_along_axis(
        transform_hits, dominant[:, :, np.newaxis], axis=2
    ).squeeze(axis=2).astype(np.float64)
    purity = np.zeros((h, w), dtype=np.float64)
    purity[hit_mask] = dominant_counts[hit_mask] / total_per_pixel[hit_mask]

    # Which transforms actually own pixels?
    active_transforms = []
    cluster_masks = {}
    for t in range(n_transforms):
        t_mask = (dominant == t) & hit_mask
        if t_mask.sum() >= 50:  # minimum cluster size
            active_transforms.append(t)
            cluster_masks[t] = t_mask

    n_clusters = len(active_transforms)

    if n_clusters == 0:
        return dict(
            tf_coverage=total_coverage,
            tf_n_clusters=0,
            tf_avg_purity=0.0,
            tf_symmetry_best=0.0,
            tf_balance=0.0,
            tf_separation=0.0,
        )

    # Per-cluster metrics
    cluster_sizes = []
    symmetry_scores = []
    cluster_purities = []

    for t in active_transforms:
        mask = cluster_masks[t]
        cl_coverage = float(mask.sum()) / n_pixels
        cluster_sizes.append(cl_coverage)

        # Purity of this cluster
        cl_purity = float(purity[mask].mean())
        cluster_purities.append(cl_purity)

        # Symmetry of this cluster's spatial shape
        sym = _cluster_symmetry(mask)
        sym_best = max(sym['rotational'], sym['reflective'])
        symmetry_scores.append(sym_best)

    # Aggregates
    cluster_sizes = np.array(cluster_sizes)
    size_weights = cluster_sizes / cluster_sizes.sum()

    # Size-weighted average purity
    avg_purity = float(np.sum(np.array(cluster_purities) * size_weights))

    # Best symmetry across clusters
    best_symmetry = max(symmetry_scores) if symmetry_scores else 0.0

    # Balance: how evenly distributed are cluster sizes?
    # Entropy of size distribution, normalized to [0, 1]
    if n_clusters > 1:
        eps = 1e-10
        size_entropy = -np.sum(size_weights * np.log2(size_weights + eps))
        max_entropy = np.log2(n_clusters)
        balance = float(size_entropy / max_entropy) if max_entropy > 0 else 0.0
    else:
        balance = 0.0

    # Separation: how spatially distinct are the clusters?
    # Compute fraction of boundary pixels (pixels where a neighbor has a
    # different dominant transform) — fewer boundaries = cleaner separation.
    padded = np.full((h + 2, w + 2), -1, dtype=dominant.dtype)
    padded[1:-1, 1:-1] = dominant
    padded[1:-1, 1:-1][~hit_mask] = -1
    boundary = np.zeros((h, w), dtype=bool)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        neighbor = padded[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx]
        boundary |= hit_mask & (neighbor != dominant) & (neighbor >= 0)
    boundary_frac = float(boundary.sum()) / max(hit_mask.sum(), 1)
    # Invert: fewer boundaries = higher separation score
    separation = float(1.0 - min(boundary_frac * 5.0, 1.0))

    return dict(
        tf_coverage=total_coverage,
        tf_n_clusters=n_clusters,
        tf_avg_purity=avg_purity,
        tf_symmetry_best=best_symmetry,
        tf_balance=balance,
        tf_separation=separation,
    )
