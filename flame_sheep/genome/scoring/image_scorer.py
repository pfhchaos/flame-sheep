"""Image-based genome scorer — operates on rendered RGBA images in HSL space.

Scores what the human actually sees, not the raw histogram.
L channel for structure (edges, coverage), H channel for color organization.

Usage:
    from .image_scorer import score_from_image
    scores = score_from_image(rgba_image)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label


def rgb_to_hsl(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert RGB uint8 image to H (0-360), S (0-1), L (0-1) float32 arrays.

    Vectorized numpy implementation. Input shape: (H, W, 3) or (H, W, 4).
    """
    rgb = img[:, :, :3].astype(np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Lightness
    L = (cmax + cmin) / 2.0

    # Saturation
    S = np.zeros_like(L)
    mask = delta > 0
    low = mask & (L <= 0.5)
    high = mask & (L > 0.5)
    S[low] = delta[low] / (cmax[low] + cmin[low] + 1e-10)
    S[high] = delta[high] / (2.0 - cmax[high] - cmin[high] + 1e-10)

    # Hue
    H = np.zeros_like(L)
    r_max = mask & (cmax == r)
    g_max = mask & (cmax == g)
    b_max = mask & (cmax == b)
    H[r_max] = 60.0 * (((g[r_max] - b[r_max]) / (delta[r_max] + 1e-10)) % 6)
    H[g_max] = 60.0 * (((b[g_max] - r[g_max]) / (delta[g_max] + 1e-10)) + 2)
    H[b_max] = 60.0 * (((r[b_max] - g[b_max]) / (delta[b_max] + 1e-10)) + 4)
    H = H % 360

    return H, S, L


def _sobel_magnitude(channel: np.ndarray) -> np.ndarray:
    """Compute Sobel gradient magnitude on a 2D float array."""
    if channel.shape[0] < 3 or channel.shape[1] < 3:
        return np.zeros_like(channel)
    gy = channel[2:, 1:-1] - channel[:-2, 1:-1]
    gx = channel[1:-1, 2:] - channel[1:-1, :-2]
    mag = np.sqrt(gx**2 + gy**2)
    # Pad back to original size
    result = np.zeros_like(channel)
    result[1:-1, 1:-1] = mag
    return result


def score_from_image(img: np.ndarray) -> dict[str, float]:
    """Score a rendered RGBA genome image in perceptual space.

    Args:
        img: uint8 array, shape (H, W, 4) RGBA

    Returns:
        Dict of metric name → float score (0-1 normalized where possible).
    """
    from PIL import Image

    # Downscale to 512×512 for speed
    if img.shape[0] > 512 or img.shape[1] > 512:
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((512, 512), Image.LANCZOS)
        img = np.array(pil_img)

    h, w = img.shape[:2]
    n_pixels = h * w

    # Convert to HSL
    H, S, L = rgb_to_hsl(img)

    # Lit pixel mask (non-black)
    lit = L > 0.02
    n_lit = lit.sum()

    if n_lit < n_pixels * 0.001:
        # Nearly empty image
        return dict(
            img_coverage=0.0, img_structural_edges=0.0,
            img_color_edges=0.0, img_color_regions=0.0,
            img_color_coherence=0.0, img_color_variety=0.0,
        )

    # === L Channel: Structure ===

    # 1. Coverage
    coverage = float(n_lit) / n_pixels

    # 2. Structural edge sharpness (Sobel on L, normalized by coverage)
    L_grad = _sobel_magnitude(L)
    # Gradient is zero-padded at edges; use inner lit mask
    inner_lit = lit[1:-1, 1:-1]
    inner_grad = L_grad[1:-1, 1:-1]
    lit_grad = inner_grad[inner_lit]
    if len(lit_grad) > 0:
        structural_edges = float(lit_grad.mean())
        # Normalize — empirically, mean gradient ~0.15 is sharp for flame fractals
        structural_edges = min(structural_edges / 0.15, 1.0)
    else:
        structural_edges = 0.0

    # === H Channel: Color Structure ===

    # Only analyze hue where pixels are lit and have some saturation
    color_mask = lit & (S > 0.1)
    n_color = color_mask.sum()

    if n_color < 100:
        return dict(
            img_coverage=coverage, img_structural_edges=structural_edges,
            img_color_edges=0.0, img_color_regions=0.0,
            img_color_coherence=0.0, img_color_variety=0.0,
        )

    # 3. Color edge sharpness (Sobel on H, circular-aware)
    # Hue is circular (0=360), so we use sin/cos decomposition
    H_rad = np.deg2rad(H)
    H_sin = np.sin(H_rad) * lit.astype(np.float32)
    H_cos = np.cos(H_rad) * lit.astype(np.float32)
    sin_grad = _sobel_magnitude(H_sin)
    cos_grad = _sobel_magnitude(H_cos)
    hue_grad = np.sqrt(sin_grad**2 + cos_grad**2)

    inner_color_mask = color_mask[1:-1, 1:-1]
    inner_hue_grad = hue_grad[1:-1, 1:-1]
    color_grad_vals = inner_hue_grad[inner_color_mask]
    if len(color_grad_vals) > 0:
        color_edges = float(color_grad_vals.mean())
        color_edges = min(color_edges / 0.3, 1.0)
    else:
        color_edges = 0.0

    # 4. Color region analysis
    # Quantize hue into 12 bins (30° each)
    hue_bins = (H / 30.0).astype(int) % 12

    total_components = 0
    coherence_scores = []
    active_bins = 0

    for bin_idx in range(12):
        bin_mask = color_mask & (hue_bins == bin_idx)
        bin_count = bin_mask.sum()
        if bin_count < 20:
            continue

        active_bins += 1
        labels, n_components = label(bin_mask)
        total_components += n_components

        if n_components > 0:
            sizes = np.bincount(labels.ravel())[1:]  # skip background
            largest = sizes.max()
            coherence_scores.append(float(largest) / bin_count)

    # 5. Color region count (normalized)
    if active_bins > 0:
        # More regions per active bin = more complex color structure
        regions_per_bin = total_components / active_bins
        color_regions = min(regions_per_bin / 5.0, 1.0)
    else:
        color_regions = 0.0

    # 6. Color coherence (mean of per-bin coherence)
    if coherence_scores:
        color_coherence = float(np.mean(coherence_scores))
    else:
        color_coherence = 0.0

    # 7. Color variety (entropy of hue histogram over lit pixels)
    hue_hist = np.bincount(hue_bins[color_mask].ravel(), minlength=12).astype(np.float64)
    hue_total = hue_hist.sum()
    if hue_total > 0:
        p = hue_hist / hue_total
        p = p[p > 0]
        color_variety = float(-np.sum(p * np.log(p)) / np.log(12))
    else:
        color_variety = 0.0

    # 8. Euclidean color edge — channel-agnostic edge sharpness
    #    sqrt(R_grad^2 + G_grad^2 + B_grad^2), color-agnostic structure metric
    rgb = img[:, :, :3].astype(np.float32) / 255.0
    r_grad = _sobel_magnitude(rgb[:, :, 0])[1:-1, 1:-1]
    g_grad = _sobel_magnitude(rgb[:, :, 1])[1:-1, 1:-1]
    b_grad = _sobel_magnitude(rgb[:, :, 2])[1:-1, 1:-1]
    euc_grad = np.sqrt(r_grad**2 + g_grad**2 + b_grad**2)
    euc_vals = euc_grad[inner_lit]
    if len(euc_vals) > 0:
        euclidean_edges = float(euc_vals.mean())
        euclidean_edges = min(euclidean_edges / 0.15, 1.0)
    else:
        euclidean_edges = 0.0

    return dict(
        img_coverage=coverage,
        img_structural_edges=structural_edges,
        img_euclidean_edges=euclidean_edges,
        img_color_edges=color_edges,
        img_color_regions=color_regions,
        img_color_coherence=color_coherence,
        img_color_variety=color_variety,
    )
