"""
Symmetry detection on flame fractal histograms.

Computes multiple symmetry metrics from a 2D hit-count grid:
  - Rotational symmetry (n-fold, n=2..8)
  - Reflective symmetry (swept across angles)
  - Radial symmetry (concentric ring correlation)
  - Fractal dimension (box-counting)

The symmetry score is the max across all detected symmetry types.
These are descriptors, not direct fitness terms — high symmetry is
striking but deliberate asymmetry can also look good.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve


def symmetry_scores(hit_grid: np.ndarray) -> dict[str, float]:
    """Compute all symmetry metrics from a hit-count histogram.

    Args:
        hit_grid: 2D array of hit counts (any resolution).

    Returns dict with keys:
        rotational_best  -- best n-fold rotational score (0..1)
        rotational_n     -- which n scored best
        reflective_best  -- best reflective score across angles (0..1)
        reflective_angle -- angle of best reflection axis (degrees)
        radial           -- radial/mandala symmetry (0..1)
        fractal_dim      -- box-counting fractal dimension (~1.0-2.0)
        symmetry_max     -- max of rotational, reflective, radial
    """
    if hit_grid.sum() == 0:
        return dict(rotational_best=0.0, rotational_n=0,
                    reflective_best=0.0, reflective_angle=0.0,
                    radial=0.0, periodic=0.0, periodic_freq=0.0,
                    fractal_dim=0.0, self_similarity=0.0, symmetry_max=0.0)

    # Normalize to float, log-scale for perceptual uniformity
    grid = np.log1p(hit_grid.astype(np.float64))
    grid /= grid.max() + 1e-10

    # Coverage gate: sparse histograms produce meaningless symmetry scores
    h, w = hit_grid.shape
    coverage = float(np.count_nonzero(hit_grid)) / (h * w)

    # Center on attractor centroid for symmetry checks
    centered = _center_on_attractor(grid)

    rot_score, rot_n = _rotational_symmetry(centered)
    ref_score, ref_angle = _reflective_symmetry(centered)
    rad_score = _radial_symmetry(centered)
    per_score, per_freq = _periodic_structure(grid)  # translation-invariant
    self_sim = _self_similarity(grid)
    fdim = _fractal_dimension(hit_grid)

    # Attenuate symmetry for degenerate histograms — sparse or uniform
    # coverage < 5% = collapsed, coverage > 80% = uniform fill
    coverage_quality = min(1.0, coverage / 0.05) * min(1.0, (1.0 - coverage) / 0.2)

    return dict(
        rotational_best=rot_score,
        rotational_n=rot_n,
        reflective_best=ref_score,
        reflective_angle=ref_angle,
        radial=rad_score,
        periodic=per_score,
        periodic_freq=per_freq,
        fractal_dim=fdim,
        self_similarity=self_sim,
        symmetry_max=max(rot_score, ref_score, rad_score, per_score) * coverage_quality,
    )


def _rotational_symmetry(grid: np.ndarray) -> tuple[float, int]:
    """Test n-fold rotational symmetry for n=2..8.

    Rotates the grid by 360/n degrees and measures correlation
    with the original. Returns (best_score, best_n).
    """
    h, w = grid.shape
    best_score = 0.0
    best_n = 0

    for n in range(2, 9):
        angle = 360.0 / n
        rotated = ndimage.rotate(grid, angle, reshape=False, order=1, mode='constant')
        # Crop to valid region (rotation can introduce border artifacts)
        margin = int(max(h, w) * 0.1)
        inner = slice(margin, -margin or None)
        g = grid[inner, inner]
        r = rotated[inner, inner]

        if g.size == 0:
            continue

        # Normalized cross-correlation
        g_mean = g.mean()
        r_mean = r.mean()
        g_std = g.std()
        r_std = r.std()

        if g_std < 1e-10 or r_std < 1e-10:
            continue

        ncc = float(np.mean((g - g_mean) * (r - r_mean)) / (g_std * r_std))
        ncc = max(0.0, ncc)  # clamp negatives

        if ncc > best_score:
            best_score = ncc
            best_n = n

    return best_score, best_n


def _reflective_symmetry(grid: np.ndarray, n_angles: int = 18) -> tuple[float, float]:
    """Test reflective symmetry across multiple axes.

    Sweeps angles in 10-degree increments (0 = vertical axis,
    90 = horizontal axis). Returns (best_score, best_angle_degrees).
    """
    h, w = grid.shape
    best_score = 0.0
    best_angle = 0.0

    for i in range(n_angles):
        angle = i * 180.0 / n_angles

        # Rotate so the reflection axis is vertical, flip, rotate back
        if abs(angle) > 0.1:
            rotated = ndimage.rotate(grid, angle, reshape=False, order=1, mode='constant')
        else:
            rotated = grid

        flipped = rotated[:, ::-1]

        # Compare in the inner region
        margin = int(max(h, w) * 0.1)
        inner = slice(margin, -margin or None)
        g = rotated[inner, inner]
        f = flipped[inner, inner]

        if g.size == 0:
            continue

        g_std = g.std()
        f_std = f.std()
        if g_std < 1e-10 or f_std < 1e-10:
            continue

        ncc = float(np.mean((g - g.mean()) * (f - f.mean())) / (g_std * f_std))
        ncc = max(0.0, ncc)

        if ncc > best_score:
            best_score = ncc
            best_angle = angle

    return best_score, best_angle


def _radial_symmetry(grid: np.ndarray, n_rings: int = 8,
                     n_sectors: int = 12) -> float:
    """Measure radial/mandala symmetry.

    Combines two signals:
    1. Angular uniformity — each ring is evenly filled around the circle
    2. Radial profile structure — rings have different intensities (not flat)

    Random noise scores low because it has no radial profile structure.
    A single point scores low because most rings are empty.
    A bullseye or mandala scores high on both.
    """
    h, w = grid.shape
    cy, cx = h / 2.0, w / 2.0
    max_r = min(cy, cx)

    ys, xs = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    angle = np.arctan2(ys - cy, xs - cx)  # -pi to pi

    ring_means = []
    ring_uniformity = []
    for i in range(n_rings):
        r_lo = (i + 1) * max_r / (n_rings + 1)
        r_hi = (i + 2) * max_r / (n_rings + 1)
        ring_mask = (dist >= r_lo) & (dist < r_hi)

        if ring_mask.sum() < n_sectors:
            continue

        ring_total = grid[ring_mask].sum()
        if ring_total < 1e-10:
            ring_means.append(0.0)
            continue

        ring_means.append(float(grid[ring_mask].mean()))

        # Angular uniformity within this ring
        sector_sums = np.zeros(n_sectors)
        for s in range(n_sectors):
            a_lo = -np.pi + s * 2 * np.pi / n_sectors
            a_hi = a_lo + 2 * np.pi / n_sectors
            sector_mask = ring_mask & (angle >= a_lo) & (angle < a_hi)
            if sector_mask.any():
                sector_sums[s] = grid[sector_mask].sum()

        p = sector_sums / (sector_sums.sum() + 1e-10)
        entropy = -np.sum(np.where(p > 0, p * np.log(p + 1e-10), 0))
        max_entropy = np.log(n_sectors)
        ring_uniformity.append(entropy / max_entropy)

    if len(ring_uniformity) < 2:
        return 0.0

    # Angular uniformity: mean across rings
    uniformity = float(np.mean(ring_uniformity))

    # Radial profile structure: how much ring means vary
    # Random noise has flat profile (all rings similar) → low structure
    # Mandalas have peaks and troughs → high structure
    rm = np.array(ring_means)
    if rm.max() < 1e-10:
        return 0.0
    rm_norm = rm / rm.max()
    profile_var = float(rm_norm.std())
    # Normalize: std > 0.3 = strong radial profile
    profile_score = min(1.0, profile_var / 0.3)

    # Both needed: uniform sectors AND structured radial profile
    return uniformity * profile_score


def _periodic_structure(grid: np.ndarray) -> tuple[float, float]:
    """Detect periodic/tiling structure via spatial power spectrum.

    Computes the 2D FFT power spectrum and looks for peaks above
    the noise floor. Strong peaks = repeating spatial pattern
    (stripes, grids, lattices).

    Returns (score, dominant_frequency) where score is 0..1 and
    frequency is in cycles per image width.
    """
    h, w = grid.shape

    # 2D FFT power spectrum
    fft = np.fft.fft2(grid)
    power = np.abs(np.fft.fftshift(fft))**2

    # Zero out DC component and very low frequencies (< 2 cycles)
    cy, cx = h // 2, w // 2
    min_freq_px = 2
    ys, xs = np.mgrid[0:h, 0:w]
    dist_from_center = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    power[dist_from_center < min_freq_px] = 0

    # Also ignore very high frequencies (noise)
    max_freq_px = min(h, w) // 4
    power[dist_from_center > max_freq_px] = 0

    if power.max() < 1e-10:
        return 0.0, 0.0

    # Find the dominant peak
    peak_idx = np.unravel_index(np.argmax(power), power.shape)
    peak_val = power[peak_idx]

    # Score: fraction of total spectral energy in the top few peaks
    # Periodic patterns concentrate energy; noise spreads it uniformly
    sorted_power = np.sort(power.ravel())[::-1]
    total = sorted_power.sum()
    if total < 1e-10:
        return 0.0, 0.0
    # Energy in top 5 peaks vs total
    top_energy = sorted_power[:5].sum()
    concentration = top_energy / total

    # Normalize: >0.5 concentration is very periodic, <0.1 is noise
    score = min(1.0, max(0.0, (concentration - 0.1) / 0.4))

    # Dominant frequency in cycles per image width
    freq = float(dist_from_center[peak_idx])

    return score, freq


def _fractal_dimension(hit_grid: np.ndarray) -> float:
    """Estimate fractal dimension via box-counting.

    Counts how many boxes of size s contain at least one hit,
    for geometrically decreasing s. The slope of log(count) vs
    log(1/s) approximates the fractal dimension.

    Returns D in roughly [1.0, 2.0]. D~1.0 = simple curve,
    D~1.5-1.8 = rich fractal, D~2.0 = space-filling.
    """
    binary = (hit_grid > 0).astype(np.uint8)
    if binary.sum() == 0:
        return 0.0

    h, w = binary.shape
    max_dim = min(h, w)

    # Box sizes: powers of 2 from 2 up to max_dim/2
    sizes = []
    s = 2
    while s <= max_dim // 2:
        sizes.append(s)
        s *= 2

    if len(sizes) < 2:
        return 1.0

    counts = []
    for s in sizes:
        # Count non-empty boxes of size s
        # Reshape into s×s blocks and check if any pixel is hit
        nh = (h // s) * s
        nw = (w // s) * s
        trimmed = binary[:nh, :nw]
        blocks = trimmed.reshape(nh // s, s, nw // s, s)
        n_occupied = int((blocks.any(axis=(1, 3))).sum())
        if n_occupied > 0:
            counts.append(n_occupied)
        else:
            counts.append(1)

    # Linear regression of log(count) vs log(1/s)
    log_inv_s = np.log(1.0 / np.array(sizes[:len(counts)]))
    log_count = np.log(np.array(counts, dtype=np.float64))

    if len(log_inv_s) < 2:
        return 1.0

    # Least squares fit
    A = np.vstack([log_inv_s, np.ones(len(log_inv_s))]).T
    result = np.linalg.lstsq(A, log_count, rcond=None)
    slope = float(result[0][0])

    # Clamp to reasonable range
    return max(0.5, min(2.0, slope))


def _center_on_attractor(grid: np.ndarray) -> np.ndarray:
    """Shift grid so the attractor centroid is at the image center.

    Flame fractals can have their symmetry center anywhere.
    Rolling the grid to center the mass lets rotational/reflective/radial
    checks find symmetry that would be missed about the image center.
    """
    h, w = grid.shape
    total = grid.sum()
    if total < 1e-10:
        return grid
    ys, xs = np.mgrid[0:h, 0:w]
    cx = int(np.sum(xs * grid) / total)
    cy = int(np.sum(ys * grid) / total)
    shift_x = w // 2 - cx
    shift_y = h // 2 - cy
    return np.roll(np.roll(grid, shift_x, axis=1), shift_y, axis=0)


def _self_similarity(grid: np.ndarray) -> float:
    """Measure multi-scale self-similarity via template matching.

    Downsamples the grid by 2x and 4x, then finds the best-matching
    region in the original via normalized cross-correlation. High NCC
    means a small piece of the fractal looks like the whole — the
    defining property of IFS.

    Returns 0..1, higher = more self-similar.
    """
    h, w = grid.shape
    if h < 8 or w < 8:
        return 0.0

    scores = []
    for factor in [2, 4]:
        small = ndimage.zoom(grid, 1.0 / factor, order=1)
        sh, sw = small.shape
        if sh < 4 or sw < 4:
            continue

        # Normalize template
        s_mean = small.mean()
        s_std = small.std()
        if s_std < 1e-10:
            continue
        s_norm = (small - s_mean) / s_std

        # Sliding NCC via FFT convolution
        # For each position, compute correlation with normalized template
        g_local_sum = fftconvolve(grid, np.ones_like(small), mode='valid')
        g_local_sq = fftconvolve(grid**2, np.ones_like(small), mode='valid')
        n = small.size
        g_local_mean = g_local_sum / n
        g_local_var = g_local_sq / n - g_local_mean**2
        g_local_std = np.sqrt(np.maximum(g_local_var, 0))

        cross = fftconvolve(grid, s_norm[::-1, ::-1], mode='valid')
        safe_std = np.where(g_local_std > 1e-10, g_local_std, 1.0)
        ncc = np.where(g_local_std > 1e-10, cross / (n * safe_std), 0.0)

        # Best match — but exclude the trivially centered match
        # (the whole image downsampled matches itself at center)
        best = float(ncc.max())
        scores.append(max(0.0, best))

    return float(np.mean(scores)) if scores else 0.0
