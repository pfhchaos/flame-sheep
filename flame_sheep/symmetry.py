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

import numpy as np
from scipy import ndimage


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
                    fractal_dim=0.0, symmetry_max=0.0)

    # Normalize to float, log-scale for perceptual uniformity
    grid = np.log1p(hit_grid.astype(np.float64))
    grid /= grid.max() + 1e-10

    rot_score, rot_n = _rotational_symmetry(grid)
    ref_score, ref_angle = _reflective_symmetry(grid)
    rad_score = _radial_symmetry(grid)
    per_score, per_freq = _periodic_structure(grid)
    fdim = _fractal_dimension(hit_grid)

    return dict(
        rotational_best=rot_score,
        rotational_n=rot_n,
        reflective_best=ref_score,
        reflective_angle=ref_angle,
        radial=rad_score,
        periodic=per_score,
        periodic_freq=per_freq,
        fractal_dim=fdim,
        symmetry_max=max(rot_score, ref_score, rad_score, per_score),
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


def _radial_symmetry(grid: np.ndarray, n_rings: int = 16) -> float:
    """Measure radial/mandala symmetry via concentric ring correlation.

    Computes mean intensity per ring at increasing radii from center.
    High correlation between adjacent rings = radial symmetry.
    """
    h, w = grid.shape
    cy, cx = h / 2.0, w / 2.0
    max_r = min(cy, cx)

    ys, xs = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xs - cx)**2 + (ys - cy)**2)

    ring_means = []
    for i in range(n_rings):
        r_lo = i * max_r / n_rings
        r_hi = (i + 1) * max_r / n_rings
        mask = (dist >= r_lo) & (dist < r_hi)
        if mask.any():
            ring_means.append(float(grid[mask].mean()))
        else:
            ring_means.append(0.0)

    if len(ring_means) < 3:
        return 0.0

    # Measure how structured the radial profile is:
    # high variance in ring means = concentric structure (rings, mandalas)
    # low variance = uniform fill or random
    ring_arr = np.array(ring_means)
    if ring_arr.max() < 1e-10:
        return 0.0

    ring_arr /= ring_arr.max()
    # Coefficient of variation — how much the rings differ from each other
    mean = ring_arr.mean()
    if mean < 1e-10:
        return 0.0
    cv = float(ring_arr.std() / mean)
    # Normalize: CV > 0.5 = strong ring structure
    return min(1.0, max(0.0, cv / 0.5))


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
