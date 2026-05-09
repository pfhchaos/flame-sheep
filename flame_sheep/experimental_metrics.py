"""Experimental image-based metrics for genome scoring.

"Throw at wall" phase — measure everything plausible from the rendered
images, figure out what predicts aesthetic quality later.

All functions take a grayscale uint8 numpy array (from tonemapped PNG)
and return a dict of metric name → float value.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _to_gray(img: np.ndarray) -> np.ndarray:
    """Ensure image is 2D grayscale float [0, 1]."""
    if img.ndim == 3:
        # RGBA or RGB → luminance
        img = img[..., :3].mean(axis=-1)
    return img.astype(np.float64) / 255.0 if img.dtype == np.uint8 else img


def _lit_mask(gray: np.ndarray, threshold: float = 0.02) -> np.ndarray:
    """Boolean mask of pixels above background threshold."""
    return gray > threshold


def spatial_frequency_spectrum(img: np.ndarray) -> dict[str, float]:
    """FFT energy distribution — ratio of high-frequency to low-frequency.

    High ratio = fine detail (filaments, texture).
    Low ratio = smooth blobs.
    Also reports the peak spatial frequency scale.
    """
    gray = _to_gray(img)
    h, w = gray.shape

    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift) ** 2

    # Radial frequency bins
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    max_r = min(cx, cy)

    # Radial power spectrum
    radial_power = np.zeros(max_r)
    for ri in range(max_r):
        mask = r == ri
        if mask.any():
            radial_power[ri] = magnitude[mask].mean()

    if radial_power.sum() < 1e-10:
        return {'freq_high_ratio': 0.0, 'freq_peak_scale': 0.0}

    # Split at 1/4 of max frequency
    split = max_r // 4
    low_energy = radial_power[:split].sum()
    high_energy = radial_power[split:].sum()
    ratio = high_energy / max(low_energy, 1e-10)

    # Peak frequency (excluding DC)
    peak_freq = np.argmax(radial_power[1:]) + 1
    peak_scale = float(peak_freq) / max_r  # normalized 0..1

    return {
        'freq_high_ratio': float(min(ratio, 10.0)),  # cap outliers
        'freq_peak_scale': float(peak_scale),
    }


def lacunarity(img: np.ndarray, scales: tuple[int, ...] = (4, 8, 16, 32)) -> dict[str, float]:
    """Gliding-box lacunarity — measures gap/hole distribution.

    Complementary to fractal dimension: fractals with the same dimension
    can have very different lacunarity (clustered vs uniform gaps).
    """
    gray = _to_gray(img)
    binary = _lit_mask(gray).astype(np.float64)
    h, w = binary.shape

    lac_values = []
    for box_size in scales:
        if box_size >= h or box_size >= w:
            continue
        # Box sums via cumulative sum
        integral = binary.cumsum(axis=0).cumsum(axis=1)
        # Pad for box sum computation
        pad = np.zeros((h + 1, w + 1))
        pad[1:, 1:] = integral
        box_sums = (pad[box_size:, box_size:]
                    - pad[:-box_size, box_size:]
                    - pad[box_size:, :-box_size]
                    + pad[:-box_size, :-box_size])
        if box_sums.size == 0:
            continue
        mean_s = box_sums.mean()
        if mean_s < 1e-10:
            continue
        var_s = box_sums.var()
        lac_values.append(var_s / (mean_s ** 2))

    return {
        'lacunarity': float(np.mean(lac_values)) if lac_values else 0.0,
    }


def radial_density_profile(img: np.ndarray) -> dict[str, float]:
    """How density falls off from centroid.

    Returns slope of log-density vs log-radius (power law fit)
    and r² goodness of fit. Steep negative slope = concentrated.
    Flat slope = diffuse.
    """
    gray = _to_gray(img)
    h, w = gray.shape

    # Find centroid
    total = gray.sum()
    if total < 1e-10:
        return {'radial_slope': 0.0, 'radial_r2': 0.0}

    ys, xs = np.mgrid[:h, :w]
    cx = (xs * gray).sum() / total
    cy = (ys * gray).sum() / total

    # Radial distance from centroid
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    max_r = min(h, w) // 2

    # Bin by radius
    n_bins = min(64, max_r)
    bin_edges = np.linspace(0, max_r, n_bins + 1)
    radial_density = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        if mask.any():
            radial_density[i] = gray[mask].mean()

    # Log-log fit (skip zero bins)
    valid = radial_density > 1e-10
    if valid.sum() < 3:
        return {'radial_slope': 0.0, 'radial_r2': 0.0}

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    log_r = np.log(bin_centers[valid] + 1)
    log_d = np.log(radial_density[valid])

    # Linear regression
    coeffs = np.polyfit(log_r, log_d, 1)
    slope = coeffs[0]

    # R² goodness of fit
    predicted = np.polyval(coeffs, log_r)
    ss_res = ((log_d - predicted) ** 2).sum()
    ss_tot = ((log_d - log_d.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-10)

    return {
        'radial_slope': float(slope),
        'radial_r2': float(max(r2, 0.0)),
    }


def compactness(img: np.ndarray) -> dict[str, float]:
    """Perimeter² / area of the lit region.

    Low = compact blob. High = filamentous/spindly.
    Circle = 4π ≈ 12.6. Fractal filaments >> 100.
    """
    gray = _to_gray(img)
    binary = _lit_mask(gray)

    area = binary.sum()
    if area < 10:
        return {'compactness': 0.0}

    # Perimeter via erosion
    eroded = ndimage.binary_erosion(binary)
    perimeter = (binary & ~eroded).sum()

    c = float(perimeter ** 2) / float(area)
    # Normalize: divide by 4π (circle baseline), cap at reasonable max
    c_norm = min(c / (4 * np.pi), 100.0)

    return {'compactness': float(c_norm)}


def bounding_box_aspect(img: np.ndarray) -> dict[str, float]:
    """Aspect ratio of the bounding box of lit pixels.

    1.0 = square. >1 = landscape. <1 = portrait.
    Returned as log2 so it's symmetric around 0.
    """
    gray = _to_gray(img)
    lit = _lit_mask(gray)

    coords = np.argwhere(lit)
    if len(coords) < 2:
        return {'bbox_aspect': 0.0}

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    height = y_max - y_min + 1
    width = x_max - x_min + 1

    if height < 1 or width < 1:
        return {'bbox_aspect': 0.0}

    return {'bbox_aspect': float(np.log2(width / height))}


def filament_count(img: np.ndarray) -> dict[str, float]:
    """Count distinct filament endpoints via morphological skeleton.

    More endpoints = more branching tendrils = more visual complexity.
    """
    gray = _to_gray(img)
    binary = _lit_mask(gray)

    # Skeletonize
    try:
        from skimage.morphology import skeletonize
        skeleton = skeletonize(binary)
    except ImportError:
        # Fallback: thin via erosion (crude but works)
        skeleton = binary.copy()
        for _ in range(10):
            eroded = ndimage.binary_erosion(skeleton)
            if not eroded.any():
                break
            skeleton = eroded

    # Count endpoints: skeleton pixels with exactly 1 neighbor
    if not skeleton.any():
        return {'filament_count': 0.0}

    # Convolve with 3x3 kernel to count neighbors
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0
    neighbor_count = ndimage.convolve(skeleton.astype(np.float64), kernel,
                                       mode='constant', cval=0)
    endpoints = ((skeleton) & (neighbor_count == 1)).sum()

    return {'filament_count': float(endpoints)}


def contrast_ratio(img: np.ndarray) -> dict[str, float]:
    """Dynamic range of pixel intensities within lit area.

    High contrast = rich density variation. Low = flat/uniform.
    Measured as ratio of 95th to 5th percentile in log space.
    """
    gray = _to_gray(img)
    lit = gray[_lit_mask(gray)]

    if len(lit) < 10:
        return {'contrast_ratio': 0.0}

    p5 = np.percentile(lit, 5)
    p95 = np.percentile(lit, 95)

    if p5 < 1e-6:
        p5 = 1e-6

    ratio = np.log10(p95 / p5)
    return {'contrast_ratio': float(max(ratio, 0.0))}


def angular_uniformity(img: np.ndarray, n_bins: int = 36) -> dict[str, float]:
    """Entropy of angular histogram from centroid.

    High entropy = uniform spread in all directions.
    Low entropy = preferred orientations (streaky/directional).
    """
    gray = _to_gray(img)
    h, w = gray.shape

    total = gray.sum()
    if total < 1e-10:
        return {'angular_uniformity': 0.0}

    ys, xs = np.mgrid[:h, :w]
    cx = (xs * gray).sum() / total
    cy = (ys * gray).sum() / total

    # Angle from centroid
    angles = np.arctan2(ys - cy, xs - cx)  # -π to π
    angles = (angles + np.pi) / (2 * np.pi)  # 0 to 1

    # Weighted histogram
    bins = np.linspace(0, 1, n_bins + 1)
    hist = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (angles >= bins[i]) & (angles < bins[i + 1])
        hist[i] = gray[mask].sum()

    hist_sum = hist.sum()
    if hist_sum < 1e-10:
        return {'angular_uniformity': 0.0}

    # Normalized entropy
    p = hist / hist_sum
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    max_entropy = np.log(n_bins)

    return {'angular_uniformity': float(entropy / max_entropy) if max_entropy > 0 else 0.0}


def swept_symmetry(img: np.ndarray) -> dict[str, float]:
    """Rotational symmetry of the rotation-accumulated envelope.

    Tests 2-fold through 8-fold by rotating and correlating.
    """
    gray = _to_gray(img)
    h, w = gray.shape

    if gray.sum() < 1e-10:
        return {'sw_rotational': 0.0}

    best = 0.0
    for n_fold in range(2, 9):
        angle = 360.0 / n_fold
        rotated = ndimage.rotate(gray, angle, reshape=False, order=1, mode='constant')
        # Correlation coefficient
        flat_orig = gray.ravel()
        flat_rot = rotated.ravel()
        denom = np.sqrt((flat_orig ** 2).sum() * (flat_rot ** 2).sum())
        if denom < 1e-10:
            continue
        corr = np.dot(flat_orig, flat_rot) / denom
        best = max(best, corr)

    return {'sw_rotational': float(best)}


def swept_coverage(img: np.ndarray) -> dict[str, float]:
    """Fraction of canvas covered by the rotating fractal."""
    gray = _to_gray(img)
    lit = _lit_mask(gray)
    total = lit.size
    return {'sw_coverage': float(lit.sum() / total) if total > 0 else 0.0}


def score_all(static_img: np.ndarray, swept_img: np.ndarray | None = None) -> dict[str, float]:
    """Run all experimental metrics on the provided images.

    Args:
        static_img: Grayscale or RGBA image from static render.
        swept_img: Grayscale or RGBA image from swept render (optional).

    Returns:
        Dict of all metric names → float values.
    """
    scores = {}
    scores.update(spatial_frequency_spectrum(static_img))
    scores.update(lacunarity(static_img))
    scores.update(radial_density_profile(static_img))
    scores.update(compactness(static_img))
    scores.update(bounding_box_aspect(static_img))
    scores.update(filament_count(static_img))
    scores.update(contrast_ratio(static_img))
    scores.update(angular_uniformity(static_img))

    if swept_img is not None:
        scores.update(swept_symmetry(swept_img))
        scores.update(swept_coverage(swept_img))

    return scores
