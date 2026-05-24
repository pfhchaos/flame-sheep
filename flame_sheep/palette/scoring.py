"""Palette quality metrics.

`score_palette` returns a dict of {contrast, saturation, harmony,
smoothness, fitness} — used by Library.save_palette to stamp scores
on persisted palettes.
"""

from __future__ import annotations

import numpy as np


def score_palette(palette: np.ndarray) -> dict[str, float]:
    """
    Compute quality metrics for a palette.

    palette: shape (256, 3) float32, RGB values in [0, 1].

    Returns dict with keys:
      contrast   -- luminance range (0=flat gray, 1=full black-to-white)
      saturation -- average color saturation (0=grayscale, 1=vivid)
      harmony    -- how well colors form coherent groups (0=noise, 1=clean scheme)
      smoothness -- gradient continuity (0=banded, 1=smooth transitions)
      fitness    -- weighted composite
    """
    p = palette.astype(np.float64)

    # -- contrast: range of perceived luminance --
    luminance = 0.299 * p[:, 0] + 0.587 * p[:, 1] + 0.114 * p[:, 2]
    contrast = float(luminance.max() - luminance.min())

    # -- saturation: average HSV saturation --
    maxc = p.max(axis=1)
    minc = p.min(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        sat = np.where(maxc > 1e-6, (maxc - minc) / maxc, 0.0)
    saturation = float(sat.mean())

    # -- harmony: cluster colors with simple histogram, reward 2-5 dominant hues --
    # Convert to hue (0-360)
    delta = maxc - minc
    hue = np.zeros(256)
    mask = delta > 1e-6
    r, g, b = p[:, 0], p[:, 1], p[:, 2]
    # Red is max
    m = mask & (maxc == r)
    hue[m] = (60 * ((g[m] - b[m]) / delta[m]) % 360)
    # Green is max
    m = mask & (maxc == g)
    hue[m] = 60 * ((b[m] - r[m]) / delta[m]) + 120
    # Blue is max
    m = mask & (maxc == b)
    hue[m] = 60 * ((r[m] - g[m]) / delta[m]) + 240
    hue = hue % 360

    # Hue histogram — count dominant hue groups
    hue_bins = np.histogram(hue[mask], bins=12, range=(0, 360))[0]
    if hue_bins.sum() > 0:
        hue_probs = hue_bins / hue_bins.sum()
        n_dominant = int(np.sum(hue_probs > 0.1))  # bins with >10% of mass
        # 2-5 dominant hues is ideal
        if n_dominant < 2:
            harmony = 0.3  # too monotone
        elif n_dominant <= 5:
            harmony = 1.0 - abs(n_dominant - 3) * 0.15  # peak at 3
        else:
            harmony = max(0.0, 1.0 - (n_dominant - 5) * 0.2)  # too many
    else:
        harmony = 0.3  # all achromatic

    # -- smoothness: average step size between adjacent entries --
    diffs = np.sqrt(np.sum(np.diff(p, axis=0) ** 2, axis=1))
    if len(diffs) > 0:
        mean_diff = diffs.mean()
        if mean_diff > 1e-6:
            # Low coefficient of variation = smooth, high = banded
            cv = float(diffs.std() / mean_diff)
            smoothness = max(0.0, 1.0 - cv / 2.0)
        else:
            smoothness = 1.0  # all same color, technically smooth
    else:
        smoothness = 0.0

    fitness = (
        contrast * 0.25
        + saturation * 0.25
        + harmony * 0.25
        + smoothness * 0.25
    )

    return dict(
        contrast=contrast,
        saturation=saturation,
        harmony=harmony,
        smoothness=smoothness,
        fitness=fitness,
    )
