"""CPU approximation of variation functions for viability testing and scoring.

Only implements variations that can blow up or produce large excursions.
Safe/bounded variations fall through to linear (identity * w).

Used by Genome.is_viable() and BackgroundScorer for CPU chaos game.
"""

import numpy as np


def apply_variations_cpu(variations: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Apply all active variations weighted, matching GPU behavior.

    Args:
        variations: per-variation weights (NUM_VARIATIONS,)
        x, y: input point (post-affine)

    Returns:
        (x', y') blended result
    """
    rx, ry = 0.0, 0.0
    for var_idx in range(len(variations)):
        w = float(variations[var_idx])
        if w < 1e-6:
            continue
        vx, vy = apply_variation_cpu(var_idx, x, y, w)
        rx += vx
        ry += vy
    return rx, ry


def apply_variation_cpu(var_idx: int, x: float, y: float, w: float) -> tuple[float, float]:
    """Apply a single variation on the CPU. Returns (x', y')."""
    r = np.sqrt(x*x + y*y) + 1e-10
    th = np.arctan2(x, y)  # flam3 convention

    if var_idx == 0:   # linear
        return w*x, w*y
    elif var_idx == 1: # sinusoidal
        return w*np.sin(x), w*np.sin(y)
    elif var_idx == 2: # spherical — 1/r², can blow up near origin
        r2 = x*x + y*y + 1e-10
        return w*x/r2, w*y/r2
    elif var_idx == 3: # swirl
        rr = x*x + y*y
        return w*(x*np.sin(rr) - y*np.cos(rr)), w*(x*np.cos(rr) + y*np.sin(rr))
    elif var_idx == 9: # spiral — w/r, blows up near origin
        return (w/r)*(np.cos(th) + np.sin(r)), (w/r)*(np.sin(th) - np.cos(r))
    elif var_idx == 10: # hyperbolic — sin/r, can be large
        return w*np.sin(th)/r, w*np.cos(th)*r
    elif var_idx == 13: # julia
        sqr = w * np.sqrt(r)
        t2  = th * 0.5
        return sqr*np.cos(t2), sqr*np.sin(t2)
    elif var_idx == 18: # exponential — exp(x), very dangerous
        scale = w * np.exp(min(x - 1.0, 10.0))  # clamp to avoid overflow
        return scale * np.cos(np.pi * y), scale * np.sin(np.pi * y)
    elif var_idx == 19: # power
        rp = np.power(max(r, 1e-10), np.sin(th))
        return w*rp*np.cos(th), w*rp*np.sin(th)
    elif var_idx == 34: # tangent — tan(y) blows up at pi/2
        cy = np.cos(y)
        if abs(cy) < 1e-6:
            return w*x, w*10.0  # clamp
        return w*np.sin(x)/max(abs(cy), 1e-6), w*np.tan(y)
    elif var_idx == 35: # cross — 1/(x²-y²)², blows up on diagonals
        d = x*x - y*y
        if abs(d) < 1e-6:
            return w*x, w*y
        s = 1.0 / (d*d + 1e-6)
        return w*s*x, w*s*y
    elif var_idx == 32: # julian — like julia but with nth root
        sqr = w * np.sqrt(r)
        t2 = th * 0.5  # simplified: power=2
        return sqr*np.cos(t2), sqr*np.sin(t2)
    elif var_idx == 33: # juliascope — same danger profile as julian
        sqr = w * np.sqrt(r)
        t2 = th * 0.5
        return sqr*np.cos(t2), sqr*np.sin(t2)
    elif var_idx == 42: # icon — complex polynomial, can blow up
        # Simplified: just rotate by degree, bounded approximation
        return w*x, w*y
    elif var_idx == 43: # sattractor — pure rotation, always bounded
        return w*x, w*y
    elif var_idx == 44: # wallpaper — affine group element, bounded
        return w*x, w*y
    elif var_idx == 45: # frieze — affine group element, bounded
        return w*x, w*y
    else:
        # treat unknown/safe variations as linear for viability purposes
        return w*x, w*y
