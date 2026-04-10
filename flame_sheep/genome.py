"""
Flame fractal genome — IFS parameter set and mutation logic.

A genome defines a complete flame fractal:
  - N transforms, each with:
      - affine coefficients [a, b, c, d, e, f]
      - variation weights (which nonlinear functions to apply, and how much)
      - color index (0..1, blended during chaos game)
      - probability weight (how often this transform is chosen)
  - global color palette (256 RGB entries)
  - global params: zoom, rotation, center

Mutation is driven by beat detection from audio.py:
  - kick  -> crossbreed with a new random genome (large mutation)
  - snare -> shift color palette
  - hihat -> perturb affine coefficients slightly
"""

import numpy as np
from dataclasses import dataclass, field

# Variation function indices — matches order in flame.comp
class Variation:
    LINEAR      = 0
    SINUSOIDAL  = 1
    SPHERICAL   = 2
    SWIRL       = 3
    HORSESHOE   = 4
    POLAR       = 5
    HANDKERCHIEF= 6
    HEART       = 7
    DISK        = 8
    SPIRAL      = 9
    HYPERBOLIC  = 10
    DIAMOND     = 11
    EX          = 12
    JULIA       = 13
    BENT        = 14
    WAVES       = 15
    FISHEYE     = 16
    POPCORN     = 17
    EXPONENTIAL = 18
    POWER       = 19
    COSINE      = 20
    RINGS       = 21
    FAN         = 22
    BLOB        = 23
    PDJ         = 24
    FAN2        = 25
    RINGS2      = 26
    EYEFISH     = 27
    BUBBLE      = 28
    CYLINDER    = 29

NUM_VARIATIONS = 30
MAX_TRANSFORMS = 6


@dataclass
class Transform:
    """One IFS function: affine transform + variation blend + color."""
    # Affine coefficients: x' = a*x + b*y + c, y' = d*x + e*y + f
    affine: np.ndarray = field(default_factory=lambda: np.array([1,0,0,0,1,0], dtype=np.float32))
    # Variation weights — how much of each variation to blend
    variations: np.ndarray = field(default_factory=lambda: np.zeros(NUM_VARIATIONS, dtype=np.float32))
    # Color index blended during chaos game
    color: float = 0.0
    # Probability weight for this transform being chosen
    weight: float = 1.0

    @classmethod
    def random(cls, rng: np.random.Generator) -> 'Transform':
        t = cls()
        # Random affine — keep it contractive (det < 1) to ensure attractor exists
        while True:
            a, b, d, e = rng.uniform(-1, 1, 4)
            M = np.array([[a, b], [d, e]])
            # contractivity: all singular values must be < 1
            # determinant < 1 is necessary but not sufficient (shear can still stretch)
            if np.max(np.linalg.svd(M, compute_uv=False)) < 0.9:
                break
        c, f = rng.uniform(-1, 1, 2)
        t.affine = np.array([a, b, c, d, e, f], dtype=np.float32)

        # Pick 1-2 variations with random weights
        n_vars = rng.integers(1, 3)
        chosen = rng.choice(NUM_VARIATIONS, n_vars, replace=False)
        weights = rng.uniform(0.3, 1.0, n_vars)
        weights /= weights.sum()
        t.variations[chosen] = weights

        t.color = float(rng.uniform(0, 1))
        t.weight = float(rng.uniform(0.5, 2.0))
        return t


@dataclass
class Genome:
    """Complete flame fractal parameter set."""
    transforms: list[Transform] = field(default_factory=list)
    palette: np.ndarray = field(default_factory=lambda: np.zeros((256, 3), dtype=np.float32))
    zoom: float = 1.0
    rotation: float = 0.0
    center: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))

    @classmethod
    def random(cls, rng: np.random.Generator | None = None, n_transforms: int | None = None) -> 'Genome':
        if rng is None:
            rng = np.random.default_rng()
        # Keep generating until we get a genome whose attractor fits in view
        for _ in range(50):
            g = cls()
            n = n_transforms or int(rng.integers(2, MAX_TRANSFORMS + 1))
            g.transforms = [Transform.random(rng) for _ in range(n)]
            g.palette = _random_palette(rng)
            g.zoom = float(rng.uniform(0.8, 1.5))
            g.rotation = float(rng.uniform(0, 2 * np.pi))
            g.center = rng.uniform(-0.5, 0.5, 2).astype(np.float32)
            if g.is_viable():
                return g
        # Fallback: return last attempt anyway, better than hanging
        return g

    def is_viable(self, n_test: int = 2000, bound: float = 4.0) -> bool:
        """Quick CPU chaos game including variations to check attractor stays in bounds.
        Returns True if enough points land inside the viewport."""
        rng  = np.random.default_rng()
        x, y = 0.0, 0.0
        hits = 0
        weights = np.array([tr.weight for tr in self.transforms], dtype=np.float64)
        weights /= weights.sum()
        cumw = np.cumsum(weights)

        for i in range(n_test):
            r    = rng.random()
            tidx = int(np.searchsorted(cumw, r))
            tidx = min(tidx, len(self.transforms) - 1)
            tr   = self.transforms[tidx]
            a, b, c, d, e, f = tr.affine
            nx = a*x + b*y + c
            ny = d*x + e*y + f

            # Apply dominant variation (highest weight) — approximates GPU behavior
            # without reimplementing all 30 variations in Python
            best_var = int(np.argmax(tr.variations))
            w = float(tr.variations[best_var])
            if w > 0.0:
                nx, ny = _apply_variation_cpu(best_var, nx, ny, w)

            x, y = nx, ny

            # bail on NaN/Inf immediately
            if not (np.isfinite(x) and np.isfinite(y)):
                return False

            if i > 20:
                if abs(x) < bound and abs(y) < bound:
                    hits += 1
                elif abs(x) > 1e6 or abs(y) > 1e6:
                    return False

        return hits > (n_test - 20) * 0.5

    def lerp(self, other: 'Genome', t: float) -> 'Genome':
        """Linear interpolation toward another genome. Used for smooth morphing."""
        # TODO: handle differing transform counts (pad shorter with identity)
        result = Genome()
        n = min(len(self.transforms), len(other.transforms))
        result.transforms = []
        for i in range(n):
            tr = Transform()
            tr.affine    = _lerp_arr(self.transforms[i].affine,      other.transforms[i].affine,      t)
            tr.variations= _lerp_arr(self.transforms[i].variations,  other.transforms[i].variations,  t)
            tr.color     = float(self.transforms[i].color  * (1-t) + other.transforms[i].color  * t)
            tr.weight    = float(self.transforms[i].weight * (1-t) + other.transforms[i].weight * t)
            result.transforms.append(tr)
        result.palette   = _lerp_arr(self.palette,  other.palette,  t)
        result.zoom      = float(self.zoom     * (1-t) + other.zoom     * t)
        result.rotation  = float(self.rotation * (1-t) + other.rotation * t)
        result.center    = _lerp_arr(self.center,   other.center,   t)
        return result

    def to_gpu_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Pack genome into flat arrays for GPU upload.

        Returns:
            affines:    (MAX_TRANSFORMS, 6)  float32
            variations: (MAX_TRANSFORMS, NUM_VARIATIONS) float32
            colors:     (MAX_TRANSFORMS,)    float32  [color index per transform]
            weights:    (MAX_TRANSFORMS,)    float32  [normalized probabilities]
        """
        n = len(self.transforms)
        affines    = np.zeros((MAX_TRANSFORMS, 6),             dtype=np.float32)
        variations = np.zeros((MAX_TRANSFORMS, NUM_VARIATIONS),dtype=np.float32)
        colors     = np.zeros(MAX_TRANSFORMS,                  dtype=np.float32)
        weights    = np.zeros(MAX_TRANSFORMS,                  dtype=np.float32)

        for i, tr in enumerate(self.transforms[:MAX_TRANSFORMS]):
            affines[i]    = tr.affine
            variations[i] = tr.variations
            colors[i]     = tr.color
            weights[i]    = tr.weight

        # normalize weights to probabilities
        w_sum = weights[:n].sum()
        if w_sum > 0:
            weights[:n] /= w_sum

        return affines, variations, colors, weights


def _apply_variation_cpu(var_idx: int, x: float, y: float, w: float) -> tuple[float, float]:
    """CPU approximation of key variation functions for viability testing.
    Only implements variations that can blow up or produce large excursions.
    Safe/bounded variations fall through to linear (identity * w)."""
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
    else:
        # treat unknown/safe variations as linear for viability purposes
        return w*x, w*y


def _lerp_arr(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (a * (1 - t) + b * t).astype(a.dtype)


def _random_palette(rng: np.random.Generator) -> np.ndarray:
    """Generate a smooth random color palette by interpolating random control points."""
    n_points = rng.integers(3, 7)
    control  = rng.uniform(0, 1, (n_points, 3)).astype(np.float32)
    palette  = np.zeros((256, 3), dtype=np.float32)
    for i in range(256):
        t        = i / 255.0 * (n_points - 1)
        lo, hi   = int(t), min(int(t) + 1, n_points - 1)
        palette[i] = _lerp_arr(control[lo], control[hi], t - lo)
    return palette
