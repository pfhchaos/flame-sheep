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

from .variations import (
    Variation, NUM_VARIATIONS, MAX_VAR_PARAMS, PARAMETRIC_VARIATIONS,
    random_var_params, apply_variation_cpu,
)

# Re-export for backwards compatibility
_PARAMETRIC_VARIATIONS = PARAMETRIC_VARIATIONS
_apply_variation_cpu = apply_variation_cpu

MAX_TRANSFORMS = 6
MAX_ACTIVE_VARS = 8  # max active variations per transform (for GPU loop)

# GPU var_params slot table: {param_name: (slot_index, default_value)}
# Must match the layout in flame.comp
_VAR_PARAM_SLOTS = {
    'julian_power': (0, 3.0), 'julian_dist': (1, 1.0),
    'splits_x': (2, 0.5), 'splits_y': (3, 0.5),
    'curl_c1': (4, 0.0), 'curl_c2': (5, 0.0),
    'rect_x': (6, 0.5), 'rect_y': (7, 0.5),
    'check_size': (8, 1.0), 'check_x': (9, 0.0), 'check_y': (10, 0.0),
    'hex_size': (11, 1.0),
    'kal_pull': (12, 0.0), 'kal_rotate': (13, 0.0), 'kal_n': (14, 6.0),
    'icon_degree': (16, 4.0), 'icon_lambda': (17, 0.0),
    'icon_alpha': (18, 0.0), 'icon_beta': (19, 0.0),
    'icon_gamma': (20, 0.0), 'icon_omega': (21, 0.0),
    'sat_m': (22, 4.0),
    'wallpaper_group': (23, 0.0), 'frieze_group': (24, 0.0),
}


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
    # Per-variation parameters (e.g. julian_power, splits_x)
    var_params: dict = field(default_factory=dict)

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

        # Initialize params for parametric variations
        for v in chosen:
            params = random_var_params(int(v), rng)
            t.var_params.update(params)

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
        # existing fast bounds check stays as-is for now
        # TODO: once _aesthetic_score_cpu is implemented, consider:
        # scores = self._aesthetic_score_cpu(n_test)
        # return scores['coverage'] > 0.05 and scores['entropy'] > 0.1
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

    def distance(self, other: 'Genome') -> float:
        """
        Perceptual distance between two genomes, in [0, 1].

        Combines:
          - weighted affine+variation distance (how different the IFS shapes are)
          - palette distance (how different the colours look)

        The affine/variation feature vector is built by sorting transforms by
        their normalised selection weight (heaviest first) then concatenating
        [weight, affine×6, variations×30] per slot, zero-padded to MAX_TRANSFORMS.
        Sorting by weight makes the comparison order-independent: the most
        influential transforms are aligned regardless of list order.
        """
        def _feature_vec(g: 'Genome') -> np.ndarray:
            n = len(g.transforms)
            # Build variation array directly from transforms for distance calc
            affines = np.zeros((MAX_TRANSFORMS, 6), dtype=np.float32)
            variations = np.zeros((MAX_TRANSFORMS, NUM_VARIATIONS), dtype=np.float32)
            weights = np.zeros(MAX_TRANSFORMS, dtype=np.float32)
            for i, tr in enumerate(g.transforms[:MAX_TRANSFORMS]):
                affines[i] = tr.affine
                variations[i] = tr.variations
                weights[i] = tr.weight
            w_sum = weights[:n].sum()
            if w_sum > 0:
                weights[:n] /= w_sum
            # sort slots by descending weight so dominant transforms align
            order = np.argsort(-weights[:n])
            rows = []
            for i in range(MAX_TRANSFORMS):
                if i < n:
                    s = order[i]
                else:
                    s = i  # zero slot
                w   = weights[s] if i < n else 0.0
                aff = affines[s]            # (6,)  already in [-1,1] ish
                var = variations[s]         # (30,) already sum-to-1
                rows.append(np.concatenate([[w], aff, var]))
            return np.concatenate(rows).astype(np.float64)

        va = _feature_vec(self)
        vb = _feature_vec(other)

        # L2 distance, normalised by the max possible range
        # affine values are ~[-1,1], weight ~[0,1], variations ~[0,1]
        # vector length = MAX_TRANSFORMS * (1 + 6 + 30) = 6*37 = 222
        vec_len = MAX_TRANSFORMS * (1 + 6 + NUM_VARIATIONS)
        iff_dist = float(np.linalg.norm(va - vb)) / np.sqrt(vec_len * 4.0)  # 4≈max sq diff
        iff_dist = min(1.0, iff_dist)

        # Mean absolute palette difference (already in [0,1])
        pal_dist = float(np.mean(np.abs(self.palette.astype(np.float64)
                                        - other.palette.astype(np.float64))))

        # Weighted combination — IFS shape matters more than colour
        return float(0.7 * iff_dist + 0.3 * pal_dist)

    def lerp(self, other: 'Genome', t: float) -> 'Genome':
        """
        Linear interpolation toward another genome. Used for smooth morphing.
        
        Transform count mismatch strategy: pad the shorter genome with identity
        transforms at weight=0 before interpolating. This means the extra
        transforms in the longer genome fade in/out smoothly rather than
        appearing/disappearing abruptly.
        
        Identity transform: affine=[1,0,0,0,1,0], variations=linear only,
        weight=0 (won't be selected), color=0.
        
        Edge case: if both genomes have 0 transforms, returns empty genome.
        """
        result = Genome()
        
        na = len(self.transforms)
        nb = len(other.transforms)
        n  = max(na, nb)
        
        # TODO: consider matching transforms by similarity before padding
        # (Hungarian algorithm on distance matrix) to reduce visual discontinuity
        # when transform counts differ by more than 1. Current approach may cause
        # visible jumps when a high-weight transform in the longer genome
        # has no good match in the shorter one.
        
        def _identity_transform() -> Transform:
            tr = Transform()
            tr.affine     = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
            tr.variations = np.zeros(NUM_VARIATIONS, dtype=np.float32)
            tr.variations[Variation.LINEAR] = 1.0
            tr.color  = 0.0
            tr.weight = 0.0  # zero weight: won't affect chaos game
            return tr
        
        a_transforms = list(self.transforms)  + [_identity_transform()] * max(0, nb - na)
        b_transforms = list(other.transforms) + [_identity_transform()] * max(0, na - nb)
        
        result.transforms = []
        for ta, tb in zip(a_transforms, b_transforms):
            tr = Transform()
            tr.affine     = _lerp_arr(ta.affine,     tb.affine,     t)
            tr.variations = _lerp_arr(ta.variations, tb.variations, t)
            tr.color      = float(ta.color  * (1-t) + tb.color  * t)
            tr.weight     = float(ta.weight * (1-t) + tb.weight * t)
            # Lerp var_params — union of keys, missing = 0
            all_keys = set(ta.var_params) | set(tb.var_params)
            tr.var_params = {
                k: ta.var_params.get(k, 0.0) * (1-t) + tb.var_params.get(k, 0.0) * t
                for k in all_keys
            }
            result.transforms.append(tr)
        
        result.palette  = _lerp_arr(self.palette,  other.palette,  t)
        result.zoom     = float(self.zoom     * (1-t) + other.zoom     * t)
        result.rotation = float(self.rotation * (1-t) + other.rotation * t)
        result.center   = _lerp_arr(self.center,   other.center,   t)
        return result

    def to_gpu_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Pack genome into flat arrays for GPU upload.

        Returns:
            affines:     (MAX_TRANSFORMS, 6) float32
            active_vars: (MAX_TRANSFORMS, MAX_ACTIVE_VARS, 2) float32
                         Each slot is (var_index, weight). Index < 0 = unused.
            colors:      (MAX_TRANSFORMS,) float32 [color index per transform]
            weights:     (MAX_TRANSFORMS,) float32 [normalized probabilities]
            var_params:  (MAX_TRANSFORMS, MAX_VAR_PARAMS) float32
                         Per-transform variation parameters.
        """
        n = len(self.transforms)
        affines     = np.zeros((MAX_TRANSFORMS, 6), dtype=np.float32)
        # Pack active variations: (index, weight) pairs, -1 marks end
        active_vars = np.full((MAX_TRANSFORMS, MAX_ACTIVE_VARS, 2), -1.0, dtype=np.float32)
        colors      = np.zeros(MAX_TRANSFORMS, dtype=np.float32)
        weights     = np.zeros(MAX_TRANSFORMS, dtype=np.float32)
        var_params  = np.zeros((MAX_TRANSFORMS, MAX_VAR_PARAMS), dtype=np.float32)

        for i, tr in enumerate(self.transforms[:MAX_TRANSFORMS]):
            affines[i] = tr.affine
            colors[i]  = tr.color
            weights[i] = tr.weight

            # Find active variations (weight > 0) and pack them
            active_indices = np.where(tr.variations > 1e-6)[0]
            for j, var_idx in enumerate(active_indices[:MAX_ACTIVE_VARS]):
                active_vars[i, j, 0] = float(var_idx)
                active_vars[i, j, 1] = tr.variations[var_idx]

            # Pack variation parameters into fixed layout (slot table)
            vp = tr.var_params
            for name, (slot, default) in _VAR_PARAM_SLOTS.items():
                var_params[i, slot] = vp.get(name, default)

        # normalize weights to probabilities
        w_sum = weights[:n].sum()
        if w_sum > 0:
            weights[:n] /= w_sum

        return affines, active_vars, colors, weights, var_params


    def aesthetic_score(self, renderer=None, n_test: int = 5000) -> dict[str, float]:
        """
        Compute aesthetic quality metrics for this genome.
        
        Two modes:
          - renderer=None: CPU-only metrics from a quick chaos game.
            Fast (~10ms), suitable for pre-filtering large batches of random genomes.
          - renderer=FlameRenderer: GPU histogram metrics from a real render pass.
            Slower but accurate. Use for scoring candidates before display.
        
        Returns a dict of named scores, all normalized to [0, 1] where 1 is better.
        Intentionally not combined into a single score — callers should weight
        these based on context (e.g. audio-coherent scoring weights differently
        than pure aesthetic scoring).
        
        Keys:
          coverage      -- fraction of viewport pixels hit (0=empty, 1=full)
          entropy       -- spatial information content (0=collapsed, 1=noise)
          color_entropy -- palette utilization (0=monochrome, 1=full range)
          balance       -- how centered/symmetric the attractor is
          complexity    -- estimated fractal dimension proxy
        """
        if renderer is not None:
            return self._aesthetic_score_gpu(renderer)
        return self._aesthetic_score_cpu(n_test)

    def _aesthetic_score_cpu(self, n_test: int = 5000) -> dict[str, float]:
        """
        CPU chaos game aesthetic scorer.

        Runs n_test iterations (after 20-step fuse) and builds a coarse
        histogram over a 64x64 grid. Computes metrics from that histogram.

        NOTE: Only applies the dominant variation per transform (same approximation
        as is_viable). Results are directionally correct but not pixel-accurate.
        Treat as a fast pre-filter, not ground truth.
        """
        grid_size = 64
        fuse = 20
        bound = 4.0
        rng = np.random.default_rng()

        hit_grid = np.zeros((grid_size, grid_size), dtype=np.float64)
        color_grid = np.zeros((grid_size, grid_size), dtype=np.float64)

        weights = np.array([tr.weight for tr in self.transforms], dtype=np.float64)
        weights /= weights.sum()
        cumw = np.cumsum(weights)

        x, y, c = 0.0, 0.0, 0.5

        for i in range(fuse + n_test):
            r = rng.random()
            tidx = min(int(np.searchsorted(cumw, r)), len(self.transforms) - 1)
            tr = self.transforms[tidx]
            a, b, cc, d, e, f = tr.affine
            nx = a * x + b * y + cc
            ny = d * x + e * y + f

            best_var = int(np.argmax(tr.variations))
            w = float(tr.variations[best_var])
            if w > 0.0:
                nx, ny = _apply_variation_cpu(best_var, nx, ny, w)

            x, y = nx, ny
            c = (c + tr.color) * 0.5

            if not (np.isfinite(x) and np.isfinite(y)):
                break

            if i >= fuse and abs(x) < bound and abs(y) < bound:
                gx = int((x + bound) / (2 * bound) * grid_size)
                gy = int((y + bound) / (2 * bound) * grid_size)
                gx = max(0, min(grid_size - 1, gx))
                gy = max(0, min(grid_size - 1, gy))
                hit_grid[gy, gx] += 1.0
                color_grid[gy, gx] += c

        return _score_from_histogram(hit_grid, color_grid)

    def _aesthetic_score_gpu(self, renderer) -> dict[str, float]:
        """
        GPU histogram aesthetic scorer.

        Reads back the renderer's histogram SSBO after a render pass and
        computes metrics from the full-resolution data.

        More accurate than CPU scorer but requires a full render pass (~16ms).
        Use for final scoring of candidates, not bulk pre-filtering.
        Includes symmetry metrics (tier 2) since we have a real histogram.
        """
        hit_counts, color_accs = renderer.histogram_data()
        hit_grid = hit_counts.astype(np.float64)
        # Recover average color index per pixel (undo COLOR_SCALE packing)
        COLOR_SCALE = 1_000_000.0
        with np.errstate(divide='ignore', invalid='ignore'):
            color_grid = np.where(
                hit_counts > 0,
                color_accs.astype(np.float64) / (hit_counts.astype(np.float64) * COLOR_SCALE),
                0.0,
            )
        scores = _score_from_histogram(hit_grid, color_grid)
        scores.update(_score_symmetry(hit_grid))
        return scores

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

    return dict(
        coverage=coverage,
        entropy=entropy,
        color_entropy=color_entropy,
        balance=balance,
        complexity=complexity,
    )


def _score_symmetry(hit_grid: np.ndarray) -> dict[str, float]:
    """Compute symmetry metrics from a hit-count histogram.

    Separate from _score_from_histogram because symmetry detection
    is tier 2 (moderate cost) — not run during fast genome generation,
    only during GPU scoring or idle background passes.
    """
    from .symmetry import symmetry_scores
    sym = symmetry_scores(hit_grid)
    return dict(
        symmetry_max=sym['symmetry_max'],
        rotational=sym['rotational_best'],
        reflective=sym['reflective_best'],
        radial=sym['radial'],
        periodic=sym['periodic'],
        fractal_dim=sym['fractal_dim'],
        self_similarity=sym['self_similarity'],
    )


    # _apply_variation_cpu is re-exported from .variations for backwards compat


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
