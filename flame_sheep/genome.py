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
  - low  -> crossbreed with a new random genome (large mutation)
  - mid  -> shift color palette
  - high -> perturb affine coefficients slightly
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .variations import (
    Variation, NUM_VARIATIONS, MAX_VAR_PARAMS, MAX_PARAMS_PER_VAR,
    SLOT_SIZE, PARAMETRIC_VARIATIONS, VAR_PARAMS_SPEC,
    random_var_params, apply_variation_cpu, apply_variations_cpu,
)

if TYPE_CHECKING:
    from .renderer import FlameRenderer

# Re-export for backwards compatibility
_PARAMETRIC_VARIATIONS = PARAMETRIC_VARIATIONS
_apply_variation_cpu = apply_variation_cpu

MAX_TRANSFORMS = 6
MAX_ACTIVE_VARS = 8  # max active variations per transform (for GPU loop)


@dataclass
class Transform:
    """One IFS function: affine transform + variation blend + color.

    Full pipeline per transform:
        pre_variations → affine → variations → post_affine

    Pre-variations and post-affine are optional (None = identity/skip).
    """
    # Affine coefficients: x' = a*x + b*y + c, y' = d*x + e*y + f
    affine: np.ndarray = field(default_factory=lambda: np.array([1,0,0,0,1,0], dtype=np.float32))
    # Post-affine: applied after variations (None = identity)
    post_affine: np.ndarray | None = None
    # Variation weights — how much of each variation to blend
    variations: np.ndarray = field(default_factory=lambda: np.zeros(NUM_VARIATIONS, dtype=np.float32))
    # Pre-affine variation weights (None = no pre-variations)
    pre_variations: np.ndarray | None = None
    # Color index blended during chaos game
    color: float = 0.0
    # Color blend speed: color = speed * xform.color + (1-speed) * prev_color
    color_speed: float = 0.5
    # Probability weight for this transform being chosen
    weight: float = 1.0
    # Per-variation parameters (e.g. julian_power, splits_x)
    var_params: dict[str, float] = field(default_factory=dict)

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
        t.color_speed = float(rng.uniform(0.0, 1.0))
        t.weight = float(rng.uniform(0.5, 2.0))

        # ~25% chance of post_affine (near-identity, contractive)
        if rng.random() < 0.25:
            while True:
                pa = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
                pa += rng.uniform(-0.3, 0.3, 6).astype(np.float32)
                M = np.array([[pa[0], pa[1]], [pa[3], pa[4]]])
                if np.max(np.linalg.svd(M, compute_uv=False)) < 0.9:
                    break
            t.post_affine = pa

        return t


@dataclass
class Genome:
    """Complete flame fractal parameter set."""
    transforms: list[Transform] = field(default_factory=list)
    # Final xform: always applied after the selected transform, not weight-selected.
    # None = no final xform (most genomes). 44% of Electric Sheep use one.
    final_xform: Transform | None = None
    palette: np.ndarray = field(default_factory=lambda: np.zeros((256, 3), dtype=np.float32))
    zoom: float = 1.0
    rotation: float = 0.0
    center: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    # Database ID (set when loaded from library, None for ephemeral genomes)
    db_id: int | None = None
    # flam3 tone mapping parameters (used for esheep rendering)
    flam3_brightness: float = 4.0
    flam3_gamma: float = 4.0

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
            # ~15% chance of final_xform (never weight-selected)
            if rng.random() < 0.15:
                g.final_xform = Transform.random(rng)
                g.final_xform.weight = 0.0
            if g.is_viable() and g.survey_and_correct():
                return g
        # Fallback: return last attempt anyway, better than hanging
        g.survey_and_correct()
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
            nx, ny = x, y

            # 1. Pre-variations (before affine)
            if tr.pre_variations is not None:
                best_pre = int(np.argmax(tr.pre_variations))
                wp = float(tr.pre_variations[best_pre])
                if wp > 0.0:
                    nx, ny = _apply_variation_cpu(best_pre, nx, ny, wp, tr.affine)

            # 2. Affine
            a, b, c, d, e, f = tr.affine
            nx, ny = a*nx + b*ny + c, d*nx + e*ny + f

            # 3. Variations (dominant only for speed)
            best_var = int(np.argmax(tr.variations))
            w = float(tr.variations[best_var])
            if w > 0.0:
                nx, ny = _apply_variation_cpu(best_var, nx, ny, w, tr.affine)

            # 4. Post-affine
            if tr.post_affine is not None:
                pa, pb, pc_, pd, pe, pf = tr.post_affine
                nx, ny = pa*nx + pb*ny + pc_, pd*nx + pe*ny + pf

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

    def survey_attractor(self, n_test: int = 5000, survey_bound: float = 8.0) -> dict:
        """Run a cheap CPU chaos game to find the attractor's centroid and bounding box.

        Uses a 2x-wide viewport to catch off-center attractors. Collects raw
        point positions (not grid cells) for accurate centroid estimation.

        Returns dict with: centroid_x, centroid_y, bbox_min_x, bbox_min_y,
        bbox_max_x, bbox_max_y, coverage, in_viewport.
        """
        rng = np.random.default_rng()
        x, y = 0.0, 0.0
        fuse = 20
        xs, ys = [], []

        weights = np.array([tr.weight for tr in self.transforms], dtype=np.float64)
        if weights.sum() == 0:
            return {'centroid_x': 0, 'centroid_y': 0, 'coverage': 0, 'in_viewport': False}
        weights /= weights.sum()
        cumw = np.cumsum(weights)

        for i in range(fuse + n_test):
            r = rng.random()
            tidx = min(int(np.searchsorted(cumw, r)), len(self.transforms) - 1)
            tr = self.transforms[tidx]
            nx, ny = x, y

            if tr.pre_variations is not None:
                best_pre = int(np.argmax(tr.pre_variations))
                wp = float(tr.pre_variations[best_pre])
                if wp > 0.0:
                    nx, ny = _apply_variation_cpu(best_pre, nx, ny, wp, tr.affine)

            a, b, c, d, e, f = tr.affine
            nx, ny = a*nx + b*ny + c, d*nx + e*ny + f

            best_var = int(np.argmax(tr.variations))
            w = float(tr.variations[best_var])
            if w > 0.0:
                nx, ny = _apply_variation_cpu(best_var, nx, ny, w, tr.affine)

            if tr.post_affine is not None:
                pa, pb, pc_, pd, pe, pf = tr.post_affine
                nx, ny = pa*nx + pb*ny + pc_, pd*nx + pe*ny + pf

            x, y = nx, ny
            if not (np.isfinite(x) and np.isfinite(y)):
                return {'centroid_x': 0, 'centroid_y': 0, 'coverage': 0, 'in_viewport': False}

            if i >= fuse and abs(x) < survey_bound and abs(y) < survey_bound:
                xs.append(x)
                ys.append(y)

        if len(xs) < 10:
            return {'centroid_x': 0, 'centroid_y': 0, 'coverage': 0, 'in_viewport': False}

        xs_arr = np.array(xs, dtype=np.float64)
        ys_arr = np.array(ys, dtype=np.float64)

        centroid_x = float(np.mean(xs_arr))
        centroid_y = float(np.mean(ys_arr))

        # Bounding box from 5th/95th percentile (robust to outliers)
        bbox_min_x = float(np.percentile(xs_arr, 5))
        bbox_max_x = float(np.percentile(xs_arr, 95))
        bbox_min_y = float(np.percentile(ys_arr, 5))
        bbox_max_y = float(np.percentile(ys_arr, 95))

        # Coverage: fraction of points in the display viewport (bound=4.0)
        display_bound = 4.0
        in_display = np.sum((np.abs(xs_arr) < display_bound) & (np.abs(ys_arr) < display_bound))
        coverage = float(in_display / len(xs_arr))

        # Grid-based cell coverage: how many 64×64 cells have at least one hit.
        # Flying dots occupy <10 cells, real fractals occupy 50+.
        grid_size = 64
        display_mask = (np.abs(xs_arr) < display_bound) & (np.abs(ys_arr) < display_bound)
        if display_mask.any():
            gx = np.clip(((xs_arr[display_mask] + display_bound) / (2 * display_bound) * grid_size).astype(int),
                         0, grid_size - 1)
            gy = np.clip(((ys_arr[display_mask] + display_bound) / (2 * display_bound) * grid_size).astype(int),
                         0, grid_size - 1)
            occupied_cells = len(set(zip(gx.tolist(), gy.tolist())))
        else:
            occupied_cells = 0

        return {
            'centroid_x': centroid_x, 'centroid_y': centroid_y,
            'bbox_min_x': bbox_min_x, 'bbox_min_y': bbox_min_y,
            'bbox_max_x': bbox_max_x, 'bbox_max_y': bbox_max_y,
            'coverage': coverage,
            'occupied_cells': occupied_cells,
            'in_viewport': coverage > 0.1,
        }

    def correct_framing(self, survey: dict, margin: float = 1.2) -> None:
        """Correct center and zoom so the attractor is well-framed.

        Mutates self in place. Must account for rotation: the shader applies
        rotation before center offset, so we need to rotate the centroid.
        """
        cx = survey['centroid_x']
        cy = survey['centroid_y']

        # Rotate centroid to match shader coordinate system
        cos_r = np.cos(self.rotation)
        sin_r = np.sin(self.rotation)
        self.center = np.array([
            cos_r * cx - sin_r * cy,
            sin_r * cx + cos_r * cy,
        ], dtype=np.float32)

        # Zoom from bounding box extent — fit the larger axis
        extent_x = max(survey['bbox_max_x'] - survey['bbox_min_x'], 0.1)
        extent_y = max(survey['bbox_max_y'] - survey['bbox_min_y'], 0.1)
        extent = max(extent_x, extent_y)
        # Target: attractor fills ~1/margin of viewport. ES median zoom ~0.25.
        # TODO: independent x/y zoom needs shader change (uniform float → vec2)
        target_zoom = 2.0 / (extent * margin)
        self.zoom = float(np.clip(target_zoom, 0.1, 1.5))

    def check_stability(self, n_angles: int = 8, n_test: int = 3000,
                        max_bbox_ratio: float = 10.0,
                        min_cells: int = 5) -> bool:
        """Check attractor stability across rotation angles.

        Rejects:
          - Flying dots: occupied_cells < min_cells (tiny attractors)
          - Pulsars: bbox area ratio > max_bbox_ratio across angles
        Fails early on first bad angle for speed.
        """
        base_rotation = self.rotation
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False) + base_rotation

        min_area = float('inf')
        max_area = 0.0

        for angle in angles:
            original = self.rotation
            self.rotation = float(angle)
            survey = self.survey_attractor(n_test=n_test)
            self.rotation = original

            if not survey.get('in_viewport', False) or 'bbox_min_x' not in survey:
                return False

            # Cell coverage check — flying dots hit <10 cells
            if survey.get('occupied_cells', 0) < min_cells:
                return False

            extent_x = survey['bbox_max_x'] - survey['bbox_min_x']
            extent_y = survey['bbox_max_y'] - survey['bbox_min_y']
            area = extent_x * extent_y

            if area < 0.01:
                return False

            min_area = min(min_area, area)
            max_area = max(max_area, area)

            if min_area > 0 and max_area / min_area > max_bbox_ratio:
                return False

        return True

    def survey_and_correct(self) -> bool:
        """Survey the attractor, correct framing, and check stability.

        Returns False if attractor is not in viewport or fails stability check.
        """
        survey = self.survey_attractor()
        if not survey['in_viewport']:
            return False
        self.correct_framing(survey)
        if not self.check_stability():
            return False
        return True

    def jitter(self, rng: np.random.Generator, scale: float = 0.1) -> 'Genome':
        """Create a mutated copy with gaussian noise on parameters.

        Jitters var_params, post_affine coefficients, and final_xform.
        Small chance of structural mutations (add/remove post_affine or final_xform).

        Args:
            rng: numpy random generator
            scale: noise magnitude as fraction of parameter range (default 10%)
        """
        from .variations._params import jitter_var_params
        import copy
        g = copy.deepcopy(self)
        for tr in g.transforms:
            if tr.var_params:
                tr.var_params = jitter_var_params(tr.var_params, rng, scale=scale)
            # Jitter color_speed
            tr.color_speed = float(np.clip(
                tr.color_speed + rng.normal(0, scale * 0.5), 0.0, 1.0))
            # Jitter post_affine coefficients
            if tr.post_affine is not None:
                tr.post_affine = tr.post_affine + rng.normal(0, scale * 0.3, 6).astype(np.float32)
            # Structural: 5% chance to add/remove post_affine
            elif rng.random() < 0.05:
                tr.post_affine = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
                tr.post_affine += rng.uniform(-0.1, 0.1, 6).astype(np.float32)
            elif tr.post_affine is not None and rng.random() < 0.05:
                tr.post_affine = None
        # Jitter final_xform
        if g.final_xform is not None:
            if g.final_xform.var_params:
                g.final_xform.var_params = jitter_var_params(
                    g.final_xform.var_params, rng, scale=scale)
            if g.final_xform.post_affine is not None:
                g.final_xform.post_affine = (
                    g.final_xform.post_affine + rng.normal(0, scale * 0.3, 6).astype(np.float32))
        # Structural: 5% chance to add/remove final_xform
        if g.final_xform is None and rng.random() < 0.05:
            g.final_xform = Transform.random(rng)
            g.final_xform.weight = 0.0
        elif g.final_xform is not None and rng.random() < 0.05:
            g.final_xform = None
        return g

    def distance(self, other: 'Genome') -> float:
        """Perceptual distance between two genomes, in [0, 1].

        Uses variation-aware transform matching: greedy Jaccard alignment
        of active variation sets, then normalized parameter distance on
        matched transforms + penalty for unmatched.

        Palette is excluded — it's driven independently by the palette axis.

        Raw transition distance (unbounded) is mapped to [0, 1] via
        d / (d + 1), where typical "good transition" distances are 0-1
        and structurally incompatible pairs saturate near 1.0.
        """
        from .transition import compute_transition_distance

        raw = compute_transition_distance(self, other)
        return float(raw / (raw + 1.0))

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
            tr.color_speed = float(ta.color_speed * (1-t) + tb.color_speed * t)
            tr.weight     = float(ta.weight * (1-t) + tb.weight * t)
            # Lerp var_params — union of keys, missing = 0
            all_keys = set(ta.var_params) | set(tb.var_params)
            tr.var_params = {
                k: ta.var_params.get(k, 0.0) * (1-t) + tb.var_params.get(k, 0.0) * t
                for k in all_keys
            }
            # Post-affine: lerp with identity fallback
            _identity_affine = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
            if ta.post_affine is not None or tb.post_affine is not None:
                pa_a = ta.post_affine if ta.post_affine is not None else _identity_affine
                pa_b = tb.post_affine if tb.post_affine is not None else _identity_affine
                tr.post_affine = _lerp_arr(pa_a, pa_b, t)
            result.transforms.append(tr)

        # Final xform: lerp with identity fallback
        if self.final_xform is not None or other.final_xform is not None:
            fa = self.final_xform if self.final_xform is not None else _identity_transform()
            fb = other.final_xform if other.final_xform is not None else _identity_transform()
            ft = Transform()
            ft.affine = _lerp_arr(fa.affine, fb.affine, t)
            ft.variations = _lerp_arr(fa.variations, fb.variations, t)
            ft.color = float(fa.color * (1-t) + fb.color * t)
            ft.color_speed = float(fa.color_speed * (1-t) + fb.color_speed * t)
            ft.weight = 0.0
            all_keys = set(fa.var_params) | set(fb.var_params)
            ft.var_params = {
                k: fa.var_params.get(k, 0.0) * (1-t) + fb.var_params.get(k, 0.0) * t
                for k in all_keys
            }
            if fa.post_affine is not None or fb.post_affine is not None:
                _identity_affine = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
                pa_a = fa.post_affine if fa.post_affine is not None else _identity_affine
                pa_b = fb.post_affine if fb.post_affine is not None else _identity_affine
                ft.post_affine = _lerp_arr(pa_a, pa_b, t)
            result.final_xform = ft

        result.palette  = _lerp_arr(self.palette,  other.palette,  t)
        result.zoom     = float(self.zoom     * (1-t) + other.zoom     * t)
        result.rotation = float(self.rotation * (1-t) + other.rotation * t)
        result.center   = _lerp_arr(self.center,   other.center,   t)
        return result

    def rotated(self, angle: float) -> 'Genome':
        """Return a copy with all affine transforms rotated by angle (radians).

        Pre-multiplies each transform's full 2x3 affine matrix by a rotation
        matrix R, including the translation column. This rotates the entire
        attractor as a coherent shape. A full 2π rotation returns to the
        original attractor. Matches flam3's flam3_rotate().
        """
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        def _rotate_affine_coeffs(aff):
            a, b, c, d, e, f = aff
            return np.array([
                cos_a * a - sin_a * d,
                cos_a * b - sin_a * e,
                cos_a * c - sin_a * f,
                sin_a * a + cos_a * d,
                sin_a * b + cos_a * e,
                sin_a * c + cos_a * f,
            ], dtype=np.float32)

        def _rotate_transform(tr):
            rt = Transform()
            rt.affine = _rotate_affine_coeffs(tr.affine)
            if tr.post_affine is not None:
                rt.post_affine = _rotate_affine_coeffs(tr.post_affine)
            rt.variations = tr.variations.copy()
            if tr.pre_variations is not None:
                rt.pre_variations = tr.pre_variations.copy()
            rt.color = tr.color
            rt.weight = tr.weight
            rt.var_params = dict(tr.var_params)
            return rt

        result = Genome()
        result.transforms = [_rotate_transform(tr) for tr in self.transforms]
        if self.final_xform is not None:
            result.final_xform = _rotate_transform(self.final_xform)
        result.palette = self.palette.copy()
        result.zoom = self.zoom
        result.rotation = self.rotation
        result.center = self.center.copy()
        return result

    @staticmethod
    def _pack_variations(tr: 'Transform', var_array: np.ndarray,
                         variations: np.ndarray) -> None:
        """Pack active variations from a weight array into a GPU slot row."""
        active_indices = np.where(np.abs(variations) > 1e-6)[0]
        for j, var_idx in enumerate(active_indices[:MAX_ACTIVE_VARS]):
            base = j * SLOT_SIZE
            var_array[base + 0] = float(var_idx)
            var_array[base + 1] = variations[var_idx]
            spec = VAR_PARAMS_SPEC.get(int(var_idx), [])
            for k, param_name in enumerate(spec[:MAX_PARAMS_PER_VAR]):
                var_array[base + 2 + k] = tr.var_params.get(param_name, 0.0)

    def to_gpu_arrays(self) -> dict:
        """Pack genome into flat arrays for GPU upload.

        All buffers have MAX_TRANSFORMS+1 slots — slot MAX_TRANSFORMS is for
        the final xform (identity/empty when absent).
        """
        n = len(self.transforms)
        n_slots = MAX_TRANSFORMS + 1  # +1 for final xform

        affines     = np.zeros((n_slots, 6), dtype=np.float32)
        active_vars = np.full((n_slots, MAX_ACTIVE_VARS * SLOT_SIZE),
                              -1.0, dtype=np.float32)
        colors      = np.zeros(n_slots, dtype=np.float32)
        color_speeds = np.full(n_slots, 0.5, dtype=np.float32)
        weights     = np.zeros(MAX_TRANSFORMS, dtype=np.float32)

        # Post-affines default to identity
        IDENTITY = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
        post_affines = np.tile(IDENTITY, (n_slots, 1))

        # Pre-variations default to empty (all -1)
        pre_active_vars = np.full((n_slots, MAX_ACTIVE_VARS * SLOT_SIZE),
                                  -1.0, dtype=np.float32)

        for i, tr in enumerate(self.transforms[:MAX_TRANSFORMS]):
            affines[i] = tr.affine
            colors[i]  = tr.color
            color_speeds[i] = tr.color_speed
            weights[i] = tr.weight

            # Main variations
            self._pack_variations(tr, active_vars[i], tr.variations)

            # Post-affine (identity if None)
            if tr.post_affine is not None:
                post_affines[i] = tr.post_affine

            # Pre-variations (empty if None)
            if tr.pre_variations is not None:
                self._pack_variations(tr, pre_active_vars[i], tr.pre_variations)

        # normalize weights to probabilities
        w_sum = weights[:n].sum()
        if w_sum > 0:
            weights[:n] /= w_sum

        # Final xform goes in slot MAX_TRANSFORMS
        has_final = self.final_xform is not None
        if has_final:
            ft = self.final_xform
            fidx = MAX_TRANSFORMS
            affines[fidx] = ft.affine
            colors[fidx] = ft.color
            color_speeds[fidx] = ft.color_speed
            self._pack_variations(ft, active_vars[fidx], ft.variations)
            if ft.post_affine is not None:
                post_affines[fidx] = ft.post_affine
            if ft.pre_variations is not None:
                self._pack_variations(ft, pre_active_vars[fidx], ft.pre_variations)

        return {
            'affines': affines,
            'active_vars': active_vars,
            'colors': colors,
            'color_speeds': color_speeds,
            'weights': weights,
            'post_affines': post_affines,
            'pre_active_vars': pre_active_vars,
            'has_final_xform': has_final,
        }


    def aesthetic_score(self, renderer: FlameRenderer | None = None, n_test: int = 5000) -> dict[str, float]:
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
            nx, ny = x, y

            # 1. Pre-variations
            if tr.pre_variations is not None:
                nx, ny = apply_variations_cpu(tr.pre_variations, nx, ny, tr.affine)

            # 2. Affine
            a, b, cc, d, e, f = tr.affine
            nx, ny = a * nx + b * ny + cc, d * nx + e * ny + f

            # 3. Variations
            nx, ny = apply_variations_cpu(tr.variations, nx, ny, tr.affine)

            # 4. Post-affine
            if tr.post_affine is not None:
                pa, pb, pc_, pd, pe, pf = tr.post_affine
                nx, ny = pa * nx + pb * ny + pc_, pd * nx + pe * ny + pf

            # 5. Final xform
            if self.final_xform is not None:
                ft = self.final_xform
                if ft.pre_variations is not None:
                    nx, ny = apply_variations_cpu(ft.pre_variations, nx, ny, ft.affine)
                fa, fb, fc, fd, fe, ff = ft.affine
                nx, ny = fa * nx + fb * ny + fc, fd * nx + fe * ny + ff
                nx, ny = apply_variations_cpu(ft.variations, nx, ny, ft.affine)
                if ft.post_affine is not None:
                    fpa, fpb, fpc, fpd, fpe, fpf = ft.post_affine
                    nx, ny = fpa * nx + fpb * ny + fpc, fpd * nx + fpe * ny + fpf
                c = (c + ft.color) * 0.5

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

        scores = _score_from_histogram(hit_grid, color_grid)

        # Coverage stability: run chaos game at multiple rotation angles
        # and measure how much coverage varies. Low variance = stable wallpaper.
        n_angles = 8
        n_per_angle = max(500, n_test // n_angles)
        coverages = []
        for ai in range(n_angles):
            angle = self.rotation + (2.0 * np.pi * ai / n_angles)
            cos_r, sin_r = np.cos(angle), np.sin(angle)
            mini_grid = np.zeros((grid_size, grid_size), dtype=np.float64)
            rx, ry = 0.0, 0.0
            for i in range(fuse + n_per_angle):
                r = rng.random()
                tidx = min(int(np.searchsorted(cumw, r)), len(self.transforms) - 1)
                tr = self.transforms[tidx]
                nx, ny = rx, ry
                if tr.pre_variations is not None:
                    nx, ny = apply_variations_cpu(tr.pre_variations, nx, ny, tr.affine)
                a, b, cc, d, e, f = tr.affine
                nx, ny = a * nx + b * ny + cc, d * nx + e * ny + f
                nx, ny = apply_variations_cpu(tr.variations, nx, ny, tr.affine)
                if tr.post_affine is not None:
                    pa, pb, pc_, pd, pe, pf = tr.post_affine
                    nx, ny = pa * nx + pb * ny + pc_, pd * nx + pe * ny + pf
                rx, ry = nx, ny
                if not (np.isfinite(rx) and np.isfinite(ry)):
                    break
                if i >= fuse:
                    # Apply rotation before viewport mapping
                    px = cos_r * rx - sin_r * ry
                    py = sin_r * rx + cos_r * ry
                    if abs(px) < bound and abs(py) < bound:
                        gx = int((px + bound) / (2 * bound) * grid_size)
                        gy = int((py + bound) / (2 * bound) * grid_size)
                        gx = max(0, min(grid_size - 1, gx))
                        gy = max(0, min(grid_size - 1, gy))
                        mini_grid[gy, gx] += 1.0
            cov = float(np.count_nonzero(mini_grid)) / (grid_size * grid_size)
            coverages.append(cov)

        mean_cov = np.mean(coverages)
        if mean_cov > 0:
            # Coefficient of variation: std/mean. Low = stable. Invert to 0-1 scale.
            cv = float(np.std(coverages) / mean_cov)
            scores['coverage_stability'] = 1.0 / (1.0 + cv * 5.0)
        else:
            scores['coverage_stability'] = 0.0

        return scores

    def _aesthetic_score_gpu(self, renderer: FlameRenderer) -> dict[str, float]:
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
