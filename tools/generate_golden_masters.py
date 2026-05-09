#!/usr/bin/env python3
"""Generate golden master data for regression tests.

Run this manually when variations or symmetry metrics change.
Visually verify the output, then check in tests/golden/*.npz.

Usage:
    python tools/generate_golden_masters.py
    python tools/generate_golden_masters.py --verify  # compare to existing
"""

import sys
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flame_sheep.variations import (
    Variation, NUM_VARIATIONS, apply_variation_cpu,
)
from flame_sheep.variations._cpu import _current_var_params
import flame_sheep.variations._cpu as cpu_mod
from flame_sheep.symmetry import symmetry_scores

# Add tests/ to path for shared fixtures
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests'))
from variation_fixtures import (
    GOLDEN_POINTS, TEST_AFFINE, AFFINE_VARIATIONS, RANDOM_VARIATIONS,
    PARAM_FIXTURES, VAR_POINTS as EXTRA_VAR_POINTS,
)

GOLDEN_DIR = Path(__file__).resolve().parent.parent / 'tests' / 'golden'

# Fixed test points for transform verification.
# Chosen to exercise conditional branches in variations:
#   - All four quadrants (bent, bent2, separation sign checks)
#   - Inside and outside unit circle (loonie, whorl, flipcircle)
#   - Near grid boundaries (rectangles, cell, modulus, checks)
#   - Near origin (spherical, spiral, scry singularities)
#   - Large radius (exponential blowup guards)
TEST_POINTS = [
    ( 1.0,   0.5),    # Q1, r>1
    ( 0.0,   0.0),    # origin — degenerate/EPS guards
    (-0.3,   0.7),    # Q2
    ( 0.5,  -0.5),    # Q4, r≈0.71 (inside unit circle)
    ( 2.0,   1.0),    # Q1, large r
    (-0.8,  -0.4),    # Q3 (both negative — bent/bent2 branch)
    ( 0.05,  0.03),   # near origin but nonzero (singularity probes)
    ( 0.95,  0.31),   # r≈1.0 from inside (loonie/whorl boundary)
    ( 1.05,  0.31),   # r≈1.0 from outside
    ( 0.49,  0.49),   # grid boundary (floor transitions)
    ( 0.51,  0.51),   # grid boundary other side
    (-1.5,   0.8),    # Q2, large negative x
]

# Fixed affine for affine-reading variations (15/17/21/22)
# [a, b, c, d, e, f]
TEST_AFFINE = np.array([0.8, 0.5, 0.3, -0.2, 0.6, 0.5], dtype=np.float32)

# Affine-reading variations — need an affine, not params
AFFINE_VARIATIONS = {Variation.WAVES, Variation.POPCORN,
                     Variation.RINGS, Variation.FAN}

# Fixed params for parametric variations
PARAM_FIXTURES = {
    # waves/popcorn/rings/fan (15/17/21/22) now read from affine — no params
    # Their parameterized counterparts are at new indices:
    Variation.WAVES_PARAM: {'waves_freq_x': 0.5, 'waves_freq_y': 0.3,
                            'waves_amp_x': 0.8, 'waves_amp_y': 0.6},
    Variation.POPCORN_PARAM: {'popcorn_cx': 0.3, 'popcorn_cy': 0.5},
    Variation.RINGS_PARAM: {'rings_c': 0.4},
    Variation.FAN_PARAM: {'fan_c': 0.3, 'fan_f': 0.5},
    Variation.WAVES2: {'waves2_scalex': 0.05, 'waves2_scaley': 0.05,
                       'waves2_freqx': 7.0, 'waves2_freqy': 13.0},
    Variation.JULIAN: {'julian_power': 4.0, 'julian_dist': 1.0},
    Variation.JULIASCOPE: {'julian_power': 4.0, 'julian_dist': 1.0},
    Variation.SPLITS: {'splits_x': 0.5, 'splits_y': 0.3},
    Variation.CURL: {'curl_c1': 0.5, 'curl_c2': -0.3},
    Variation.RECTANGLES: {'rect_x': 0.8, 'rect_y': 0.6},
    Variation.CHECKS: {'check_size': 1.5, 'check_x': 0.2, 'check_y': 0.1},
    Variation.HEX_MODULUS: {'hex_size': 1.0},
    Variation.KALEIDOSCOPE: {'kal_pull': 0.3, 'kal_rotate': 1.0, 'kal_n': 6.0},
    Variation.ICON: {'icon_degree': 4.0, 'icon_lambda': 1.5,
                     'icon_alpha': 0.5, 'icon_beta': 0.3,
                     'icon_gamma': 0.1, 'icon_omega': 0.2},
    Variation.SATTRACTOR: {'sat_m': 6.0},
    Variation.WALLPAPER: {'wallpaper_group': 0.0},
    Variation.FRIEZE: {'frieze_group': 0.0},
    # batch 2 parametric variations
    Variation.RADIAL_BLUR: {'radial_blur_angle': 0.5},
    Variation.PERSPECTIVE: {'perspective_angle': 0.8, 'perspective_dist': 2.0},
    Variation.SUPER_SHAPE: {'super_shape_rnd': 0.3, 'super_shape_m': 4.0,
                            'super_shape_n1': 2.0, 'super_shape_n2': 2.0,
                            'super_shape_n3': 2.0, 'super_shape_holes': 0.0},
    Variation.PIE: {'pie_slices': 6.0, 'pie_rotation': 0.0, 'pie_thickness': 0.5},
    Variation.PARABOLA: {'parabola_height': 1.0, 'parabola_width': 1.0},
    Variation.CONIC: {'conic_eccentricity': 1.0, 'conic_holes': 0.0},
    Variation.ESCHER: {'escher_beta': 0.3},
    Variation.OSCILLOSCOPE: {'osc_separation': 1.0, 'osc_frequency': 3.14159,
                             'osc_amplitude': 1.0, 'osc_damping': 0.1},
    Variation.CURVE: {'curve_xamp': 0.5, 'curve_yamp': 0.3,
                      'curve_xlength': 1.0, 'curve_ylength': 1.0},
    Variation.WEDGE_JULIA: {'wedge_julia_angle': 0.3, 'wedge_julia_count': 2.0,
                            'wedge_julia_power': 4.0, 'wedge_julia_dist': 1.0},
    Variation.WEDGE: {'wedge_angle': 0.3, 'wedge_hole': 0.0,
                      'wedge_count': 2.0, 'wedge_swirl': 0.5},
    Variation.WEDGE_SPH: {'wedge_sph_angle': 0.3, 'wedge_sph_hole': 0.0,
                          'wedge_sph_count': 2.0, 'wedge_sph_swirl': 0.5},
    Variation.LAZYSUSAN: {'lazysusan_x': 0.1, 'lazysusan_y': 0.1,
                          'lazysusan_spin': 0.5, 'lazysusan_space': 0.2,
                          'lazysusan_twist': 0.3},
    Variation.MODULUS_FUNC: {'modulus_x': 0.5, 'modulus_y': 0.5},
    Variation.BENT2: {'bent2_x': 1.5, 'bent2_y': -0.5},
    Variation.BIPOLAR: {'bipolar_shift': 0.3},
    Variation.FLUX: {'flux_spread': 0.5},
    Variation.SPLIT: {'split_xsize': 0.5, 'split_ysize': 0.5},
    Variation.SEPARATION: {'separation_x': 0.5, 'separation_y': 0.5,
                           'separation_xinside': 0.2, 'separation_yinside': 0.3},
    Variation.POPCORN2: {'popcorn2_x': 0.1, 'popcorn2_y': 0.1, 'popcorn2_c': 3.0},
}

# Variations that use randomness internally
RANDOM_VARIATIONS = {Variation.JULIA, Variation.SATTRACTOR, Variation.WALLPAPER,
                     Variation.FRIEZE, Variation.JULIAN, Variation.JULIASCOPE,
                     Variation.ICON, Variation.CPOW,
                     # batch 2 RNG variations
                     Variation.BLUR, Variation.GAUSSIAN_BLUR, Variation.RADIAL_BLUR,
                     Variation.NOISE, Variation.PIE, Variation.ARCH, Variation.PARABOLA,
                     Variation.RAYS, Variation.CONIC, Variation.SQUARE,
                     Variation.TWINTRIAN, Variation.SUPER_SHAPE,
                     Variation.WEDGE_JULIA, Variation.BLADE, Variation.FLOWER,
                     Variation.LISSAJOUS, Variation.EPISPIRAL, Variation.BOARDERS}


def generate_transform_golden():
    """Generate golden master for all CPU variations.

    Uses GOLDEN_POINTS as baseline plus per-variation edge case points
    from variation_fixtures.VAR_POINTS. This ensures golden masters
    cover the same edge cases used for GPU/CPU comparison.
    """
    data = {}
    for var_idx in range(NUM_VARIATIONS):
        params = PARAM_FIXTURES.get(var_idx, {})
        cpu_mod._current_var_params = params
        affine = TEST_AFFINE if var_idx in AFFINE_VARIATIONS else None

        # Use variation-specific points if available, plus generic
        points = list(GOLDEN_POINTS)
        if var_idx in EXTRA_VAR_POINTS:
            existing = set(points)
            for pt in EXTRA_VAR_POINTS[var_idx]:
                if pt not in existing:
                    points.append(pt)
                    existing.add(pt)

        for px, py in points:
            if var_idx in RANDOM_VARIATIONS:
                np.random.seed(42)
            rx, ry = apply_variation_cpu(var_idx, px, py, 1.0, affine)
            key = f'v{var_idx}_{px}_{py}'
            data[key] = np.array([rx, ry], dtype=np.float64)

    return data


def make_hex_pattern(size=128):
    """6 dots in a hexagon — should score high on 6-fold rotational."""
    grid = np.zeros((size, size), dtype=np.float64)
    cx, cy = size // 2, size // 2
    r = size // 4
    for k in range(6):
        angle = k * np.pi / 3
        x = int(cx + r * np.cos(angle))
        y = int(cy + r * np.sin(angle))
        # Draw a small blob at each point
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if 0 <= x+dx < size and 0 <= y+dy < size:
                    grid[y+dy, x+dx] = 1.0
    return grid


def make_cross_pattern(size=128):
    """Cross — 4-fold rotational + reflective."""
    grid = np.zeros((size, size), dtype=np.float64)
    cx, cy = size // 2, size // 2
    thickness = 3
    arm = size // 4
    grid[cy-thickness:cy+thickness, cx-arm:cx+arm] = 1.0
    grid[cy-arm:cy+arm, cx-thickness:cx+thickness] = 1.0
    return grid


def make_diagonal(size=128):
    """Diagonal line — low symmetry."""
    grid = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for d in range(-2, 3):
            j = i + d
            if 0 <= j < size:
                grid[i, j] = 1.0
    return grid


def make_rings(size=128):
    """Concentric rings — high radial symmetry."""
    grid = np.zeros((size, size), dtype=np.float64)
    cx, cy = size // 2, size // 2
    ys, xs = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    # Rings at r=10, 20, 30, 40
    for r in [10, 20, 30, 40]:
        grid[(dist >= r-1.5) & (dist <= r+1.5)] = 1.0
    return grid


def make_uniform(size=128):
    """Uniform fill — high coverage, low symmetry_max."""
    return np.ones((size, size), dtype=np.float64)


def make_single_point(size=128):
    """Single pixel — degenerate."""
    grid = np.zeros((size, size), dtype=np.float64)
    grid[size//2, size//2] = 1.0
    return grid


SYMMETRY_PATTERNS = {
    'hex_6fold': make_hex_pattern,
    'cross_4fold': make_cross_pattern,
    'diagonal_line': make_diagonal,
    'concentric_rings': make_rings,
    'uniform_fill': make_uniform,
    'single_point': make_single_point,
}


def generate_symmetry_golden():
    """Generate golden master for symmetry metric patterns."""
    data = {}
    for name, make_fn in SYMMETRY_PATTERNS.items():
        grid = make_fn()
        scores = symmetry_scores(grid.astype(np.uint32))
        # Store both the pattern and the scores
        data[f'{name}_grid'] = grid
        # Store scores as individual keys since npz doesn't handle dicts
        for k, v in scores.items():
            data[f'{name}_{k}'] = np.array(v, dtype=np.float64)
        print(f'{name}:')
        for k, v in sorted(scores.items()):
            print(f'  {k}: {v}')
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true',
                        help='Compare to existing golden files')
    args = parser.parse_args()

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    # Transforms
    print('=== Transform Golden Masters ===')
    transform_data = generate_transform_golden()
    transform_path = GOLDEN_DIR / 'transforms.npz'
    if args.verify and transform_path.exists():
        golden = dict(np.load(transform_path))
        mismatches = 0
        for key in transform_data:
            if key not in golden:
                print(f'  NEW: {key}')
                mismatches += 1
            elif not np.allclose(transform_data[key], golden[key], atol=1e-6):
                print(f'  CHANGED: {key}: {golden[key]} → {transform_data[key]}')
                mismatches += 1
        print(f'{len(transform_data)} transforms, {mismatches} mismatches')
    else:
        np.savez(transform_path, **transform_data)
        print(f'Saved {len(transform_data)} transforms to {transform_path}')

    # Symmetry
    print('\n=== Symmetry Golden Masters ===')
    symmetry_data = generate_symmetry_golden()
    symmetry_path = GOLDEN_DIR / 'symmetry.npz'
    if args.verify and symmetry_path.exists():
        golden = dict(np.load(symmetry_path))
        mismatches = 0
        for key in symmetry_data:
            if key not in golden:
                print(f'  NEW: {key}')
                mismatches += 1
            elif not np.allclose(symmetry_data[key], golden[key], atol=1e-4):
                print(f'  CHANGED: {key}: {golden[key]} → {symmetry_data[key]}')
                mismatches += 1
        print(f'{len(symmetry_data)} values, {mismatches} mismatches')
    else:
        np.savez(symmetry_path, **symmetry_data)
        print(f'Saved {len(symmetry_data)} values to {symmetry_path}')


if __name__ == '__main__':
    main()
