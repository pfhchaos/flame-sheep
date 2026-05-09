"""Golden master regression tests.

Compares current transform and symmetry metric outputs against
stored baselines. Run tools/generate_golden_masters.py to regenerate
after intentional changes. Use the viewer to verify visually before
checking in new golden files.
"""

from pathlib import Path

import numpy as np
import pytest

from flame_sheep.variations import (
    Variation, NUM_VARIATIONS, apply_variation_cpu,
)
import flame_sheep.variations._cpu as cpu_mod
from flame_sheep.symmetry import symmetry_scores

GOLDEN_DIR = Path(__file__).parent / 'golden'

# Same fixtures as the generator — must match exactly
TEST_POINTS = [
    ( 1.0,   0.5),
    ( 0.0,   0.0),
    (-0.3,   0.7),
    ( 0.5,  -0.5),
    ( 2.0,   1.0),
    (-0.8,  -0.4),
    ( 0.05,  0.03),
    ( 0.95,  0.31),
    ( 1.05,  0.31),
    ( 0.49,  0.49),
    ( 0.51,  0.51),
    (-1.5,   0.8),
]

# Fixed affine for affine-reading variations (15/17/21/22)
TEST_AFFINE = np.array([0.8, 0.5, 0.3, -0.2, 0.6, 0.5], dtype=np.float32)

AFFINE_VARIATIONS = {Variation.WAVES, Variation.POPCORN,
                     Variation.RINGS, Variation.FAN}

PARAM_FIXTURES = {
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
    # batch 2 parametric
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

RANDOM_VARIATIONS = {Variation.JULIA, Variation.SATTRACTOR, Variation.WALLPAPER,
                     Variation.FRIEZE, Variation.JULIAN, Variation.JULIASCOPE,
                     Variation.ICON, Variation.CPOW,
                     Variation.BLUR, Variation.GAUSSIAN_BLUR, Variation.RADIAL_BLUR,
                     Variation.NOISE, Variation.PIE, Variation.ARCH, Variation.PARABOLA,
                     Variation.RAYS, Variation.CONIC, Variation.SQUARE,
                     Variation.TWINTRIAN, Variation.SUPER_SHAPE,
                     Variation.WEDGE_JULIA, Variation.BLADE, Variation.FLOWER,
                     Variation.LISSAJOUS, Variation.EPISPIRAL, Variation.BOARDERS}


@pytest.fixture(scope='module')
def transform_golden():
    path = GOLDEN_DIR / 'transforms.npz'
    if not path.exists():
        pytest.skip('Golden master not generated — run tools/generate_golden_masters.py')
    return dict(np.load(path))


@pytest.fixture(scope='module')
def symmetry_golden():
    path = GOLDEN_DIR / 'symmetry.npz'
    if not path.exists():
        pytest.skip('Golden master not generated — run tools/generate_golden_masters.py')
    return dict(np.load(path))


class TestTransformGoldenMasters:
    """Verify CPU variation outputs match stored golden values."""

    @pytest.mark.parametrize('var_idx', range(NUM_VARIATIONS))
    def test_variation_matches_golden(self, var_idx, transform_golden):
        params = PARAM_FIXTURES.get(var_idx, {})
        cpu_mod._current_var_params = params
        affine = TEST_AFFINE if var_idx in AFFINE_VARIATIONS else None

        # Test all points that have golden data for this variation
        tested = 0
        for key, expected in transform_golden.items():
            if not key.startswith(f'v{var_idx}_'):
                continue
            # Parse point from key: v{idx}_{px}_{py}
            parts = key.split('_', 1)[1]  # everything after vN_
            # Handle negative numbers: split on _ but rejoin negative signs
            coords = parts.split('_')
            # Reconstruct px, py from the key
            # Keys look like: v0_1.0_0.5 or v0_-0.3_0.7
            i = 0
            vals = []
            while i < len(coords):
                if coords[i] == '' and i + 1 < len(coords):
                    # Negative number: empty string before the minus
                    vals.append(float('-' + coords[i + 1]))
                    i += 2
                else:
                    vals.append(float(coords[i]))
                    i += 1
            if len(vals) != 2:
                continue
            px, py = vals

            if var_idx in RANDOM_VARIATIONS:
                np.random.seed(42)
            rx, ry = apply_variation_cpu(var_idx, px, py, 1.0, affine)

            np.testing.assert_allclose(
                [rx, ry], expected, atol=1e-6,
                err_msg=f'Variation {var_idx} at ({px}, {py})')
            tested += 1

        assert tested > 0, f'No golden data for variation {var_idx}'


class TestSymmetryGoldenMasters:
    """Verify symmetry metrics match stored golden values for known patterns."""

    def _scores(self, golden, pattern_name):
        """Extract score dict from golden data for a pattern."""
        keys = [k for k in golden if k.startswith(f'{pattern_name}_')
                and not k.endswith('_grid')]
        return {k.replace(f'{pattern_name}_', ''): float(golden[k])
                for k in keys}

    def test_hex_6fold_rotational(self, symmetry_golden):
        scores = self._scores(symmetry_golden, 'hex_6fold')
        assert scores['rotational_n'] == 6
        assert scores['rotational_best'] > 0.5

    def test_cross_4fold_rotational(self, symmetry_golden):
        scores = self._scores(symmetry_golden, 'cross_4fold')
        assert scores['rotational_n'] == 4
        assert scores['rotational_best'] > 0.5
        assert scores['reflective_best'] > 0.5

    def test_concentric_rings_high_radial(self, symmetry_golden):
        scores = self._scores(symmetry_golden, 'concentric_rings')
        assert scores['radial'] > 0.9

    def test_uniform_fill_low_symmetry_max(self, symmetry_golden):
        scores = self._scores(symmetry_golden, 'uniform_fill')
        assert scores['symmetry_max'] < 0.01

    def test_single_point_low_symmetry_max(self, symmetry_golden):
        scores = self._scores(symmetry_golden, 'single_point')
        assert scores['symmetry_max'] < 0.01

    def test_diagonal_has_2fold_rotation(self, symmetry_golden):
        scores = self._scores(symmetry_golden, 'diagonal_line')
        assert scores['rotational_n'] == 2

    @pytest.mark.parametrize('pattern', [
        'hex_6fold', 'cross_4fold', 'diagonal_line',
        'concentric_rings', 'uniform_fill', 'single_point',
    ])
    def test_scores_match_golden(self, pattern, symmetry_golden):
        """All metric values match stored golden within tolerance."""
        grid = symmetry_golden[f'{pattern}_grid']
        current = symmetry_scores(grid.astype(np.uint32))

        for key, expected in self._scores(symmetry_golden, pattern).items():
            actual = current.get(key, 0.0)
            if isinstance(expected, float) and abs(expected) > 1e-6:
                np.testing.assert_allclose(
                    actual, expected, rtol=0.01,
                    err_msg=f'{pattern}.{key}')
