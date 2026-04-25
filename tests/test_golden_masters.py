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

# Same fixtures as the generator
TEST_POINTS = [
    (1.0, 0.5),
    (0.0, 0.0),
    (-0.3, 0.7),
    (0.5, -0.5),
    (2.0, 1.0),
]

PARAM_FIXTURES = {
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
}

RANDOM_VARIATIONS = {Variation.SATTRACTOR, Variation.WALLPAPER, Variation.FRIEZE,
                     Variation.JULIAN, Variation.JULIASCOPE}


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

        for px, py in TEST_POINTS:
            key = f'v{var_idx}_{px}_{py}'
            if key not in transform_golden:
                pytest.skip(f'No golden data for {key}')

            if var_idx in RANDOM_VARIATIONS:
                np.random.seed(42)
            rx, ry = apply_variation_cpu(var_idx, px, py, 1.0)
            expected = transform_golden[key]

            np.testing.assert_allclose(
                [rx, ry], expected, atol=1e-6,
                err_msg=f'Variation {var_idx} at ({px}, {py})')


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
