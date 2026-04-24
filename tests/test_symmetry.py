"""Tests for symmetry detection on histograms."""

import numpy as np
import pytest

from flame_sheep.symmetry import (
    symmetry_scores,
    _rotational_symmetry,
    _reflective_symmetry,
    _radial_symmetry,
    _periodic_structure,
    _fractal_dimension,
)


def _make_grid(size: int = 64) -> np.ndarray:
    """Empty grid."""
    return np.zeros((size, size), dtype=np.float64)


class TestRotationalSymmetry:

    def test_circle_is_symmetric(self):
        """A centered circle has high rotational symmetry."""
        grid = _make_grid(64)
        ys, xs = np.mgrid[0:64, 0:64]
        dist = np.sqrt((xs - 32)**2 + (ys - 32)**2)
        grid[dist < 20] = 1.0
        score, n = _rotational_symmetry(np.log1p(grid))
        assert score > 0.7

    def test_random_is_not_symmetric(self):
        rng = np.random.default_rng(42)
        grid = rng.random((64, 64))
        score, n = _rotational_symmetry(grid)
        assert score < 0.3

    def test_4fold_detected(self):
        """A pattern with 4-fold rotational symmetry should score high at n=4."""
        grid = _make_grid(64)
        # Draw a cross pattern — 4-fold symmetric
        grid[28:36, :] = 1.0
        grid[:, 28:36] = 1.0
        score, n = _rotational_symmetry(np.log1p(grid))
        assert score > 0.5
        assert n in (2, 4)  # 4-fold also passes 2-fold

    def test_empty_grid(self):
        grid = _make_grid(64)
        score, n = _rotational_symmetry(grid)
        # All zeros — correlation undefined but shouldn't crash
        assert score >= 0.0


class TestReflectiveSymmetry:

    def test_vertical_mirror(self):
        grid = _make_grid(64)
        # Fill left half with pattern, mirror to right
        rng = np.random.default_rng(1)
        grid[:, :32] = rng.random((64, 32))
        grid[:, 32:] = grid[:, 31::-1]
        score, angle = _reflective_symmetry(np.log1p(grid))
        assert score > 0.5

    def test_random_not_reflective(self):
        rng = np.random.default_rng(42)
        grid = rng.random((64, 64))
        score, angle = _reflective_symmetry(grid)
        assert score < 0.3

    def test_horizontal_mirror(self):
        grid = _make_grid(64)
        # Symmetric about horizontal axis
        grid[:32, :] = np.random.default_rng(1).random((32, 64))
        grid[32:, :] = grid[31::-1, :]
        score, angle = _reflective_symmetry(np.log1p(grid))
        assert score > 0.5


class TestRadialSymmetry:

    def test_concentric_rings(self):
        grid = _make_grid(64)
        ys, xs = np.mgrid[0:64, 0:64]
        dist = np.sqrt((xs - 32)**2 + (ys - 32)**2)
        # Alternating rings — strong radial pattern
        grid = (np.sin(dist * 0.8) > 0).astype(np.float64)
        score = _radial_symmetry(np.log1p(grid))
        assert score > 0.2

    def test_random_not_radial(self):
        rng = np.random.default_rng(42)
        grid = rng.random((64, 64))
        score = _radial_symmetry(grid)
        assert score < 0.5

    def test_single_point(self):
        grid = _make_grid(64)
        grid[32, 32] = 1.0
        score = _radial_symmetry(np.log1p(grid))
        # Single point has trivial radial symmetry
        assert isinstance(score, float)


class TestPeriodicStructure:

    def test_stripes_are_periodic(self):
        grid = _make_grid(64)
        # Vertical stripes with period 8
        for i in range(0, 64, 8):
            grid[:, i:i+2] = 1.0
        score, freq = _periodic_structure(np.log1p(grid))
        assert score > 0.3

    def test_grid_is_periodic(self):
        grid = _make_grid(64)
        # Checkerboard with period 8
        for i in range(0, 64, 8):
            for j in range(0, 64, 8):
                grid[i:i+4, j:j+4] = 1.0
        score, freq = _periodic_structure(np.log1p(grid))
        assert score > 0.3

    def test_random_not_periodic(self):
        rng = np.random.default_rng(42)
        grid = rng.random((64, 64))
        score, freq = _periodic_structure(grid)
        assert score < 0.3

    def test_empty_not_periodic(self):
        grid = _make_grid(64)
        score, freq = _periodic_structure(grid)
        assert score == 0.0


class TestFractalDimension:

    def test_line_low_dimension(self):
        grid = _make_grid(128)
        # Diagonal line — D ~ 1.0
        for i in range(128):
            grid[i, i] = 1
        fd = _fractal_dimension(grid)
        assert 0.8 < fd < 1.3

    def test_filled_square_high_dimension(self):
        grid = np.ones((128, 128), dtype=np.uint32)
        fd = _fractal_dimension(grid)
        assert 1.8 < fd < 2.1

    def test_sierpinski_middle_dimension(self):
        """Sierpinski triangle has D ~ 1.585."""
        size = 128
        grid = np.zeros((size, size), dtype=np.uint32)
        # Generate Sierpinski via chaos game
        rng = np.random.default_rng(42)
        vertices = np.array([[0, size-1], [size-1, size-1], [size//2, 0]])
        x, y = size // 2, size // 2
        for _ in range(50000):
            v = vertices[rng.integers(3)]
            x = (x + v[0]) // 2
            y = (y + v[1]) // 2
            if 0 <= x < size and 0 <= y < size:
                grid[y, x] = 1
        fd = _fractal_dimension(grid)
        assert 1.3 < fd < 1.9  # should be near 1.585

    def test_empty_returns_zero(self):
        grid = _make_grid(64).astype(np.uint32)
        fd = _fractal_dimension(grid)
        assert fd == 0.0


class TestSymmetryScores:

    def test_returns_all_keys(self):
        grid = np.ones((32, 32), dtype=np.uint32)
        scores = symmetry_scores(grid)
        expected = {'rotational_best', 'rotational_n', 'reflective_best',
                    'reflective_angle', 'radial', 'periodic', 'periodic_freq',
                    'fractal_dim', 'self_similarity', 'symmetry_max'}
        assert set(scores.keys()) == expected

    def test_empty_grid_all_zero(self):
        grid = np.zeros((32, 32), dtype=np.uint32)
        scores = symmetry_scores(grid)
        assert scores['symmetry_max'] == 0.0
        assert scores['fractal_dim'] == 0.0

    def test_symmetry_max_is_max(self):
        rng = np.random.default_rng(42)
        grid = rng.integers(0, 100, (64, 64), dtype=np.uint32)
        scores = symmetry_scores(grid)
        assert scores['symmetry_max'] == max(
            scores['rotational_best'],
            scores['reflective_best'],
            scores['radial'],
            scores['periodic'],
        )

    def test_all_scores_in_range(self):
        rng = np.random.default_rng(42)
        grid = rng.integers(0, 100, (64, 64), dtype=np.uint32)
        scores = symmetry_scores(grid)
        for key in ('rotational_best', 'reflective_best', 'radial',
                     'periodic', 'symmetry_max'):
            assert 0.0 <= scores[key] <= 1.0, f'{key}={scores[key]}'
        assert 0.0 <= scores['fractal_dim'] <= 2.0
