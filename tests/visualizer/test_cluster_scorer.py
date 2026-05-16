"""
Tests for cluster_scorer: color-index clustering and transform-based scoring.
"""

import numpy as np
import pytest

from flame_sheep.cluster_scorer import (
    _find_c_clusters,
    _sobel_magnitude,
    _cluster_symmetry,
    _cluster_edges,
    score_from_clusters,
    score_from_transform_hits,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ----------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------

class TestSobelMagnitude:
    def test_flat_is_zero(self):
        flat = np.ones((32, 32), dtype=np.float64)
        grad = _sobel_magnitude(flat)
        np.testing.assert_allclose(grad, 0.0, atol=1e-10)

    def test_vertical_edge(self):
        img = np.zeros((32, 32), dtype=np.float64)
        img[:, 16:] = 1.0
        grad = _sobel_magnitude(img)
        # Maximum gradient should be at column 16
        assert grad[:, 16].max() > 0

    def test_small_array(self):
        tiny = np.ones((2, 2), dtype=np.float64)
        grad = _sobel_magnitude(tiny)
        np.testing.assert_array_equal(grad, 0.0)


class TestClusterSymmetry:
    def test_symmetric_circle(self):
        """A circle should have high symmetry."""
        mask = np.zeros((64, 64), dtype=bool)
        cy, cx = 32, 32
        ys, xs = np.mgrid[0:64, 0:64]
        mask[((ys - cy)**2 + (xs - cx)**2) < 15**2] = True
        sym = _cluster_symmetry(mask)
        assert sym['rotational'] > 0.8
        assert sym['reflective'] > 0.8

    def test_asymmetric_blob(self, rng):
        """A random blob should have lower symmetry than a circle."""
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:30, 5:15] = True
        mask[40:55, 35:60] = True
        sym = _cluster_symmetry(mask)
        # Not perfectly symmetric — below circle's >0.8
        assert sym['rotational'] < 0.7

    def test_empty_mask(self):
        mask = np.zeros((32, 32), dtype=bool)
        sym = _cluster_symmetry(mask)
        assert sym['rotational'] == 0.0
        assert sym['reflective'] == 0.0


class TestFindCClusters:
    def test_two_distinct_peaks(self):
        """Two groups of pixels at c=0.2 and c=0.8 should produce 2 clusters."""
        h, w = 64, 64
        avg_c = np.zeros((h, w), dtype=np.float32)
        hit_mask = np.zeros((h, w), dtype=bool)
        # Group A at c=0.2
        avg_c[:32, :] = 0.2
        hit_mask[:32, :] = True
        # Group B at c=0.8
        avg_c[32:, :] = 0.8
        hit_mask[32:, :] = True
        clusters = _find_c_clusters(avg_c, hit_mask)
        assert len(clusters) == 2

    def test_uniform_one_cluster(self):
        """Uniform c values should produce one cluster."""
        h, w = 64, 64
        avg_c = np.full((h, w), 0.5, dtype=np.float32)
        hit_mask = np.ones((h, w), dtype=bool)
        clusters = _find_c_clusters(avg_c, hit_mask)
        assert len(clusters) >= 1

    def test_too_few_pixels_returns_empty(self):
        h, w = 64, 64
        avg_c = np.zeros((h, w), dtype=np.float32)
        hit_mask = np.zeros((h, w), dtype=bool)
        hit_mask[0, :10] = True  # only 10 pixels
        avg_c[0, :10] = 0.5
        clusters = _find_c_clusters(avg_c, hit_mask)
        assert len(clusters) == 0

    def test_cluster_masks_are_subsets_of_hit_mask(self, rng):
        h, w = 64, 64
        avg_c = rng.uniform(0, 1, (h, w)).astype(np.float32)
        hit_mask = rng.random((h, w)) > 0.3
        clusters = _find_c_clusters(avg_c, hit_mask)
        for cl in clusters:
            # Every pixel in cluster mask should also be in hit_mask
            assert np.all(cl['mask'] <= hit_mask)


# ----------------------------------------------------------------
# score_from_clusters
# ----------------------------------------------------------------

class TestScoreFromClusters:
    def test_output_keys(self, rng):
        hit_grid = rng.uniform(0, 1000, (64, 64))
        color_grid = hit_grid * rng.uniform(0, 1, (64, 64))
        scores = score_from_clusters(hit_grid, color_grid)
        expected = {'cl_coverage', 'cl_edge_sharpness', 'cl_symmetry_best',
                    'cl_cluster_count', 'cl_dominance', 'cl_balance'}
        assert expected <= set(scores.keys())

    def test_values_bounded(self, rng):
        hit_grid = rng.uniform(0, 1000, (64, 64))
        color_grid = hit_grid * rng.uniform(0, 1, (64, 64))
        scores = score_from_clusters(hit_grid, color_grid)
        assert 0.0 <= scores['cl_coverage'] <= 1.0
        assert 0.0 <= scores['cl_edge_sharpness'] <= 1.0
        assert 0.0 <= scores['cl_symmetry_best'] <= 1.0
        assert 0.0 <= scores['cl_dominance'] <= 1.0
        assert 0.0 <= scores['cl_balance'] <= 1.0

    def test_empty_histogram(self):
        hit_grid = np.zeros((64, 64))
        color_grid = np.zeros((64, 64))
        scores = score_from_clusters(hit_grid, color_grid)
        assert scores['cl_coverage'] == 0.0
        assert scores['cl_cluster_count'] == 0

    def test_store_detail(self, rng):
        hit_grid = rng.uniform(0, 1000, (64, 64))
        color_grid = hit_grid * rng.uniform(0, 1, (64, 64))
        scores = score_from_clusters(hit_grid, color_grid, store_detail=True)
        assert 'cluster_detail' in scores
        import json
        detail = json.loads(scores['cluster_detail'])
        assert isinstance(detail, list)

    def test_balanced_image_high_balance(self):
        """A centered, uniform image should score high balance."""
        h, w = 64, 64
        hit_grid = np.zeros((h, w), dtype=np.float64)
        # Fill center symmetrically
        hit_grid[16:48, 16:48] = 100.0
        color_grid = hit_grid * 0.5
        scores = score_from_clusters(hit_grid, color_grid)
        assert scores['cl_balance'] > 0.8


# ----------------------------------------------------------------
# score_from_transform_hits
# ----------------------------------------------------------------

class TestScoreFromTransformHits:
    def test_output_keys(self, rng):
        h, w, n = 64, 64, 3
        hit_grid = rng.uniform(0, 1000, (h, w))
        transform_hits = rng.integers(0, 500, (h, w, n), dtype=np.uint32)
        scores = score_from_transform_hits(hit_grid, transform_hits)
        expected = {'tf_coverage', 'tf_n_clusters', 'tf_avg_purity',
                    'tf_symmetry_best', 'tf_balance', 'tf_separation'}
        assert expected <= set(scores.keys())

    def test_values_bounded(self, rng):
        h, w, n = 64, 64, 3
        hit_grid = rng.uniform(100, 1000, (h, w))
        transform_hits = rng.integers(10, 500, (h, w, n), dtype=np.uint32)
        scores = score_from_transform_hits(hit_grid, transform_hits)
        assert 0.0 <= scores['tf_coverage'] <= 1.0
        assert 0.0 <= scores['tf_avg_purity'] <= 1.0
        assert 0.0 <= scores['tf_balance'] <= 1.0
        assert 0.0 <= scores['tf_separation'] <= 1.0

    def test_empty_histogram(self):
        hit_grid = np.zeros((64, 64))
        transform_hits = np.zeros((64, 64, 3), dtype=np.uint32)
        scores = score_from_transform_hits(hit_grid, transform_hits)
        assert scores['tf_coverage'] == 0.0
        assert scores['tf_n_clusters'] == 0

    def test_single_transform(self, rng):
        """With only 1 transform, n_transforms < 2 returns early."""
        h, w = 64, 64
        hit_grid = rng.uniform(100, 1000, (h, w))
        transform_hits = rng.integers(10, 500, (h, w, 1), dtype=np.uint32)
        scores = score_from_transform_hits(hit_grid, transform_hits)
        assert scores['tf_n_clusters'] == 0

    def test_pure_clusters_high_purity(self):
        """When each transform owns its own region exclusively, purity should be high."""
        h, w, n = 64, 64, 2
        hit_grid = np.ones((h, w), dtype=np.float64) * 100
        transform_hits = np.zeros((h, w, n), dtype=np.uint32)
        # Left half = transform 0, right half = transform 1
        transform_hits[:, :32, 0] = 100
        transform_hits[:, 32:, 1] = 100
        scores = score_from_transform_hits(hit_grid, transform_hits)
        assert scores['tf_avg_purity'] > 0.9
        assert scores['tf_n_clusters'] == 2

    def test_mixed_transforms_lower_purity(self, rng):
        """When transforms overlap heavily, purity should be lower."""
        h, w, n = 64, 64, 3
        hit_grid = np.ones((h, w), dtype=np.float64) * 300
        # All transforms hit everywhere equally
        transform_hits = np.full((h, w, n), 100, dtype=np.uint32)
        scores = score_from_transform_hits(hit_grid, transform_hits)
        # Purity should be ~1/3 (each pixel's dominant is arbitrary)
        assert scores['tf_avg_purity'] < 0.5

    def test_clean_separation_high_score(self):
        """Well-separated clusters should have high separation score."""
        h, w, n = 64, 64, 2
        hit_grid = np.ones((h, w), dtype=np.float64) * 100
        transform_hits = np.zeros((h, w, n), dtype=np.uint32)
        transform_hits[:, :30, 0] = 100
        transform_hits[:, 34:, 1] = 100
        # Gap in columns 30-33 (no hits)
        hit_grid[:, 30:34] = 0
        scores = score_from_transform_hits(hit_grid, transform_hits)
        assert scores['tf_separation'] > 0.5
