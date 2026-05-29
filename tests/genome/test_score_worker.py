"""
Tests for cpu_score_worker._score_genome: scoring pipeline, histogram handling.
"""

import io
import struct
import zlib

import numpy as np
import pytest
from PIL import Image

from flame_sheep.genome.score_worker import _score_genome
from flame_sheep.genome.scoring.scoring_channels import pack_histogram, pack_static_histogram


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _make_png(w=512, h=512, mode='RGB'):
    """Generate a minimal PNG blob."""
    img = Image.new(mode, (w, h), color=(128,) * len(mode))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def _make_hist_static(h=512, w=512, rng=None):
    """Make a v2-style hist_static blob (no header, raw zlib)."""
    if rng is None:
        rng = np.random.default_rng(42)
    hits = rng.integers(0, 10000, (h, w), dtype=np.uint32)
    colors = rng.integers(0, 10_000_000, (h, w), dtype=np.uint32)
    return zlib.compress(hits.tobytes() + colors.tobytes())


def _make_hist_transform_v2(h=512, w=512, n_transforms=3, rng=None):
    """Make a v2-style hist_transform blob (no header)."""
    if rng is None:
        rng = np.random.default_rng(42)
    data = rng.integers(0, 500, (h, w, n_transforms), dtype=np.uint32)
    return zlib.compress(data.tobytes())


def _make_hist_transform_v4(h=512, w=512, n_transforms=3, rng=None):
    """Make a v4-style hist_transform blob (with dimension header)."""
    if rng is None:
        rng = np.random.default_rng(42)
    data = rng.integers(0, 500, (h, w, n_transforms), dtype=np.uint32)
    return pack_histogram(data.reshape(h, w * n_transforms))


# ----------------------------------------------------------------
# Basic scoring
# ----------------------------------------------------------------

class TestScoreGenome:
    def test_returns_dict(self):
        static = _make_png()
        scores = _score_genome(static, None)
        assert isinstance(scores, dict)

    def test_includes_detail_sensitivity(self):
        static = _make_png()
        scores = _score_genome(static, None)
        assert 'detail_sensitivity' in scores

    def test_with_swept(self):
        static = _make_png()
        swept = _make_png(mode='L')
        scores = _score_genome(static, swept)
        assert isinstance(scores, dict)


# ----------------------------------------------------------------
# Histogram-based scoring
# ----------------------------------------------------------------

class TestHistogramScoring:
    def test_with_hist_static(self, rng):
        static = _make_png()
        hist = _make_hist_static(rng=rng)
        scores = _score_genome(static, None, hist_static=hist)
        # Should include cluster scores
        assert 'cl_coverage' in scores

    def test_with_hist_static_and_transform_v2(self, rng):
        """v2 transform blob (no header) should work."""
        static = _make_png()
        hist_static = _make_hist_static(rng=rng)
        hist_tf = _make_hist_transform_v2(n_transforms=3, rng=rng)
        scores = _score_genome(static, None,
                               hist_static=hist_static,
                               hist_transform=hist_tf)
        assert 'tf_n_clusters' in scores

    def test_with_hist_static_and_transform_v4(self, rng):
        """v4 transform blob (with header) should work — the bug that was fixed."""
        static = _make_png()
        hist_static = _make_hist_static(rng=rng)
        # Pack as v4: use pack_histogram which adds dimension header
        h, w, n_transforms = 512, 512, 4
        data = rng.integers(0, 500, (h, w, n_transforms), dtype=np.uint32)
        header = struct.pack('II', h, w)
        hist_tf = zlib.compress(header + data.tobytes())
        scores = _score_genome(static, None,
                               hist_static=hist_static,
                               hist_transform=hist_tf)
        assert 'tf_n_clusters' in scores

    def test_header_detection_does_not_strip_v2(self, rng):
        """v2 blobs where first 8 bytes happen to NOT match dimensions
        should not have header stripped."""
        static = _make_png()
        hist_static = _make_hist_static(rng=rng)
        hist_tf = _make_hist_transform_v2(n_transforms=6, rng=rng)
        # This should parse correctly as v2 (no header)
        scores = _score_genome(static, None,
                               hist_static=hist_static,
                               hist_transform=hist_tf)
        assert scores['tf_n_clusters'] >= 0

    def test_different_transform_counts(self, rng):
        """Should handle varying numbers of transforms."""
        static = _make_png()
        hist_static = _make_hist_static(rng=rng)
        for n_tf in [2, 3, 5, 6]:
            hist_tf = _make_hist_transform_v2(n_transforms=n_tf, rng=rng)
            scores = _score_genome(static, None,
                                   hist_static=hist_static,
                                   hist_transform=hist_tf)
            assert 'tf_n_clusters' in scores


# ----------------------------------------------------------------
# Score completeness
# ----------------------------------------------------------------

class TestScoreCompleteness:
    def test_all_score_keys_present(self, rng):
        """Full scoring with all blobs should produce a complete score dict."""
        static = _make_png()
        swept = _make_png(mode='L')
        hist_static = _make_hist_static(rng=rng)
        hist_tf = _make_hist_transform_v2(n_transforms=3, rng=rng)
        scores = _score_genome(static, swept,
                               hist_static=hist_static,
                               hist_transform=hist_tf)
        # Core image-based scores
        assert 'coverage' in scores or 'cl_coverage' in scores
        # Cluster scores
        assert 'cl_coverage' in scores
        # Transform scores
        assert 'tf_coverage' in scores
        # Detail
        assert 'detail_sensitivity' in scores

    def test_all_values_finite(self, rng):
        static = _make_png()
        hist_static = _make_hist_static(rng=rng)
        scores = _score_genome(static, None, hist_static=hist_static)
        for k, v in scores.items():
            if isinstance(v, (int, float)):
                assert np.isfinite(v), f'{k} is not finite: {v}'


# ============================================================================
# _compute_cnn_weights_hash — pure-function hash that includes both weights
# AND normalization version. Tracking just weights misses input-pipeline
# changes that change scores without changing the model file. The 2026-05
# incident (3000 genomes with legacy scores frozen under v3-normalized
# hash) is exactly what tests in this class are guarding against.
# ============================================================================

class TestComputeCnnWeightsHash:

    def test_same_weights_same_norm_same_hash(self):
        from flame_sheep.genome.score_worker import _compute_cnn_weights_hash
        h1 = _compute_cnn_weights_hash(b'fake weights bytes', 'v3')
        h2 = _compute_cnn_weights_hash(b'fake weights bytes', 'v3')
        assert h1 == h2

    def test_different_weights_different_hash(self):
        from flame_sheep.genome.score_worker import _compute_cnn_weights_hash
        h1 = _compute_cnn_weights_hash(b'weights A', 'v3')
        h2 = _compute_cnn_weights_hash(b'weights B', 'v3')
        assert h1 != h2

    def test_same_weights_different_norm_different_hash(self):
        """The May 2026 incident shape — weights file unchanged but
        normalization upgraded from legacy to v3. Scores produced before
        vs after are NOT equivalent and must hash differently."""
        from flame_sheep.genome.score_worker import _compute_cnn_weights_hash
        h_legacy = _compute_cnn_weights_hash(b'weights', None)
        h_v3 = _compute_cnn_weights_hash(b'weights', 'v3')
        assert h_legacy != h_v3

    def test_none_normalization_tags_as_legacy(self):
        """normalization=None ↔ normalization='legacy_v0' — same hash
        (legacy weights load as None, but the tag is canonicalized)."""
        from flame_sheep.genome.score_worker import _compute_cnn_weights_hash
        h_none = _compute_cnn_weights_hash(b'weights', None)
        h_explicit = _compute_cnn_weights_hash(b'weights', 'legacy_v0')
        assert h_none == h_explicit

    def test_hex_output_is_16_chars(self):
        from flame_sheep.genome.score_worker import _compute_cnn_weights_hash
        h = _compute_cnn_weights_hash(b'whatever', 'v1')
        assert len(h) == 16
        assert all(c in '0123456789abcdef' for c in h)

    def test_different_norm_versions_distinct(self):
        from flame_sheep.genome.score_worker import _compute_cnn_weights_hash
        weights = b'fixed weights'
        hashes = [_compute_cnn_weights_hash(weights, v)
                  for v in ['v1', 'v2', 'v3', 'legacy_v0']]
        assert len(set(hashes)) == 4, 'each normalization version should produce a distinct hash'
