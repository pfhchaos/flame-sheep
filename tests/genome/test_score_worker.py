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

    @pytest.mark.slow  # 4x full histogram clustering pipeline — ~9s total
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


# ============================================================================
# _discover_aux_models — scans aux_models/ dir, loads each .npz via the
# injected load_fn, returns {stem: model} dict. Tests use a fake load_fn
# returning sentinel strings so we don't need real CNN weights.
# ============================================================================

class TestDiscoverAuxModels:
    """Aux-model discovery for multi-model active learning. The
    score_worker scans ~/.local/share/flame-sheep/aux_models/ on
    startup; each .npz file becomes a named aux model whose scores
    populate the cnn_scores_detail JSON for the compare-mode
    multi-strategy active-learning framework."""

    @pytest.fixture
    def log(self):
        import logging
        return logging.getLogger('test_discover_aux_models')

    def test_missing_dir_returns_empty(self, tmp_path, log):
        """Common case: user hasn't set up aux models yet."""
        from flame_sheep.genome.score_worker import _discover_aux_models
        missing = tmp_path / 'nonexistent'
        result = _discover_aux_models(missing, lambda p: 'fake', log)
        assert result == {}

    def test_empty_dir_returns_empty(self, tmp_path, log):
        from flame_sheep.genome.score_worker import _discover_aux_models
        result = _discover_aux_models(tmp_path, lambda p: 'fake', log)
        assert result == {}

    def test_discovers_all_npz_files(self, tmp_path, log):
        from flame_sheep.genome.score_worker import _discover_aux_models
        # Stage three .npz files
        for name in ('alpha', 'beta', 'gamma'):
            (tmp_path / f'{name}.npz').write_bytes(b'fake')
        # Plus a non-.npz file that should be ignored
        (tmp_path / 'readme.txt').write_text('not a model')

        # load_fn returns the path's stem as the "model" sentinel
        result = _discover_aux_models(tmp_path, lambda p: p.stem, log)
        assert set(result.keys()) == {'alpha', 'beta', 'gamma'}
        assert result['alpha'] == 'alpha'
        assert result['beta'] == 'beta'

    def test_load_failure_skips_and_continues(self, tmp_path, log):
        """A single broken model file shouldn't kill discovery —
        other files still load."""
        from flame_sheep.genome.score_worker import _discover_aux_models
        (tmp_path / 'good.npz').write_bytes(b'fake')
        (tmp_path / 'broken.npz').write_bytes(b'fake')
        (tmp_path / 'also_good.npz').write_bytes(b'fake')

        def flaky_load(path):
            if path.stem == 'broken':
                raise RuntimeError('bad weights')
            return path.stem

        result = _discover_aux_models(tmp_path, flaky_load, log)
        assert set(result.keys()) == {'good', 'also_good'}
        # Broken is omitted; not in dict
        assert 'broken' not in result

    def test_symlinks_resolve(self, tmp_path, log):
        """Common deployment pattern: aux_models/curriculum.npz is a
        symlink to a checkpoint dir elsewhere on disk. The discovery
        should treat the symlink as a normal .npz file (stem = the
        symlink's name, not the target's)."""
        from flame_sheep.genome.score_worker import _discover_aux_models
        # Create a real file and a symlink pointing at it
        target = tmp_path / 'actual_model.npz'
        target.write_bytes(b'fake weights')
        link = tmp_path / 'aliased.npz'
        link.symlink_to(target)

        result = _discover_aux_models(tmp_path, lambda p: p.name, log)
        # Both the real file AND the symlink should be discovered as
        # independent models (the user might want both names available)
        assert 'actual_model' in result
        assert 'aliased' in result

    def test_dotfiles_are_loaded(self, tmp_path, log):
        """Documents actual behavior: `.glob('*.npz')` MATCHES hidden
        files starting with '.'. A `.foo.npz` will get loaded as an
        aux model named '.foo'. If this becomes undesirable (e.g. it
        picks up editor backups) the discovery function would need
        an explicit not-startswith('.') filter."""
        from flame_sheep.genome.score_worker import _discover_aux_models
        (tmp_path / 'visible.npz').write_bytes(b'fake')
        (tmp_path / '.hidden.npz').write_bytes(b'fake')

        result = _discover_aux_models(tmp_path, lambda p: p.stem, log)
        assert 'visible' in result
        # CURRENT BEHAVIOR: dotfiles ARE loaded.
        assert '.hidden' in result

    def test_load_fn_receives_path(self, tmp_path, log):
        """Sanity: load_fn gets the full Path, not just the name."""
        from flame_sheep.genome.score_worker import _discover_aux_models
        (tmp_path / 'thing.npz').write_bytes(b'fake')
        seen_paths = []

        def capture_load(p):
            seen_paths.append(p)
            return 'fake'

        _discover_aux_models(tmp_path, capture_load, log)
        assert len(seen_paths) == 1
        assert seen_paths[0] == tmp_path / 'thing.npz'
        # And the path is absolute (or at least the function received
        # a Path, not a string)
        from pathlib import Path
        assert isinstance(seen_paths[0], Path)
