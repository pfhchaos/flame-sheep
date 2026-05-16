"""
Tests for scoring_channels: pack/unpack roundtrips, normalization, raw histogram I/O.
"""

import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from flame_sheep.scoring_channels import (
    COLOR_SCALE,
    pack_histogram,
    pack_static_histogram,
    unpack_histogram,
    unpack_static_histogram,
    save_raw_histograms,
    load_raw_histograms,
    normalize_channels,
)


# ----------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_histograms(rng):
    """Small 32x32 histogram arrays for fast tests."""
    h, w = 32, 32
    static_hits = rng.integers(0, 10_000, (h, w), dtype=np.uint32)
    static_colors = np.clip(
        static_hits.astype(np.float64) * rng.uniform(0, 1, (h, w)) * COLOR_SCALE,
        0, np.iinfo(np.uint32).max).astype(np.uint32)
    swept_hits = rng.integers(0, 5_000, (h, w), dtype=np.uint32)
    first_hit = rng.integers(0, 256, (h, w), dtype=np.uint8)
    return static_hits, static_colors, swept_hits, first_hit


# ----------------------------------------------------------------
# pack/unpack roundtrips
# ----------------------------------------------------------------

class TestPackHistogram:
    def test_roundtrip_uint32(self, rng):
        arr = rng.integers(0, 100_000, (64, 64), dtype=np.uint32)
        blob = pack_histogram(arr)
        recovered = unpack_histogram(blob)
        np.testing.assert_array_equal(arr, recovered)

    def test_roundtrip_preserves_shape(self, rng):
        for shape in [(16, 32), (64, 64), (128, 256)]:
            arr = rng.integers(0, 100, shape, dtype=np.uint32)
            recovered = unpack_histogram(pack_histogram(arr))
            assert recovered.shape == shape

    def test_zeros(self):
        arr = np.zeros((32, 32), dtype=np.uint32)
        recovered = unpack_histogram(pack_histogram(arr))
        np.testing.assert_array_equal(arr, recovered)

    def test_header_contains_dimensions(self, rng):
        arr = rng.integers(0, 100, (48, 64), dtype=np.uint32)
        blob = pack_histogram(arr)
        import zlib
        raw = zlib.decompress(blob)
        h, w = struct.unpack('II', raw[:8])
        assert h == 48
        assert w == 64


class TestPackStaticHistogram:
    def test_roundtrip(self, rng):
        hits = rng.integers(0, 10_000, (64, 64), dtype=np.uint32)
        colors = rng.integers(0, 10_000_000, (64, 64), dtype=np.uint32)
        blob = pack_static_histogram(hits, colors)
        rec_hits, rec_colors = unpack_static_histogram(blob)
        np.testing.assert_array_equal(hits, rec_hits)
        np.testing.assert_array_equal(colors, rec_colors)

    def test_roundtrip_with_float_input(self, rng):
        """pack_static_histogram casts to uint32 — floats should survive."""
        hits = rng.uniform(0, 10_000, (32, 32)).astype(np.float64)
        colors = rng.uniform(0, 10_000_000, (32, 32)).astype(np.float64)
        blob = pack_static_histogram(hits, colors)
        rec_hits, rec_colors = unpack_static_histogram(blob)
        np.testing.assert_array_equal(hits.astype(np.uint32), rec_hits)
        np.testing.assert_array_equal(colors.astype(np.uint32), rec_colors)

    def test_zeros(self):
        hits = np.zeros((32, 32), dtype=np.uint32)
        colors = np.zeros((32, 32), dtype=np.uint32)
        rec_hits, rec_colors = unpack_static_histogram(
            pack_static_histogram(hits, colors))
        np.testing.assert_array_equal(hits, rec_hits)
        np.testing.assert_array_equal(colors, rec_colors)


# ----------------------------------------------------------------
# save/load raw histograms (.npz)
# ----------------------------------------------------------------

class TestRawHistogramIO:
    def test_roundtrip_without_first_hit(self, small_histograms):
        static_hits, static_colors, swept_hits, _ = small_histograms
        with tempfile.NamedTemporaryFile(suffix='.npz') as f:
            save_raw_histograms(f.name, static_hits, static_colors, swept_hits)
            loaded = load_raw_histograms(f.name)
        np.testing.assert_array_equal(loaded['static_hits'], static_hits)
        np.testing.assert_array_equal(loaded['static_colors'], static_colors)
        np.testing.assert_array_equal(loaded['swept_hits'], swept_hits)
        assert 'first_hit' not in loaded

    def test_roundtrip_with_first_hit(self, small_histograms):
        static_hits, static_colors, swept_hits, first_hit = small_histograms
        with tempfile.NamedTemporaryFile(suffix='.npz') as f:
            save_raw_histograms(f.name, static_hits, static_colors, swept_hits,
                                first_hit=first_hit)
            loaded = load_raw_histograms(f.name)
        np.testing.assert_array_equal(loaded['first_hit'], first_hit)


# ----------------------------------------------------------------
# normalize_channels
# ----------------------------------------------------------------

class TestNormalizeChannels:
    def test_output_shape(self, small_histograms):
        static_hits, static_colors, swept_hits, first_hit = small_histograms
        result = normalize_channels(static_hits, static_colors, swept_hits,
                                    first_hit=first_hit, output_size=64)
        assert result.shape == (4, 64, 64)

    def test_output_shape_default_size(self, small_histograms):
        static_hits, static_colors, swept_hits, _ = small_histograms
        result = normalize_channels(static_hits, static_colors, swept_hits,
                                    output_size=256)
        assert result.shape == (4, 256, 256)

    def test_output_dtype(self, small_histograms):
        static_hits, static_colors, swept_hits, _ = small_histograms
        result = normalize_channels(static_hits, static_colors, swept_hits,
                                    output_size=32)
        assert result.dtype == np.float32

    def test_output_range_zero_to_one(self, small_histograms):
        static_hits, static_colors, swept_hits, first_hit = small_histograms
        result = normalize_channels(static_hits, static_colors, swept_hits,
                                    first_hit=first_hit, output_size=32)
        # All channels should be in [0, 1] (allowing small float error)
        assert result.min() >= -1e-6
        assert result.max() <= 1.0 + 1e-6

    def test_all_zeros_produces_zeros(self):
        h, w = 32, 32
        zeros = np.zeros((h, w), dtype=np.uint32)
        result = normalize_channels(zeros, zeros, zeros, output_size=32)
        np.testing.assert_array_equal(result, 0.0)

    def test_no_first_hit_gives_zero_alpha(self, small_histograms):
        static_hits, static_colors, swept_hits, _ = small_histograms
        result = normalize_channels(static_hits, static_colors, swept_hits,
                                    first_hit=None, output_size=32)
        # A channel (index 3) should be all zeros when no first_hit
        np.testing.assert_array_equal(result[3], 0.0)

    def test_channel_order_HSLA(self, small_histograms):
        """Verify channels are H, S, L, A as documented."""
        static_hits, static_colors, swept_hits, first_hit = small_histograms
        result = normalize_channels(static_hits, static_colors, swept_hits,
                                    first_hit=first_hit, output_size=32)
        # L channel (index 2) should be nonzero where hits exist
        # since static_hits has values > 0
        assert result[2].sum() > 0
        # A channel (index 3) should be nonzero since first_hit has values
        assert result[3].sum() > 0

    def test_deterministic(self, small_histograms):
        static_hits, static_colors, swept_hits, first_hit = small_histograms
        r1 = normalize_channels(static_hits, static_colors, swept_hits,
                                first_hit=first_hit, output_size=32)
        r2 = normalize_channels(static_hits, static_colors, swept_hits,
                                first_hit=first_hit, output_size=32)
        np.testing.assert_array_equal(r1, r2)

    def test_resize_to_larger(self, small_histograms):
        """32x32 input resized to 128x128 output."""
        static_hits, static_colors, swept_hits, _ = small_histograms
        result = normalize_channels(static_hits, static_colors, swept_hits,
                                    output_size=128)
        assert result.shape == (4, 128, 128)

    def test_same_size_no_resize(self):
        """When input matches output_size, no resize needed."""
        h, w = 64, 64
        rng = np.random.default_rng(99)
        hits = rng.integers(1, 1000, (h, w), dtype=np.uint32)
        colors = (hits * rng.uniform(0.2, 0.8, (h, w)) * COLOR_SCALE).astype(np.uint32)
        swept = rng.integers(1, 500, (h, w), dtype=np.uint32)
        result = normalize_channels(hits, colors, swept, output_size=64)
        assert result.shape == (4, 64, 64)
        # L channel should preserve structure — nonzero where hits are nonzero
        assert (result[2] > 0).any()

    def test_h_channel_bounded_by_palette(self, rng):
        """H channel = color_acc / (hits * COLOR_SCALE), clipped to [0,1]."""
        h, w = 16, 16
        hits = np.full((h, w), 100, dtype=np.uint32)
        # Colors at exactly half palette: 0.5 * hits * COLOR_SCALE
        colors = (0.5 * hits * COLOR_SCALE).astype(np.uint32)
        swept = np.ones((h, w), dtype=np.uint32)
        result = normalize_channels(hits, colors, swept, output_size=16)
        # H channel should be ~0.5
        h_channel = result[0]
        np.testing.assert_allclose(h_channel, 0.5, atol=0.01)

    def test_gamma_boosts_faint_structure(self, rng):
        """With gamma > 1, faint hits should be boosted relative to linear."""
        h, w = 16, 16
        # A few pixels with low hit counts
        hits = np.full((h, w), 1, dtype=np.uint32)
        hits[0, 0] = 10000  # one bright pixel for max reference
        colors = np.zeros((h, w), dtype=np.uint32)
        swept = np.zeros((h, w), dtype=np.uint32)
        result = normalize_channels(hits, colors, swept, output_size=16)
        # L channel: faint pixels should be brighter than linear would give
        # Linear: 1/10000 = 0.0001. With log+gamma, should be much higher.
        faint_brightness = result[2, 8, 8]  # a pixel with hits=1
        assert faint_brightness > 0.1  # gamma boost makes this visible
