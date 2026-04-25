"""Tests for spectral band definitions and utilities.

Covers frequency mask generation, A-weighting curve, band definitions,
and SpringBand physics.
"""

import numpy as np
import pytest

from flame_sheep.audio._constants import SAMPLE_RATE, N_BINS, FREQS, FFT_SIZE
from flame_sheep.audio._bands import (
    make_mask, make_weights, a_weight_curve, A_WEIGHTS,
    BAND_RANGES, BAND_MASKS, DETECTION_BANDS,
    DEFAULT_RANGES, ALLOWED_RANGES,
    SpringBand,
)


class TestMakeMask:
    """Test boolean mask generation from frequency ranges."""

    def test_shape(self):
        mask = make_mask(100, 1000)
        assert mask.shape == (N_BINS,)
        assert mask.dtype == bool

    def test_includes_range(self):
        mask = make_mask(100, 1000)
        # Bins within range should be True
        in_range = (FREQS >= 100) & (FREQS < 1000)
        np.testing.assert_array_equal(mask, in_range)

    def test_excludes_outside(self):
        mask = make_mask(100, 200)
        # DC (0 Hz) should be excluded
        assert not mask[0]
        # Very high bins should be excluded
        assert not mask[-1]

    def test_full_range(self):
        mask = make_mask(0, SAMPLE_RATE / 2 + 1)
        assert mask.all()

    def test_empty_range(self):
        """Range with no bins should be all False."""
        mask = make_mask(99999, 99999.1)
        assert not mask.any()


class TestMakeWeights:
    """Test normalized weight generation."""

    def test_sums_to_one(self):
        w = make_weights(100, 1000)
        assert abs(w.sum() - 1.0) < 1e-5

    def test_zero_outside_range(self):
        w = make_weights(5000, 10000)
        low_bins = FREQS < 5000
        assert (w[low_bins] == 0).all()

    def test_uniform_within_range(self):
        w = make_weights(100, 1000)
        active = w[w > 0]
        # All active weights should be equal
        assert np.allclose(active, active[0])

    def test_empty_range_all_zero(self):
        w = make_weights(99999, 99999.1)
        assert (w == 0).all()


class TestAWeighting:
    """Test A-weighting curve properties."""

    def test_1khz_is_unity(self):
        """A-weighting is normalized so 1kHz = 1.0."""
        idx = np.argmin(np.abs(FREQS - 1000))
        assert abs(A_WEIGHTS[idx] - 1.0) < 0.05

    def test_low_freq_attenuated(self):
        """Sub-100Hz should be attenuated relative to 1kHz."""
        idx_50hz = np.argmin(np.abs(FREQS - 50))
        idx_1khz = np.argmin(np.abs(FREQS - 1000))
        assert A_WEIGHTS[idx_50hz] < A_WEIGHTS[idx_1khz]

    def test_high_freq_attenuated(self):
        """Very high frequencies should be attenuated."""
        idx_15khz = np.argmin(np.abs(FREQS - 15000))
        idx_1khz = np.argmin(np.abs(FREQS - 1000))
        assert A_WEIGHTS[idx_15khz] < A_WEIGHTS[idx_1khz]

    def test_peak_near_2_4khz(self):
        """A-weighting peaks around 2-4kHz (ear sensitivity peak)."""
        mask_peak = (FREQS >= 2000) & (FREQS <= 5000)
        mask_low = (FREQS >= 50) & (FREQS <= 200)
        assert A_WEIGHTS[mask_peak].max() > A_WEIGHTS[mask_low].max()

    def test_shape_and_dtype(self):
        assert A_WEIGHTS.shape == (N_BINS,)
        assert A_WEIGHTS.dtype == np.float32

    def test_no_negative_values(self):
        assert (A_WEIGHTS >= 0).all()


class TestBandDefinitions:
    """Verify band range consistency."""

    def test_detection_bands_subset_of_band_ranges(self):
        for name in DETECTION_BANDS:
            assert name in BAND_RANGES

    def test_band_masks_match_ranges(self):
        for name, (lo, hi) in BAND_RANGES.items():
            expected = make_mask(lo, hi)
            np.testing.assert_array_equal(BAND_MASKS[name], expected)

    def test_detection_bands_have_defaults_and_allowed(self):
        for name in DETECTION_BANDS:
            assert name in DEFAULT_RANGES, f'{name} missing from DEFAULT_RANGES'
            assert name in ALLOWED_RANGES, f'{name} missing from ALLOWED_RANGES'

    def test_default_within_allowed(self):
        for name in DETECTION_BANDS:
            d_lo, d_hi = DEFAULT_RANGES[name]
            a_lo, a_hi = ALLOWED_RANGES[name]
            assert d_lo >= a_lo, f'{name} default lo {d_lo} < allowed {a_lo}'
            # Note: kick default_hi (200) intentionally exceeds allowed_hi (150)
            # because the kick band's static range is wider than the spring's
            # drift range. Skip this check for kick.
            if name != 'kick':
                assert d_hi <= a_hi, f'{name} default hi {d_hi} > allowed {a_hi}'

    def test_no_detection_band_gaps(self):
        """Adjacent detection bands should be contiguous or overlapping."""
        bands = [DEFAULT_RANGES[n] for n in DETECTION_BANDS]
        bands.sort(key=lambda r: r[0])
        for i in range(len(bands) - 1):
            _, hi = bands[i]
            lo_next, _ = bands[i + 1]
            assert lo_next <= hi, \
                f'Gap between {bands[i]} and {bands[i+1]}'


class TestSpringBand:
    """Test spring-model adaptive band."""

    def test_initial_center(self):
        sb = SpringBand('kick')
        lo, hi = DEFAULT_RANGES['kick']
        assert sb.center == pytest.approx((lo + hi) / 2.0)

    def test_initial_mask_matches_default(self):
        sb = SpringBand('kick')
        lo, hi = DEFAULT_RANGES['kick']
        expected = make_mask(lo, hi)
        np.testing.assert_array_equal(sb.mask, expected)

    def test_anchor_pulls_toward_default(self):
        # Use snare — has wide allowed range (150-2000) vs kick's narrow (25-150)
        sb = SpringBand('snare')
        # Shift center away from default
        sb.center = sb.default_center + 200
        original = sb.center
        sb.apply_forces(anchor_k=1.0, flux_k=0.0,
                        neighbors=[], repulsion_k=0.0)
        # Should move toward default
        assert abs(sb.center - sb.default_center) < abs(original - sb.default_center)

    def test_center_stays_in_allowed_range(self):
        sb = SpringBand('snare')
        # Force it manually out of range
        sb.center = 999999
        sb.apply_forces(anchor_k=0.01, flux_k=0.0,
                        neighbors=[], repulsion_k=0.0)
        half = sb.width
        assert sb.center >= sb.lo_allowed + half
        assert sb.center <= sb.hi_allowed - half

    def test_repulsion_pushes_bands_apart(self):
        kick = SpringBand('kick')
        snare = SpringBand('snare')
        # Move snare center close to kick
        snare.center = kick.center + 10
        original_dist = abs(kick.center - snare.center)

        kick.apply_forces(anchor_k=0.0, flux_k=0.0,
                          neighbors=[snare], repulsion_k=100.0)
        new_dist = abs(kick.center - snare.center)
        assert new_dist > original_dist

    def test_reset_restores_defaults(self):
        sb = SpringBand('kick')
        sb.center = 999
        sb.width = 1
        sb._flux_ema[:] = 1.0
        sb.reset()
        assert sb.center == sb.default_center
        assert sb.width == sb.default_width
        assert sb._flux_ema.sum() == 0

    def test_flux_centroid_default_when_empty(self):
        sb = SpringBand('kick')
        assert sb.flux_centroid() == sb.default_center

    def test_flux_ema_accumulates(self):
        sb = SpringBand('kick')
        flux = np.zeros(N_BINS, dtype=np.float32)
        # Put energy in a specific bin
        target_idx = np.argmin(np.abs(FREQS - 100))
        flux[target_idx] = 1.0
        # Stability=0 means fully percussive (passes through)
        stability = np.zeros(N_BINS, dtype=np.float32)
        sb.update_flux_ema(flux, stability, alpha=0.0)
        centroid = sb.flux_centroid()
        # Centroid should be near 100 Hz
        assert abs(centroid - 100) < 30
