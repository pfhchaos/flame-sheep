"""Tests for variation parameter generation and registry."""

from __future__ import annotations

import numpy as np
import pytest

from flame_sheep.variations._registry import (
    Variation, NUM_VARIATIONS, SLOT_SIZE, VAR_PARAMS_SPEC,
    PARAMETRIC_VARIATIONS,
)
from flame_sheep.variations._params import (
    random_var_params, jitter_var_params, _PARAM_RANGES, _INTEGER_PARAMS,
)


# ----------------------------------------------------------------
# Registry
# ----------------------------------------------------------------

class TestVariationRegistry:

    def test_num_variations_positive(self):
        assert NUM_VARIATIONS > 0

    def test_all_variation_indices_contiguous(self):
        """Variation indices should be 0..NUM_VARIATIONS-1 with no gaps."""
        indices = set()
        for name in dir(Variation):
            val = getattr(Variation, name)
            if isinstance(val, int) and 0 <= val < NUM_VARIATIONS:
                indices.add(val)
        assert indices == set(range(NUM_VARIATIONS))

    def test_slot_size_sufficient_for_params(self):
        """SLOT_SIZE must be at least 2 (index + weight) + max params."""
        max_params = max((len(spec) for spec in VAR_PARAMS_SPEC.values()), default=0)
        assert SLOT_SIZE >= 2 + max_params

    def test_parametric_variations_have_specs(self):
        """Every parametric variation should have a VAR_PARAMS_SPEC entry."""
        for var_idx in PARAMETRIC_VARIATIONS:
            assert var_idx in VAR_PARAMS_SPEC, \
                f"Variation {var_idx} is parametric but has no spec"

    def test_spec_params_are_strings(self):
        for var_idx, params in VAR_PARAMS_SPEC.items():
            assert isinstance(params, list)
            for p in params:
                assert isinstance(p, str), f"Var {var_idx} param {p} is not a string"


# ----------------------------------------------------------------
# Random parameter generation
# ----------------------------------------------------------------

class TestRandomVarParams:

    def test_non_parametric_returns_empty(self):
        rng = np.random.default_rng(42)
        # LINEAR (0) has no params
        params = random_var_params(Variation.LINEAR, rng)
        assert params == {}

    def test_parametric_returns_params(self):
        rng = np.random.default_rng(42)
        params = random_var_params(Variation.WAVES_PARAM, rng)
        assert 'waves_freq_x' in params
        assert 'waves_freq_y' in params
        assert 'waves_amp_x' in params
        assert 'waves_amp_y' in params

    def test_all_parametric_return_correct_keys(self):
        rng = np.random.default_rng(42)
        for var_idx, expected_keys in VAR_PARAMS_SPEC.items():
            params = random_var_params(var_idx, rng)
            for key in expected_keys:
                assert key in params, \
                    f"Variation {var_idx} missing param '{key}'"

    def test_values_are_finite(self):
        rng = np.random.default_rng(42)
        for var_idx in PARAMETRIC_VARIATIONS:
            params = random_var_params(var_idx, rng)
            for key, val in params.items():
                assert np.isfinite(val), \
                    f"Var {var_idx} param {key} = {val} (not finite)"

    def test_deterministic_with_seed(self):
        p1 = random_var_params(Variation.WAVES, np.random.default_rng(42))
        p2 = random_var_params(Variation.WAVES, np.random.default_rng(42))
        assert p1 == p2


# ----------------------------------------------------------------
# Jitter
# ----------------------------------------------------------------

class TestJitterVarParams:

    def test_jitter_changes_values(self):
        rng = np.random.default_rng(42)
        original = random_var_params(Variation.WAVES_PARAM, rng)
        jittered = jitter_var_params(original, np.random.default_rng(99))
        assert jittered != original
        assert set(jittered.keys()) == set(original.keys())

    def test_jitter_preserves_keys(self):
        rng = np.random.default_rng(42)
        original = random_var_params(Variation.CURL, rng)
        jittered = jitter_var_params(original, rng)
        assert set(jittered.keys()) == set(original.keys())

    def test_jitter_stays_in_range(self):
        rng = np.random.default_rng(42)
        for var_idx in PARAMETRIC_VARIATIONS:
            original = random_var_params(var_idx, rng)
            if not original:
                continue
            jittered = jitter_var_params(original, rng, scale=0.5)
            for key, val in jittered.items():
                lo, hi = _PARAM_RANGES.get(key, (-2.5, 2.5))
                assert lo <= val <= hi, \
                    f"Var {var_idx} param {key} = {val} outside [{lo}, {hi}]"

    def test_integer_params_rounded(self):
        rng = np.random.default_rng(42)
        params = {'kal_n': 6.0, 'kal_pull': 0.5, 'kal_rotate': 1.0}
        for _ in range(20):
            jittered = jitter_var_params(params, rng)
            assert jittered['kal_n'] == round(jittered['kal_n']), \
                f"kal_n should be integer, got {jittered['kal_n']}"

    def test_zero_scale_no_change(self):
        rng = np.random.default_rng(42)
        original = {'curl_c1': 0.5, 'curl_c2': -0.3}
        jittered = jitter_var_params(original, rng, scale=0.0)
        assert jittered == original

    def test_empty_params(self):
        rng = np.random.default_rng(42)
        assert jitter_var_params({}, rng) == {}


# ----------------------------------------------------------------
# Param ranges table
# ----------------------------------------------------------------

class TestParamRanges:

    def test_all_spec_params_have_ranges(self):
        """Every param in VAR_PARAMS_SPEC should have a range in _PARAM_RANGES."""
        missing = []
        for var_idx, params in VAR_PARAMS_SPEC.items():
            for p in params:
                if p not in _PARAM_RANGES:
                    missing.append(f"var {var_idx}: {p}")
        assert not missing, f"Missing ranges: {missing}"

    def test_ranges_are_valid(self):
        for name, (lo, hi) in _PARAM_RANGES.items():
            assert lo < hi, f"{name}: lo={lo} >= hi={hi}"
