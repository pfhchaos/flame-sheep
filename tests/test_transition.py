"""Tests for genome transition quality scoring."""

import numpy as np
import pytest

from flame_sheep.genome import Genome, Transform, NUM_VARIATIONS
from flame_sheep.transition import (
    variation_signature,
    signature_distance,
    match_transforms,
    compute_transition_distance,
    _active_variations,
    _transform_distance,
    _global_distance,
    FILTERED_DISTANCE,
)


def _make_transform(active_vars: list[int], weight: float = 1.0,
                     color: float = 0.5, rng=None) -> Transform:
    """Create a transform with specific active variations."""
    if rng is None:
        rng = np.random.default_rng(42)
    t = Transform()
    t.affine = rng.uniform(-1, 1, 6).astype(np.float32)
    for v in active_vars:
        t.variations[v] = rng.uniform(0.3, 1.0)
    # Normalize variation weights
    total = t.variations.sum()
    if total > 0:
        t.variations /= total
    t.weight = weight
    t.color = color
    return t


def _make_genome(transform_specs: list[list[int]], **kwargs) -> Genome:
    """Create a genome from a list of per-transform active variation indices."""
    rng = np.random.default_rng(42)
    g = Genome()
    for spec in transform_specs:
        g.transforms.append(_make_transform(spec, rng=rng))
    g.palette = rng.random((256, 3)).astype(np.float32)
    g.zoom = kwargs.get('zoom', 1.0)
    g.rotation = kwargs.get('rotation', 0.0)
    g.center = np.array(kwargs.get('center', [0.0, 0.0]), dtype=np.float32)
    return g


class TestVariationSignature:
    def test_empty_genome(self):
        g = Genome()
        assert variation_signature(g) == ''

    def test_single_variation(self):
        g = _make_genome([[26]])
        assert variation_signature(g) == '26'

    def test_multiple_transforms(self):
        g = _make_genome([[26], [26], [18, 23]])
        sig = variation_signature(g)
        # Sorted: 18, 23, 26, 26
        assert sig == '18,23,26,26'

    def test_deterministic(self):
        g = _make_genome([[5, 10], [3], [10, 20]])
        s1 = variation_signature(g)
        s2 = variation_signature(g)
        assert s1 == s2


class TestSignatureDistance:
    def test_identical(self):
        assert signature_distance('18,23,26,26', '18,23,26,26') == 0

    def test_empty(self):
        assert signature_distance('', '') == 0

    def test_one_empty(self):
        assert signature_distance('18,26', '') == 2

    def test_one_different(self):
        assert signature_distance('18,23,26', '18,23,29') == 2  # 26 gone, 29 added

    def test_count_difference(self):
        assert signature_distance('26,26', '26') == 1

    def test_completely_different(self):
        assert signature_distance('1,2,3', '10,20,30') == 6


class TestMatchTransforms:
    def test_identical_variations(self):
        a = [_make_transform([26]), _make_transform([18])]
        b = [_make_transform([18]), _make_transform([26])]
        matched, ua, ub = match_transforms(a, b)
        # Should match a[0]↔b[1] (both [26]) and a[1]↔b[0] (both [18])
        assert len(matched) == 2
        assert len(ua) == 0
        assert len(ub) == 0
        # Check correct matching
        match_dict = dict(matched)
        assert match_dict[0] == 1  # a[0]=[26] ↔ b[1]=[26]
        assert match_dict[1] == 0  # a[1]=[18] ↔ b[0]=[18]

    def test_different_counts(self):
        a = [_make_transform([26]), _make_transform([18]), _make_transform([5])]
        b = [_make_transform([26])]
        matched, ua, ub = match_transforms(a, b)
        assert len(matched) == 1
        assert len(ua) == 2
        assert len(ub) == 0

    def test_empty(self):
        matched, ua, ub = match_transforms([], [_make_transform([26])])
        assert len(matched) == 0
        assert len(ub) == 1

    def test_no_overlap(self):
        a = [_make_transform([1, 2])]
        b = [_make_transform([50, 60])]
        matched, ua, ub = match_transforms(a, b)
        # Still matches (greedy picks best, even if Jaccard=0)
        assert len(matched) == 1


class TestTransformDistance:
    def test_identical(self):
        rng = np.random.default_rng(42)
        t = _make_transform([26, 18], rng=rng)
        assert _transform_distance(t, t) == pytest.approx(0.0, abs=1e-6)

    def test_different(self):
        t1 = _make_transform([26], rng=np.random.default_rng(1))
        t2 = _make_transform([26], rng=np.random.default_rng(2))
        d = _transform_distance(t1, t2)
        assert d > 0
        assert np.isfinite(d)


class TestGlobalDistance:
    def test_identical(self):
        g = _make_genome([[26]])
        assert _global_distance(g, g) == pytest.approx(0.0, abs=1e-6)

    def test_rotation_wraparound(self):
        g1 = _make_genome([[26]], rotation=0.1)
        g2 = _make_genome([[26]], rotation=2 * np.pi - 0.1)
        d = _global_distance(g1, g2)
        # Should be close (0.2 radians apart, not 6.08)
        assert d < 0.5


class TestTransitionDistance:
    def test_identical_genomes(self):
        g = _make_genome([[26], [18]])
        d = compute_transition_distance(g, g)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_similar_genomes(self):
        g1 = _make_genome([[26], [18]])
        g2 = _make_genome([[26], [18]])
        # Same structure, same RNG → same genome → distance ≈ 0
        d = compute_transition_distance(g1, g2)
        assert d < 1.0

    def test_different_structure_filtered(self):
        g1 = _make_genome([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        g2 = _make_genome([[50, 60], [70, 80], [90, 100]])
        d = compute_transition_distance(g1, g2)
        assert d == FILTERED_DISTANCE

    def test_precomputed_signatures(self):
        g1 = _make_genome([[26], [18]])
        g2 = _make_genome([[26], [29]])
        sig1 = variation_signature(g1)
        sig2 = variation_signature(g2)
        d = compute_transition_distance(g1, g2, sig_a=sig1, sig_b=sig2)
        assert np.isfinite(d)
        assert d > 0

    def test_symmetry(self):
        g1 = _make_genome([[26], [18]])
        g2 = _make_genome([[26], [29]])
        d1 = compute_transition_distance(g1, g2)
        d2 = compute_transition_distance(g2, g1)
        assert d1 == pytest.approx(d2, abs=1e-6)
