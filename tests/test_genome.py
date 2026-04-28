"""
Tests for genome generation, validation, and mutation.
No GPU required — pure CPU/numpy.
"""

import numpy as np
import pytest
from flame_sheep.genome import (
    Genome, Transform, _apply_variation_cpu, _score_from_histogram,
    MAX_TRANSFORMS, MAX_ACTIVE_VARS, NUM_VARIATIONS
)
from flame_sheep.variations import Variation, SLOT_SIZE


RNG = np.random.default_rng(42)  # fixed seed for reproducibility


# ----------------------------------------------------------------
# Transform
# ----------------------------------------------------------------

class TestTransform:

    def test_random_is_contractive(self):
        """Random transforms must have all singular values < 0.9."""
        rng = np.random.default_rng(0)
        for _ in range(50):
            t = Transform.random(rng)
            a, b, c, d, e, f = t.affine
            M = np.array([[a, b], [d, e]])
            svs = np.linalg.svd(M, compute_uv=False)
            assert np.all(svs < 0.9), f"singular values {svs} not contractive"

    def test_random_variation_weights_sum_to_one(self):
        """Active variation weights must sum to 1.0."""
        rng = np.random.default_rng(1)
        for _ in range(20):
            t = Transform.random(rng)
            total = t.variations.sum()
            assert abs(total - 1.0) < 1e-5, f"variation weights sum to {total}"

    def test_random_color_in_range(self):
        rng = np.random.default_rng(2)
        for _ in range(20):
            t = Transform.random(rng)
            assert 0.0 <= t.color <= 1.0

    def test_affine_shape(self):
        rng = np.random.default_rng(3)
        t = Transform.random(rng)
        assert t.affine.shape == (6,)
        assert t.affine.dtype == np.float32

    def test_variations_shape(self):
        rng = np.random.default_rng(4)
        t = Transform.random(rng)
        assert t.variations.shape == (NUM_VARIATIONS,)


# ----------------------------------------------------------------
# Genome generation
# ----------------------------------------------------------------

class TestGenomeRandom:

    def test_random_produces_genome(self):
        g = Genome.random(RNG)
        assert isinstance(g, Genome)
        assert len(g.transforms) >= 2
        assert len(g.transforms) <= MAX_TRANSFORMS

    def test_random_palette_shape(self):
        g = Genome.random(RNG)
        assert g.palette.shape == (256, 3)
        assert g.palette.dtype == np.float32

    def test_random_palette_in_range(self):
        g = Genome.random(RNG)
        assert np.all(g.palette >= 0.0)
        assert np.all(g.palette <= 1.0)

    def test_random_zoom_in_range(self):
        rng = np.random.default_rng(10)
        for _ in range(20):
            g = Genome.random(rng)
            assert 0.8 <= g.zoom <= 1.5

    def test_random_is_viable(self):
        """Genome.random() should always return a viable genome."""
        rng = np.random.default_rng(20)
        for _ in range(10):
            g = Genome.random(rng)
            assert g.is_viable(), "Genome.random() returned non-viable genome"


# ----------------------------------------------------------------
# Viability
# ----------------------------------------------------------------

class TestGenomeViability:

    def test_diverging_genome_rejected(self):
        """A genome with expanding transforms should fail viability."""
        g = Genome()
        t = Transform()
        # Deliberately non-contractive: scale factor 2, translation 0.1 so
        # the origin fixed-point is avoided and points actually escape.
        t.affine = np.array([2.0, 0.0, 0.1, 0.0, 2.0, 0.1], dtype=np.float32)
        t.variations[0] = 1.0  # linear
        t.weight = 1.0
        g.transforms = [t]
        assert not g.is_viable()

    def test_sierpinski_genome_viable(self):
        """Classic Sierpinski triangle IFS should be viable."""
        g = Genome()
        corners = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        for cx, cy in corners:
            t = Transform()
            # Shrink by 0.5 toward each corner
            t.affine = np.array([0.5, 0.0, cx*0.5,
                                  0.0, 0.5, cy*0.5], dtype=np.float32)
            t.variations[0] = 1.0  # linear
            t.weight = 1.0
            g.transforms.append(t)
        assert g.is_viable()

    def test_nan_genome_rejected(self):
        """A genome producing NaN should be rejected immediately."""
        g = Genome()
        t = Transform()
        t.affine = np.array([np.nan, 0.0, 0.0, 0.0, 0.5, 0.0], dtype=np.float32)
        t.variations[0] = 1.0
        t.weight = 1.0
        g.transforms = [t]
        assert not g.is_viable()


# ----------------------------------------------------------------
# Lerp / morphing
# ----------------------------------------------------------------

class TestGenomeLerp:

    def setup_method(self):
        rng = np.random.default_rng(30)
        self.g1 = Genome.random(rng)
        self.g2 = Genome.random(rng)

    def test_lerp_t0_equals_current(self):
        result = self.g1.lerp(self.g2, 0.0)
        n = min(len(self.g1.transforms), len(self.g2.transforms))
        for i in range(n):
            np.testing.assert_allclose(
                result.transforms[i].affine,
                self.g1.transforms[i].affine,
                atol=1e-5
            )

    def test_lerp_t1_equals_target(self):
        result = self.g1.lerp(self.g2, 1.0)
        n = min(len(self.g1.transforms), len(self.g2.transforms))
        for i in range(n):
            np.testing.assert_allclose(
                result.transforms[i].affine,
                self.g2.transforms[i].affine,
                atol=1e-5
            )

    def test_lerp_midpoint(self):
        result = self.g1.lerp(self.g2, 0.5)
        n = min(len(self.g1.transforms), len(self.g2.transforms))
        for i in range(n):
            expected = (self.g1.transforms[i].affine + self.g2.transforms[i].affine) * 0.5
            np.testing.assert_allclose(result.transforms[i].affine, expected, atol=1e-5)

    def test_lerp_zoom(self):
        result = self.g1.lerp(self.g2, 0.5)
        expected = (self.g1.zoom + self.g2.zoom) * 0.5
        assert abs(result.zoom - expected) < 1e-5

    def test_lerp_palette_shape(self):
        result = self.g1.lerp(self.g2, 0.5)
        assert result.palette.shape == (256, 3)


# ----------------------------------------------------------------
# GPU array packing
# ----------------------------------------------------------------

class TestToGpuArrays:

    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(40)
        g = Genome.random(rng)
        _, _, _, weights = g.to_gpu_arrays()
        n = len(g.transforms)
        assert abs(weights[:n].sum() - 1.0) < 1e-5

    def test_affines_shape(self):
        rng = np.random.default_rng(41)
        g = Genome.random(rng)
        affines, _, _, _ = g.to_gpu_arrays()
        assert affines.shape == (MAX_TRANSFORMS, 6)
        assert affines.dtype == np.float32

    def test_variations_shape(self):
        rng = np.random.default_rng(42)
        g = Genome.random(rng)
        _, active_vars, _, _ = g.to_gpu_arrays()
        assert active_vars.shape == (MAX_TRANSFORMS, MAX_ACTIVE_VARS * SLOT_SIZE)

    def test_unused_transform_slots_zero(self):
        """Transforms beyond n_transforms should be zero."""
        rng = np.random.default_rng(43)
        g = Genome.random(rng, n_transforms=2)
        affines, _, _, weights = g.to_gpu_arrays()
        assert np.all(affines[2:] == 0.0)
        assert np.all(weights[2:] == 0.0)


# ----------------------------------------------------------------
# Genome distance
# ----------------------------------------------------------------

class TestGenomeDistance:

    def test_distance_self_is_zero(self):
        """Distance from a genome to itself must be 0."""
        rng = np.random.default_rng(50)
        g = Genome.random(rng)
        assert g.distance(g) == 0.0

    def test_distance_is_symmetric(self):
        rng = np.random.default_rng(51)
        g1 = Genome.random(rng)
        g2 = Genome.random(rng)
        assert abs(g1.distance(g2) - g2.distance(g1)) < 1e-9

    def test_distance_in_range(self):
        """Distance must always be in [0, 1]."""
        rng = np.random.default_rng(52)
        genomes = [Genome.random(rng) for _ in range(10)]
        for i in range(len(genomes)):
            for j in range(i + 1, len(genomes)):
                d = genomes[i].distance(genomes[j])
                assert 0.0 <= d <= 1.0, f"distance {d} out of [0,1]"

    def test_random_genomes_exceed_min_threshold(self):
        """Random genome pairs should nearly always be visually distinct."""
        rng = np.random.default_rng(53)
        MIN = 0.15
        above = sum(
            Genome.random(rng).distance(Genome.random(rng)) >= MIN
            for _ in range(20)
        )
        # Allow some near-duplicates by chance, but majority must differ
        assert above >= 10, f"only {above}/20 pairs exceeded min distance {MIN}"

    def test_lerp_midpoint_closer_to_both_endpoints(self):
        """A lerp midpoint should be closer to each endpoint than they are to each other."""
        rng = np.random.default_rng(54)
        g1 = Genome.random(rng)
        g2 = Genome.random(rng)
        mid = g1.lerp(g2, 0.5)
        d12  = g1.distance(g2)
        d1m  = g1.distance(mid)
        d2m  = g2.distance(mid)
        assert d1m < d12, f"midpoint not closer to g1: d1m={d1m:.3f} d12={d12:.3f}"
        assert d2m < d12, f"midpoint not closer to g2: d2m={d2m:.3f} d12={d12:.3f}"


# ----------------------------------------------------------------
# Aesthetic scoring
# ----------------------------------------------------------------

class TestScoreFromHistogram:
    """Property tests for _score_from_histogram — no GPU needed."""

    def test_empty_histogram_all_zeros(self):
        hits = np.zeros((64, 64))
        colors = np.zeros((64, 64))
        scores = _score_from_histogram(hits, colors)
        for k, v in scores.items():
            assert v == 0.0, f"{k} should be 0 for empty histogram, got {v}"

    def test_single_pixel(self):
        hits = np.zeros((64, 64))
        colors = np.zeros((64, 64))
        hits[32, 32] = 100.0
        colors[32, 32] = 0.5
        scores = _score_from_histogram(hits, colors)
        assert scores['coverage'] == pytest.approx(1 / (64 * 64))
        assert scores['entropy'] < 0.01
        assert scores['complexity'] < 0.01

    def test_uniform_histogram(self):
        hits = np.ones((64, 64))
        colors = np.linspace(0, 1, 64 * 64).reshape(64, 64)
        scores = _score_from_histogram(hits, colors)
        assert scores['coverage'] == pytest.approx(1.0)
        assert scores['entropy'] > 0.99
        assert scores['complexity'] < 0.1  # uniform = no structure

    def test_all_scores_in_range(self):
        rng = np.random.default_rng(70)
        for _ in range(10):
            g = Genome.random(rng)
            scores = g.aesthetic_score()
            for k, v in scores.items():
                if k.startswith('centroid_offset'):
                    assert -1.0 <= v <= 1.0, f"{k}={v} out of [-1,1]"
                else:
                    assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"

    def test_corner_cluster_low_balance(self):
        hits = np.zeros((64, 64))
        hits[0:4, 0:4] = 50.0  # top-left corner
        colors = np.full((64, 64), 0.5)
        scores = _score_from_histogram(hits, colors)
        assert scores['balance'] < 0.7

    def test_centered_cluster_high_balance(self):
        hits = np.zeros((64, 64))
        hits[30:34, 30:34] = 50.0  # near center
        colors = np.full((64, 64), 0.5)
        scores = _score_from_histogram(hits, colors)
        assert scores['balance'] > 0.9

    def test_monochrome_low_color_entropy(self):
        hits = np.ones((64, 64))
        colors = np.full((64, 64), 0.5)  # same color everywhere
        scores = _score_from_histogram(hits, colors)
        assert scores['color_entropy'] < 0.1

    def test_diverse_colors_high_color_entropy(self):
        hits = np.ones((64, 64))
        colors = np.linspace(0, 1, 64 * 64).reshape(64, 64)
        scores = _score_from_histogram(hits, colors)
        assert scores['color_entropy'] > 0.8

    def test_structured_beats_uniform_complexity(self):
        """A fractal-like pattern should have higher complexity than uniform."""
        # Uniform
        uniform_hits = np.ones((64, 64))
        uniform_scores = _score_from_histogram(uniform_hits, np.zeros((64, 64)))

        # Structured: exponential falloff from center (like a real attractor)
        y, x = np.mgrid[0:64, 0:64]
        structured_hits = np.exp(-0.01 * ((x - 32)**2 + (y - 32)**2))
        structured_hits *= 1000
        structured_scores = _score_from_histogram(structured_hits, np.zeros((64, 64)))

        assert structured_scores['complexity'] > uniform_scores['complexity']


class TestAestheticScoreCpu:
    """Integration tests for the full CPU scoring pipeline."""

    def test_returns_all_keys(self):
        rng = np.random.default_rng(80)
        g = Genome.random(rng)
        scores = g.aesthetic_score()
        expected = {'coverage', 'entropy', 'color_entropy', 'balance', 'complexity',
                    'centroid_offset_x', 'centroid_offset_y'}
        assert set(scores.keys()) == expected

    def test_different_genomes_different_scores(self):
        rng = np.random.default_rng(81)
        scores = [Genome.random(rng).aesthetic_score() for _ in range(10)]
        # At least some metrics should vary across genomes
        varying = 0
        for key in scores[0]:
            values = [s[key] for s in scores]
            if max(values) - min(values) > 0.01:
                varying += 1
        assert varying >= 3, \
            f"only {varying}/{len(scores[0])} metrics varied across 10 genomes"


# ----------------------------------------------------------------
# GPU packing
# ----------------------------------------------------------------

class TestGpuPacking:

    def test_active_vars_has_correct_indices(self):
        """Active vars should contain the variation indices and weights."""
        g = Genome()
        t = Transform()
        t.variations = np.zeros(NUM_VARIATIONS, dtype=np.float32)
        t.variations[Variation.CURL] = 0.7
        t.variations[Variation.SPLITS] = 0.3
        g.transforms = [t]
        _, active_vars, _, _ = g.to_gpu_arrays()
        # First transform — two active vars at slot 0 and slot 1
        idx0 = int(active_vars[0, 0 * SLOT_SIZE])
        idx1 = int(active_vars[0, 1 * SLOT_SIZE])
        indices = sorted([idx0, idx1])
        assert Variation.SPLITS in indices
        assert Variation.CURL in indices

    def test_params_packed_inline_with_variation(self):
        """Params should be packed right after (index, weight) in each slot."""
        g = Genome()
        t = Transform()
        t.variations = np.zeros(NUM_VARIATIONS, dtype=np.float32)
        t.variations[Variation.CURL] = 1.0
        t.var_params = {'curl_c1': 0.42, 'curl_c2': -0.77}
        g.transforms = [t]
        _, active_vars, _, _ = g.to_gpu_arrays()
        # Curl is the only active variation, so it's at slot 0
        assert int(active_vars[0, 0]) == Variation.CURL
        assert abs(active_vars[0, 1] - 1.0) < 1e-6  # weight
        assert abs(active_vars[0, 2] - 0.42) < 1e-6  # curl_c1 = param 0
        assert abs(active_vars[0, 3] - (-0.77)) < 1e-6  # curl_c2 = param 1

    def test_unused_slots_negative(self):
        """Unused variation slots should have index < 0."""
        g = Genome()
        t = Transform()
        t.variations = np.zeros(NUM_VARIATIONS, dtype=np.float32)
        t.variations[Variation.LINEAR] = 1.0
        g.transforms = [t]
        _, active_vars, _, _ = g.to_gpu_arrays()
        # Slot 0 has LINEAR
        assert int(active_vars[0, 0]) == Variation.LINEAR
        # Slot 1 should be unused (index < 0)
        assert active_vars[0, 1 * SLOT_SIZE] < 0

    def test_icon_params_packed(self):
        """Icon variation should have all 6 params packed inline."""
        g = Genome()
        t = Transform()
        t.variations = np.zeros(NUM_VARIATIONS, dtype=np.float32)
        t.variations[Variation.ICON] = 1.0
        t.var_params = {
            'icon_degree': 5.0, 'icon_lambda': 1.5,
            'icon_alpha': -0.3, 'icon_beta': 0.7,
            'icon_gamma': 0.1, 'icon_omega': -0.2,
        }
        g.transforms = [t]
        _, active_vars, _, _ = g.to_gpu_arrays()
        # Icon at slot 0, params at offsets 2-7
        assert int(active_vars[0, 0]) == Variation.ICON
        assert abs(active_vars[0, 2] - 5.0) < 1e-6   # degree
        assert abs(active_vars[0, 3] - 1.5) < 1e-6   # lambda
        assert abs(active_vars[0, 4] - (-0.3)) < 1e-6 # alpha
        assert abs(active_vars[0, 5] - 0.7) < 1e-6   # beta
        assert abs(active_vars[0, 6] - 0.1) < 1e-6   # gamma
        assert abs(active_vars[0, 7] - (-0.2)) < 1e-6 # omega

    def test_weights_normalized(self):
        """Transform weights should sum to 1.0 after packing."""
        g = Genome()
        g.transforms = [Transform(), Transform(), Transform()]
        g.transforms[0].weight = 2.0
        g.transforms[1].weight = 3.0
        g.transforms[2].weight = 5.0
        _, _, _, weights = g.to_gpu_arrays()
        assert abs(weights[:3].sum() - 1.0) < 1e-6


# ----------------------------------------------------------------
# Variation CPU implementations
# ----------------------------------------------------------------

class TestVariationCpu:

    def test_linear(self):
        x, y = _apply_variation_cpu(0, 1.0, 2.0, 1.0)
        assert abs(x - 1.0) < 1e-6
        assert abs(y - 2.0) < 1e-6

    def test_linear_weight(self):
        x, y = _apply_variation_cpu(0, 1.0, 2.0, 0.5)
        assert abs(x - 0.5) < 1e-6
        assert abs(y - 1.0) < 1e-6

    def test_sinusoidal(self):
        x, y = _apply_variation_cpu(1, np.pi/2, 0.0, 1.0)
        assert abs(x - 1.0) < 1e-6
        assert abs(y - 0.0) < 1e-6

    def test_exponential_clamped(self):
        """Exponential with large x should not overflow."""
        x, y = _apply_variation_cpu(18, 1000.0, 0.0, 1.0)
        assert np.isfinite(x)
        assert np.isfinite(y)

    def test_spherical_near_origin(self):
        """Spherical near origin should not produce infinity."""
        x, y = _apply_variation_cpu(2, 1e-8, 1e-8, 1.0)
        assert np.isfinite(x)
        assert np.isfinite(y)

    def test_sattractor_finite(self):
        from flame_sheep.variations._cpu import _current_var_params
        import flame_sheep.variations._cpu as cpu_mod
        cpu_mod._current_var_params = {'sat_m': 6.0}
        x, y = _apply_variation_cpu(43, 1.0, 0.5, 1.0)
        assert np.isfinite(x) and np.isfinite(y)

    def test_wallpaper_all_groups_finite(self):
        import flame_sheep.variations._cpu as cpu_mod
        for group in range(17):
            cpu_mod._current_var_params = {'wallpaper_group': float(group)}
            for _ in range(20):
                x, y = _apply_variation_cpu(44, 1.0, 0.5, 1.0)
                assert np.isfinite(x) and np.isfinite(y), \
                    f"wallpaper group {group} produced non-finite output"

    def test_frieze_all_groups_finite(self):
        import flame_sheep.variations._cpu as cpu_mod
        for group in range(7):
            cpu_mod._current_var_params = {'frieze_group': float(group)}
            for _ in range(20):
                x, y = _apply_variation_cpu(45, 1.0, 0.5, 1.0)
                assert np.isfinite(x) and np.isfinite(y), \
                    f"frieze group {group} produced non-finite output"


class TestSymmetryGroupData:

    def test_wallpaper_group_count(self):
        from flame_sheep.variations._symmetry_groups import WALLPAPER_GROUPS
        assert len(WALLPAPER_GROUPS) == 17

    def test_frieze_group_count(self):
        from flame_sheep.variations._symmetry_groups import FRIEZE_GROUPS
        assert len(FRIEZE_GROUPS) == 7

    def test_all_transforms_are_6_tuples(self):
        from flame_sheep.variations._symmetry_groups import WALLPAPER_GROUPS, FRIEZE_GROUPS
        for i, group in enumerate(WALLPAPER_GROUPS):
            for j, elem in enumerate(group):
                assert len(elem) == 6, f"wallpaper[{i}][{j}] has {len(elem)} elements"
        for i, group in enumerate(FRIEZE_GROUPS):
            for j, elem in enumerate(group):
                assert len(elem) == 6, f"frieze[{i}][{j}] has {len(elem)} elements"

    def test_expected_element_counts(self):
        from flame_sheep.variations._symmetry_groups import WALLPAPER_GROUPS, FRIEZE_GROUPS
        # From JWildfire source
        wp_expected = [2, 2, 2, 2, 2, 4, 4, 8, 4, 4, 8, 8, 6, 12, 12, 12, 24]
        for i, (group, expected) in enumerate(zip(WALLPAPER_GROUPS, wp_expected)):
            assert len(group) == expected, \
                f"wallpaper[{i}] has {len(group)} elements, expected {expected}"

    def test_glsl_generation(self):
        from flame_sheep.variations._symmetry_groups import generate_glsl
        glsl = generate_glsl()
        assert 'WALLPAPER_COUNTS' in glsl
        assert 'FRIEZE_COUNTS' in glsl
        assert 'WALLPAPER_DATA' in glsl
        assert 'FRIEZE_DATA' in glsl

    def test_rebuild_with_different_step(self):
        from flame_sheep.variations._symmetry_groups import (
            WALLPAPER_GROUPS, _build_wallpaper, S
        )
        original_c = WALLPAPER_GROUPS[0][0][2]  # first group, first elem, c component
        bigger = _build_wallpaper(1.0)
        assert bigger[0][0][2] != original_c, "Different S should produce different translations"
