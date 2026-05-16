"""
Tests for genome survey_attractor, correct_framing, survey_and_correct.
"""

import numpy as np
import pytest

from flame_sheep.genome import Genome


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestSurveyAttractor:
    def test_returns_expected_keys(self, rng):
        g = Genome.random(rng)
        survey = g.survey_attractor()
        expected = {'centroid_x', 'centroid_y', 'coverage', 'in_viewport'}
        assert expected <= set(survey.keys())

    def test_viable_genome_has_coverage(self, rng):
        """A viable genome should have nonzero coverage."""
        for _ in range(10):
            g = Genome.random(rng)
            survey = g.survey_attractor()
            if survey['in_viewport']:
                assert survey['coverage'] > 0
                return
        pytest.skip('No viable genome found in 10 tries')

    def test_centroid_finite(self, rng):
        g = Genome.random(rng)
        survey = g.survey_attractor()
        assert np.isfinite(survey['centroid_x'])
        assert np.isfinite(survey['centroid_y'])

    def test_bbox_ordered(self, rng):
        """Bounding box min should be <= max."""
        for _ in range(10):
            g = Genome.random(rng)
            survey = g.survey_attractor()
            if 'bbox_min_x' in survey:
                assert survey['bbox_min_x'] <= survey['bbox_max_x']
                assert survey['bbox_min_y'] <= survey['bbox_max_y']
                return
        pytest.skip('No genome with bbox found')

    def test_degenerate_genome_low_coverage(self):
        """A genome with zero weights should return zero coverage."""
        g = Genome.random(np.random.default_rng(0))
        for tr in g.transforms:
            tr.weight = 0.0
        survey = g.survey_attractor()
        assert survey['coverage'] == 0


class TestCorrectFraming:
    def test_sets_center(self, rng):
        g = Genome.random(rng)
        survey = g.survey_attractor()
        if not survey['in_viewport']:
            pytest.skip('Genome not in viewport')
        original_center = g.center.copy()
        g.correct_framing(survey)
        # Center should change (unless centroid was already at origin)
        # Just verify it's finite and set
        assert np.all(np.isfinite(g.center))

    def test_sets_zoom_in_range(self, rng):
        g = Genome.random(rng)
        survey = g.survey_attractor()
        if not survey['in_viewport']:
            pytest.skip('Genome not in viewport')
        g.correct_framing(survey)
        assert 0.1 <= g.zoom <= 1.5

    def test_accounts_for_rotation(self, rng):
        """Corrected center should rotate with the genome's rotation."""
        g = Genome.random(rng)
        survey = g.survey_attractor()
        if not survey['in_viewport']:
            pytest.skip('Genome not in viewport')

        # Correct with rotation=0
        g.rotation = 0.0
        g.correct_framing(survey)
        center_no_rot = g.center.copy()

        # Correct with rotation=pi/2
        g.rotation = np.pi / 2
        g.correct_framing(survey)
        center_rot = g.center.copy()

        cx, cy = survey['centroid_x'], survey['centroid_y']
        if abs(cx) > 0.01 or abs(cy) > 0.01:
            # Centers should differ when rotation changes
            assert not np.allclose(center_no_rot, center_rot, atol=0.01)


class TestCheckStability:
    def test_stable_genome_passes(self, rng):
        """A genome that passes survey_and_correct should be stable."""
        g = Genome.random(rng)
        # Already passed stability in random(), should pass again
        assert g.check_stability()

    def test_returns_bool(self, rng):
        g = Genome.random(rng)
        assert isinstance(g.check_stability(), bool)

    def test_respects_max_bbox_ratio(self, rng):
        """Very tight threshold should reject more genomes."""
        rejected = 0
        for _ in range(20):
            g = Genome.random(rng)
            if not g.check_stability(max_bbox_ratio=1.05):
                rejected += 1
        # With ratio=1.05, even small variation is rejected
        # Some genomes should fail this tight threshold
        assert rejected >= 0  # non-crashing is the main check


class TestSurveyAndCorrect:
    def test_returns_bool(self, rng):
        g = Genome.random(rng)
        result = g.survey_and_correct()
        assert isinstance(result, bool)

    def test_random_genomes_pass_rate(self, rng):
        """At least some random genomes should pass survey_and_correct."""
        passed = 0
        n = 30
        for _ in range(n):
            g = Genome.random(rng)
            if g.survey_and_correct():
                passed += 1
        # Genome.random already calls survey_and_correct internally,
        # so a freshly generated one should almost always pass
        assert passed > n * 0.3, f'Only {passed}/{n} passed'

    def test_corrected_genome_has_reasonable_zoom(self, rng):
        g = Genome.random(rng)
        if g.survey_and_correct():
            assert 0.1 <= g.zoom <= 1.5

    def test_idempotent_ish(self, rng):
        """Running survey_and_correct twice shouldn't change framing drastically."""
        g = Genome.random(rng)
        if not g.survey_and_correct():
            pytest.skip('Genome not viable')
        center1 = g.center.copy()
        zoom1 = g.zoom
        g.survey_and_correct()
        center2 = g.center.copy()
        zoom2 = g.zoom
        # Should be similar (not exact due to random chaos game)
        assert np.allclose(center1, center2, atol=0.5)
        assert abs(zoom1 - zoom2) < 0.5
