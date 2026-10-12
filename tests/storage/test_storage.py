"""
Tests for persistent storage: genome serialization, loops, motion fields, ratings.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from flame_sheep.genome import Genome, Transform
from flame_sheep.storage import (
    Library, compute_motion_field, motion_field_coherence,
    motion_field_to_blob, motion_field_from_blob,
    _genome_to_json, _genome_from_json,
    score_loop,
    MOTION_GRID,
)


def _mk_genome_with_color_speeds(speeds):
    """Lightweight genome (no viability loop / scoring) whose transforms carry
    the given ``color_speed`` values. One transform per value."""
    trs = []
    for i, cs in enumerate(speeds):
        t = Transform()
        t.variations[i % t.variations.shape[0]] = 1.0
        t.weight = 1.0 + 0.1 * i
        t.color = 0.1 * i
        t.color_speed = float(cs)
        trs.append(t)
    return Genome(transforms=trs)


@pytest.fixture
def tmp_lib():
    """Library backed by a temp directory — cleaned up after test."""
    with tempfile.TemporaryDirectory() as d:
        lib = Library(data_dir=Path(d))
        yield lib
        lib.close()


# ----------------------------------------------------------------
# Genome serialization
# ----------------------------------------------------------------

class TestGenomeSerialization:

    def test_roundtrip_preserves_transforms(self):
        rng = np.random.default_rng(1)
        g = Genome.random(rng)
        restored = _genome_from_json(_genome_to_json(g))
        assert len(restored.transforms) == len(g.transforms)
        for orig, rest in zip(g.transforms, restored.transforms):
            np.testing.assert_allclose(rest.affine, orig.affine, atol=1e-5)
            np.testing.assert_allclose(rest.variations, orig.variations, atol=1e-5)
            assert abs(rest.color - orig.color) < 1e-5
            assert abs(rest.color_speed - orig.color_speed) < 1e-5
            assert abs(rest.weight - orig.weight) < 1e-5

    def test_roundtrip_preserves_palette(self):
        rng = np.random.default_rng(2)
        g = Genome.random(rng)
        restored = _genome_from_json(_genome_to_json(g))
        np.testing.assert_allclose(restored.palette, g.palette, atol=1e-5)

    def test_roundtrip_preserves_global_params(self):
        rng = np.random.default_rng(3)
        g = Genome.random(rng)
        restored = _genome_from_json(_genome_to_json(g))
        assert abs(restored.zoom - g.zoom) < 1e-5
        assert abs(restored.rotation - g.rotation) < 1e-5
        np.testing.assert_allclose(restored.center, g.center, atol=1e-5)

    # ---- color_speed catalog bug-class guard ----
    #
    # color_speed is a genuinely-varying, GPU-uploaded field (random on new
    # genomes, jittered in breeding, interpolated in morphs). An earlier DB
    # serializer serialized `color` but not `color_speed`, so any genome
    # round-tripped through the catalog silently lost it and reset to the 0.5
    # default. These guard the catalog save->load path directly, so they stay
    # green whether color_speed support comes from the standalone DB fix or from
    # the unified core serializer, and go RED if color_speed is dropped from
    # whichever serializer the catalog uses.

    def test_color_speed_survives_json_roundtrip(self):
        """NON-default color_speeds (never 0.5) must survive the catalog JSON
        round-trip — a 'defaults-anyway' pass cannot hide a drop."""
        speeds = [0.0, 0.17, 0.83, 1.0]
        g = _mk_genome_with_color_speeds(speeds)
        # Sanity: nothing here is the 0.5 default.
        assert all(abs(s - 0.5) > 1e-6 for s in speeds)
        restored = _genome_from_json(_genome_to_json(g))
        got = [tr.color_speed for tr in restored.transforms]
        np.testing.assert_allclose(got, speeds, atol=1e-6)

    def test_color_speed_persists_in_serialized_blob(self):
        """The field is actually written into the JSON blob (not merely rebuilt
        from a default on load)."""
        g = _mk_genome_with_color_speeds([0.83])
        blob = json.loads(_genome_to_json(g))
        assert abs(blob['transforms'][0]['color_speed'] - 0.83) < 1e-6

    def test_color_speed_survives_library_roundtrip(self, tmp_lib):
        """Full public catalog path: save_genome -> load_genome keeps a
        non-default color_speed."""
        g = _mk_genome_with_color_speeds([0.11, 0.92])
        scores = dict(coverage=0.0, entropy=0.0, color_entropy=0.0,
                      balance=0.0, complexity=0.0)
        gid = tmp_lib.save_genome(g, scores=scores)
        loaded = tmp_lib.load_genome(gid)
        np.testing.assert_allclose(
            [tr.color_speed for tr in loaded.transforms], [0.11, 0.92], atol=1e-6)

    def test_old_blob_without_color_speed_loads_as_default(self):
        """Backward compat: an old-format blob that predates color_speed loads
        with the field default (0.5), no error. Simulated by stripping the key
        from a freshly-serialized blob — exactly what an old row looks like."""
        g = _mk_genome_with_color_speeds([0.83, 0.2])
        data = json.loads(_genome_to_json(g))
        for td in data['transforms']:
            td.pop('color_speed', None)       # old rows have no such key
        restored = _genome_from_json(json.dumps(data))
        assert all(tr.color_speed == 0.5 for tr in restored.transforms)

    def test_db_id_not_in_blob(self):
        """Identity is the row key, not genome data — it must never appear in the
        serialized catalog blob (library.load_genome stamps it from the row)."""
        g = _mk_genome_with_color_speeds([0.3])
        g.db_id = 12345
        blob = json.loads(_genome_to_json(g))
        assert 'db_id' not in blob
        assert _genome_from_json(_genome_to_json(g)).db_id is None


# ----------------------------------------------------------------
# Motion field
# ----------------------------------------------------------------

class TestMotionField:

    def test_shape(self):
        rng = np.random.default_rng(10)
        a = Genome.random(rng)
        b = Genome.random(rng)
        mf = compute_motion_field(a, b)
        assert mf.shape == (MOTION_GRID, MOTION_GRID, 2)
        assert mf.dtype == np.float32

    def test_self_transition_near_zero(self):
        """Motion field from a genome to itself should be near zero."""
        rng = np.random.default_rng(11)
        g = Genome.random(rng)
        mf = compute_motion_field(g, g)
        assert np.max(np.abs(mf)) < 1.5  # not exactly zero due to RNG in chaos game

    def test_blob_roundtrip(self):
        field = np.random.randn(MOTION_GRID, MOTION_GRID, 2).astype(np.float32)
        restored = motion_field_from_blob(motion_field_to_blob(field))
        np.testing.assert_array_equal(restored, field)

    def test_coherence_identical(self):
        field = np.random.randn(MOTION_GRID, MOTION_GRID, 2).astype(np.float32)
        assert motion_field_coherence(field, field) > 0.9

    def test_coherence_opposite(self):
        field = np.random.randn(MOTION_GRID, MOTION_GRID, 2).astype(np.float32)
        assert motion_field_coherence(field, -field) < -0.9

    def test_coherence_in_range(self):
        rng = np.random.default_rng(12)
        for _ in range(10):
            a = rng.standard_normal((MOTION_GRID, MOTION_GRID, 2)).astype(np.float32)
            b = rng.standard_normal((MOTION_GRID, MOTION_GRID, 2)).astype(np.float32)
            c = motion_field_coherence(a, b)
            assert -1.0 <= c <= 1.0


# ----------------------------------------------------------------
# Library — genomes
# ----------------------------------------------------------------

class TestLibraryGenomes:

    def test_save_and_load(self, tmp_lib):
        rng = np.random.default_rng(20)
        g = Genome.random(rng)
        gid = tmp_lib.save_genome(g)
        loaded = tmp_lib.load_genome(gid)
        assert len(loaded.transforms) == len(g.transforms)
        np.testing.assert_allclose(loaded.palette, g.palette, atol=1e-5)

    def test_scores_stored(self, tmp_lib):
        rng = np.random.default_rng(21)
        g = Genome.random(rng)
        scores = g.aesthetic_score()
        gid = tmp_lib.save_genome(g, scores=scores)
        stored = tmp_lib.genome_scores(gid)
        # save_genome only persists the columns it has — some derived scores
        # (e.g. coverage_stability) are transient filtering signals and
        # never round-trip. Verify every persisted column matches its source
        # value rather than asserting every aesthetic_score key persists.
        for k in stored:
            if k in scores and stored[k] is not None:
                assert abs(stored[k] - scores[k]) < 1e-5, f"{k} mismatch"

    def test_load_missing_raises(self, tmp_lib):
        with pytest.raises(KeyError):
            tmp_lib.load_genome(9999)

    def test_genome_count(self, tmp_lib):
        rng = np.random.default_rng(22)
        assert tmp_lib.genome_count() == 0
        tmp_lib.save_genome(Genome.random(rng))
        tmp_lib.save_genome(Genome.random(rng))
        assert tmp_lib.genome_count() == 2

    @pytest.mark.slow  # 10x Genome.random() — pure-Python chaos-game survey, ~15s total
    def test_top_genomes(self, tmp_lib):
        rng = np.random.default_rng(23)
        for _ in range(10):
            tmp_lib.save_genome(Genome.random(rng))
        top = tmp_lib.top_genomes(n=5)
        assert len(top) <= 5
        # Should be sorted descending by composite fitness
        if len(top) >= 2:
            def composite(s):
                return s['entropy'] + s['color_entropy'] + s['balance'] * 0.5 + s['complexity']
            for i in range(len(top) - 1):
                assert composite(top[i][1]) >= composite(top[i + 1][1])


# ----------------------------------------------------------------
# Library — loops
# ----------------------------------------------------------------

class TestLibraryLoops:

    @pytest.mark.slow  # Genome.random() x4 + motion-field compute — pure-Python chaos game, ~7s
    def test_save_and_load_loop(self, tmp_lib):
        rng = np.random.default_rng(30)
        gids = [tmp_lib.save_genome(Genome.random(rng)) for _ in range(4)]
        loop_id = tmp_lib.save_loop(gids, name='test loop')
        loaded = tmp_lib.load_loop(loop_id)
        assert len(loaded) == 4
        assert [item[0] for item in loaded] == gids
        # Each item should have a motion field
        for gid, genome, mf in loaded:
            assert mf is not None
            assert mf.shape == (MOTION_GRID, MOTION_GRID, 2)

    @pytest.mark.slow  # Genome.random() x6 + motion-field compute — pure-Python chaos game, ~11s
    def test_loop_preserves_order(self, tmp_lib):
        rng = np.random.default_rng(31)
        gids = [tmp_lib.save_genome(Genome.random(rng)) for _ in range(6)]
        loop_id = tmp_lib.save_loop(gids)
        loaded = tmp_lib.load_loop(loop_id)
        assert [item[0] for item in loaded] == gids

    def test_load_missing_loop_raises(self, tmp_lib):
        with pytest.raises(KeyError):
            tmp_lib.load_loop(9999)

    @pytest.mark.slow  # Genome.random() x3 + motion-field compute — pure-Python chaos game, ~5s
    def test_loop_count(self, tmp_lib):
        rng = np.random.default_rng(32)
        assert tmp_lib.loop_count() == 0
        gids = [tmp_lib.save_genome(Genome.random(rng)) for _ in range(3)]
        tmp_lib.save_loop(gids)
        assert tmp_lib.loop_count() == 1


# ----------------------------------------------------------------
# Library — ratings
# ----------------------------------------------------------------

class TestLibraryRatings:

    def test_like_dislike(self, tmp_lib):
        rng = np.random.default_rng(40)
        gid = tmp_lib.save_genome(Genome.random(rng))
        tmp_lib.rate('genome', gid, +1)
        tmp_lib.rate('genome', gid, +1)
        tmp_lib.rate('genome', gid, -1)
        assert tmp_lib.net_rating('genome', gid) == 1

    def test_unrated_is_zero(self, tmp_lib):
        assert tmp_lib.net_rating('genome', 9999) == 0


# ----------------------------------------------------------------
# Loop-level fitness
# ----------------------------------------------------------------

class TestScoreLoop:

    @pytest.mark.slow  # Genome.random() x4 + motion-field compute — pure-Python chaos game, ~5s
    def test_returns_all_keys(self):
        rng = np.random.default_rng(50)
        genomes = [Genome.random(rng) for _ in range(4)]
        mfs = [compute_motion_field(genomes[i], genomes[(i+1) % 4]) for i in range(4)]
        scores = score_loop(genomes, mfs)
        expected = {'mean_coherence', 'min_coherence', 'diversity', 'palette_flow', 'smoothness', 'fitness'}
        assert set(scores.keys()) == expected

    @pytest.mark.slow  # Genome.random() x4 + motion-field compute — pure-Python chaos game, ~5s
    def test_all_values_finite(self):
        rng = np.random.default_rng(51)
        genomes = [Genome.random(rng) for _ in range(4)]
        mfs = [compute_motion_field(genomes[i], genomes[(i+1) % 4]) for i in range(4)]
        scores = score_loop(genomes, mfs)
        for k, v in scores.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"

    def test_identical_genomes_low_diversity(self):
        rng = np.random.default_rng(52)
        g = Genome.random(rng)
        genomes = [g, g, g, g]
        mfs = [np.zeros((MOTION_GRID, MOTION_GRID, 2), dtype=np.float32)] * 4
        scores = score_loop(genomes, mfs)
        assert scores['diversity'] == 0.0


@pytest.mark.slow
class TestLoopFitnessStorage:

    def test_fitness_stored_on_save(self, tmp_lib):
        rng = np.random.default_rng(60)
        gids = [tmp_lib.save_genome(Genome.random(rng)) for _ in range(4)]
        loop_id = tmp_lib.save_loop(gids)
        fitness = tmp_lib.loop_fitness(loop_id)
        assert 'fitness' in fitness
        assert 'diversity' in fitness
        assert all(np.isfinite(v) for v in fitness.values())

    def test_user_rating_affects_fitness(self, tmp_lib):
        rng = np.random.default_rng(61)
        gids = [tmp_lib.save_genome(Genome.random(rng)) for _ in range(4)]
        loop_id = tmp_lib.save_loop(gids)
        before = tmp_lib.loop_fitness(loop_id)['fitness']
        tmp_lib.rate('loop', loop_id, +1)
        tmp_lib.rate('loop', loop_id, +1)
        tmp_lib.update_loop_fitness(loop_id)
        after = tmp_lib.loop_fitness(loop_id)['fitness']
        assert after > before

    def test_top_loops_sorted(self, tmp_lib):
        rng = np.random.default_rng(62)
        for _ in range(5):
            gids = [tmp_lib.save_genome(Genome.random(rng)) for _ in range(4)]
            tmp_lib.save_loop(gids)
        top = tmp_lib.top_loops(n=5)
        for i in range(len(top) - 1):
            assert top[i][1]['fitness'] >= top[i+1][1]['fitness']

    def test_loop_genome_ids(self, tmp_lib):
        rng = np.random.default_rng(63)
        gids = [tmp_lib.save_genome(Genome.random(rng)) for _ in range(4)]
        loop_id = tmp_lib.save_loop(gids)
        assert tmp_lib.loop_genome_ids(loop_id) == gids

    def test_genome_db_id_roundtrip(self, tmp_lib):
        """Genome.db_id should be set after save + load."""
        rng = np.random.default_rng(70)
        genome = Genome.random(rng)
        assert genome.db_id is None
        gid = tmp_lib.save_genome(genome)
        loaded = tmp_lib.load_genome(gid)
        assert loaded.db_id == gid

    def test_genome_rating(self, tmp_lib):
        """Direct genome ratings should accumulate."""
        rng = np.random.default_rng(71)
        gid = tmp_lib.save_genome(Genome.random(rng))
        assert tmp_lib.net_rating('genome', gid) == 0
        tmp_lib.rate('genome', gid, +1)
        tmp_lib.rate('genome', gid, +1)
        tmp_lib.rate('genome', gid, -1)
        assert tmp_lib.net_rating('genome', gid) == 1
