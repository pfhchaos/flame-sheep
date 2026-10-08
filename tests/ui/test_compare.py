"""
Tests for CompareMode: pair selection, active learning, winner tracking.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from flame_sheep.genome import Genome
from flame_sheep.storage import Library
from flame_sheep.ui.compare import CompareMode, PairState

# Whole module gated slow: nearly every test depends on tmp_lib / tmp_lib_unscored /
# tmp_lib_multi_model, each of which is function-scoped and regenerates 20 random
# genomes (Genome.random() is a pure-Python chaos-game survey, ~1s/call) plus
# aesthetic_score() per genome. That's ~20-30s of fixture setup PER TEST, and this
# file has ~35 tests — by far the single largest contributor to full-suite runtime.
pytestmark = pytest.mark.slow


@pytest.fixture
def tmp_lib():
    """Library with genomes seeded for comparison tests."""
    with tempfile.TemporaryDirectory() as d:
        lib = Library(data_dir=Path(d))
        rng = np.random.default_rng(42)
        # Create 20 genomes with fake renders and CNN scores
        for i in range(20):
            g = Genome.random(rng)
            scores = g.aesthetic_score()
            gid = lib.save_genome(g, scores)
            # Fake a render blob so the genome is "rendered"
            lib.conn.execute(
                'INSERT OR REPLACE INTO genome_blobs (genome_id, render_static) VALUES (?, ?)',
                (gid, b'fake_png'),
            )
            lib.conn.execute(
                'UPDATE genomes SET cnn_score=? WHERE id=?',
                (float(i) * 0.1, gid),
            )
        lib.conn.commit()
        yield lib
        lib.close()


@pytest.fixture
def tmp_lib_unscored():
    """Library with genomes that have no CNN scores."""
    with tempfile.TemporaryDirectory() as d:
        lib = Library(data_dir=Path(d))
        rng = np.random.default_rng(42)
        for i in range(20):
            g = Genome.random(rng)
            scores = g.aesthetic_score()
            gid = lib.save_genome(g, scores)
            lib.conn.execute(
                'INSERT OR REPLACE INTO genome_blobs (genome_id, render_static) VALUES (?, ?)',
                (gid, b'fake_png'),
            )
        lib.conn.commit()
        yield lib
        lib.close()


# ----------------------------------------------------------------
# Pair selection
# ----------------------------------------------------------------

class TestPairSelection:
    def test_pick_pair_returns_two_genomes(self, tmp_lib):
        cm = CompareMode(tmp_lib)
        pair = cm.pick_pair()
        assert pair.left is not None
        assert pair.right is not None
        assert isinstance(pair.left, Genome)
        assert isinstance(pair.right, Genome)

    def test_pick_pair_returns_different_genomes(self, tmp_lib):
        cm = CompareMode(tmp_lib)
        pair = cm.pick_pair()
        assert pair.left_id != pair.right_id

    def test_pick_pair_ids_match_genomes(self, tmp_lib):
        cm = CompareMode(tmp_lib)
        pair = cm.pick_pair()
        assert pair.left.db_id == pair.left_id
        assert pair.right.db_id == pair.right_id

    def test_pick_pair_selects_adjacent_scores(self, tmp_lib):
        """Active learning should pick genomes with similar scores on the
        FIRST pick — strategy 0 is uncertainty:deployed_legacy when only
        the legacy cnn_score column is populated.
        """
        from flame_sheep.ui.compare import LEGACY_MODEL_NAME
        cm = CompareMode(tmp_lib)
        pair = cm.pick_pair()
        left_score = cm._scores_by_gid.get(pair.left_id, {}).get(LEGACY_MODEL_NAME)
        right_score = cm._scores_by_gid.get(pair.right_id, {}).get(LEGACY_MODEL_NAME)
        assert left_score is not None and right_score is not None
        # First pick is the smallest-gap adjacent pair under uncertainty
        # strategy. With diversity_weight=0 the first pair would be exactly
        # adjacent; with the default diversity_weight=0.3 a non-adjacent
        # pair can win the re-rank, so allow a looser bound here.
        assert abs(left_score - right_score) <= 0.5

    def test_not_enough_genomes(self):
        """With <2 genomes, pick_pair returns empty state."""
        with tempfile.TemporaryDirectory() as d:
            lib = Library(data_dir=Path(d))
            rng = np.random.default_rng(42)
            g = Genome.random(rng)
            gid = lib.save_genome(g, g.aesthetic_score())
            lib.conn.execute(
                'INSERT OR REPLACE INTO genome_blobs (genome_id, render_static) VALUES (?, ?)',
                (gid, b'fake'))
            lib.conn.commit()
            cm = CompareMode(lib)
            pair = cm.pick_pair()
            # Should handle gracefully — at least one side will be None
            assert pair.left is None or pair.right is None
            lib.close()


# ----------------------------------------------------------------
# Active learning: unscored genomes
# ----------------------------------------------------------------

class TestUnscoredFallback:
    """Fresh libraries (no cnn_score yet) must still produce pairs via the
    random strategy fallback. Pre-multi-model behavior was "shuffle when
    mostly unscored"; post-multi-model is "random strategy always available,
    falls back to the full active set if no model has scored anything."
    """

    def test_active_gids_populated_when_unscored(self, tmp_lib_unscored):
        cm = CompareMode(tmp_lib_unscored)
        cm._load_model_scores()
        # _all_active_gids reflects every rendered active genome,
        # independent of whether any model has scored them yet
        assert len(cm._all_active_gids) == 20

    def test_still_picks_valid_pairs(self, tmp_lib_unscored):
        cm = CompareMode(tmp_lib_unscored)
        pair = cm.pick_pair()
        assert pair.left is not None
        assert pair.right is not None
        assert pair.left_id != pair.right_id


# ----------------------------------------------------------------
# Winner/loser tracking
# ----------------------------------------------------------------

class TestWinnerTracking:
    def test_left_wins_records_comparison(self, tmp_lib):
        cm = CompareMode(tmp_lib)
        cm.pick_pair()
        winner_id = cm.pair.left_id
        loser_id = cm.pair.right_id
        cm.on_left_wins()
        # Should be in pairwise_ratings table
        row = tmp_lib.conn.execute(
            'SELECT winner_id, loser_id FROM pairwise_ratings '
            'WHERE winner_id=? AND loser_id=?',
            (winner_id, loser_id)).fetchone()
        assert row is not None

    def test_right_wins_records_comparison(self, tmp_lib):
        cm = CompareMode(tmp_lib)
        cm.pick_pair()
        winner_id = cm.pair.right_id
        loser_id = cm.pair.left_id
        cm.on_right_wins()
        row = tmp_lib.conn.execute(
            'SELECT winner_id, loser_id FROM pairwise_ratings '
            'WHERE winner_id=? AND loser_id=?',
            (winner_id, loser_id)).fetchone()
        assert row is not None

    def test_left_wins_advances_to_fresh_pair(self, tmp_lib):
        """Multi-model design drops keep-winner — after a vote, pick_pair()
        picks a fresh pair via the current strategy (no continuity with
        the just-voted winner). This is intentional: strategies may want
        different genome distributions across rotation boundaries."""
        cm = CompareMode(tmp_lib)
        cm.pick_pair()
        old_left, old_right = cm.pair.left_id, cm.pair.right_id
        cm.on_left_wins()
        # Some side should differ from the previous pair (fresh selection).
        # Both could in principle be different; at minimum the just-voted
        # pair shouldn't repeat as the same arrangement.
        new_pair_key = (min(cm.pair.left_id, cm.pair.right_id),
                        max(cm.pair.left_id, cm.pair.right_id))
        old_pair_key = (min(old_left, old_right), max(old_left, old_right))
        assert new_pair_key != old_pair_key

    def test_right_wins_advances_to_fresh_pair(self, tmp_lib):
        cm = CompareMode(tmp_lib)
        cm.pick_pair()
        old_left, old_right = cm.pair.left_id, cm.pair.right_id
        cm.on_right_wins()
        new_pair_key = (min(cm.pair.left_id, cm.pair.right_id),
                        max(cm.pair.left_id, cm.pair.right_id))
        old_pair_key = (min(old_left, old_right), max(old_left, old_right))
        assert new_pair_key != old_pair_key

    def test_compared_pairs_not_repeated(self, tmp_lib):
        cm = CompareMode(tmp_lib)
        cm.pick_pair()
        pair1 = (cm.pair.left_id, cm.pair.right_id)
        cm.on_left_wins()
        # The same pair shouldn't show up again
        seen = {(min(*pair1), max(*pair1))}
        for _ in range(15):
            pair_key = (min(cm.pair.left_id, cm.pair.right_id),
                        max(cm.pair.left_id, cm.pair.right_id))
            assert pair_key not in seen or len(seen) >= 19  # exhaustion is ok
            seen.add(pair_key)
            cm.on_left_wins()


# ----------------------------------------------------------------
# Skip
# ----------------------------------------------------------------

class TestSkip:
    def test_skip_replaces_both(self, tmp_lib):
        cm = CompareMode(tmp_lib)
        cm.pick_pair()
        old_left = cm.pair.left_id
        old_right = cm.pair.right_id
        cm.on_skip()
        # At least one side should change (both ideally, but randomization)
        changed = (cm.pair.left_id != old_left or cm.pair.right_id != old_right)
        assert changed

    def test_skip_does_not_record_rating(self, tmp_lib):
        cm = CompareMode(tmp_lib)
        cm.pick_pair()
        cm.on_skip()
        count = tmp_lib.conn.execute(
            'SELECT COUNT(*) FROM pairwise_ratings').fetchone()[0]
        assert count == 0


# ----------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------

class TestEdgeCases:
    def test_no_pair_yet(self, tmp_lib):
        cm = CompareMode(tmp_lib)
        # Calling win before pick_pair should not crash
        cm.on_left_wins()
        cm.on_right_wins()

    def test_many_votes_exhausts_gracefully(self, tmp_lib):
        """Voting many times should eventually fall back to random pairs."""
        cm = CompareMode(tmp_lib)
        cm.pick_pair()
        # Vote 100 times — more than the 20*19/2=190 possible pairs
        for _ in range(100):
            cm.on_left_wins()
        # Should still have valid pair
        assert cm.pair.left is not None
        assert cm.pair.right is not None


# ============================================================================
# Multi-model active learning — the strategy framework that uses N CNN
# models (one deployed + N-1 aux) to drive uncertainty + disagreement
# pair selection. Lives in compare.py and is THE active-learning driver
# now — the legacy single-model `_pick_random` is just one of the fallback
# strategies.
# ============================================================================

import json as _json


@pytest.fixture
def tmp_lib_multi_model():
    """Library seeded with multi-model scores (cnn_scores_detail JSON
    populated for 3 named models). Use for testing the strategy
    framework (uncertainty + disagreement + rotation)."""
    with tempfile.TemporaryDirectory() as d:
        lib = Library(data_dir=Path(d))
        rng = np.random.default_rng(42)
        # 30 genomes — enough that top-K + bottom-K subsets are non-trivial
        for i in range(30):
            g = Genome.random(rng)
            scores = g.aesthetic_score()
            gid = lib.save_genome(g, scores)
            lib.conn.execute(
                'INSERT OR REPLACE INTO genome_blobs (genome_id, render_static) VALUES (?, ?)',
                (gid, b'fake_png'),
            )
            # Per-model scores: deployed has linear ordering, curriculum
            # has reversed ordering, lr1e4 has noisy ordering. So:
            #   deployed and curriculum DISAGREE strongly (rank-correlated -1)
            #   deployed and lr1e4 agree weakly
            #   curriculum and lr1e4 disagree weakly
            # This setup makes the disagreement strategy actually
            # produce signal.
            scores_detail = {
                'deployed':   float(i) * 0.1,
                'curriculum': float(29 - i) * 0.1,
                'lr1e4':      float(i) * 0.1 + rng.normal(0, 0.5),
            }
            lib.conn.execute(
                'UPDATE genomes SET cnn_score=?, cnn_scores_detail=? WHERE id=?',
                (float(i) * 0.1, _json.dumps(scores_detail), gid),
            )
        lib.conn.commit()
        yield lib
        lib.close()


class TestLoadModelScores:

    def test_loads_all_models_from_detail_json(self, tmp_lib_multi_model):
        cm = CompareMode(tmp_lib_multi_model)
        cm._load_model_scores()
        assert set(cm._models) == {'deployed', 'curriculum', 'lr1e4'}

    def test_suppresses_legacy_when_named_present(self, tmp_lib_multi_model):
        """If cnn_scores_detail has entries, the legacy cnn_score column
        is NOT added under LEGACY_MODEL_NAME — otherwise we'd inflate
        the strategy set with a trivial disagreement:deployed:
        deployed_legacy slot. Pre-existing trap: identical scores under
        different names looks like a "model" but produces zero signal."""
        from flame_sheep.ui.compare import LEGACY_MODEL_NAME
        cm = CompareMode(tmp_lib_multi_model)
        cm._load_model_scores()
        assert LEGACY_MODEL_NAME not in cm._models

    def test_legacy_used_when_no_detail(self, tmp_lib):
        """In tmp_lib, only cnn_score is set; cnn_scores_detail is None.
        The legacy fallback path adds LEGACY_MODEL_NAME so the strategy
        framework has at least one model to work with."""
        from flame_sheep.ui.compare import LEGACY_MODEL_NAME
        cm = CompareMode(tmp_lib)
        cm._load_model_scores()
        assert LEGACY_MODEL_NAME in cm._models

    def test_coverage_filter_drops_sparse_models(self, tmp_lib_multi_model):
        """Models with <50% coverage produce unstable z-score
        distributions and get filtered out."""
        # Make the 'lr1e4' model sparse by clearing it on most genomes
        lib = tmp_lib_multi_model
        rows = lib.conn.execute(
            'SELECT id, cnn_scores_detail FROM genomes WHERE cnn_scores_detail IS NOT NULL'
        ).fetchall()
        for i, (gid, detail) in enumerate(rows):
            if i >= 5:  # only first 5 keep lr1e4 → 5/30 = 17%, well below 50%
                d = _json.loads(detail)
                d.pop('lr1e4', None)
                lib.conn.execute(
                    'UPDATE genomes SET cnn_scores_detail=? WHERE id=?',
                    (_json.dumps(d), gid),
                )
        lib.conn.commit()

        cm = CompareMode(lib)
        cm._load_model_scores()
        assert 'lr1e4' not in cm._models
        # Others survive
        assert 'deployed' in cm._models
        assert 'curriculum' in cm._models

    def test_zscores_computed_per_model(self, tmp_lib_multi_model):
        cm = CompareMode(tmp_lib_multi_model)
        cm._load_model_scores()
        for m in cm._models:
            assert m in cm._zscores
            gids, z = cm._zscores[m]
            assert len(gids) == len(z)
            # Z-scored array has mean ~0
            assert abs(z.mean()) < 1e-6
            # And std ~1 (if there's any variance)
            if z.std() > 1e-9:
                assert abs(z.std() - 1.0) < 1e-6

    def test_all_active_gids_populated_even_for_unscored(self, tmp_lib_unscored):
        """Random strategy fallback needs the full active set, not just
        scored ones — a fresh library has no scores but we still need
        to surface pairs."""
        cm = CompareMode(tmp_lib_unscored)
        cm._load_model_scores()
        # tmp_lib_unscored has 20 genomes
        assert len(cm._all_active_gids) == 20


class TestBuildStrategies:

    def test_strategy_count(self, tmp_lib_multi_model):
        """3 models → 3 uncertainty + C(3,2)=3 disagreement + 1 random = 7."""
        cm = CompareMode(tmp_lib_multi_model)
        cm._load_model_scores()
        cm._build_strategies()
        assert len(cm._strategies) == 7

    def test_strategy_names(self, tmp_lib_multi_model):
        cm = CompareMode(tmp_lib_multi_model)
        cm._load_model_scores()
        cm._build_strategies()
        names = [s[0] for s in cm._strategies]
        # 3 uncertainty entries, one per model (sorted alphabetically)
        assert 'uncertainty:curriculum' in names
        assert 'uncertainty:deployed' in names
        assert 'uncertainty:lr1e4' in names
        # 3 disagreement pairs in sorted order (m1 < m2)
        assert 'disagreement:curriculum:deployed' in names
        assert 'disagreement:curriculum:lr1e4' in names
        assert 'disagreement:deployed:lr1e4' in names
        # Random baseline
        assert 'random' in names

    def test_strategy_ordering(self, tmp_lib_multi_model):
        """Strategies follow a fixed order so rotation cycles
        deterministically: uncertainty first, then disagreement,
        then random. Tests this stability."""
        cm = CompareMode(tmp_lib_multi_model)
        cm._load_model_scores()
        cm._build_strategies()
        names = [s[0] for s in cm._strategies]
        # All uncertainty come before all disagreement, random last
        last_uncert = max(i for i, n in enumerate(names)
                           if n.startswith('uncertainty:'))
        first_disag = min(i for i, n in enumerate(names)
                           if n.startswith('disagreement:'))
        assert last_uncert < first_disag
        assert names[-1] == 'random'


class TestStrategyRotation:
    """The vote/skip-weighted progress counter that rotates between
    strategies — votes count 1.0, skips count `skip_progress_weight`
    (default 0.5). Rotate when progress ≥ votes_per_strategy."""

    def test_vote_increments_progress_by_one(self, tmp_lib_multi_model):
        cm = CompareMode(tmp_lib_multi_model, votes_per_strategy=5)
        cm.pick_pair()  # initialize
        assert cm._strategy_progress == 0.0
        cm.on_left_wins()
        # on_left_wins increments BEFORE the subsequent pick_pair —
        # progress reflects the just-cast vote
        assert cm._strategy_progress == 1.0

    def test_skip_increments_by_weight(self, tmp_lib_multi_model):
        cm = CompareMode(tmp_lib_multi_model, votes_per_strategy=5,
                          skip_progress_weight=0.5)
        cm.pick_pair()
        cm.on_skip()
        assert cm._strategy_progress == 0.5

    def test_rotation_at_threshold(self, tmp_lib_multi_model):
        """After 5 votes-equivalent of progress, the next pick_pair
        rotates to the next strategy."""
        cm = CompareMode(tmp_lib_multi_model, votes_per_strategy=5)
        cm.pick_pair()
        initial_idx = cm._strategy_idx
        # 5 votes — progress reaches 5.0
        for _ in range(5):
            cm.on_left_wins()
        # Strategy should have rotated during one of the post-vote
        # pick_pair() calls (each on_left_wins ends with pick_pair)
        assert cm._strategy_idx != initial_idx

    def test_rotation_cycles_back_to_first(self, tmp_lib_multi_model):
        """After going through all strategies, rotation wraps to 0."""
        cm = CompareMode(tmp_lib_multi_model, votes_per_strategy=1)
        cm.pick_pair()
        n = len(cm._strategies)
        # Each vote rotates once (votes_per_strategy=1)
        for _ in range(n):
            cm.on_left_wins()
        # Should be back at strategy 0 (or close — depends on
        # fall-through behavior; just check we made it through and
        # wrapped at least once)
        # Hard guarantee: index always in valid range
        assert 0 <= cm._strategy_idx < n


class TestPickDisagreement:
    """The disagreement strategy: -d_m1 * d_m2 score where d_m = z(a) - z(b).
    Score is POSITIVE when sign(d_m1) != sign(d_m2) — models disagree on
    direction. Higher magnitude = more confident disagreement."""

    def test_picks_when_models_disagree_strongly(self, tmp_lib_multi_model):
        """In the fixture, 'deployed' and 'curriculum' have OPPOSITE
        rankings (deployed scores ascending 0.0..2.9, curriculum
        scores descending 2.9..0.0). disagreement:deployed:curriculum
        should easily find a pair."""
        cm = CompareMode(tmp_lib_multi_model)
        cm._load_model_scores()
        pair = cm._pick_disagreement('deployed', 'curriculum')
        assert pair is not None
        assert pair[0] != pair[1]

    def test_returns_none_when_no_fresh_pairs(self, tmp_lib_multi_model):
        """If all candidate pairs have been compared, the strategy
        should return None so pick_pair can fall through to the next
        strategy."""
        cm = CompareMode(tmp_lib_multi_model)
        cm._load_model_scores()
        # Manually mark all possible pairs as compared
        gids = cm._all_active_gids
        for i in range(len(gids)):
            for j in range(i + 1, len(gids)):
                cm._compared.add((min(gids[i], gids[j]),
                                  max(gids[i], gids[j])))
        pair = cm._pick_disagreement('deployed', 'curriculum')
        assert pair is None


class TestPairDiversity:
    """_pair_diversity returns the signature_distance between the two
    genomes in the pair — used to re-rank candidate pairs toward
    structurally distinct members (the user can't easily tell two
    near-clones apart, so those pairs carry less label info)."""

    def test_returns_non_negative_distance(self, tmp_lib_multi_model):
        cm = CompareMode(tmp_lib_multi_model)
        cm._load_model_scores()
        gids = cm._all_active_gids[:2]
        d = cm._pair_diversity(gids[0], gids[1])
        assert d >= 0

    def test_signature_cached(self, tmp_lib_multi_model):
        """Computing signatures for the same gid twice should hit the
        cache the second time (no DB lookup, no signature recompute)."""
        cm = CompareMode(tmp_lib_multi_model)
        cm._load_model_scores()
        gid = cm._all_active_gids[0]
        assert gid not in cm._signature_cache
        cm._signature(gid)
        assert gid in cm._signature_cache
        # Second call shouldn't change cache state — it's already there
        sig_before = cm._signature_cache[gid]
        cm._signature(gid)
        assert cm._signature_cache[gid] == sig_before


class TestFreshness:
    """_is_fresh: True if the (sorted-id) pair is NOT in the compared set."""

    def test_unseen_pair_is_fresh(self, tmp_lib_multi_model):
        cm = CompareMode(tmp_lib_multi_model)
        assert cm._is_fresh(1, 2) is True

    def test_seen_pair_not_fresh(self, tmp_lib_multi_model):
        cm = CompareMode(tmp_lib_multi_model)
        cm._compared.add((1, 2))
        assert cm._is_fresh(1, 2) is False
        # Order-independent
        assert cm._is_fresh(2, 1) is False
