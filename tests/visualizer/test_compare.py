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
