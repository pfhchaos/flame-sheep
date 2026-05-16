"""
Tests for CompareMode: pair selection, active learning, winner tracking.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from flame_sheep.genome import Genome
from flame_sheep.storage import Library
from flame_sheep.compare import CompareMode, PairState


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
                'UPDATE genomes SET render_static=?, cnn_score=? WHERE id=?',
                (b'fake_png', float(i) * 0.1, gid),
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
                'UPDATE genomes SET render_static=? WHERE id=?',
                (b'fake_png', gid),
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
        """Active learning should pick genomes with similar scores."""
        cm = CompareMode(tmp_lib)
        cm._build_candidates()
        pair = cm.pick_pair()
        # Get the scores
        left_score = None
        right_score = None
        for gid, score in cm._candidates:
            if gid == pair.left_id:
                left_score = score
            if gid == pair.right_id:
                right_score = score
        assert left_score is not None and right_score is not None
        # Should be adjacent in sorted order — gap should be small
        assert abs(left_score - right_score) <= 0.15  # one step is 0.1

    def test_not_enough_genomes(self):
        """With <2 genomes, pick_pair returns empty state."""
        with tempfile.TemporaryDirectory() as d:
            lib = Library(data_dir=Path(d))
            rng = np.random.default_rng(42)
            g = Genome.random(rng)
            gid = lib.save_genome(g, g.aesthetic_score())
            lib.conn.execute(
                'UPDATE genomes SET render_static=? WHERE id=?',
                (b'fake', gid))
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
    def test_shuffles_when_mostly_unscored(self, tmp_lib_unscored):
        """When >50% unscored, should shuffle instead of sorting by score."""
        cm = CompareMode(tmp_lib_unscored)
        cm._build_candidates()
        # Should still have candidates
        assert len(cm._candidates) == 20

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

    def test_left_wins_keeps_winner(self, tmp_lib):
        cm = CompareMode(tmp_lib)
        cm.pick_pair()
        winner_id = cm.pair.left_id
        cm.on_left_wins()
        # Left (winner) should remain
        assert cm.pair.left_id == winner_id

    def test_right_wins_keeps_winner(self, tmp_lib):
        cm = CompareMode(tmp_lib)
        cm.pick_pair()
        winner_id = cm.pair.right_id
        cm.on_right_wins()
        # Right (winner) should remain
        assert cm.pair.right_id == winner_id

    def test_winner_gets_new_opponent(self, tmp_lib):
        cm = CompareMode(tmp_lib)
        cm.pick_pair()
        old_right = cm.pair.right_id
        cm.on_left_wins()
        # Right side should be replaced with a new genome
        assert cm.pair.right_id != old_right or cm.pair.right_id is None

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
