"""Comparison mode for pairwise genome rating.

Manages pair selection, active learning, and state for A/B comparison.
The renderer handles the visual split — this module handles the logic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .genome import Genome

log = logging.getLogger(__name__)

if __name__ != '__main__':
    from .storage import Library


@dataclass
class PairState:
    """Current comparison pair."""
    left: Genome | None = None
    right: Genome | None = None
    left_id: int | None = None
    right_id: int | None = None


class CompareMode:
    """Active-learning pairwise comparison manager.

    Picks pairs where the CNN scorer is most uncertain (smallest score gap).
    Tracks which genomes have been compared to avoid repeats.
    """

    def __init__(self, lib: Library, score_fn=None):
        self.lib = lib
        self._score_fn = score_fn  # callable: Genome -> float (CNN score)
        self.pair = PairState()
        self._compared: set[tuple[int, int]] = set()
        self._score_cache: dict[int, float] = {}
        self._candidates: list[tuple[int, float]] = []  # (genome_id, score)
        self._candidate_idx = 0
        self.rng = np.random.default_rng()

    def set_score_fn(self, fn) -> None:
        """Set the scoring function (genome_id -> float)."""
        self._score_fn = fn

    def _build_candidates(self) -> None:
        """Score all genomes and sort by score for pair selection."""
        # Get all genomes with renders
        rows = self.lib.conn.execute(
            '''SELECT id, cnn_score FROM genomes
               WHERE render_static IS NOT NULL
                 AND COALESCE(archived, 0) = 0
               ORDER BY id'''
        ).fetchall()

        scored = []
        n_scored = 0
        for gid, cnn_score in rows:
            if cnn_score is not None:
                scored.append((gid, float(cnn_score)))
                n_scored += 1
            elif self._score_fn is not None:
                try:
                    g = self.lib.load_genome(gid)
                    s = self._score_fn(g)
                    scored.append((gid, s))
                    n_scored += 1
                except Exception:
                    continue
            else:
                # No score — assign random value for diversity
                scored.append((gid, self.rng.random()))

        if n_scored < len(scored) * 0.5:
            # Most genomes unscored — shuffle for diversity instead of
            # clustering at score=0 which causes repeated pairs
            self.rng.shuffle(scored)
            log.info(f'[compare] {n_scored}/{len(scored)} scored, using random order')
        else:
            scored.sort(key=lambda x: x[1])
        self._candidates = scored
        self._candidate_idx = 0
        log.info(f'[compare] built candidate pool: {len(scored)} genomes')

    def pick_pair(self) -> PairState:
        """Select a pair where the scorer is most uncertain.

        Finds adjacent genomes in the score ranking (smallest gap)
        that haven't been compared yet.
        """
        if not self._candidates:
            self._build_candidates()

        if len(self._candidates) < 2:
            log.warning('[compare] not enough candidates for comparison')
            return self.pair

        # Find the pair with smallest score gap that hasn't been compared
        best_pair = None
        best_gap = float('inf')

        # Search from where we left off to avoid always showing the same pair
        n = len(self._candidates)
        for offset in range(n - 1):
            i = (self._candidate_idx + offset) % (n - 1)
            gid_a, score_a = self._candidates[i]
            gid_b, score_b = self._candidates[i + 1]

            pair_key = (min(gid_a, gid_b), max(gid_a, gid_b))
            if pair_key in self._compared:
                continue

            gap = abs(score_a - score_b)
            if gap < best_gap:
                best_gap = gap
                best_pair = (gid_a, gid_b)
                self._candidate_idx = i + 1
                break  # take the first uncompared adjacent pair

        if best_pair is None:
            # All adjacent pairs compared — pick random
            gid_a, gid_b = self.rng.choice(n, 2, replace=False)
            best_pair = (self._candidates[gid_a][0], self._candidates[gid_b][0])
            log.debug('[compare] all adjacent pairs seen, using random')

        # Load the genomes
        gid_a, gid_b = best_pair
        # Randomize left/right so position doesn't bias
        if self.rng.random() > 0.5:
            gid_a, gid_b = gid_b, gid_a

        self.pair.left = self.lib.load_genome(gid_a)
        self.pair.right = self.lib.load_genome(gid_b)
        self.pair.left_id = gid_a
        self.pair.right_id = gid_b

        log.info(f'[compare] pair: #{gid_a} vs #{gid_b} (gap={best_gap:.3f})')
        return self.pair

    def on_left_wins(self) -> None:
        """Record left genome as winner."""
        if self.pair.left_id is None or self.pair.right_id is None:
            return
        self._record(self.pair.left_id, self.pair.right_id)
        # Keep winner, replace loser
        old_right = self.pair.right_id
        self._pick_opponent('right')
        log.info(f'[compare] left #{self.pair.left_id} wins over #{old_right}')

    def on_right_wins(self) -> None:
        """Record right genome as winner."""
        if self.pair.left_id is None or self.pair.right_id is None:
            return
        self._record(self.pair.right_id, self.pair.left_id)
        # Keep winner, replace loser
        old_left = self.pair.left_id
        self._pick_opponent('left')
        log.info(f'[compare] right #{self.pair.right_id} wins over #{old_left}')

    def on_skip(self) -> None:
        """Skip this pair — replace both."""
        self.pick_pair()

    def _record(self, winner_id: int, loser_id: int) -> None:
        """Store pairwise comparison result."""
        pair_key = (min(winner_id, loser_id), max(winner_id, loser_id))
        self._compared.add(pair_key)

        # Store in pairwise_ratings table
        self.lib.conn.execute(
            '''INSERT INTO pairwise_ratings (winner_id, loser_id, source)
               VALUES (?, ?, 'compare')''',
            (winner_id, loser_id),
        )
        self.lib.conn.commit()

    def _pick_opponent(self, replace_side: str) -> None:
        """Replace the losing side with a new genome near the winner in score."""
        winner_id = self.pair.left_id if replace_side == 'right' else self.pair.right_id

        # Find the winner's position in the candidate list
        winner_score = None
        for i, (gid, score) in enumerate(self._candidates):
            if gid == winner_id:
                winner_score = score
                break

        if winner_score is None:
            # Winner not in candidate list — pick random
            self.pick_pair()
            return

        # Find nearest uncompared genome to the winner
        best = None
        best_gap = float('inf')
        for gid, score in self._candidates:
            if gid == winner_id or gid == self.pair.left_id or gid == self.pair.right_id:
                continue
            pair_key = (min(gid, winner_id), max(gid, winner_id))
            if pair_key in self._compared:
                continue
            gap = abs(score - winner_score)
            if gap < best_gap:
                best_gap = gap
                best = gid

        if best is None:
            self.pick_pair()
            return

        genome = self.lib.load_genome(best)
        if replace_side == 'left':
            self.pair.left = genome
            self.pair.left_id = best
        else:
            self.pair.right = genome
            self.pair.right_id = best

        log.debug(f'[compare] new opponent #{best} (gap={best_gap:.3f})')
