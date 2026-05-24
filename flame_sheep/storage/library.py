"""Library class — the SQLite-backed persistence layer.

`score_loop` lives here temporarily; it moves to `loops/scoring.py`
in Stage 4 when the loops/ package is created.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

from ..genome import Genome
from .._motion_field_tmp import (
    compute_motion_field,
    motion_field_coherence,
    motion_field_from_blob,
    motion_field_to_blob,
)
from ..palette.scoring import score_palette
from .schema import (
    BLOB_COLS,
    NORMALIZATION_VERSION,
    _connect,
)
from .serialization import _genome_from_json, _genome_to_json

log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Loop-level fitness
# ------------------------------------------------------------------

def score_loop(genomes: list[Genome],
               motion_fields: list[np.ndarray],
               structure: str = 'cyclic') -> dict[str, float]:
    """
    Compute automated fitness metrics for a loop.

    Parameters
    ----------
    genomes : list[Genome]
        The genomes in loop order.
    motion_fields : list[np.ndarray]
        Motion field for each transition (len == len(genomes), wrapping).
    structure : str
        'cyclic', 'palindrome', or 'rondo'.

    Returns
    -------
    dict with keys:
      mean_coherence -- average motion coherence between consecutive transitions
      min_coherence  -- worst-case coherence (one bad jerk tanks this)
      diversity      -- visual variety across genomes in the loop
      palette_flow   -- smoothness of color transitions
      fitness        -- weighted composite of the above
    """
    n = len(genomes)

    # -- motion coherence --
    coherences = []
    for i in range(len(motion_fields)):
        mf_a = motion_fields[i]
        mf_b = motion_fields[(i + 1) % len(motion_fields)]
        coherences.append(motion_field_coherence(mf_a, mf_b))
    mean_coh = float(np.mean(coherences)) if coherences else 0.0
    min_coh = float(np.min(coherences)) if coherences else 0.0

    # -- diversity: average pairwise distance between genomes --
    # High = the loop covers interesting visual ground
    # Low = all genomes look similar (boring)
    if n >= 2:
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(genomes[i].distance(genomes[j]))
        diversity = float(np.mean(dists))
    else:
        diversity = 0.0

    # -- palette flow: how smoothly colors transition around the loop --
    # Compare average palette color between consecutive genomes.
    # Small steps = smooth flow, big jumps = jarring.
    palette_diffs = []
    for i in range(n):
        p_a = genomes[i].palette.mean(axis=0)       # average RGB
        p_b = genomes[(i + 1) % n].palette.mean(axis=0)
        palette_diffs.append(float(np.linalg.norm(p_a - p_b)))
    if palette_diffs:
        # Low variance in step size = smooth; normalize by mean to get consistency
        mean_diff = np.mean(palette_diffs)
        if mean_diff > 1e-6:
            # Consistency: 1 = perfectly even steps, 0 = erratic
            consistency = 1.0 - float(np.std(palette_diffs) / mean_diff)
            consistency = max(0.0, consistency)
            # Moderate step size is best — too small = no change, too big = jarring
            # Peak at ~0.3 palette distance per step
            step_quality = 1.0 - abs(mean_diff - 0.3) / 0.3
            step_quality = max(0.0, min(1.0, step_quality))
            palette_flow = (consistency + step_quality) / 2.0
        else:
            palette_flow = 0.0  # no color change at all
    else:
        palette_flow = 0.0

    # -- smoothness: evenness of consecutive genome distances --
    # Penalizes loops where one transition is a huge jump (the snap problem).
    # Measures coefficient of variation of step distances — 0 = all equal, high = uneven.
    step_dists = []
    for i in range(n):
        step_dists.append(genomes[i].distance(genomes[(i + 1) % n]))
    if step_dists:
        mean_step = np.mean(step_dists)
        max_step = max(step_dists)
        if mean_step > 1e-6:
            # Ratio of worst step to mean — 1.0 = perfectly even, 0 = one step dominates
            smoothness = 1.0 - (max_step - mean_step) / max_step
            smoothness = max(0.0, smoothness)
        else:
            smoothness = 1.0
    else:
        smoothness = 0.0

    # -- composite fitness (structure-dependent weights) --
    if structure == 'palindrome':
        # Palindrome: wrap-around less important (reversal is smooth),
        # bidirectional coherence matters, smoothness still king
        fitness = (
            smoothness * 0.30
            + mean_coh * 0.30   # higher — coherence in both directions
            + diversity * 0.20
            + palette_flow * 0.15
            + min_coh * 0.05    # lower — wrap transition less critical
        )
    elif structure == 'rondo':
        # Rondo: home genome quality matters, diversity of episodes matters
        # min_coherence less important (transitions to/from home vary)
        fitness = (
            smoothness * 0.25
            + mean_coh * 0.20
            + diversity * 0.30   # higher — episodes should be distinct
            + palette_flow * 0.15
            + min_coh * 0.10
        )
    else:  # cyclic
        fitness = (
            smoothness * 0.30
            + mean_coh * 0.25
            + diversity * 0.20
            + palette_flow * 0.15
            + min_coh * 0.10
        )

    return dict(
        mean_coherence=mean_coh,
        min_coherence=min_coh,
        diversity=diversity,
        palette_flow=palette_flow,
        smoothness=smoothness,
        fitness=fitness,
    )


# ------------------------------------------------------------------
# Database operations
# ------------------------------------------------------------------

class Library:
    """Interface to the genome/loop database."""

    def __init__(self, data_dir: Path | None = None):
        self.conn = _connect(data_dir)
        self._data_dir = data_dir

    def close(self) -> None:
        self.conn.close()

    # -- Genomes --

    def save_genome(self, genome: Genome, scores: dict[str, float] | None = None) -> int:
        """Store a genome, return its ID."""
        from ..transition import variation_signature

        params = _genome_to_json(genome)
        if scores is None:
            scores = genome.aesthetic_score()
        sig = variation_signature(genome)
        n_xforms = len(genome.transforms)
        cur = self.conn.execute(
            '''INSERT INTO genomes (params, coverage, entropy, color_entropy, balance, complexity,
                                    edge_sharpness, contour_coherence,
                                    symmetry_max, rotational, reflective, radial, periodic, fractal_dim,
                                    centroid_x, centroid_y, score_version,
                                    variation_signature, n_transforms, framing_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (params, scores['coverage'], scores['entropy'],
             scores['color_entropy'], scores['balance'], scores['complexity'],
             scores.get('edge_sharpness', 0.0), scores.get('contour_coherence', 0.0),
             scores.get('symmetry_max'), scores.get('rotational'),
             scores.get('reflective'), scores.get('radial'),
             scores.get('periodic'), scores.get('fractal_dim'),
             scores.get('centroid_offset_x'), scores.get('centroid_offset_y'),
             1, sig, n_xforms, 1),
        )
        self.conn.commit()
        return cur.lastrowid

    # --- Blob storage (separate table) ---

    def load_genome_blobs(self, genome_id: int,
                          cols: list[str] | None = None) -> dict:
        """Fetch named blob columns for a genome.

        cols: subset of BLOB_COLS. None = all columns.
        Returns {col_name: bytes | None}. Missing rows return all-None.
        """
        if cols is None:
            cols = list(BLOB_COLS)
        cols_sql = ', '.join(cols)
        row = self.conn.execute(
            f'SELECT {cols_sql} FROM genome_blobs WHERE genome_id = ?',
            (genome_id,)
        ).fetchone()
        if row is None:
            return {c: None for c in cols}
        return dict(zip(cols, row))

    def has_render(self, genome_id: int) -> bool:
        """Quick existence check — does this genome have a stored render?"""
        row = self.conn.execute(
            'SELECT render_static IS NOT NULL FROM genome_blobs '
            'WHERE genome_id = ?', (genome_id,)
        ).fetchone()
        return bool(row and row[0])

    def save_genome_blobs(self, genome_id: int, **blobs) -> None:
        """INSERT OR REPLACE blob columns for a genome.

        Pass any subset of the 6 blob columns as keyword arguments.
        Existing values for unspecified columns are PRESERVED (unlike
        INSERT OR REPLACE which would null them).
        """
        invalid = set(blobs) - set(BLOB_COLS)
        if invalid:
            raise ValueError(f'unknown blob columns: {invalid}')

        # Fetch existing row so we don't null out unspecified columns
        existing = self.load_genome_blobs(genome_id)
        existing.update(blobs)

        cols = list(BLOB_COLS)
        placeholders = ', '.join(['?'] * len(cols))
        cols_sql = ', '.join(cols)
        self.conn.execute(
            f'INSERT OR REPLACE INTO genome_blobs '
            f'(genome_id, {cols_sql}) VALUES (?, {placeholders})',
            (genome_id, *(existing[c] for c in cols))
        )
        self.conn.commit()

    # --- Channel stats / normalization ---

    def save_channel_stats(self, genome_id: int,
                           means: np.ndarray, stds: np.ndarray) -> None:
        """Store per-channel mean/std for a genome. means/stds: (4,) arrays."""
        self.conn.execute(
            '''UPDATE genome_blobs
                  SET mean_h=?, mean_s=?, mean_l=?, mean_a=?,
                      std_h=?,  std_s=?,  std_l=?,  std_a=?
                WHERE genome_id=?''',
            (float(means[0]), float(means[1]), float(means[2]), float(means[3]),
             float(stds[0]), float(stds[1]), float(stds[2]), float(stds[3]),
             genome_id),
        )
        self.conn.commit()

    def get_normalization(self, version: str = NORMALIZATION_VERSION):
        """Read aggregated normalization stats from metadata.

        Returns (mean, std) as (4,) np.float32 arrays, or None if the
        version isn't stored yet (run tools/compute_normalization.py first).
        """
        row = self.conn.execute(
            "SELECT value FROM metadata WHERE key=?",
            (f'normalization_{version}',)
        ).fetchone()
        if row is None:
            return None
        try:
            data = json.loads(row[0])
            return (np.asarray(data['mean'], dtype=np.float32),
                    np.asarray(data['std'], dtype=np.float32))
        except (json.JSONDecodeError, KeyError) as e:
            log.error(f'malformed normalization_{version} metadata: {e}')
            return None

    def set_normalization(self, mean: np.ndarray, std: np.ndarray,
                          n_samples: int,
                          version: str = NORMALIZATION_VERSION) -> None:
        """Store aggregated normalization stats in metadata."""
        payload = json.dumps({
            'mean': [float(x) for x in mean],
            'std': [float(x) for x in std],
            'n_samples': int(n_samples),
            'version': version,
        })
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (f'normalization_{version}', payload),
        )
        self.conn.commit()

    # --- Generational system ---

    def get_current_generation(self) -> int:
        """Read the current generation counter from metadata.

        Used by rating-insertion code paths to stamp new ratings with
        the current model+population generation, and by training code
        to filter thumbs to intra-generation synthesis. Returns 0 if
        the metadata key is missing (databases that predate the
        generational migration).
        """
        row = self.conn.execute(
            "SELECT value FROM metadata WHERE key='current_generation'"
        ).fetchone()
        if row is None:
            return 0
        try:
            return int(row[0])
        except (ValueError, TypeError):
            log.error('malformed current_generation metadata: %r', row[0])
            return 0

    def advance_generation(self) -> int:
        """Increment current_generation. Call after training a new model
        + breeding a new population. Returns the new generation number."""
        new_gen = self.get_current_generation() + 1
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES "
            "('current_generation', ?)",
            (str(new_gen),),
        )
        self.conn.commit()
        return new_gen

    # ------------------------------------------------------------------
    # Generation-advance latches
    #
    # State machine in metadata that orchestrates the post-train pipeline
    # without a long-running coordinator process. Each background worker
    # reads its trigger state, runs its step, CAS-flips to the next state.
    # See docs/generational_architecture.md and flame_sheep/gen_advance.py.
    #
    # States:
    #   idle               — no advance in progress (default)
    #   awaiting_rescore   — new weights deployed, waiting for full rescore
    #   awaiting_breed     — rescore done, mass breeder runs next
    #   awaiting_score_new — bred genomes need scoring against new weights
    #   awaiting_prune     — new genomes scored, pruner runs next
    # ------------------------------------------------------------------

    def get_gen_advance_state(self) -> str:
        """Current latch state. Defaults to 'idle' if metadata missing."""
        row = self.conn.execute(
            "SELECT value FROM metadata WHERE key='gen_advance_state'"
        ).fetchone()
        return row[0] if row else 'idle'

    def get_gen_advance_target(self) -> int | None:
        """The new generation number the in-flight advance is targeting.
        None when state is 'idle' or no target was set."""
        row = self.conn.execute(
            "SELECT value FROM metadata WHERE key='gen_advance_target'"
        ).fetchone()
        if row is None:
            return None
        try:
            return int(row[0])
        except (ValueError, TypeError):
            return None

    def cas_gen_advance_state(self, expected: str, new: str,
                              target: int | None = None) -> bool:
        """Atomic compare-and-set on gen_advance_state.

        Returns True iff the state was 'expected' and got flipped to 'new'.
        Two workers racing on the same transition will see one True and
        one False — only the True-getter runs the work.

        If target is provided, also writes gen_advance_target in the same
        transaction. Pass target on the *first* transition (idle →
        awaiting_rescore) so downstream workers can read it.
        """
        # SQLite's UPDATE on a single row is atomic. Use the row count to
        # detect the race. INSERT-OR-REPLACE handles the missing-row case
        # for the very first transition (no row → can't CAS, so we seed).
        cur = self.conn.cursor()
        # Ensure the row exists with current value (no-op if already there).
        cur.execute(
            "INSERT OR IGNORE INTO metadata (key, value) "
            "VALUES ('gen_advance_state', 'idle')"
        )
        cur.execute(
            "UPDATE metadata SET value=? "
            "WHERE key='gen_advance_state' AND value=?",
            (new, expected),
        )
        won = cur.rowcount == 1
        if won and target is not None:
            cur.execute(
                "INSERT OR REPLACE INTO metadata (key, value) "
                "VALUES ('gen_advance_target', ?)",
                (str(target),),
            )
        self.conn.commit()
        return won

    def clear_gen_advance(self) -> None:
        """Reset latches to idle and drop the target. Used by the final
        step of the pipeline after advance_generation(), and by the
        recovery CLI when something gets stuck."""
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) "
            "VALUES ('gen_advance_state', 'idle')"
        )
        self.conn.execute(
            "DELETE FROM metadata WHERE key='gen_advance_target'"
        )
        self.conn.commit()

    def retrain_recommendation(self,
                               fixed_threshold: int = 2000,
                               relative_threshold: float = 1.5) -> dict:
        """Heuristic for whether enough new judgments warrant a new model gen.

        Each thumb (rating row) and each compare-mode pair counts as one
        judgment, since they carry equivalent independent information per
        click (a thumb is one absolute judgment about a single genome; a
        compare is one relative judgment between two). The K*M synthesized
        pair count from thumbs is NOT what's measured here — that's a
        redundancy-inflated number; we care about independent observations.

        Geometric throughout: threshold = max(fixed_threshold,
        relative_threshold * judgments_prev_gen). fixed_threshold is the
        floor; from gen 2 onward the 1.5x multiplier dominates.

        Gen 0 is excluded from prev_gen calculations — it's pre-generational
        backfill (existing pairwise ratings stamped 0 at migration), not a
        real cycled generation. Using it as the basis for gen 1's threshold
        would yield a pathologically high number (1.5x of bootstrap pool).

        Also short-circuits to should_retrain=False whenever a generation
        advance is already in flight (gen_advance_state != 'idle'). No
        point recommending a retrain we're literally executing.

        Returns a dict suitable for logging — current_gen, this_gen_count,
        threshold, should_retrain, and a human-readable message.
        """
        gen = self.get_current_generation()
        thumbs_this = self.conn.execute(
            "SELECT COUNT(*) FROM ratings WHERE generation=? AND target_type='genome'",
            (gen,)
        ).fetchone()[0]
        pairwise_this = self.conn.execute(
            "SELECT COUNT(*) FROM pairwise_ratings WHERE generation=?",
            (gen,)
        ).fetchone()[0]
        judgments_this = thumbs_this + pairwise_this

        # Previous generation's total judgments for the relative threshold.
        # Gen 0 is excluded — it's bootstrap backfill, not a real cycled
        # generation. For gen 1, prev_basis collapses to 0 → threshold
        # falls to the fixed_threshold floor.
        if gen >= 2:
            prev_thumbs = self.conn.execute(
                "SELECT COUNT(*) FROM ratings WHERE generation=? AND target_type='genome'",
                (gen - 1,)
            ).fetchone()[0]
            prev_pairwise = self.conn.execute(
                "SELECT COUNT(*) FROM pairwise_ratings WHERE generation=?",
                (gen - 1,)
            ).fetchone()[0]
            judgments_prev = prev_thumbs + prev_pairwise
        else:
            judgments_prev = 0

        # All-generations total — diagnostic only.
        total = self.conn.execute(
            "SELECT COUNT(*) FROM ratings WHERE target_type='genome'"
        ).fetchone()[0] + self.conn.execute(
            "SELECT COUNT(*) FROM pairwise_ratings"
        ).fetchone()[0]

        threshold = max(fixed_threshold,
                        int(relative_threshold * judgments_prev))
        mode = 'geometric'

        # Suppress recommendation if a gen advance is already in flight —
        # the retrain has happened (or is happening); recommending it again
        # is just noise and a foot-gun for any tool that auto-acts on this.
        advancing = self.get_gen_advance_state() != 'idle'
        should = (judgments_this >= threshold) and not advancing

        if advancing:
            tail = 'gen advance in flight, suppressing'
        elif should:
            tail = 'RETRAIN RECOMMENDED'
        else:
            tail = 'more data needed'
        msg = (f'gen {gen}: {judgments_this} new judgments '
               f'({thumbs_this} thumbs + {pairwise_this} pairwise), '
               f'threshold={threshold} ({mode}, total={total}), {tail}')
        return {
            'current_gen': gen,
            'judgments_this_gen': judgments_this,
            'thumbs_this_gen': thumbs_this,
            'pairwise_this_gen': pairwise_this,
            'judgments_prev_gen': judgments_prev,
            'all_gen_total': total,
            'threshold': threshold,
            'threshold_mode': mode,
            'advancing': advancing,
            'should_retrain': should,
            'message': msg,
        }

    def load_genome(self, genome_id: int) -> Genome:
        """Load a genome by ID.

        Center and zoom are baked into params from survey_and_correct()
        at creation time.
        """
        row = self.conn.execute(
            'SELECT params FROM genomes WHERE id = ?',
            (genome_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f'No genome with id {genome_id}')
        genome = _genome_from_json(row[0])
        genome.db_id = genome_id
        return genome

    def genome_scores(self, genome_id: int) -> dict[str, float]:
        """Get stored aesthetic scores for a genome."""
        row = self.conn.execute(
            'SELECT coverage, entropy, color_entropy, balance, complexity,'
            ' edge_sharpness, contour_coherence, centroid_x, centroid_y'
            ' FROM genomes WHERE id = ?',
            (genome_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f'No genome with id {genome_id}')
        return dict(coverage=row[0], entropy=row[1], color_entropy=row[2],
                    balance=row[3], complexity=row[4],
                    edge_sharpness=row[5] or 0.0, contour_coherence=row[6] or 0.0,
                    centroid_offset_x=row[7], centroid_offset_y=row[8])

    def top_genomes(self, n: int = 20, min_coverage: float = 0.01) -> list[tuple[int, dict]]:
        """Return top N genomes ranked by CNN score (preferred) or composite."""
        rows = self.conn.execute(
            '''SELECT id, coverage, entropy, color_entropy, balance, complexity,
                      cnn_score
               FROM genomes
               WHERE coverage >= ? AND COALESCE(archived, 0) = 0
               ORDER BY COALESCE(cnn_score, -999) DESC,
                        (entropy + color_entropy + balance * 0.5 + complexity) DESC
               LIMIT ?''',
            (min_coverage, n),
        ).fetchall()
        return [
            (r[0], dict(coverage=r[1], entropy=r[2], color_entropy=r[3],
                        balance=r[4], complexity=r[5], cnn_score=r[6]))
            for r in rows
        ]

    def genome_count(self) -> int:
        return self.conn.execute('SELECT COUNT(*) FROM genomes').fetchone()[0]

    def genome_loop_counts(self) -> dict[int, int]:
        """Count how many loops each genome appears in."""
        rows = self.conn.execute(
            'SELECT genome_id, COUNT(DISTINCT loop_id) FROM loop_items GROUP BY genome_id'
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def genome_transition_counts(self) -> dict[int, int]:
        """Count cached transition neighbors per genome (excluding self-edges)."""
        rows = self.conn.execute(
            '''SELECT genome_a, COUNT(*) FROM genome_transitions
               WHERE genome_a != genome_b
               GROUP BY genome_a'''
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def cnn_disagreement(self, min_models: int = 2) -> list[tuple[int, float]]:
        """Genomes with max score spread across model versions.

        Requires store_cnn_detail to be enabled. Returns
        (genome_id, max_score - min_score) sorted by disagreement descending.
        """
        rows = self.conn.execute(
            'SELECT id, cnn_scores_detail FROM genomes WHERE cnn_scores_detail IS NOT NULL'
        ).fetchall()
        results = []
        for gid, detail_json in rows:
            detail = json.loads(detail_json)
            if len(detail) < min_models:
                continue
            scores = list(detail.values())
            spread = max(scores) - min(scores)
            results.append((gid, spread))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # -- Transitions --

    def store_transition(self, genome_a: int, genome_b: int, distance: float) -> None:
        """Cache a pairwise transition distance (both directions).
        Skips zero-distance self-transitions — no useful information."""
        if distance <= 0 or genome_a == genome_b:
            return
        self.conn.execute(
            'INSERT OR REPLACE INTO genome_transitions (genome_a, genome_b, distance) VALUES (?, ?, ?)',
            (genome_a, genome_b, distance),
        )
        self.conn.execute(
            'INSERT OR REPLACE INTO genome_transitions (genome_a, genome_b, distance) VALUES (?, ?, ?)',
            (genome_b, genome_a, distance),
        )

    def nearest_transitions(self, genome_id: int, n: int = 10) -> list[tuple[int, float]]:
        """Return the N nearest genomes by transition distance."""
        rows = self.conn.execute(
            '''SELECT genome_b, distance FROM genome_transitions
               WHERE genome_a = ?
               ORDER BY distance ASC
               LIMIT ?''',
            (genome_id, n),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    # -- Loops --

    def save_loop(self, genome_ids: list[int],
                  motion_fields: list[np.ndarray] | None = None,
                  name: str | None = None,
                  loop_type: str = 'cyclic',
                  parent_a: int | None = None,
                  parent_b: int | None = None) -> int:
        """
        Store a loop as an ordered sequence of genome IDs.

        motion_fields: list of 3x3x2 arrays, one per transition.
                       Length should equal len(genome_ids) (last wraps to first).
                       If None, motion fields are computed automatically.
        loop_type: 'cyclic', 'palindrome', or 'rondo'.
        parent_a/b: loop IDs of breeding parents (for tracking lineage).
        """
        genomes = [self.load_genome(gid) for gid in genome_ids]

        if motion_fields is None:
            motion_fields = []
            for i in range(len(genome_ids)):
                motion_fields.append(compute_motion_field(
                    genomes[i], genomes[(i + 1) % len(genomes)]
                ))

        scores = score_loop(genomes, motion_fields, structure=loop_type)

        cur = self.conn.execute(
            '''INSERT INTO loops (name, fitness, mean_coherence, min_coherence,
                                  diversity, palette_flow, smoothness, loop_type,
                                  parent_a, parent_b)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (name, scores['fitness'], scores['mean_coherence'],
             scores['min_coherence'], scores['diversity'],
             scores['palette_flow'], scores['smoothness'], loop_type,
             parent_a, parent_b),
        )
        loop_id = cur.lastrowid

        for i, gid in enumerate(genome_ids):
            mf = motion_fields[i] if i < len(motion_fields) else None
            blob = motion_field_to_blob(mf) if mf is not None else None
            self.conn.execute(
                'INSERT INTO loop_items (loop_id, position, genome_id, motion_field) VALUES (?, ?, ?, ?)',
                (loop_id, i, gid, blob),
            )
        self.conn.commit()
        return loop_id

    def loop_type(self, loop_id: int) -> str:
        """Get the structure type of a loop ('cyclic', 'palindrome', 'rondo')."""
        row = self.conn.execute(
            'SELECT loop_type FROM loops WHERE id = ?', (loop_id,)
        ).fetchone()
        return row[0] if row and row[0] else 'cyclic'

    def load_loop(self, loop_id: int) -> list[tuple[int, Genome, np.ndarray | None]]:
        """
        Load a loop: returns [(genome_id, Genome, motion_field), ...] in order.
        """
        rows = self.conn.execute(
            '''SELECT li.genome_id, li.motion_field
               FROM loop_items li
               WHERE li.loop_id = ?
               ORDER BY li.position''',
            (loop_id,),
        ).fetchall()
        if not rows:
            raise KeyError(f'No loop with id {loop_id}')
        result = []
        for gid, mf_blob in rows:
            genome = self.load_genome(gid)
            mf = motion_field_from_blob(mf_blob) if mf_blob else None
            result.append((gid, genome, mf))
        return result

    def loop_count(self) -> int:
        return self.conn.execute('SELECT COUNT(*) FROM loops').fetchone()[0]

    def loop_fitness(self, loop_id: int) -> dict[str, float]:
        """Get stored fitness scores for a loop."""
        row = self.conn.execute(
            '''SELECT fitness, mean_coherence, min_coherence, diversity, palette_flow, smoothness
               FROM loops WHERE id = ?''',
            (loop_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f'No loop with id {loop_id}')
        return dict(fitness=row[0], mean_coherence=row[1], min_coherence=row[2],
                    diversity=row[3], palette_flow=row[4], smoothness=row[5])

    def update_loop_fitness(self, loop_id: int) -> None:
        """Recompute and store loop fitness from coherence + genome quality + votes.

        Components:
          - auto: mean_coherence + diversity + palette_flow + smoothness
          - genome_quality: mean genome fitness across loop genomes
          - transition_quality: smoothness of consecutive genome transitions
          - user_bonus: clamped loop vote (-1/0/+1) * 0.5
        """
        user_bonus = self.net_rating('loop', loop_id) * 0.5
        row = self.conn.execute(
            'SELECT mean_coherence, diversity, palette_flow, smoothness FROM loops WHERE id = ?',
            (loop_id,),
        ).fetchone()
        if row is None:
            return
        auto = (row[0] or 0) + (row[1] or 0) + (row[2] or 0) + (row[3] or 0)

        # Mean genome fitness across loop
        genome_ids = self.loop_genome_ids(loop_id)
        if genome_ids:
            genome_scores = [self.genome_fitness(gid) for gid in genome_ids]
            genome_quality = sum(genome_scores) / len(genome_scores)
        else:
            genome_quality = 0.0

        # Mean transition quality between consecutive genomes in loop
        transition_quality = 0.0
        if len(genome_ids) >= 2:
            dists = []
            for i in range(len(genome_ids)):
                ga = genome_ids[i]
                gb = genome_ids[(i + 1) % len(genome_ids)]
                row_t = self.conn.execute(
                    'SELECT distance FROM genome_transitions WHERE genome_a=? AND genome_b=?',
                    (ga, gb),
                ).fetchone()
                if row_t is not None:
                    dists.append(row_t[0])
            if dists:
                mean_dist = sum(dists) / len(dists)
                transition_quality = 1.0 / (1.0 + mean_dist)

        fitness = auto + genome_quality * 0.3 + transition_quality * 0.2 + user_bonus
        self.conn.execute(
            'UPDATE loops SET fitness = ? WHERE id = ?',
            (fitness, loop_id),
        )
        self.conn.commit()

    def top_loops(self, n: int = 10) -> list[tuple[int, dict]]:
        """Return top N loops ranked by fitness."""
        rows = self.conn.execute(
            '''SELECT id, fitness, mean_coherence, min_coherence, diversity, palette_flow, smoothness
               FROM loops
               ORDER BY fitness DESC
               LIMIT ?''',
            (n,),
        ).fetchall()
        return [
            (r[0], dict(fitness=r[1], mean_coherence=r[2], min_coherence=r[3],
                        diversity=r[4], palette_flow=r[5], smoothness=r[6]))
            for r in rows
        ]

    def loop_genome_ids(self, loop_id: int) -> list[int]:
        """Get just the genome IDs for a loop, in order."""
        rows = self.conn.execute(
            'SELECT genome_id FROM loop_items WHERE loop_id = ? ORDER BY position',
            (loop_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def all_genome_ids_in_loops(self) -> list[int]:
        """Return all unique genome IDs that appear in at least one loop."""
        rows = self.conn.execute(
            'SELECT DISTINCT genome_id FROM loop_items'
        ).fetchall()
        return [r[0] for r in rows]

    def nearest_loop_transitions(self, genome_id: int,
                                  exclude_loop: int | None = None,
                                  exclude_loops: set[int] | None = None,
                                  n: int = 3) -> list[tuple[int, float, float, int]]:
        """Find nearest loops via the transition graph in a single query.

        Returns [(loop_id, fitness, transition_distance, entry_genome_id)]
        sorted by distance, deduplicated to one entry per loop.
        """
        exclude = set(exclude_loops or ())
        if exclude_loop is not None:
            exclude.add(exclude_loop)

        # Join transitions → loop_items → loops to find reachable loops.
        rows = self.conn.execute(
            '''SELECT li.loop_id, l.fitness, gt.distance, gt.genome_b
               FROM genome_transitions gt
               JOIN loop_items li ON li.genome_id = gt.genome_b
               JOIN loops l ON l.id = li.loop_id
               WHERE gt.genome_a = ?
               ORDER BY gt.distance ASC''',
            (genome_id,),
        ).fetchall()

        # Deduplicate: keep closest entry per loop, skip excluded
        seen = set()
        result = []
        for lid, fitness, dist, entry_gid in rows:
            if lid in exclude or lid in seen:
                continue
            seen.add(lid)
            result.append((lid, fitness or 0.0, dist, entry_gid))
            if len(result) >= n:
                break

        return result

    def loops_containing(self, genome_id: int) -> list[tuple[int, float]]:
        """Find loops that contain a genome. Returns [(loop_id, fitness)]."""
        rows = self.conn.execute(
            '''SELECT DISTINCT li.loop_id, l.fitness
               FROM loop_items li
               JOIN loops l ON l.id = li.loop_id
               WHERE li.genome_id = ?
               ORDER BY l.fitness DESC''',
            (genome_id,),
        ).fetchall()
        return [(r[0], r[1] or 0.0) for r in rows]

    def delete_loop(self, loop_id: int) -> None:
        """Delete a loop and its items (CASCADE)."""
        self.conn.execute('DELETE FROM loops WHERE id = ?', (loop_id,))
        self.conn.commit()

    def delete_loops(self, loop_ids: list[int]) -> int:
        """Delete multiple loops. Returns count deleted."""
        if not loop_ids:
            return 0
        placeholders = ','.join('?' * len(loop_ids))
        # Clear parent references that point to loops being deleted
        self.conn.execute(
            f'UPDATE loops SET parent_a = NULL WHERE parent_a IN ({placeholders})',
            loop_ids)
        self.conn.execute(
            f'UPDATE loops SET parent_b = NULL WHERE parent_b IN ({placeholders})',
            loop_ids)
        cur = self.conn.execute(
            f'DELETE FROM loops WHERE id IN ({placeholders})', loop_ids)
        self.conn.commit()
        return cur.rowcount

    # -- Palettes (graph nodes) --

    def save_palette(self, palette: np.ndarray) -> int:
        """Store a palette with fitness scores, return its ID. palette: shape (256, 3) float32."""
        data = palette.astype(np.float32).tobytes()
        mean_rgb = palette.mean(axis=0).astype(np.float32).tobytes()
        scores = score_palette(palette)
        cur = self.conn.execute(
            '''INSERT INTO palettes (data, mean_rgb, contrast, saturation, harmony, smoothness, fitness)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (data, mean_rgb, scores['contrast'], scores['saturation'],
             scores['harmony'], scores['smoothness'], scores['fitness']),
        )
        self.conn.commit()
        return cur.lastrowid

    def load_palette(self, palette_id: int) -> np.ndarray:
        """Load a palette by ID. Returns shape (256, 3) float32."""
        row = self.conn.execute(
            'SELECT data FROM palettes WHERE id = ?', (palette_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f'No palette with id {palette_id}')
        return np.frombuffer(row[0], dtype=np.float32).reshape(256, 3).copy()

    def palette_count(self) -> int:
        return self.conn.execute('SELECT COUNT(*) FROM palettes').fetchone()[0]

    def all_palette_ids(self) -> list[int]:
        rows = self.conn.execute('SELECT id FROM palettes').fetchall()
        return [r[0] for r in rows]

    def palette_distance(self, id_a: int, id_b: int) -> float:
        """Perceptual distance between two palettes. Cheap — uses mean RGB."""
        rows = self.conn.execute(
            'SELECT id, mean_rgb FROM palettes WHERE id IN (?, ?)',
            (id_a, id_b),
        ).fetchall()
        if len(rows) != 2:
            raise KeyError(f'Palette {id_a} or {id_b} not found')
        by_id = {r[0]: np.frombuffer(r[1], dtype=np.float32) for r in rows}
        return float(np.linalg.norm(by_id[id_a] - by_id[id_b]))

    def palette_neighbors(self, palette_id: int, min_dist: float = 0.05,
                          max_dist: float = 0.4,
                          min_fitness: float = 0.0) -> list[tuple[int, float]]:
        """
        Find palettes within a distance range, optionally filtered by fitness.
        Returns [(id, distance), ...] sorted by distance.
        """
        source = self.conn.execute(
            'SELECT mean_rgb FROM palettes WHERE id = ?', (palette_id,)
        ).fetchone()
        if source is None:
            raise KeyError(f'No palette with id {palette_id}')
        source_rgb = np.frombuffer(source[0], dtype=np.float32)

        rows = self.conn.execute(
            'SELECT id, mean_rgb FROM palettes WHERE id != ? AND COALESCE(fitness, 0) >= ?',
            (palette_id, min_fitness),
        ).fetchall()
        neighbors = []
        for pid, mean_blob in rows:
            other_rgb = np.frombuffer(mean_blob, dtype=np.float32)
            d = float(np.linalg.norm(source_rgb - other_rgb))
            if min_dist <= d <= max_dist:
                neighbors.append((pid, d))
        neighbors.sort(key=lambda x: x[1])
        return neighbors

    # -- Ratings --

    def rate(self, target_type: str, target_id: int, rating: int) -> None:
        """Record a like (+1) or dislike (-1).

        Loop votes are clamped to -1/0/+1 (replaces previous vote on
        the loop). Loop votes do NOT propagate to constituent genomes:
        a loop rating expresses opinion about the loop's gestalt
        (motion, flow, transition coherence) which is genuinely
        different from per-genome aesthetic judgment. Propagating
        loop ratings to genome ratings would mislabel transitional
        or flat individual genomes as "liked" just because they
        appeared in a liked loop, and vice versa — exactly the
        structural label noise that poisoned gen-0 thumbs training
        and made the CNN scorer's random-init landing point heavily
        anti-correlated with quality.

        Direct genome thumbs-up additionally triggers immediate breeding
        of ~7 jittered children (see breed_thumbsup_children).
        """
        if target_type == 'loop':
            # Clamp: replace any existing vote on this loop
            self.conn.execute(
                'DELETE FROM ratings WHERE target_type = ? AND target_id = ?',
                (target_type, target_id),
            )
        # Stamp the current generation at insert time. Thumbs are corpus-
        # relative and only meaningful as training signal within their
        # own generation (see docs/generational_architecture.md).
        gen = self.get_current_generation()
        self.conn.execute(
            'INSERT INTO ratings (target_type, target_id, rating, source, generation) '
            'VALUES (?, ?, ?, ?, ?)',
            (target_type, target_id, rating, 'direct', gen),
        )
        self.conn.commit()

        # Thumbs-up breeder — direct genome upvotes only.
        # Spawned as a separate subprocess (not a thread) because
        # Genome.jitter + survey_and_correct is CPU-bound Python and
        # shares the GIL with the render thread when run as a
        # background thread. Empirically c695a12's threaded version
        # still caused visible render-thread stalls even though rate()
        # returned in microseconds. The subprocess pays ~50ms python
        # startup once per upvote, then runs entirely outside the
        # foreground process's GIL.
        # See flame_sheep.thumbsup_breed.
        if target_type == 'genome' and rating > 0:
            import subprocess
            parent_id = target_id
            try:
                args = [
                    sys.executable, '-m', 'flame_sheep.thumbsup_breed',
                    '--parent-id', str(parent_id),
                ]
                if self._data_dir is not None:
                    args += ['--data-dir', str(self._data_dir)]
                # Detached so the parent doesn't wait on it. stdout/stderr
                # inherited from parent so the child's [thumbsup-breed]
                # log line appears in the wallpaper's terminal too.
                subprocess.Popen(args, start_new_session=True)
            except Exception:
                log.exception('[thumbsup-breed] failed to spawn for genome #%d',
                              parent_id)

    def breed_thumbsup_children(self, parent_genome_id: int,
                                n_children: int = 7,
                                jitter_scale: float = 0.1) -> list[int]:
        """Breed N jittered children of a thumbs-upped genome.

        Each child is a small perturbation of the parent (Genome.jitter
        with the given scale). Children are inserted into the genomes
        table with no render — the gpu_render_worker will pick them up
        and render them in the background. Once rendered, they become
        eligible candidates for compare-mode pairing and CNN scoring.

        Children where survey_and_correct() rejects the framing (e.g.,
        attractor escaped viewport from too-aggressive jitter) are
        silently skipped; the n_children request is a target, not a
        guarantee.

        Returns the list of new genome IDs.

        Self-limiting: near-duplicate offspring will be deduped at the
        next generation boundary (or whenever similarity-pruning runs).
        """
        parent = self.load_genome(parent_genome_id)
        rng = np.random.default_rng()
        new_ids: list[int] = []
        for _ in range(n_children):
            child = parent.jitter(rng, scale=jitter_scale)
            try:
                if not child.survey_and_correct():
                    continue  # framing/stability rejection
            except Exception:
                log.debug('[thumbsup-breed] survey failed, skipping a child',
                          exc_info=True)
                continue
            gid = self.save_genome(child)
            new_ids.append(gid)
        return new_ids

    def net_rating(self, target_type: str, target_id: int) -> int:
        """Sum of ratings for a target."""
        row = self.conn.execute(
            'SELECT COALESCE(SUM(rating), 0) FROM ratings WHERE target_type = ? AND target_id = ?',
            (target_type, target_id),
        ).fetchone()
        return row[0]

    def genome_fitness(self, genome_id: int) -> float:
        """Compute genome fitness from aesthetic scores + symmetry + user votes.

        Uses CNN score as primary signal when available (trained on 13 years
        of Electric Sheep crowd ratings). Falls back to cluster-based or
        histogram-based heuristics for unscored genomes.
        """
        row = self.conn.execute(
            '''SELECT coverage, entropy, color_entropy, balance, complexity,
                      edge_sharpness, contour_coherence,
                      symmetry_max, fractal_dim, cnn_score
               FROM genomes WHERE id = ?''',
            (genome_id,),
        ).fetchone()
        if row is None:
            return 0.0

        coverage = row[0] or 0
        color_entropy = row[2] or 0
        fractal_dim = row[8] or 1.0
        cnn_score = row[9]

        if cnn_score is not None:
            # CNN score is the primary aesthetic signal.
            # Raw scores are unbounded; typical range roughly -2 to +5.
            aesthetic = cnn_score
        else:
            # Fallback: hand-crafted heuristics for genomes not yet CNN-scored.
            # Coverage sweet spot: 0.15-0.40 is ideal
            if coverage < 0.05:
                coverage_score = coverage * 4
            elif coverage < 0.15:
                coverage_score = 0.2 + (coverage - 0.05) * 4
            elif coverage <= 0.40:
                coverage_score = 0.6 + (coverage - 0.15) * 1.6
            else:
                coverage_score = max(0, 1.0 - (coverage - 0.40) * 2)

            cl_row = self.conn.execute(
                'SELECT cl_coverage, cl_edge_sharpness, cl_symmetry_best, cl_dominance'
                ' FROM genomes WHERE id = ?',
                (genome_id,),
            ).fetchone()

            if cl_row and cl_row[0] is not None:
                cl_cov = cl_row[0] or 0
                cl_edges = cl_row[1] or 0
                cl_sym = cl_row[2] or 0
                eps = 0.01
                aesthetic = cl_cov * (cl_edges + eps) * (cl_sym + eps)
            else:
                eps = 0.01
                aesthetic = (
                    max(coverage_score, eps) ** 1.0
                    * max(color_entropy, eps) ** 1.0
                )

        # User signal: net votes propagated from loop ratings
        votes = self.net_rating('genome', genome_id)
        return aesthetic + votes * 0.5
