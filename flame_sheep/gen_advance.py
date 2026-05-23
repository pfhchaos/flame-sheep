"""Generation-advance orchestration: rescore-done check, bulk breeder, rank pruner.

The latch state machine itself lives on `Library` (storage.py). This module
provides the per-step logic each worker invokes when it sees its trigger
state. Post-reorg this file's contents split along object lines (breed →
genome/, prune-priority → genome/, sequencing → runtime/), but keeping it
together today makes the single transition easy to follow.

See docs/generational_architecture.md.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .storage import Library

log = logging.getLogger(__name__)


# Latch state names — keep in sync with Library docstring.
STATE_IDLE = 'idle'
STATE_AWAITING_RESCORE = 'awaiting_rescore'
STATE_AWAITING_BREED = 'awaiting_breed'
STATE_AWAITING_SCORE_NEW = 'awaiting_score_new'
STATE_AWAITING_PRUNE = 'awaiting_prune'

ALL_STATES = (
    STATE_IDLE, STATE_AWAITING_RESCORE, STATE_AWAITING_BREED,
    STATE_AWAITING_SCORE_NEW, STATE_AWAITING_PRUNE,
)

# Breed mix (must sum to 1.0). Order matters only for log readability.
BREED_MIX = {
    'jitter_top': 0.40,    # exploit current taste
    'lerp_top_top': 0.30,  # combine winners
    'jitter_mid': 0.20,    # explore around uncertain
    'random': 0.10,        # escape local optima
}
JITTER_TOP_SCALE = 0.1
JITTER_MID_SCALE = 0.25

# Bulk-breed sizing. Floor is the "you can't go below this if you want
# enough genomes to vote on" value (~800 gives ~5 judgments per genome at
# the next-gen 4k-judgment threshold). Disagreement scaling adds breadth.
BREED_FLOOR = 800
BREED_ALPHA = 500.0    # multiplier on mean |Δscore| in [0,1]
TOP_POOL_SIZE = 200    # genomes considered "top" for jitter/lerp parents
MID_POOL_SIZE = 400    # ranked 200-600 are "mid" for wider-jitter parents

# Population target after pruning. >3K because gen-1 throughput was painful
# at smaller pop sizes.
TARGET_POP_SIZE = 3000
# When pop is below target, under-prune so we drift toward target.
UNDER_PRUNE_RATIO = 0.75


def rescore_complete(conn: sqlite3.Connection, weights_hash: str) -> bool:
    """True iff every active *rendered* genome has been scored against
    the deployed weights. Use for awaiting_rescore (post-train sweep over
    the existing rendered pool — nothing new is being added)."""
    row = conn.execute(
        'SELECT COUNT(*) FROM genomes g '
        'JOIN genome_blobs b ON b.genome_id = g.id '
        'WHERE COALESCE(g.archived, 0) = 0 '
        '  AND b.render_static IS NOT NULL '
        '  AND (g.cnn_weights_hash IS NULL OR g.cnn_weights_hash != ?)',
        (weights_hash,)
    ).fetchone()
    return row[0] == 0


def all_active_fully_scored(conn: sqlite3.Connection, weights_hash: str) -> bool:
    """True iff every active genome — rendered or not — has a matching
    cnn_weights_hash. Use for awaiting_score_new where we must wait for
    BOTH render_worker (to render freshly-bred genomes) AND score_worker
    (to score them). Unrendered genomes have no blob row or NULL
    render_static; they count as stale until rendered+scored.
    """
    row = conn.execute(
        'SELECT COUNT(*) FROM genomes g '
        'WHERE COALESCE(g.archived, 0) = 0 '
        '  AND (g.cnn_weights_hash IS NULL OR g.cnn_weights_hash != ?)',
        (weights_hash,)
    ).fetchone()
    return row[0] == 0


def compute_breed_count(disagreement: float, current_pop: int) -> int:
    """Number of children to breed this generation.

    Floor: BREED_FLOOR. Adds BREED_ALPHA * disagreement for exploration of
    territory the new model now scores differently. Adds a population
    deficit term so an under-target pop catches up over a few generations.
    """
    deficit = max(0, TARGET_POP_SIZE - current_pop)
    base = BREED_FLOOR + int(BREED_ALPHA * disagreement) + deficit // 4
    return base


def measure_disagreement(conn: sqlite3.Connection,
                          prev_score_col: str = 'prev_cnn_score',
                          new_score_col: str = 'cnn_score') -> float:
    """Mean |new_score - old_score| over active genomes that have both.

    Returns 0.0 if the prev-score column doesn't exist (pre-snapshot
    schema) or nothing comparable is present. Caller treats 0.0 as
    "no disagreement adjustment" so breed_count falls back to the
    floor + deficit term.

    TODO: add `prev_cnn_score` column + score_worker snapshot to make
    this meaningful. For now the floor carries the count.
    """
    try:
        row = conn.execute(
            f'SELECT AVG(ABS({new_score_col} - {prev_score_col})) '
            f'FROM genomes WHERE COALESCE(archived, 0) = 0 '
            f'  AND {new_score_col} IS NOT NULL '
            f'  AND {prev_score_col} IS NOT NULL'
        ).fetchone()
    except sqlite3.OperationalError:
        return 0.0
    return float(row[0]) if row and row[0] is not None else 0.0


def _select_parents_by_score(conn: sqlite3.Connection,
                              offset: int, limit: int) -> list[int]:
    """Top-N (or middle-N if offset>0) active rendered genomes by CNN score."""
    return [r[0] for r in conn.execute(
        'SELECT g.id FROM genomes g '
        'JOIN genome_blobs b ON b.genome_id = g.id '
        'WHERE COALESCE(g.archived, 0) = 0 '
        '  AND b.render_static IS NOT NULL '
        '  AND g.cnn_score IS NOT NULL '
        'ORDER BY g.cnn_score DESC LIMIT ? OFFSET ?',
        (limit, offset)
    ).fetchall()]


def bulk_breed(lib: 'Library', total_count: int) -> dict:
    """Generate `total_count` new genomes mixed across four sources.

    Returns a dict with per-source counts and the list of new IDs.
    Children that fail survey_and_correct (framing/stability) are
    silently dropped — the target count is best-effort, not guaranteed.
    """
    rng = np.random.default_rng()
    n_jitter_top = int(total_count * BREED_MIX['jitter_top'])
    n_lerp = int(total_count * BREED_MIX['lerp_top_top'])
    n_jitter_mid = int(total_count * BREED_MIX['jitter_mid'])
    n_random = total_count - n_jitter_top - n_lerp - n_jitter_mid

    top_ids = _select_parents_by_score(lib.conn, offset=0, limit=TOP_POOL_SIZE)
    mid_ids = _select_parents_by_score(lib.conn, offset=TOP_POOL_SIZE,
                                       limit=MID_POOL_SIZE)
    log.info('[bulk-breed] parents available: %d top, %d mid',
             len(top_ids), len(mid_ids))

    counts = {'jitter_top': 0, 'lerp_top_top': 0, 'jitter_mid': 0,
              'random': 0, 'rejected': 0}
    new_ids: list[int] = []

    def _try_save(child) -> bool:
        try:
            if not child.survey_and_correct():
                counts['rejected'] += 1
                return False
        except Exception:
            log.debug('[bulk-breed] survey failed', exc_info=True)
            counts['rejected'] += 1
            return False
        new_ids.append(lib.save_genome(child))
        return True

    # 1. Jitter top
    if top_ids:
        for _ in range(n_jitter_top):
            parent = lib.load_genome(int(rng.choice(top_ids)))
            if _try_save(parent.jitter(rng, scale=JITTER_TOP_SCALE)):
                counts['jitter_top'] += 1

    # 2. Lerp top × top
    if len(top_ids) >= 2:
        for _ in range(n_lerp):
            a, b = rng.choice(top_ids, size=2, replace=False)
            pa = lib.load_genome(int(a))
            pb = lib.load_genome(int(b))
            try:
                child = pa.lerp(pb, float(rng.uniform(0.3, 0.7)))
            except Exception:
                log.debug('[bulk-breed] lerp failed', exc_info=True)
                counts['rejected'] += 1
                continue
            if _try_save(child):
                counts['lerp_top_top'] += 1

    # 3. Jitter mid (wider scale)
    if mid_ids:
        for _ in range(n_jitter_mid):
            parent = lib.load_genome(int(rng.choice(mid_ids)))
            if _try_save(parent.jitter(rng, scale=JITTER_MID_SCALE)):
                counts['jitter_mid'] += 1

    # 4. Random
    from .genome import Genome
    for _ in range(n_random):
        if _try_save(Genome.random(rng)):
            counts['random'] += 1

    log.info('[bulk-breed] produced %d/%d children: %s',
             len(new_ids), total_count, counts)
    return {'ids': new_ids, 'counts': counts}


def _active_pop_size(conn: sqlite3.Connection) -> int:
    return conn.execute(
        'SELECT COUNT(*) FROM genomes WHERE COALESCE(archived, 0) = 0'
    ).fetchone()[0]


def rank_prune(lib: 'Library', current_gen: int) -> dict:
    """Archive low-rank active genomes to bring pop near TARGET_POP_SIZE.

    Priority (lower = pruned first):
      1. CNN score (low first)
      2. Vote count (few first — less invested signal)
      3. Age (older first — younger genomes deserve time to be evaluated)

    Exemptions:
      - Current-gen upvoted genomes (their score is pre-vote-stale)
      - TODO: graph-critical genomes (transition-graph dependency tracked
        elsewhere; placeholder skip until that's wired through)

    Under-target pop → prunes UNDER_PRUNE_RATIO of the would-be count so
    population drifts toward TARGET_POP_SIZE rather than holding flat.
    """
    conn = lib.conn
    current_pop = _active_pop_size(conn)
    excess = current_pop - TARGET_POP_SIZE
    if excess <= 0:
        # Under target: under-prune. Even at-target, take a small bite to
        # rotate the bottom — but don't grow if we're below.
        prune_target = max(0, int(-excess * (1 - UNDER_PRUNE_RATIO)))
    else:
        prune_target = excess

    if prune_target <= 0:
        log.info('[rank-prune] pop=%d ≤ target=%d, no pruning this advance',
                 current_pop, TARGET_POP_SIZE)
        return {'archived': 0, 'pop_before': current_pop,
                'pop_after': current_pop}

    # Current-gen upvotes — exemption set
    upvoted_now = set(r[0] for r in conn.execute(
        "SELECT target_id FROM ratings "
        "WHERE target_type='genome' AND generation=? "
        "GROUP BY target_id HAVING SUM(rating) > 0",
        (current_gen,)
    ).fetchall())

    # Rank-able candidates: active, scored. Order by combined priority.
    # vote_count is a sum of |rating| across all gens — proxy for "human
    # has looked at this".
    candidates = conn.execute('''
        SELECT g.id, g.cnn_score,
               COALESCE((SELECT SUM(ABS(rating)) FROM ratings r
                         WHERE r.target_type='genome' AND r.target_id=g.id), 0) AS votes,
               g.id AS age_proxy
          FROM genomes g
         WHERE COALESCE(g.archived, 0) = 0
           AND g.cnn_score IS NOT NULL
         ORDER BY g.cnn_score ASC, votes ASC, age_proxy ASC
    ''').fetchall()

    to_archive: list[int] = []
    for gid, _score, _votes, _age in candidates:
        if gid in upvoted_now:
            continue
        to_archive.append(gid)
        if len(to_archive) >= prune_target:
            break

    if to_archive:
        conn.executemany(
            "UPDATE genomes SET archived=1, archive_reason='rank_prune' "
            "WHERE id=?",
            [(gid,) for gid in to_archive]
        )
        conn.commit()

    pop_after = _active_pop_size(conn)
    log.info('[rank-prune] archived %d/%d targeted (pop %d → %d, target %d)',
             len(to_archive), prune_target, current_pop, pop_after,
             TARGET_POP_SIZE)
    return {'archived': len(to_archive), 'pop_before': current_pop,
            'pop_after': pop_after}
