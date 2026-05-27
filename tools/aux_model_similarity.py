"""Compare aux model scores: how different is each from each other?

For the multi-model active learning framework: an aux model that's
essentially a duplicate of another wastes a rotation slot in
compare-mode (the disagreement strategy involving the duplicate pair
produces no signal — both models always agree). This tool reports
pairwise similarity metrics so the user can decide whether to drop
redundant aux models.

Three metrics per model pair, each capturing a different sense of
"are these the same model":

  - Pearson r — linear correlation on raw scores. Robust to scale
    differences (e.g., a model that's 2x the magnitude of another but
    perfectly correlated reads r=1.0). What "are these measuring the
    same thing on average" looks like.

  - Spearman ρ — rank correlation. The relevant signal for the
    uncertainty-strategy: if two models produce identical RANKINGS,
    "uncertainty in model A" picks the same pairs as "uncertainty in
    model B" — duplicate work even if raw scores differ.

  - Pairwise agreement — for each pair (a, b) of genomes, do both
    models pick the same winner? This is the load-bearing metric for
    disagreement-strategy in compare-mode: agreement = 1.0 means
    disagreement strategy produces zero useful signal.

High values across all three = duplicate. Worth dropping the redundant
one. Threshold for "duplicate" is fuzzy; >0.95 on all three is a safe
bet for "remove."

Usage:
    python tools/aux_model_similarity.py
    python tools/aux_model_similarity.py --include-primary
        # include the legacy `cnn_score` column as "primary" for full
        # N+1 comparison (most useful — answers "is this aux model
        # different from what's already deployed?")
    python tools/aux_model_similarity.py --pairs 5000
        # number of random pairs to use for pairwise agreement
        # (O(N^2) over all 3000 active genomes would be 9M pairs)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flame_sheep.storage import _db_path


def load_scores(db_path: str, include_primary: bool) -> tuple[dict, list[str]]:
    """Return ({gid: {model_name: score}}, sorted_model_list)."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        '''SELECT g.id, g.cnn_score, g.cnn_scores_detail
             FROM genomes g
             JOIN genome_blobs b ON b.genome_id = g.id
            WHERE b.render_static IS NOT NULL
              AND COALESCE(g.archived, 0) = 0'''
    ).fetchall()
    conn.close()
    scores_by_gid: dict[int, dict[str, float]] = {}
    for gid, cnn_score, detail_json in rows:
        d = {}
        if detail_json:
            try:
                parsed = json.loads(detail_json)
                for k, v in parsed.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        d[k] = float(v)
            except (json.JSONDecodeError, TypeError):
                pass
        if include_primary and cnn_score is not None and 'primary' not in d:
            d['primary'] = float(cnn_score)
        if d:
            scores_by_gid[gid] = d

    coverage: dict[str, int] = {}
    for d in scores_by_gid.values():
        for k in d:
            coverage[k] = coverage.get(k, 0) + 1
    total = len(scores_by_gid)
    models = sorted(m for m, c in coverage.items() if c >= total * 0.5)
    return scores_by_gid, models


def pairwise_agreement(score_a: np.ndarray, score_b: np.ndarray,
                       n_pairs: int, rng: np.random.Generator) -> float:
    """For n_pairs random (i, j), check sign(score_a[i] - score_a[j])
    == sign(score_b[i] - score_b[j]). Returns fraction agreeing."""
    n = len(score_a)
    if n < 2:
        return float('nan')
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    mask = i != j
    i = i[mask]
    j = j[mask]
    if len(i) == 0:
        return float('nan')
    da = np.sign(score_a[i] - score_a[j])
    db = np.sign(score_b[i] - score_b[j])
    nonzero = (da != 0) & (db != 0)
    if nonzero.sum() == 0:
        return float('nan')
    return float((da[nonzero] == db[nonzero]).mean())


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation. scipy.stats.spearmanr equivalent without import."""
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    parser = argparse.ArgumentParser(description='Aux-model pairwise similarity')
    parser.add_argument('--db', type=Path, default=None)
    parser.add_argument('--include-primary', action='store_true',
                        help='Include the legacy cnn_score column as "primary"')
    parser.add_argument('--pairs', type=int, default=5000,
                        help='Random pairs for pairwise agreement metric')
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='Flag pairs whose ALL metrics exceed this value')
    args = parser.parse_args()

    db = str(args.db) if args.db else str(_db_path())
    scores_by_gid, models = load_scores(db, args.include_primary)
    print(f'Loaded {len(scores_by_gid)} genomes with scores from {db}')
    print(f'Models: {models}')
    print()

    # Build per-model arrays — only over genomes that have ALL models scored,
    # otherwise correlations are over different supports and not comparable
    common_gids = [gid for gid, d in scores_by_gid.items()
                   if all(m in d for m in models)]
    print(f'Genomes with scores under all {len(models)} models: '
          f'{len(common_gids)} (using this intersection for fair comparison)')
    print()

    model_arrs = {
        m: np.array([scores_by_gid[g][m] for g in common_gids],
                    dtype=np.float64)
        for m in models
    }
    rng = np.random.default_rng(42)

    print(f'{"":>14} | {"vs":>14} | {"pearson":>8} {"spearman":>9} {"pair-agree":>11}')
    print('-' * 65)
    flagged: list[tuple[str, str, float, float, float]] = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            a, b = model_arrs[m1], model_arrs[m2]
            r_p = float(np.corrcoef(a, b)[0, 1])
            r_s = spearman(a, b)
            r_a = pairwise_agreement(a, b, args.pairs, rng)
            print(f'{m1:>14} | {m2:>14} | {r_p:>8.3f} {r_s:>9.3f} {r_a:>11.3f}')
            if r_p >= args.threshold and r_s >= args.threshold and r_a >= args.threshold:
                flagged.append((m1, m2, r_p, r_s, r_a))

    if flagged:
        print()
        print(f'⚠ FLAGGED — all 3 metrics ≥ {args.threshold} (likely duplicates):')
        for m1, m2, p, s, a in flagged:
            print(f'  {m1} ↔ {m2}: pearson={p:.3f} spearman={s:.3f} '
                  f'pair-agree={a:.3f}')
        print('  Consider removing one to free a rotation slot in compare-mode.')
    else:
        print()
        print(f'✓ No model pairs exceed all-three-≥{args.threshold} threshold. '
              f'All aux models add measurable signal.')


if __name__ == '__main__':
    main()
