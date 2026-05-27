"""Analyze blind_eval discordant pairs: where does each model uniquely win?

Reads blind_eval_results.json, partitions pairs into four buckets:
  - both: both models matched user
  - only_curriculum: curriculum matched, deployed didn't
  - only_deployed: deployed matched, curriculum didn't
  - neither: both wrong (shared failure mode)

Renders the two discordant buckets as side-by-side composites
(WINNER | LOSER) into /tmp/discordant/{only_curriculum,only_deployed}/
so the user can scan for visual patterns: do curriculum's unique wins
cluster on a recognizable failure mode of deployed (gating signal)?
Do deployed's unique wins cluster differently?

Also prints summary stats per bucket: score ranges, score gaps, source
genome-id distribution.

Usage:
    python tools/discordant_pair_analysis.py /tmp/blind_eval/blind_eval_results.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flame_sheep.storage import _db_path


def render_side_by_side(conn, gid_winner: int, gid_loser: int,
                        out_path: Path, gap_px: int = 8) -> bool:
    """Save WINNER | LOSER as one composite PNG. Returns False if blob missing."""
    rows = conn.execute(
        'SELECT genome_id, render_static FROM genome_blobs '
        'WHERE genome_id IN (?, ?) AND render_static IS NOT NULL',
        (gid_winner, gid_loser),
    ).fetchall()
    blobs = {gid: blob for gid, blob in rows}
    if gid_winner not in blobs or gid_loser not in blobs:
        return False
    from io import BytesIO
    w_img = Image.open(BytesIO(blobs[gid_winner])).convert('RGB')
    l_img = Image.open(BytesIO(blobs[gid_loser])).convert('RGB')
    h = max(w_img.height, l_img.height)
    w = w_img.width + gap_px + l_img.width
    composite = Image.new('RGB', (w, h), (0, 0, 0))
    composite.paste(w_img, (0, 0))
    composite.paste(l_img, (w_img.width + gap_px, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(out_path)
    return True


def main():
    parser = argparse.ArgumentParser(description='Analyze discordant pairs')
    parser.add_argument('results_json', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('/tmp/discordant'))
    parser.add_argument('--db', type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.results_json.read_text())
    pairs = data.get('pairs', [])
    print(f'Loaded {len(pairs)} votes from {args.results_json}')
    print(f'Models: A={data["label_a"]!r}  B={data["label_b"]!r}')
    print()

    db_path = str(args.db) if args.db else str(_db_path())
    conn = sqlite3.connect(db_path)

    buckets: dict[str, list[dict]] = {
        'both': [], 'only_a': [], 'only_b': [], 'neither': []
    }

    for p in pairs:
        user = p.get('user')
        if user not in ('a', 'b'):
            continue
        a_pick = 'a' if p['score_a_for_a'] > p['score_a_for_b'] else 'b'
        b_pick = 'a' if p['score_b_for_a'] > p['score_b_for_b'] else 'b'
        a_match = (a_pick == user)
        b_match = (b_pick == user)
        if a_match and b_match:
            buckets['both'].append(p)
        elif a_match:
            buckets['only_a'].append(p)
        elif b_match:
            buckets['only_b'].append(p)
        else:
            buckets['neither'].append(p)

    print('=== Bucket sizes ===')
    for k in ('both', 'only_a', 'only_b', 'neither'):
        label = {
            'both': f'both matched',
            'only_a': f'only {data["label_a"]} matched',
            'only_b': f'only {data["label_b"]} matched',
            'neither': f'neither matched',
        }[k]
        print(f'  {label}: {len(buckets[k])}')
    print()

    # Per-bucket score stats
    for k, label in [('only_a', f'only {data["label_a"]}'),
                     ('only_b', f'only {data["label_b"]}'),
                     ('neither', 'neither')]:
        pp = buckets[k]
        if not pp:
            continue
        # Score gap magnitudes per model on these pairs
        a_gaps = [abs(p['score_a_for_a'] - p['score_a_for_b']) for p in pp]
        b_gaps = [abs(p['score_b_for_a'] - p['score_b_for_b']) for p in pp]
        print(f'=== {label} ({len(pp)} pairs) ===')
        print(f'  {data["label_a"]} score gap (winner-loser): '
              f'min={min(a_gaps):.3f}  median={np.median(a_gaps):.3f}  '
              f'max={max(a_gaps):.3f}')
        print(f'  {data["label_b"]} score gap (winner-loser): '
              f'min={min(b_gaps):.3f}  median={np.median(b_gaps):.3f}  '
              f'max={max(b_gaps):.3f}')
        # Existing-cnn-score percentile of involved genomes — does each
        # bucket pull from a particular score region? (genomes table has
        # no generation column; cnn_score serves as a corpus-position proxy)
        gids = [g for p in pp for g in (p['gid_a'], p['gid_b'])]
        score_rows = conn.execute(
            f'SELECT id, cnn_score FROM genomes WHERE id IN '
            f'({",".join("?" * len(gids))})', gids
        ).fetchall()
        scores = [s for _, s in score_rows if s is not None]
        if scores:
            print(f'  involved genome cnn_score: '
                  f'min={min(scores):.2f}  median={np.median(scores):.2f}  '
                  f'max={max(scores):.2f}  (n={len(scores)})')
        print()

    # Render discordant pairs side-by-side
    print('=== Rendering discordant pairs to disk ===')
    for k, label in [('only_a', data['label_a']),
                     ('only_b', data['label_b'])]:
        out_dir = args.output_dir / f'only_{label}'
        out_dir.mkdir(parents=True, exist_ok=True)
        # Clean old contents
        for old in out_dir.glob('*.png'):
            old.unlink()
        n_rendered = 0
        for p in buckets[k]:
            user = p['user']
            if user == 'a':
                winner, loser = p['gid_a'], p['gid_b']
            else:
                winner, loser = p['gid_b'], p['gid_a']
            fname = f'n{p["n"]:02d}_winner{winner}_vs_loser{loser}.png'
            if render_side_by_side(conn, winner, loser, out_dir / fname):
                n_rendered += 1
        print(f'  {label}: {n_rendered} pairs to {out_dir}/')
    conn.close()

    print()
    print('Each PNG: WINNER (user pick) on the LEFT, LOSER on the RIGHT.')
    print('Look for patterns: do "only curriculum" winners share a structural')
    print('property? do "only deployed" losers share a different one?')


if __name__ == '__main__':
    main()
