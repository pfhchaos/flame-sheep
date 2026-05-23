#!/usr/bin/env python3
"""Find pairs where the CNN model is confidently wrong vs. user judgment.

A "confidently wrong" pair is one where:
  - The user picked A over B (pairwise rating)
  - The model scores B > A (model disagrees)
  - The score gap is large (model is confident in its wrong answer)

These are the most informative pairs for diagnosing what systematic errors
the model is making. Render them side-by-side so visual patterns emerge:
  "model always undervalues X-type" / "model always overvalues Y-type" etc.

Output: one composite PNG per confidently-wrong pair, sorted by gap.
Browse the output directory and look for patterns in the errors.

Usage:
    python tools/disagreements.py
    python tools/disagreements.py --top-n 20 --output ~/renders/disagreements
"""
from __future__ import annotations

import argparse
import io
import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flame_sheep.storage import _db_path
from finetune_cnn_vk import load_training_pairs


def render_pair(conn, winner_id: int, loser_id: int,
                w_score: float, l_score: float,
                out_path: Path) -> bool:
    """Render winner + loser side-by-side into one PNG."""
    def fetch_render(gid):
        row = conn.execute(
            'SELECT render_static FROM genome_blobs WHERE genome_id = ?',
            (gid,)).fetchone()
        if row is None or row[0] is None:
            return None
        return Image.open(io.BytesIO(row[0]))

    w_img = fetch_render(winner_id)
    l_img = fetch_render(loser_id)
    if w_img is None or l_img is None:
        return False

    # Resize to consistent height; assume square renders
    target_size = 256
    w_img = w_img.resize((target_size, target_size))
    l_img = l_img.resize((target_size, target_size))

    composite = Image.new('RGB', (target_size * 2 + 8, target_size + 32),
                          color=(0, 0, 0))
    composite.paste(w_img, (0, 32))
    composite.paste(l_img, (target_size + 8, 32))

    # Annotate via filename — caption text in image is more work than worth
    composite.save(out_path)
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Find and render pairs where the CNN is confidently wrong.')
    parser.add_argument('--top-n', type=int, default=20,
                        help='How many most-confidently-wrong pairs to render')
    parser.add_argument('--output', type=Path,
                        default=Path.home() / 'renders' / 'disagreements',
                        help='Output directory')
    parser.add_argument('--data-mode', default='pairwise',
                        choices=['mixed', 'pairwise', 'thumbs'])
    parser.add_argument('--val-fraction', type=float, default=0.15)
    parser.add_argument('--all-pairs', action='store_true',
                        help='Use full pairwise set, not just val split')
    args = parser.parse_args()

    # Clean output dir
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    print(f'Loading pairwise data (mode={args.data_mode})...')
    all_pairs, stats = load_training_pairs(str(_db_path()), mode=args.data_mode)
    print(f'  {len(all_pairs)} total pairs')

    if args.all_pairs:
        pairs = all_pairs
    else:
        rng = np.random.default_rng(42)
        indices = np.arange(len(all_pairs))
        rng.shuffle(indices)
        split = int(len(indices) * (1 - args.val_fraction))
        pairs = [all_pairs[i] for i in indices[split:]]
        print(f'  using val split: {len(pairs)} pairs')

    # Fetch CNN scores for every genome in the pair set
    conn = sqlite3.connect(str(_db_path()))
    genome_ids = sorted(set(w for w, _, _ in pairs) | set(l for _, l, _ in pairs))
    print(f'  {len(genome_ids)} unique genomes')

    # Query scores in batches (sqlite IN clause)
    scores: dict[int, float] = {}
    for i in range(0, len(genome_ids), 500):
        batch = genome_ids[i:i + 500]
        placeholders = ','.join('?' * len(batch))
        rows = conn.execute(
            f'SELECT id, cnn_score FROM genomes WHERE id IN ({placeholders})',
            batch).fetchall()
        for gid, score in rows:
            if score is not None:
                scores[gid] = float(score)

    # Compute model's gap on each pair (positive = model agrees, negative = wrong)
    pair_diagnostics = []
    for w, l, _weight in pairs:
        if w not in scores or l not in scores:
            continue
        # gap is signed: positive if model agrees with user, negative if it disagrees
        gap = scores[w] - scores[l]
        pair_diagnostics.append((w, l, scores[w], scores[l], gap))

    # Filter to wrong (model picked loser as higher) and sort by confidence
    wrong = [(w, l, ws, ls, gap) for w, l, ws, ls, gap in pair_diagnostics
             if gap < 0]
    wrong.sort(key=lambda r: r[4])  # most negative gap first = most confidently wrong

    print(f'  model agrees with user: {sum(1 for *_, g in pair_diagnostics if g > 0)}')
    print(f'  model disagrees: {len(wrong)}')
    print(f'  rendering top {args.top_n} most confidently wrong...')

    rendered = 0
    for rank, (w, l, ws, ls, gap) in enumerate(wrong[:args.top_n], 1):
        # Filename: rank, signed gap, IDs, scores
        # Sorted alphabetically by filename → most confident wrong at top
        name = (f'rank_{rank:02d}_gap_{abs(gap):.3f}_'
                f'user_winner_{w}_({ws:.3f})_vs_loser_{l}_({ls:.3f}).png')
        out_path = args.output / name
        if render_pair(conn, w, l, ws, ls, out_path):
            print(f'  {name}')
            rendered += 1
        else:
            print(f'  SKIP {name}: missing render(s)')

    conn.close()
    print()
    print(f'Done. {rendered} pairs rendered to: {args.output}')
    print('Layout: user_winner on LEFT, user_loser on RIGHT (model thinks right > left)')
    print('Look for patterns: what does the model consistently get wrong?')


if __name__ == '__main__':
    main()
