#!/usr/bin/env python3
"""Compare new CNN weights against current deployed scores, render most-changed genomes.

Reads existing scores from genomes.cnn_score, runs new weights against all
active genomes, computes score deltas. Renders top N genomes from each
direction (biggest positive shift = "model now likes more", biggest negative
= "model now likes less"), with old + new scores in the filename.

Run this AFTER a retrain but BEFORE running score_worker against the new
weights (which would overwrite cnn_score and lose the comparison).

Usage:
    python tools/whats_changed.py \\
        --new-weights ~/datasets/esheep-cnn/cnn_scorer_finetune_05-22.npz \\
        --n 20
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

from flame_sheep.genome.scoring.cnn_scorer import load_cnn_weights_file
from flame_sheep.storage import Library, _db_path
from wallpaper_ml.vk_compute import VkCompute
from wallpaper_ml import build_cnn_scorer
from train_cnn_vk import MODEL_CONFIGS
from finetune_cnn_vk import DbImageStore


def render_one(conn, genome_id: int, output_dir: Path, prefix: str) -> bool:
    row = conn.execute(
        'SELECT render_static FROM genome_blobs WHERE genome_id = ?',
        (genome_id,)).fetchone()
    if row is None or row[0] is None:
        return False
    img = Image.open(io.BytesIO(row[0]))
    img.save(output_dir / f'{prefix}.png')
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Find genomes whose CNN score changed most after retrain.')
    parser.add_argument('--new-weights', type=Path, required=True,
                        help='Path to new .npz weights file from latest retrain')
    parser.add_argument('--n', type=int, default=20,
                        help='How many top-changed genomes to render in each direction')
    parser.add_argument('--db', type=Path,
                        default=Path.home() / '.local/share/flame-sheep/library.db')
    parser.add_argument('--model-size', choices=['25k', '55k', '100k'], default='25k')
    parser.add_argument('--mlp-head', action='store_true', default=True)
    parser.add_argument('--no-mlp-head', dest='mlp_head', action='store_false')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--output', type=Path,
                        default=Path.home() / 'renders' / 'score-changes')
    args = parser.parse_args()

    # Clean output dirs
    out_up = args.output / 'upgrades'    # model now likes more
    out_dn = args.output / 'downgrades'  # model now likes less
    for d in (out_up, out_dn):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # Read old scores from DB
    conn = sqlite3.connect(str(args.db))
    rows = conn.execute('''
        SELECT id, cnn_score FROM genomes
         WHERE COALESCE(archived, 0) = 0
           AND cnn_score IS NOT NULL
    ''').fetchall()
    old_scores: dict[int, float] = {gid: float(s) for gid, s in rows}
    print(f'Read {len(old_scores)} old scores from DB')

    # Load new weights + normalization
    weights, norm_version = load_cnn_weights_file(str(args.new_weights))
    lib = Library()
    normalization = lib.get_normalization(norm_version) if norm_version else None
    lib.close()
    print(f'Loaded new weights: {args.new_weights.name} (norm={norm_version or "legacy"})')

    # Build GPU + model + input buffer
    gpu = VkCompute()
    print(f'GPU: {gpu.device_name}')
    layers = MODEL_CONFIGS[args.model_size]
    model = build_cnn_scorer(gpu, layers, batch_size=args.batch_size,
                              image_size=args.image_size, mlp_head=args.mlp_head)
    model.load_weights(weights)
    input_buf = gpu.create_buffer(
        args.batch_size * 4 * args.image_size * args.image_size * 4)

    # Run new model on all active genomes
    store = DbImageStore(str(args.db), args.image_size, channels='domain',
                         cache_gb=2.0, normalization=normalization)
    new_scores: dict[int, float] = {}
    ids = sorted(old_scores.keys())
    print(f'Scoring {len(ids)} genomes with new weights...')

    try:
        for i in range(0, len(ids), args.batch_size):
            batch_ids = ids[i:i + args.batch_size]
            batch_imgs = store.get_batch(batch_ids)
            if len(batch_imgs) < args.batch_size:
                pad = np.zeros((args.batch_size - len(batch_imgs),
                                *batch_imgs.shape[1:]), dtype=np.float32)
                batch_imgs = np.concatenate([batch_imgs, pad])
            gpu.upload(input_buf, batch_imgs)
            out_buf = model.forward(input_buf, args.batch_size,
                                    (4, args.image_size, args.image_size))
            batch_scores = gpu.download(out_buf, np.float32, args.batch_size)
            for j, gid in enumerate(batch_ids):
                new_scores[gid] = float(batch_scores[j])
            if i % 200 == 0:
                print(f'  scored {i}/{len(ids)}')
    finally:
        store.close()
        input_buf.destroy()
        gpu.destroy()

    # Compute diffs
    diffs = [(gid, old_scores[gid], new_scores[gid],
              new_scores[gid] - old_scores[gid])
             for gid in ids if gid in new_scores]
    diffs.sort(key=lambda r: r[3])  # most-negative first

    downgrades = diffs[:args.n]       # biggest negative shifts
    upgrades = list(reversed(diffs[-args.n:]))  # biggest positive shifts, biggest first

    # Render
    print(f'\nRendering top {args.n} UPGRADES (model now likes more):')
    for rank, (gid, old, new, diff) in enumerate(upgrades, 1):
        prefix = (f'rank_{rank:02d}_diff_+{diff:+.3f}_'
                  f'id_{gid}_old_{old:.3f}_new_{new:.3f}')
        ok = render_one(conn, gid, out_up, prefix)
        print(f'  {prefix}: {"ok" if ok else "MISSING"}')

    print(f'\nRendering top {args.n} DOWNGRADES (model now likes less):')
    for rank, (gid, old, new, diff) in enumerate(downgrades, 1):
        prefix = (f'rank_{rank:02d}_diff_{diff:+.3f}_'
                  f'id_{gid}_old_{old:.3f}_new_{new:.3f}')
        ok = render_one(conn, gid, out_dn, prefix)
        print(f'  {prefix}: {"ok" if ok else "MISSING"}')

    # Summary stats
    all_diffs = np.array([d[3] for d in diffs])
    print(f'\n=== Score change distribution ===')
    print(f'  genomes scored: {len(diffs)}')
    print(f'  mean diff: {all_diffs.mean():+.3f}')
    print(f'  std diff:  {all_diffs.std():.3f}')
    print(f'  median:    {np.median(all_diffs):+.3f}')
    print(f'  max +:     {all_diffs.max():+.3f}')
    print(f'  max -:     {all_diffs.min():+.3f}')
    print(f'  |diff| > 0.5: {(np.abs(all_diffs) > 0.5).sum()} ({100*(np.abs(all_diffs) > 0.5).mean():.1f}%)')
    print(f'  |diff| > 1.0: {(np.abs(all_diffs) > 1.0).sum()} ({100*(np.abs(all_diffs) > 1.0).mean():.1f}%)')
    print(f'\nOutput: {args.output}')

    conn.close()


if __name__ == '__main__':
    main()
