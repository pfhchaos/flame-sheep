"""Backfill auxiliary-model scores into genomes.cnn_scores_detail.

For the multi-model active learning architecture: each auxiliary model
(curriculum, future axis-models, imported user-shared models) needs its
scores stored per-genome so compare-mode can use them for pair selection
strategies (uncertainty in any one model, disagreement between any pair).

Writes to cnn_scores_detail keyed by a human-readable name (NOT weights
hash — we want to query "curriculum's score" specifically). Coexists with
the hash-keyed entries the score_worker may have written.

Usage:
    python tools/backfill_aux_model_scores.py \\
        --weights ~/datasets/esheep-cnn/cnn_stage3_LSHA_personal.npz \\
        --name curriculum

    # Re-run cheap (existing key gets overwritten with fresh score)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flame_sheep.storage import _db_path


def score_all_genomes(weights_path: Path, db_path: str, batch_size: int = 8,
                      image_size: int = 256) -> dict[int, float]:
    """Run a model over all active genomes (with renders), return {gid: score}.

    Reuses the DbImageStore + Vk forward path from blind_model_eval.
    """
    from flame_sheep.genome.scoring.cnn_scorer import load_cnn_weights_file
    from flame_sheep.genome.scoring.scoring_channels import load_normalization_sidecar
    from flame_sheep.storage import NORMALIZATION_VERSION, Library
    from wallpaper_ml.vk_compute import VkCompute
    from wallpaper_ml import build_cnn_scorer
    from train_cnn_vk import MODEL_CONFIGS
    from finetune_cnn_vk import DbImageStore

    # Use Library's normalization stats (matches genome render conditions)
    lib = Library()
    normalization = lib.get_normalization(NORMALIZATION_VERSION)
    lib.close()

    # List of active genome IDs with renders
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        'SELECT b.genome_id FROM genome_blobs b '
        'JOIN genomes g ON g.id = b.genome_id '
        'WHERE b.render_static IS NOT NULL AND g.archived = 0'
    ).fetchall()
    conn.close()
    genome_ids = [r[0] for r in rows]
    print(f'  active genomes with renders: {len(genome_ids)}')

    gpu = VkCompute()
    LAYERS = list(MODEL_CONFIGS['25k'])
    model = build_cnn_scorer(gpu, LAYERS, batch_size=batch_size,
                             image_size=image_size, mlp_head=True)
    weights, _ = load_cnn_weights_file(str(weights_path))
    model.load_weights(weights)

    store = DbImageStore(db_path, image_size, channels='domain', cache_gb=2.0,
                         normalization=normalization, n_channels=4)
    input_buf = gpu.create_buffer(batch_size * 4 * image_size * image_size * 4)

    scores: dict[int, float] = {}
    n_total = len(genome_ids)
    for i in range(0, n_total, batch_size):
        batch_ids = genome_ids[i:i + batch_size]
        batch = np.zeros((batch_size, 4, image_size, image_size),
                         dtype=np.float32)
        valid_in_batch = []
        for j, gid in enumerate(batch_ids):
            img = store.get(gid)
            if img is not None:
                batch[j] = img
                valid_in_batch.append(gid)
        if not valid_in_batch:
            continue
        gpu.upload(input_buf, batch)
        out = model.forward(input_buf, batch_size,
                            (4, image_size, image_size))
        out_scores = gpu.download(out, np.float32, batch_size)
        for j, gid in enumerate(batch_ids):
            if gid in valid_in_batch:
                scores[gid] = float(out_scores[j])
        if (i // batch_size) % 50 == 0 and i > 0:
            print(f'    scored {i}/{n_total}')
    store.close()
    return scores


def write_scores(db_path: str, name: str, scores: dict[int, float]) -> int:
    """Write each {gid: score} into genomes.cnn_scores_detail[name].

    Read-modify-write per row: load existing JSON, set/overwrite the named
    key, write back. Returns count written.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    n = 0
    for gid, score in scores.items():
        row = cur.execute(
            'SELECT cnn_scores_detail FROM genomes WHERE id=?', (gid,)
        ).fetchone()
        detail = {}
        if row and row[0]:
            try:
                detail = json.loads(row[0])
            except json.JSONDecodeError:
                detail = {}
        detail[name] = score
        cur.execute(
            'UPDATE genomes SET cnn_scores_detail=? WHERE id=?',
            (json.dumps(detail), gid),
        )
        n += 1
    conn.commit()
    conn.close()
    return n


def main():
    parser = argparse.ArgumentParser(
        description='Backfill auxiliary-model scores into cnn_scores_detail'
    )
    parser.add_argument('--weights', type=Path, required=True,
                        help='Model weights (.npz)')
    parser.add_argument('--name', type=str, required=True,
                        help='Name to store under in cnn_scores_detail '
                             '(e.g. "curriculum", "deployed", "axis_color")')
    parser.add_argument('--db', type=Path, default=None,
                        help='DB path (default: library.db)')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--dry-run', action='store_true',
                        help='Compute scores but do not write to DB')
    args = parser.parse_args()

    db_path = str(args.db) if args.db else str(_db_path())
    print(f'DB: {db_path}')
    print(f'Weights: {args.weights}')
    print(f'Storing as: {args.name!r}')

    print('--- Scoring ---')
    scores = score_all_genomes(args.weights, db_path, args.batch_size)
    print(f'  scored {len(scores)} genomes')
    print(f'  range: min={min(scores.values()):.3f}  '
          f'max={max(scores.values()):.3f}  '
          f'median={np.median(list(scores.values())):.3f}')

    if args.dry_run:
        print('--dry-run: not writing')
        return

    print('--- Writing ---')
    n = write_scores(db_path, args.name, scores)
    print(f'  wrote {n} rows')


if __name__ == '__main__':
    main()
