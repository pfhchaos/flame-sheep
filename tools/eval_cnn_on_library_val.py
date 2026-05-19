#!/usr/bin/env python3
"""Evaluate one or more CNN scorer checkpoints on the same library val pairs.

Builds the same train/val split that finetune_cnn_vk.py uses (mixed mode,
seed=42 split), then scores every requested weights file against the
identical val pair set. Reports per-checkpoint val_acc so you can compare
v1 vs v3, ES-pretrained vs from-scratch, etc., on the same yardstick.

Why this matters: the corpus changed when degenerate genomes (flying
dots, pulsars) were filtered out. A model that scored 79.1% against the
old corpus may score ~50% against the current harder corpus without
having actually regressed. This tool measures every checkpoint against
the same current-corpus benchmark.

Usage:
    python tools/eval_cnn_on_library_val.py \\
        --weights ~/datasets/esheep-cnn/cnn_scorer_mlp_v10_alive.npy \\
                  ~/datasets/esheep-cnn/cnn_scorer_es_v3_lr003.npz \\
                  ~/datasets/esheep-cnn/cnn_scorer_mlp_v3_personal.npz
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flame_sheep.cnn_scorer import load_cnn_weights_file
from flame_sheep.storage import Library, NORMALIZATION_VERSION
from flame_sheep.vk_compute import VkCompute
from flame_sheep.wallpaper_ml import build_cnn_scorer
from train_cnn_vk import MODEL_CONFIGS
from finetune_cnn_vk import DbImageStore, load_training_pairs

log = logging.getLogger(__name__)


def evaluate_one(weights_path: Path, val_pairs, db_path: Path,
                  model_size: str, mlp_head: bool, batch_size: int,
                  image_size: int, gpu: VkCompute) -> tuple[float, str | None]:
    """Load weights + matching normalization, score val_pairs, return val_acc."""
    weights, norm_version = load_cnn_weights_file(str(weights_path))

    # Resolve normalization. If the weights file stamps a version, use that
    # version's stats. Legacy .npy weights without a stamp run unstandardized
    # (which is what they were trained against).
    normalization = None
    if norm_version is not None:
        lib = Library()
        normalization = lib.get_normalization(norm_version)
        lib.close()
        if normalization is None:
            log.warning('weights %s expect normalization_%s but no stats found — '
                        'running unstandardized (results will be garbage)',
                        weights_path.name, norm_version)

    # Build the model and load weights.
    layers = MODEL_CONFIGS[model_size]
    model = build_cnn_scorer(gpu, layers, batch_size=batch_size,
                              image_size=image_size, mlp_head=mlp_head)
    if len(weights) != model.param_count():
        return float('nan'), f'param mismatch (file: {len(weights)}, model: {model.param_count()})'
    model.load_weights(weights)

    # Image store with matching standardization.
    store = DbImageStore(str(db_path), image_size, channels='domain',
                         cache_gb=2.0, normalization=normalization)

    # Score every unique genome that appears in any val pair.
    val_genome_ids = sorted(set(w for w, _ in val_pairs)
                            | set(l for _, l in val_pairs))
    scores: dict[int, float] = {}
    input_buf = gpu.create_buffer(batch_size * 4 * image_size * image_size * 4)

    for i in range(0, len(val_genome_ids), batch_size):
        batch_ids = val_genome_ids[i:i + batch_size]
        batch_imgs = store.get_batch(batch_ids)
        # Pad to batch_size if needed
        if len(batch_imgs) < batch_size:
            pad = np.zeros((batch_size - len(batch_imgs), *batch_imgs.shape[1:]),
                           dtype=np.float32)
            batch_imgs = np.concatenate([batch_imgs, pad])
        gpu.upload(input_buf, batch_imgs)
        out_buf = model.forward(input_buf, batch_size,
                                (4, image_size, image_size))
        batch_scores = gpu.download(out_buf, np.float32, batch_size)
        for j, gid in enumerate(batch_ids):
            scores[gid] = float(batch_scores[j])

    correct = sum(1 for w, l in val_pairs
                  if scores.get(w, 0.0) > scores.get(l, 0.0))
    val_acc = correct / max(len(val_pairs), 1)

    input_buf.destroy()
    store.close()
    return val_acc, f'{len(weights)}p, norm={norm_version or "legacy"}'


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate CNN checkpoints on the same library val set.')
    parser.add_argument('--weights', type=Path, nargs='+', required=True,
                        help='One or more .npy/.npz weights files to compare')
    parser.add_argument('--db', type=Path,
                        default=Path.home() / '.local/share/flame-sheep/library.db',
                        help='Library DB path')
    parser.add_argument('--model-size', choices=['25k', '55k', '100k'], default='25k')
    parser.add_argument('--mlp-head', action='store_true', default=True)
    parser.add_argument('--no-mlp-head', dest='mlp_head', action='store_false')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--data-mode', default='mixed',
                        choices=['mixed', 'pairwise', 'thumbs'])
    parser.add_argument('--val-fraction', type=float, default=0.15)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s', datefmt='%H:%M:%S')

    # Build val_pairs ONCE — every weights file is scored against the same set.
    log.info('Building val pairs (mode=%s)...', args.data_mode)
    all_pairs, stats = load_training_pairs(args.db, mode=args.data_mode)
    rng = np.random.default_rng(42)  # matches finetune_cnn_vk seed
    indices = np.arange(len(all_pairs))
    rng.shuffle(indices)
    split = int(len(indices) * (1 - args.val_fraction))
    val_pairs = [all_pairs[i] for i in indices[split:]]
    log.info('  pairwise=%d, thumbs_liked=%d, thumbs_disliked=%d, val_pairs=%d',
             stats['pairwise'], stats['thumbs_liked'], stats['thumbs_disliked'],
             len(val_pairs))

    # Single GPU context shared across all evaluations.
    gpu = VkCompute()
    log.info('GPU: %s', gpu.device_name)

    print()
    print(f'{"checkpoint":<60} {"val_acc":>10}  details')
    print('-' * 100)
    for w_path in args.weights:
        try:
            val_acc, details = evaluate_one(
                w_path, val_pairs, args.db,
                args.model_size, args.mlp_head, args.batch_size,
                args.image_size, gpu)
            print(f'{w_path.name:<60} {val_acc:>10.4f}  {details}')
        except Exception as e:
            print(f'{w_path.name:<60} {"ERROR":>10}  {e}')

    gpu.destroy()


if __name__ == '__main__':
    main()
