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
from wallpaper_ml.vk_compute import VkCompute
from wallpaper_ml import build_cnn_scorer
from train_cnn_vk import MODEL_CONFIGS
from finetune_cnn_vk import DbImageStore, load_training_pairs

log = logging.getLogger(__name__)


def score_with_weights(weights_path: Path, val_pairs, val_genome_ids,
                       db_path: Path, model, input_buf, gpu, batch_size: int,
                       image_size: int) -> tuple[float, str | None]:
    """Load weights + matching normalization into the pre-built model, score val_pairs.

    Model + input_buf are built once in main() and reused across all
    checkpoints. Rebuilding the model per checkpoint creates new Vulkan
    buffer handles, which defeats the pipeline cache in VkCompute (cache
    keys include buffer handles) and bloats GPU memory with stale
    pipelines — exactly what commit 80055a3 fixed for training. load_weights
    just uploads new data into existing weight buffers, so this stays
    inside the cache's happy path.
    """
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

    if len(weights) != model.param_count():
        return float('nan'), f'param mismatch (file: {len(weights)}, model: {model.param_count()})'
    model.load_weights(weights)

    # Image store rebuilt per checkpoint because normalization may differ.
    # No GPU buffers involved — store is just a sqlite handle + an in-memory
    # cache of standardized numpy arrays.
    store = DbImageStore(str(db_path), image_size, channels='domain',
                         cache_gb=2.0, normalization=normalization)
    try:
        scores: dict[int, float] = {}
        for i in range(0, len(val_genome_ids), batch_size):
            batch_ids = val_genome_ids[i:i + batch_size]
            batch_imgs = store.get_batch(batch_ids)
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

        correct = sum(1 for w, l, _ in val_pairs
                      if scores.get(w, 0.0) > scores.get(l, 0.0))
        val_acc = correct / max(len(val_pairs), 1)
    finally:
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

    # Single GPU context + single model + single input_buf shared across
    # all evaluations. Rebuilding any of these per-checkpoint would create
    # new buffer handles and explode the pipeline cache (commit 80055a3
    # context). load_weights uploads new data into the existing weight
    # buffers — no buffer churn, no cache pollution.
    gpu = VkCompute()
    log.info('GPU: %s', gpu.device_name)
    layers = MODEL_CONFIGS[args.model_size]
    model = build_cnn_scorer(gpu, layers, batch_size=args.batch_size,
                              image_size=args.image_size, mlp_head=args.mlp_head)
    input_buf = gpu.create_buffer(args.batch_size * 4 * args.image_size * args.image_size * 4)

    # Genome ID set is identical for every checkpoint — compute once.
    val_genome_ids = sorted(set(w for w, _, _ in val_pairs)
                            | set(l for _, l, _ in val_pairs))

    try:
        print()
        print(f'{"checkpoint":<60} {"val_acc":>10}  details')
        print('-' * 100)
        for w_path in args.weights:
            try:
                val_acc, details = score_with_weights(
                    w_path, val_pairs, val_genome_ids, args.db,
                    model, input_buf, gpu, args.batch_size, args.image_size)
                print(f'{w_path.name:<60} {val_acc:>10.4f}  {details}')
            except Exception as e:
                print(f'{w_path.name:<60} {"ERROR":>10}  {e}')
    finally:
        input_buf.destroy()
        gpu.destroy()


if __name__ == '__main__':
    main()
