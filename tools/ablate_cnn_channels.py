#!/usr/bin/env python3
"""Channel ablation diagnostic for the aesthetic CNN scorer.

Reruns pairwise val accuracy with each input channel (H, S, L, A)
individually zeroed, to measure how much each channel contributes
to the model's decisions.

Inputs are post-standardization (zero-mean, unit-variance). Setting a
channel to zero = setting it to its standardized mean = "remove this
channel's information." Score delta vs baseline = how much the model
was relying on that channel.

Usage:
    python tools/ablate_cnn_channels.py \\
        --weights flame_sheep/data/cnn_scorer_personal_vk.npz
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
from flame_sheep.storage import Library
from flame_sheep.vk_compute import VkCompute
from flame_sheep.wallpaper_ml import build_cnn_scorer
from train_cnn_vk import MODEL_CONFIGS
from finetune_cnn_vk import DbImageStore, load_training_pairs


log = logging.getLogger(__name__)


# Channel order from scoring_channels.py: H, S, L, A
CHANNEL_NAMES = ['H', 'S', 'L', 'A']


def score_pairs_with_ablation(
    val_pairs, val_genome_ids, store, model, input_buf, gpu,
    batch_size: int, image_size: int, ablate_channel: int | None,
) -> tuple[float, dict[int, float]]:
    """Run pairwise eval with optional channel zeroing.

    ablate_channel=None means baseline (no ablation).
    ablate_channel=0..3 zeros that channel post-standardization.

    Returns (val_acc, scores_by_genome_id).
    """
    scores: dict[int, float] = {}
    for i in range(0, len(val_genome_ids), batch_size):
        batch_ids = val_genome_ids[i:i + batch_size]
        batch_imgs = store.get_batch(batch_ids)
        if len(batch_imgs) < batch_size:
            pad = np.zeros((batch_size - len(batch_imgs), *batch_imgs.shape[1:]),
                           dtype=np.float32)
            batch_imgs = np.concatenate([batch_imgs, pad])

        # Channel zeroing happens HERE — after standardization, before upload.
        # batch_imgs shape: (B, 4, H, W). Channel index 0=H, 1=S, 2=L, 3=A.
        if ablate_channel is not None:
            batch_imgs = batch_imgs.copy()  # don't mutate the cache
            batch_imgs[:, ablate_channel, :, :] = 0.0

        gpu.upload(input_buf, batch_imgs)
        out_buf = model.forward(input_buf, batch_size,
                                (4, image_size, image_size))
        batch_scores = gpu.download(out_buf, np.float32, batch_size)
        for j, gid in enumerate(batch_ids):
            scores[gid] = float(batch_scores[j])

    correct = sum(1 for w, l, _ in val_pairs
                  if scores.get(w, 0.0) > scores.get(l, 0.0))
    val_acc = correct / max(len(val_pairs), 1)
    return val_acc, scores


def main():
    parser = argparse.ArgumentParser(
        description='Channel ablation diagnostic for aesthetic CNN.')
    parser.add_argument('--weights', type=Path, required=True,
                        help='Path to .npz weights file (deployed model)')
    parser.add_argument('--db', type=Path,
                        default=Path.home() / '.local/share/flame-sheep/library.db')
    parser.add_argument('--model-size', choices=['25k', '55k', '100k'], default='25k')
    parser.add_argument('--mlp-head', action='store_true', default=True)
    parser.add_argument('--no-mlp-head', dest='mlp_head', action='store_false')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--data-mode', default='pairwise',
                        choices=['mixed', 'pairwise', 'thumbs'])
    parser.add_argument('--val-fraction', type=float, default=0.15)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s', datefmt='%H:%M:%S')

    # Build val_pairs once
    log.info('Building val pairs (mode=%s)...', args.data_mode)
    all_pairs, stats = load_training_pairs(args.db, mode=args.data_mode)
    rng = np.random.default_rng(42)
    indices = np.arange(len(all_pairs))
    rng.shuffle(indices)
    split = int(len(indices) * (1 - args.val_fraction))
    val_pairs = [all_pairs[i] for i in indices[split:]]
    val_genome_ids = sorted(set(w for w, _, _ in val_pairs)
                            | set(l for _, l, _ in val_pairs))
    log.info('  %d val pairs, %d unique genomes', len(val_pairs), len(val_genome_ids))

    # Load weights + normalization
    weights, norm_version = load_cnn_weights_file(str(args.weights))
    lib = Library()
    normalization = lib.get_normalization(norm_version) if norm_version else None
    lib.close()
    log.info('  weights: %s (norm=%s)', args.weights.name, norm_version or 'legacy')

    # GPU + model + input buffer (one set, reused across ablations)
    gpu = VkCompute()
    log.info('GPU: %s', gpu.device_name)
    layers = MODEL_CONFIGS[args.model_size]
    model = build_cnn_scorer(gpu, layers, batch_size=args.batch_size,
                              image_size=args.image_size, mlp_head=args.mlp_head)
    model.load_weights(weights)
    input_buf = gpu.create_buffer(args.batch_size * 4 * args.image_size * args.image_size * 4)
    store = DbImageStore(str(args.db), args.image_size, channels='domain',
                         cache_gb=2.0, normalization=normalization)

    try:
        results = []
        # Baseline first
        log.info('Running baseline (no ablation)...')
        baseline_acc, baseline_scores = score_pairs_with_ablation(
            val_pairs, val_genome_ids, store, model, input_buf, gpu,
            args.batch_size, args.image_size, ablate_channel=None)
        results.append(('baseline', baseline_acc, baseline_scores))

        # Each channel ablation
        for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
            log.info('Running ablation: zero %s channel...', ch_name)
            ablated_acc, ablated_scores = score_pairs_with_ablation(
                val_pairs, val_genome_ids, store, model, input_buf, gpu,
                args.batch_size, args.image_size, ablate_channel=ch_idx)
            results.append((f'no_{ch_name}', ablated_acc, ablated_scores))

        # Report
        print()
        print(f'{"condition":<12} {"val_acc":>10} {"delta":>10} {"mean_score":>12} {"score_shift":>12}')
        print('-' * 60)
        for name, acc, scores in results:
            mean_score = np.mean(list(scores.values()))
            if name == 'baseline':
                delta_str = '—'
                shift_str = '—'
            else:
                delta = acc - baseline_acc
                # Score shift: mean(|ablated - baseline|) across genomes
                shifts = [scores[gid] - baseline_scores[gid] for gid in scores
                          if gid in baseline_scores]
                mean_shift = np.mean(np.abs(shifts))
                delta_str = f'{delta:+.4f}'
                shift_str = f'{mean_shift:.4f}'
            print(f'{name:<12} {acc:>10.4f} {delta_str:>10} {mean_score:>12.4f} {shift_str:>12}')

        # Interpretation hints
        print()
        print('Interpretation:')
        print('  - Large negative delta = model heavily relied on that channel')
        print('  - Large score_shift = channel changes individual scores even if rank is preserved')
        print('  - Near-zero delta + near-zero shift = channel is essentially unused')

    finally:
        store.close()
        input_buf.destroy()
        gpu.destroy()


if __name__ == '__main__':
    main()
