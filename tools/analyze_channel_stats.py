#!/usr/bin/env python3
"""Diagnose channel imbalance and weight distribution in a CNN scorer.

Loads a CNN scorer's weights and inspects:
  1. Per-input-channel statistics on a batch of rendered data
     (mean, std, sentinel %, hit-only mean/std) — is each channel
     dominated by sentinels, or is the standardization landing in
     a reasonable range?
  2. Per-input-channel weight magnitude in conv0 — has the model
     concentrated weight on or away from any particular channel?
  3. Conv0 output activation stats (mean, std, dead ReLU fraction
     per output channel) — are any feature maps saturated or dead?

Handles both legacy .npy weights (v1, sentinel-included normalization)
and .npz weights (v2+, sentinel-aware). When a weights file stamps a
normalization version, this script loads the matching stats from the
Library and applies them to the input batch — so the "INPUT CHANNEL
STATS" section reflects what the model actually sees at scoring time.

Channel layout: (H=palette, S=swept density, L=log hits, A=first-hit).

Usage:
    python tools/analyze_channel_stats.py \
        --weights /home/chaos/datasets/esheep-cnn/cnn_scorer_es_v3_lr003.npz \
        --db ~/.local/share/flame-sheep/library.db \
        --model-size 25k --mlp-head --channels domain
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flame_sheep.vk_compute import VkCompute
from flame_sheep.wallpaper_ml import build_cnn_scorer
from flame_sheep.cnn_scorer import load_cnn_weights_file
from flame_sheep.scoring_channels import SENTINEL_STANDARDIZED
from flame_sheep.storage import Library, NORMALIZATION_VERSION
from train_cnn_vk import MODEL_CONFIGS
from finetune_cnn_vk import DbImageStore


CHANNEL_NAMES = ['H (palette)', 'S (swept)', 'L (log hits)', 'A (first-hit)']


def input_stats(batch: np.ndarray, sentinel_value: float | None = None) -> None:
    """Per-channel input stats. batch: [B, 4, H, W].

    If sentinel_value is provided (e.g., SENTINEL_STANDARDIZED for v2+
    sentinel-aware standardization), also reports sentinel fraction and
    hit-only mean/std — the values the conv layers actually fit against.
    """
    print()
    print("=" * 70)
    print("INPUT CHANNEL STATS")
    print("=" * 70)
    if sentinel_value is not None:
        print(f"{'channel':<18} {'mean':>9} {'std':>9} {'min':>9} {'max':>9} "
              f"{'sent %':>8} {'hit mean':>10} {'hit std':>9}")
        print("-" * 86)
        for c in range(4):
            x = batch[:, c, :, :].ravel()
            sentinel_mask = (np.abs(x - sentinel_value) < 1e-4)
            sentinel_frac = sentinel_mask.mean()
            hit_pixels = x[~sentinel_mask]
            if hit_pixels.size > 0:
                hm, hs = hit_pixels.mean(), hit_pixels.std()
            else:
                hm, hs = float('nan'), float('nan')
            print(f"{CHANNEL_NAMES[c]:<18} {x.mean():>9.3f} {x.std():>9.3f} "
                  f"{x.min():>9.3f} {x.max():>9.3f} {sentinel_frac:>7.1%} "
                  f"{hm:>10.3f} {hs:>9.3f}")
    else:
        # Legacy v1 path: no sentinel concept, fall back to near-zero metric.
        print(f"{'channel':<18} {'mean':>10} {'std':>10} {'min':>10} {'max':>10} {'near-zero':>11}")
        print("-" * 70)
        for c in range(4):
            x = batch[:, c, :, :].ravel()
            near_zero = (np.abs(x) < 0.01).mean()
            print(f"{CHANNEL_NAMES[c]:<18} {x.mean():>10.4f} {x.std():>10.4f} "
                  f"{x.min():>10.4f} {x.max():>10.4f} {near_zero:>10.1%}")


def weight_stats(kernel_w: np.ndarray) -> None:
    """Per-input-channel weight magnitude in conv0.

    kernel_w shape: (out_channels, in_channels=4, kernel_h, kernel_w)
    """
    print()
    print("=" * 70)
    print("CONV0 WEIGHT MAGNITUDES (per input channel)")
    print("=" * 70)
    print(f"{'input channel':<18} {'mean(|w|)':>12} {'std(w)':>12} {'min(w)':>10} {'max(w)':>10}")
    print("-" * 70)
    for c in range(4):
        slab = kernel_w[:, c, :, :]
        print(f"{CHANNEL_NAMES[c]:<18} {np.abs(slab).mean():>12.5f} {slab.std():>12.5f} "
              f"{slab.min():>10.4f} {slab.max():>10.4f}")


def conv0_output_stats(out: np.ndarray) -> None:
    """Per-output-channel activation stats after conv0 + ReLU.

    out shape: (B, C_out, H, W)
    """
    B, C, H, W = out.shape
    print()
    print("=" * 70)
    print(f"CONV0 OUTPUT STATS (post-ReLU, {C} output channels)")
    print("=" * 70)
    print(f"{'out ch':<8} {'mean':>10} {'std':>10} {'max':>10} {'dead %':>10} {'sat %':>10}")
    print("-" * 70)
    sat_thresh = 5.0  # arbitrary high threshold for "saturated"
    for c in range(C):
        x = out[:, c, :, :].ravel()
        dead = (x < 1e-6).mean()
        sat = (x > sat_thresh).mean()
        print(f"{c:<8} {x.mean():>10.4f} {x.std():>10.4f} {x.max():>10.4f} "
              f"{dead:>9.1%} {sat:>9.1%}")

    # Summary
    print()
    print(f"  total dead pixels: {(out < 1e-6).mean():.1%}")
    print(f"  total saturated  : {(out > sat_thresh).mean():.1%}")
    print(f"  output overall: mean={out.mean():.4f} std={out.std():.4f} max={out.max():.4f}")


def main():
    parser = argparse.ArgumentParser(description='Diagnose channel imbalance in CNN scorer')
    parser.add_argument('--weights', type=Path, required=True,
                        help='Path to .npy weights file')
    parser.add_argument('--db', type=Path, required=True,
                        help='Path to library.sqlite')
    parser.add_argument('--model-size', choices=['25k', '55k', '100k'], default='25k')
    parser.add_argument('--mlp-head', action='store_true')
    parser.add_argument('--channels', choices=['rgb', 'domain'], default='domain')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Number of genomes to sample')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--label', type=str, default=None,
                        help='Label for this run (printed in header for diff identification)')
    args = parser.parse_args()

    if args.label:
        print(f"\n{'#' * 70}")
        print(f"# {args.label}")
        print(f"# weights: {args.weights}")
        print(f"# db:      {args.db}")
        print(f"{'#' * 70}")

    rng = np.random.default_rng(args.seed)

    # --- Load weights (with version) BEFORE images, so we can pick the
    # matching normalization for the input stats reported below. ---
    weights, weights_norm_version = load_cnn_weights_file(args.weights)
    print(f"Loaded {len(weights)} params from {args.weights}  "
          f"(normalization_version={weights_norm_version or 'legacy/.npy'})")

    # --- Resolve normalization stats. Domain channels need sentinel-aware
    # standardization when the weights file expects it; without it the
    # input stats and the conv0 output stats reflect a different input
    # distribution than the model was trained on. ---
    normalization = None
    if args.channels == 'domain' and weights_norm_version is not None:
        lib_for_norm = Library(data_dir=args.db.parent
                               if args.db.is_file() else args.db)
        normalization = lib_for_norm.get_normalization(weights_norm_version)
        lib_for_norm.close()
        if normalization is None:
            print(f"WARNING: no normalization_{weights_norm_version} stats "
                  f"in {args.db}. Diagnostics will run against raw channels.",
                  file=sys.stderr)
        else:
            print(f"Using normalization {weights_norm_version}: "
                  f"mean={[f'{x:.3f}' for x in normalization[0]]}  "
                  f"std={[f'{x:.3f}' for x in normalization[1]]}")
    elif weights_norm_version is None:
        print("Legacy weights (no normalization stamp) — running raw, "
              "no standardization applied.", file=sys.stderr)

    # --- Sample some rendered genomes ---
    import sqlite3
    conn = sqlite3.connect(str(args.db))
    if args.channels == 'domain':
        rows = conn.execute(
            'SELECT genome_id FROM genome_blobs WHERE hist_static IS NOT NULL AND hist_swept IS NOT NULL'
        ).fetchall()
    else:
        rows = conn.execute(
            'SELECT genome_id FROM genome_blobs WHERE render_static IS NOT NULL'
        ).fetchall()
    conn.close()
    all_ids = [r[0] for r in rows]
    if len(all_ids) < args.batch_size:
        print(f"Only {len(all_ids)} genomes available, using all of them")
        args.batch_size = len(all_ids)
    sample_ids = rng.choice(all_ids, size=args.batch_size, replace=False).tolist()
    print(f"Sampled {len(sample_ids)} genomes from {len(all_ids)} available")

    # --- Load images with matching standardization ---
    store = DbImageStore(str(args.db), args.image_size,
                         channels=args.channels,
                         normalization=normalization)
    batch = store.get_batch(sample_ids)
    print(f"Batch shape: {batch.shape}")

    sentinel = SENTINEL_STANDARDIZED if (normalization is not None
                                         and args.channels == 'domain') else None
    input_stats(batch, sentinel_value=sentinel)

    # --- Build model and load weights ---
    gpu = VkCompute()
    layers = MODEL_CONFIGS[args.model_size]
    model = build_cnn_scorer(gpu, layers, batch_size=args.batch_size,
                              image_size=args.image_size, mlp_head=args.mlp_head)

    if len(weights) != model.param_count():
        print(f"WARNING: weights file has {len(weights)} params, "
              f"model expects {model.param_count()}", file=sys.stderr)
    model.load_weights(weights)

    # --- Conv0 weight inspection ---
    conv0 = model.layers[0]
    c_out, c_in, k = conv0.out_channels, conv0.in_channels, conv0.kernel_size
    kernel_w = gpu.download(conv0.kernel_buf, np.float32, c_out * c_in * k * k)
    kernel_w = kernel_w.reshape(c_out, c_in, k, k)
    weight_stats(kernel_w)

    # --- Forward pass and conv0 output capture ---
    BS, _, H, W = batch.shape
    input_buf = gpu.create_buffer(BS * 4 * H * W * 4)
    gpu.upload(input_buf, batch)
    _ = model.forward(input_buf, BS, (4, H, W))

    # Download conv0's output buffer (post-ReLU since relu=True by default)
    out_size = BS * c_out * conv0.out_h * conv0.out_w
    conv0_out = gpu.download(conv0.output_buf, np.float32, out_size)
    conv0_out = conv0_out.reshape(BS, c_out, conv0.out_h, conv0.out_w)
    conv0_output_stats(conv0_out)

    input_buf.destroy()
    store.close()


if __name__ == '__main__':
    main()
