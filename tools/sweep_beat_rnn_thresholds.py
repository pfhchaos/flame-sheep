#!/usr/bin/env python3
"""Per-head threshold sweep for a multidepth beat-RNN checkpoint.

Loads the model, runs one forward pass over the val set per head to
collect predicted probabilities, then evaluates F1 and F2 over a
threshold grid per head independently. Reports the best threshold
per head for each metric.

Why F1 AND F2:
  - F1 is "accuracy under symmetric utility." Maximizing F1 walks the
    model toward balanced precision/recall.
  - F2 weights recall 2x more than precision. For viewer experience
    the failure modes are asymmetric — a missed beat reads as "dead
    wallpaper at a moment your eye expected motion," a false-positive
    beat reads as "brief extra flicker, barely visible." F2 is closer
    to viewer utility.

Independent per-head sweep: each head's peak-pick happens
independently then the runtime cascade picks the most-specific fired
class. Tuning one head doesn't affect another head's peak-pick.

Usage:
    python tools/sweep_beat_rnn_thresholds.py \\
        flame_sheep/data/beat_rnn_multidepth.npz \\
        ~/datasets/beat-labels-mmap
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.train_beat_rnn_continuous import (
    peak_pick, f_measure, OUR_FPS, materialize_batch_continuous,
    MULTIDEPTH_HEAD_TO_LABEL_COL, MULTIDEPTH_HEAD_NAMES,
)
from tools.train_beat_rnn import load_dataset, make_chunks, LazyFileCache
from wallpaper_ml import (
    build_beat_crnn_multidepth, load_schema, SCHEMA_FILENAME,
)
from wallpaper_ml.vk_compute import VkCompute


SWEEP_GRID = {
    0: np.arange(0.15, 0.70, 0.05),   # head 0 = onset (label col 2)
    1: np.arange(0.15, 0.70, 0.05),   # head 1 = beat  (label col 1)
    2: np.arange(0.05, 0.55, 0.05),   # head 2 = downbeat (label col 0)
}


def _f1_f2(tp: int, fp: int, fn: int) -> tuple[float, float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    # F2 = (1 + 4) * P * R / (4 * P + R) = 5 * P * R / (4 * P + R)
    f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) else 0.0
    return precision, recall, f1, f2


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('weights', type=Path)
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('--max-val-files', type=int, default=200)
    parser.add_argument('--chunk-len', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--csv-out', type=Path,
                        default=Path('/tmp/threshold_sweep.csv'))
    args = parser.parse_args()

    print(f'Loading weights from {args.weights}...')
    ckpt = np.load(args.weights, allow_pickle=True)
    arch = ckpt['architecture'].item()
    assert arch == 'multidepth', f'expected multidepth, got {arch}'
    H = int(ckpt['hidden_size'].item())
    P = int(ckpt['proj_size'].item())
    L = int(ckpt['n_gru_layers'].item())
    n_classes = int(ckpt['n_classes'].item())
    print(f'  hidden={H} proj={P} gru_layers={L} n_classes={n_classes}')

    gpu = VkCompute()
    model = build_beat_crnn_multidepth(
        gpu, input_size=216, hidden_size=H, proj_size=P,
        n_gru_layers=L, n_heads=n_classes,
        batch_size=args.batch_size, max_seq_len=args.chunk_len)
    model.load_weights(ckpt['weights'])

    print(f'Loading schema + dataset from {args.data_dir}...')
    schema = load_schema(args.data_dir) if (args.data_dir / SCHEMA_FILENAME).exists() else None
    if schema is None:
        sys.exit(f'No {SCHEMA_FILENAME} in {args.data_dir} — packed mmap required')

    dataset = load_dataset(args.data_dir, max_files=None)
    rng = np.random.default_rng(args.seed)
    # Match training-time train/val split via deterministic shuffle.
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    n_val = max(1, int(len(dataset) * args.val_split))
    val_indices = indices[:n_val][:args.max_val_files]
    val_dataset = [dataset[i] for i in val_indices]
    print(f'  {len(val_dataset)} val files (capped from {n_val})')

    cache = LazyFileCache(max_files=200)
    val_batches = make_chunks(val_dataset, args.chunk_len,
                               args.batch_size, rng)
    print(f'  {len(val_batches)} val batches')

    # --- Forward pass: collect per-head probabilities + targets over val ---
    in_buf = gpu.create_buffer(args.batch_size * args.chunk_len * 216 * 4)

    all_probs = [[] for _ in range(n_classes)]
    all_targets = [[] for _ in range(n_classes)]

    print('Forward pass over val set...')
    for bi, batch_spec in enumerate(val_batches):
        inputs, targets = materialize_batch_continuous(
            batch_spec, val_dataset, args.chunk_len, cache,
            n_classes=3, label_key='labels_hier', schema=schema)
        B = inputs.shape[0]
        T = inputs.shape[1]
        BT = B * T

        inputs_TBF = np.ascontiguousarray(inputs.transpose(1, 0, 2))
        gpu.upload(in_buf, inputs_TBF.ravel())
        model.reset_hidden()
        head_outs = model.forward_multidepth(in_buf, B=B, T=T)

        for head_idx in range(n_classes):
            logits_flat = gpu.download(head_outs[head_idx], np.float32, BT)
            logits_TB = logits_flat.reshape(T, B)
            logits_BT = logits_TB.transpose(1, 0)
            probs_BT = 1.0 / (1.0 + np.exp(-logits_BT))
            label_col = MULTIDEPTH_HEAD_TO_LABEL_COL[head_idx]
            all_probs[head_idx].append(probs_BT)
            all_targets[head_idx].append(targets[:, :, label_col])

        if (bi + 1) % 20 == 0:
            print(f'  batch {bi+1}/{len(val_batches)}')

    # Concatenate batches: each head's arrays become (n_seqs, T)
    for h in range(n_classes):
        all_probs[h] = np.concatenate(all_probs[h], axis=0)
        all_targets[h] = np.concatenate(all_targets[h], axis=0)

    # --- Sweep ---
    print()
    print('=' * 76)

    csv_rows = []
    best_per_head: dict[str, dict] = {}

    for head_idx in range(n_classes):
        name = MULTIDEPTH_HEAD_NAMES[head_idx]
        probs = all_probs[head_idx]
        targets = all_targets[head_idx]
        n_seqs = probs.shape[0]

        print(f'\nHead {head_idx} ({name}) — {n_seqs} sequences:')
        print(f'  {"thresh":>7}  {"P":>6}  {"R":>6}  {"F1":>6}  {"F2":>6}  '
              f'{"tp":>6}  {"fp":>6}  {"fn":>6}')
        print('  ' + '-' * 64)

        # Label threshold mirrors training-time per-head defaults.
        label_th = 0.15 if head_idx == 2 else 0.30
        head_grid = SWEEP_GRID[head_idx]

        best_f1 = (-1.0, None)
        best_f2 = (-1.0, None)
        for th in head_grid:
            total_tp = total_fp = total_fn = 0
            for b in range(n_seqs):
                pred_frames = peak_pick(probs[b], threshold=float(th))
                true_frames = peak_pick(targets[b], threshold=label_th)
                pred_times = pred_frames / OUR_FPS
                true_times = true_frames / OUR_FPS
                m = f_measure(pred_times, true_times)
                total_tp += m['tp']
                total_fp += m['fp']
                total_fn += m['fn']
            p, r, f1, f2 = _f1_f2(total_tp, total_fp, total_fn)
            print(f'  {th:>7.2f}  {p:>6.3f}  {r:>6.3f}  {f1:>6.3f}  '
                  f'{f2:>6.3f}  {total_tp:>6}  {total_fp:>6}  {total_fn:>6}')
            csv_rows.append({
                'head': name, 'threshold': round(float(th), 3),
                'precision': round(p, 4), 'recall': round(r, 4),
                'f1': round(f1, 4), 'f2': round(f2, 4),
                'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
            })
            if f1 > best_f1[0]:
                best_f1 = (f1, float(th))
            if f2 > best_f2[0]:
                best_f2 = (f2, float(th))

        best_per_head[name] = {'best_f1': best_f1, 'best_f2': best_f2}
        print(f'  ⤷ best F1 = {best_f1[0]:.3f} at threshold {best_f1[1]:.2f}')
        print(f'  ⤷ best F2 = {best_f2[0]:.3f} at threshold {best_f2[1]:.2f}')

    print()
    print('=' * 76)
    print('Summary — best thresholds per head:')
    print()
    print(f'{"head":>10}  {"F1-optimal":>16}  {"F2-optimal":>16}')
    print('-' * 50)
    for name, info in best_per_head.items():
        f1_th = info['best_f1'][1]
        f2_th = info['best_f2'][1]
        print(f'{name:>10}  {f1_th:>6.2f}  (F1={info["best_f1"][0]:.3f})  '
              f'{f2_th:>6.2f}  (F2={info["best_f2"][0]:.3f})')

    print()
    print(f'CSV: {args.csv_out}')
    with open(args.csv_out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'head', 'threshold', 'precision', 'recall',
            'f1', 'f2', 'tp', 'fp', 'fn'])
        w.writeheader()
        w.writerows(csv_rows)


if __name__ == '__main__':
    main()
