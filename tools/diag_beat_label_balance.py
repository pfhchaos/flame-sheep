#!/usr/bin/env python3
"""Measure per-head positive rates in the beat-label corpus.

The multidepth trainer currently sums per-head BCE losses with uniform
weights. If positive rates across heads differ significantly, the
loudest-gradient head dominates training. This diagnostic measures the
actual rates so weighting can be set from data rather than 4/4@120bpm
math.

Reports per-head:
  - total positive frames
  - total frames
  - positive rate (positives / frames)
  - relative-to-rarest ratio
  - suggested per-head loss weight (inverse-rate, normalized so the
    rarest head = highest weight)

Run:
    python tools/diag_beat_label_balance.py ~/datasets/beat-labels/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


MULTIDEPTH_HEAD_TO_LABEL_COL = {0: 2, 1: 1, 2: 0}  # head → labels_hier col
MULTIDEPTH_HEAD_NAMES = {0: 'onset', 1: 'beat', 2: 'downbeat'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('--n-files', type=int, default=200,
                        help='Random sample size (default: 200)')
    parser.add_argument('--label-key', type=str, default='labels_hier')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    files = sorted(args.data_dir.glob('*.npz'))
    if len(files) == 0:
        print(f'no .npz files in {args.data_dir}', file=sys.stderr)
        sys.exit(1)
    sample = rng.choice(files, size=min(args.n_files, len(files)),
                         replace=False)

    head_pos = {h: 0 for h in MULTIDEPTH_HEAD_TO_LABEL_COL}
    total_frames = 0
    skipped = 0
    for i, path in enumerate(sample):
        try:
            d = np.load(path, allow_pickle=False)
            labels = d[args.label_key]
        except Exception as e:
            skipped += 1
            continue
        if labels.ndim != 2 or labels.shape[1] < 3:
            skipped += 1
            continue
        total_frames += labels.shape[0]
        for head, col in MULTIDEPTH_HEAD_TO_LABEL_COL.items():
            head_pos[head] += int((labels[:, col] > 0.5).sum())
        if (i + 1) % 50 == 0:
            print(f'  scanned {i+1}/{len(sample)}')

    if total_frames == 0:
        print('no usable label data', file=sys.stderr)
        sys.exit(1)

    print()
    print(f'Sampled {len(sample) - skipped} files '
          f'({skipped} skipped), {total_frames:,} frames')
    print()
    print(f'{"head":<10} {"pos":>10} {"rate":>10} {"vs-rarest":>12}')
    print('-' * 48)

    rates = {h: head_pos[h] / total_frames for h in head_pos}
    rarest = min(rates.values()) if min(rates.values()) > 0 else 1.0
    for head in sorted(rates, key=lambda h: -rates[h]):
        name = MULTIDEPTH_HEAD_NAMES[head]
        print(f'{name:<10} {head_pos[head]:>10,} {rates[head]*100:>9.3f}% '
              f'{rates[head]/rarest:>11.2f}x')

    print()
    print('Suggested per-head loss weights (inverse-rate, '
          'normalized so rarest head = highest):')
    print(f'  Goal: total gradient contribution per head ~= equal across heads.')
    print()

    # weights: w_h = (rate_rarest / rate_h) so the rarest head has
    # weight 1 of (rare_rate/rare_rate) but we want the OPPOSITE
    # direction — rare heads get higher weight to compensate. Use
    # 1/rate, normalized so the largest weight = ratio-vs-rarest.
    inv = {h: 1.0 / rates[h] if rates[h] > 0 else 0.0 for h in rates}
    max_inv = max(inv.values())
    norm = {h: inv[h] * (max_inv / max_inv) for h in inv}  # max → max_inv
    # Cleaner: scale so the LARGEST weight is the ratio-vs-rarest, and
    # the smallest (most common) head is 1.0.
    scale = 1.0 / min(inv.values())
    weights = {h: inv[h] * scale for h in inv}

    print(f'  {"head":<10} {"weight":>10}')
    print('  ' + '-' * 22)
    for head in sorted(weights, key=lambda h: head):  # by head idx
        name = MULTIDEPTH_HEAD_NAMES[head]
        print(f'  {name:<10} {weights[head]:>10.3f}')

    # Print in the order [onset, beat, downbeat] (head idx 0/1/2) so
    # it's drop-in for HEAD_LOSS_WEIGHTS = [...] in source.
    ordered = [weights[h] for h in sorted(weights)]
    print()
    print(f'  As HEAD_LOSS_WEIGHTS (head 0/1/2 = '
          f'{[MULTIDEPTH_HEAD_NAMES[h] for h in sorted(weights)]}):')
    print(f'  [{", ".join(f"{w:.3f}" for w in ordered)}]')


if __name__ == '__main__':
    main()
