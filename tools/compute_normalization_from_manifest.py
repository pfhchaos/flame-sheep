#!/usr/bin/env python3
"""Compute per-channel normalization stats for a manifest-based dataset.

Counterpart to tools/compute_normalization.py, which sources from the
Library DB. This one is for datasets that live as a directory of .npz
histograms + a manifest.csv — e.g., ~/datasets/esheep-cnn/.

Per-dataset stats matter because render parameters differ across sources.
ES uses different walker density / iteration counts than the user's
library; the channel distributions diverge even though the same
normalize_channels function is applied. Standardizing each dataset with
its OWN stats lets a single normalization version (formula) cover both.

Output: <data_dir>/normalization_<version>.json — picked up automatically
by train_cnn_vk.py at startup.

Usage:
    python tools/compute_normalization_from_manifest.py \\
        --data ~/datasets/esheep-cnn/
    python tools/compute_normalization_from_manifest.py \\
        --data ~/datasets/esheep-cnn/ --version v1
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flame_sheep.scoring_channels import (
    load_raw_histograms, channel_stats_hit_only, save_normalization_sidecar,
)
from flame_sheep.storage import NORMALIZATION_VERSION


def main():
    parser = argparse.ArgumentParser(
        description='Compute per-channel normalization stats for a manifest dataset.')
    parser.add_argument('--data', type=str, required=True,
                        help='Dataset directory (must contain manifest.csv and *_hist.npz)')
    parser.add_argument('--version', default=NORMALIZATION_VERSION,
                        help='Normalization version key (default: %(default)s)')
    parser.add_argument('--image-size', type=int, default=256,
                        help='Output spatial size for normalize_channels (default: 256)')
    parser.add_argument('--limit', type=int, default=0,
                        help='If >0, sample this many entries instead of all (faster spot-check)')
    args = parser.parse_args()

    data_dir = Path(args.data).expanduser()
    manifest_path = data_dir / 'manifest.csv'
    if not manifest_path.exists():
        print(f'no manifest at {manifest_path}', file=sys.stderr)
        sys.exit(1)

    entries = []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            static = row['static_path']
            hist_name = static.replace('_static.png', '_hist.npz')
            hist_path = data_dir / hist_name
            if hist_path.exists():
                entries.append(hist_path)

    if not entries:
        print(f'no *_hist.npz files referenced by {manifest_path}', file=sys.stderr)
        sys.exit(1)

    if args.limit and len(entries) > args.limit:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(entries), args.limit, replace=False)
        entries = [entries[i] for i in idx]

    print(f'Scanning {len(entries)} histogram files from {data_dir}...')

    # Aggregate per-image hit-only mean/std, then average — matches the
    # strategy in tools/compute_normalization.py which averages per-genome
    # stats already computed sentinel-aware at render time. Hit-only stats
    # collapse the library/ES corpus gap to ~5% (was ~40% with sentinels
    # included), so a single normalization is comparable across sources.
    means = np.zeros((len(entries), 4), dtype=np.float64)
    stds = np.zeros((len(entries), 4), dtype=np.float64)

    t0 = time.time()
    for i, hist_path in enumerate(entries):
        raw = load_raw_histograms(str(hist_path))
        m, s = channel_stats_hit_only(
            raw['static_hits'], raw['static_colors'],
            raw['swept_hits'], raw.get('first_hit'),
        )
        means[i] = m
        stds[i] = s
        if (i + 1) % 500 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f'  {i+1}/{len(entries)}  ({rate:.0f}/s)')

    mean = means.mean(axis=0).astype(np.float32)
    std = stds.mean(axis=0).astype(np.float32)

    print(f'\nAggregated from {len(entries)} renders:')
    print(f'  channel    mean      std')
    for i, name in enumerate(['H', 'S', 'L', 'A']):
        print(f'  {name}        {mean[i]:7.4f}   {std[i]:6.4f}')

    if (std < 1e-3).any():
        bad = [['H', 'S', 'L', 'A'][i] for i in range(4) if std[i] < 1e-3]
        print(f'WARNING: degenerate stds on channels {bad} — '
              f'standardize_channels will pass these through unchanged.',
              file=sys.stderr)

    path = save_normalization_sidecar(data_dir, args.version, mean, std, len(entries))
    print(f'\nWrote {path}')


if __name__ == '__main__':
    main()
