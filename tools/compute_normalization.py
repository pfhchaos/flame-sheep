#!/usr/bin/env python3
"""Aggregate per-genome channel stats into global normalization constants.

Reads mean_h/s/l/a and std_h/s/l/a from genome_blobs, takes the average
across all genomes, writes to metadata table as 'normalization_<version>'.

The trainer reads this at startup and standardizes inputs to zero mean /
unit variance per channel. Without that, our domain channels (which use
sentinel=1.0 for never-hit regions) have a per-image mean of 0.3–0.6 —
Kaiming weight init assumes zero-mean inputs, so the first conv produces
biased outputs and ReLU kills swathes of channels permanently.

Run this after a render-batch completes, or whenever normalize_channels
semantics change (bump NORMALIZATION_VERSION in storage.py first).

Usage:
    python tools/compute_normalization.py
    python tools/compute_normalization.py --version v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flame_sheep.storage import Library, NORMALIZATION_VERSION, STATS_COLS


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate per-genome channel stats into global normalization.')
    parser.add_argument('--version', default=NORMALIZATION_VERSION,
                        help='Normalization version key (default: %(default)s)')
    args = parser.parse_args()

    lib = Library()

    row = lib.conn.execute(
        '''SELECT AVG(mean_h), AVG(mean_s), AVG(mean_l), AVG(mean_a),
                  AVG(std_h),  AVG(std_s),  AVG(std_l),  AVG(std_a),
                  COUNT(*)
             FROM genome_blobs
            WHERE mean_h IS NOT NULL'''
    ).fetchone()
    if row is None or row[8] == 0:
        print('No per-genome stats found. Migration may not have run, or '
              'no rendered genomes exist yet.', file=sys.stderr)
        sys.exit(1)

    mean = np.array(row[0:4], dtype=np.float32)
    std = np.array(row[4:8], dtype=np.float32)
    n_samples = int(row[8])

    print(f'Aggregated from {n_samples} genomes:')
    print(f'  channel    mean      std')
    for i, ch in enumerate(['H', 'S', 'L', 'A']):
        print(f'  {ch}        {mean[i]:7.4f}   {std[i]:6.4f}')

    # Sanity: degenerate channels (std ~ 0) would make standardization
    # divide-by-zero. The standardize_channels function guards against
    # this but warn the user so they can investigate.
    if (std < 1e-3).any():
        bad = [['H', 'S', 'L', 'A'][i] for i in range(4) if std[i] < 1e-3]
        print(f'WARNING: degenerate stds on channels {bad} — '
              f'standardize_channels will pass these through unchanged.',
              file=sys.stderr)

    lib.set_normalization(mean, std, n_samples, version=args.version)
    print(f'\nStored as metadata key "normalization_{args.version}".')

    lib.close()


if __name__ == '__main__':
    main()
