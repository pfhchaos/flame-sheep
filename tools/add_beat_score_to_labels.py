#!/usr/bin/env python3
"""Add a continuous `beat_score` array to existing BeatNet label .npz files.

The original .npz files have `labels` = (T, 3) soft probabilities for
(downbeat, beat, non-beat). The continuous reformulation of training
needs a single-channel target: beat-or-downbeat probability per frame,
which is just `1 - labels[:, 2]`.

This script post-processes each existing file in-place (atomic rename),
adding `beat_score` alongside the existing keys. No BeatNet re-run
needed — we already have the soft labels.

Usage:
    python tools/add_beat_score_to_labels.py ~/datasets/beat-labels/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='Add `beat_score` channel to existing label .npz files.')
    parser.add_argument('data_dir', type=Path,
                        help='Directory containing the .npz label files')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be processed without writing')
    args = parser.parse_args()

    files = sorted(args.data_dir.glob('*.npz'))
    print(f'Found {len(files)} .npz files in {args.data_dir}')

    n_processed = 0
    n_skipped = 0
    n_already_has = 0
    for i, f in enumerate(files):
        try:
            with np.load(f) as d:
                if 'labels' not in d.files:
                    n_skipped += 1
                    continue
                if 'beat_score' in d.files:
                    n_already_has += 1
                    continue
                # Preserve all existing arrays + add beat_score
                arrs = {k: d[k] for k in d.files}
        except (KeyError, OSError, ValueError) as e:
            print(f'  WARN: skipping {f.name}: {e}', file=sys.stderr)
            n_skipped += 1
            continue

        labels = arrs['labels']  # (T, 3)
        beat_score = (1.0 - labels[:, 2]).astype(np.float32)
        arrs['beat_score'] = beat_score

        if args.dry_run:
            n_processed += 1
            continue

        # Atomic write: tmp + rename
        tmp = f.with_suffix(f.suffix + '.tmp')
        np.savez_compressed(tmp, **arrs)
        # numpy might have appended .npz to the tmp path
        actual_tmp = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + '.npz')
        actual_tmp.replace(f)
        n_processed += 1

        if (i + 1) % 200 == 0:
            print(f'  {i + 1}/{len(files)} processed')

    print()
    print(f'Done. processed={n_processed}, already_has={n_already_has}, '
          f'skipped={n_skipped}')


if __name__ == '__main__':
    main()
