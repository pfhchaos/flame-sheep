#!/usr/bin/env python3
"""Measure training-corpus RMS to calibrate AGC's target_rms.

The beat-RNN was trained on audio loaded by librosa.load (default
normalization), then run through CQT + log1p. The live AGC normalizes
PCM to its `target_rms` setting; if that setting doesn't match the
training-corpus typical RMS, the model sees out-of-distribution
input and the activation distribution shifts (currently: p90 ≈ 0.22
vs the trained 0.3+ peak threshold).

This script:
1. Reads source audio paths from the .npz label files
   (each label file stores the path it was generated from).
2. Loads a random sample with librosa.load — same path the training
   pipeline used.
3. Computes per-file RMS.
4. Reports the distribution and recommends a target_rms (median).

Usage:
    .venv/bin/python tools/calibrate_agc.py \\
        --labels-dir ~/datasets/beat-labels \\
        --sample 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='Measure training-corpus RMS for AGC target calibration')
    parser.add_argument('--labels-dir', type=Path,
                        default=Path.home() / 'datasets' / 'beat-labels',
                        help='Directory with .npz label files (each carries '
                             'the source audio path)')
    parser.add_argument('--sample', type=int, default=50,
                        help='Number of files to sample (default 50)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sr', type=int, default=48000,
                        help='Resample rate (matches training: 48000)')
    args = parser.parse_args()

    import librosa

    label_files = sorted(args.labels_dir.glob('*.npz'))
    if not label_files:
        print(f'No .npz files in {args.labels_dir}', file=sys.stderr)
        sys.exit(1)
    print(f'Found {len(label_files)} label files')

    # Extract source paths
    rng = np.random.default_rng(args.seed)
    if len(label_files) > args.sample:
        idx = rng.choice(len(label_files), size=args.sample, replace=False)
        label_files = [label_files[i] for i in sorted(idx)]
    print(f'Sampling {len(label_files)} files')

    per_file: list[tuple[str, float]] = []
    for i, lf in enumerate(label_files, 1):
        try:
            with np.load(lf) as d:
                if 'source' not in d.files:
                    continue
                src = str(d['source'])
        except (OSError, KeyError):
            continue

        src_path = Path(src)
        if not src_path.exists():
            print(f'  [{i}/{len(label_files)}] MISSING: {src_path.name[:60]}',
                  file=sys.stderr)
            continue

        try:
            audio, _ = librosa.load(str(src_path), sr=args.sr, mono=True,
                                     duration=120.0)  # cap per-file load for speed
        except Exception as e:
            print(f'  [{i}/{len(label_files)}] LOAD FAIL: {src_path.name[:50]}: {e}',
                  file=sys.stderr)
            continue

        if len(audio) < args.sr:  # need >=1s
            continue
        # RMS over the whole sample. Filters out near-silent leading/trailing
        # passages with a simple noise floor before averaging.
        nonzero = audio[np.abs(audio) > 1e-4]
        if len(nonzero) < args.sr // 10:
            continue
        rms = float(np.sqrt(np.mean(nonzero ** 2)))
        per_file.append((src_path.name, rms))
        print(f'  [{i:3d}/{len(label_files)}] {src_path.name[:55]:55s}  RMS={rms:.4f}')

    if not per_file:
        print('No files processed successfully.', file=sys.stderr)
        sys.exit(2)

    rms_arr = np.array([r for _, r in per_file])
    print()
    print('=== Per-file RMS statistics ===')
    print(f'  n         : {len(rms_arr)}')
    print(f'  mean      : {rms_arr.mean():.4f}')
    print(f'  median    : {np.median(rms_arr):.4f}')
    print(f'  std       : {rms_arr.std():.4f}')
    print(f'  p10/p90   : {np.percentile(rms_arr, 10):.4f} / '
          f'{np.percentile(rms_arr, 90):.4f}')
    print(f'  min/max   : {rms_arr.min():.4f} / {rms_arr.max():.4f}')

    median = float(np.median(rms_arr))
    print()
    print('=== Recommended AGC target_rms ===')
    print(f'  target_rms = {median:.4f}')
    print(f'  (median per-file RMS; robust to outlier tracks)')
    print()
    print(f'  Update cfg.agc.target_rms in flame_sheep_audio/config.py '
          f'DEFAULTS,\n  and the running daemon\'s AGC state file at '
          f'~/.local/share/flame-sheep/agc_state.json (the next process()\n'
          f'  call will pick up the new constants from cfg).')


if __name__ == '__main__':
    main()
