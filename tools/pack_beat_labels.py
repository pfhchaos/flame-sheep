#!/usr/bin/env python3
"""Pack BeatNet label corpus into mmap-able .npy files for fast training.

Each source .npz contains separately-stored arrays:
    spectrum (T, 108) float32  — mel features
    diff     (T, 108) float32  — onset diffs
    beat_score (T,)  float32   — continuous beat probability
    labels   (T, 3)  float32   — 3-class soft labels (unused, dropped)
    sr, hop, source            — metadata (unused for training, dropped)

We pack the three training-relevant arrays into one contiguous .npy:
    packed (T, 217) float32
        cols [0:108]   = spectrum
        cols [108:216] = diff
        col   216      = beat_score

Benefits over the source .npz:
- np.load(mmap_mode='r') returns a true memmap — slicing is a view, no
  decompression and no full-array copy. Working-set per batch drops
  from ~3.6 GB (read every file) to ~50 MB (only touched chunks).
- The materialize step can slice the packed array directly instead of
  concatenating separate spectrum + diff arrays per chunk.
- File-descriptor count = file count (vs 4× if we used a directory
  layout with one .npy per channel).

Usage:
    python tools/pack_beat_labels.py \\
        ~/datasets/beat-labels/ \\
        ~/datasets/beat-labels-mmap/

Disk cost: ~143 GB for 7521-file × 235M-frame corpus.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np


def _pack_one(args: tuple[Path, Path]) -> tuple[str, str]:
    src, dst = args
    if dst.exists():
        return ('already', '')
    try:
        with np.load(src) as d:
            if 'spectrum' not in d.files:
                return ('skip', f'{src.name}: no spectrum (not a label file)')
            if 'beat_score' not in d.files:
                return ('skip', f'{src.name}: no beat_score (run augment first)')
            spec = np.asarray(d['spectrum'], dtype=np.float32)
            diff = np.asarray(d['diff'], dtype=np.float32)
            bs = np.asarray(d['beat_score'], dtype=np.float32)
        T = spec.shape[0]
        if diff.shape != (T, 108) or spec.shape != (T, 108):
            return ('err', f'{src.name}: unexpected shapes '
                           f'spec={spec.shape} diff={diff.shape}')
        if bs.shape != (T,):
            return ('err', f'{src.name}: beat_score shape {bs.shape} != ({T},)')

        packed = np.empty((T, 217), dtype=np.float32)
        packed[:, :108] = spec
        packed[:, 108:216] = diff
        packed[:, 216] = bs

        tmp = Path(str(dst) + '.tmp')
        np.save(tmp, packed)
        # np.save may append .npy to the tmp path
        actual = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + '.npy')
        actual.replace(dst)
        return ('ok', '')
    except (KeyError, OSError, ValueError) as e:
        return ('err', f'{src.name}: {e}')


def main():
    parser = argparse.ArgumentParser(
        description='Pack BeatNet labels into mmap-able .npy files.')
    parser.add_argument('src_dir', type=Path)
    parser.add_argument('dst_dir', type=Path)
    parser.add_argument('-j', '--jobs', type=int, default=mp.cpu_count(),
                        help='Worker processes (default: CPU count)')
    args = parser.parse_args()

    args.dst_dir.mkdir(parents=True, exist_ok=True)

    src_files = sorted(p for p in args.src_dir.glob('*.npz')
                       if not p.name.endswith('.tmp.npz')
                       and not p.name.endswith('.npz.tmp'))
    pairs = [(f, args.dst_dir / (f.stem + '.npy')) for f in src_files]
    print(f'Packing {len(pairs)} files: {args.src_dir} -> {args.dst_dir}')
    print(f'Workers: {args.jobs}')

    t0 = time.time()
    counts = {'ok': 0, 'already': 0, 'skip': 0, 'err': 0}
    with mp.Pool(args.jobs) as pool:
        for i, (status, detail) in enumerate(
                pool.imap_unordered(_pack_one, pairs, chunksize=8)):
            counts[status] += 1
            if status in ('err', 'skip'):
                print(f'  {status.upper()}: {detail}', file=sys.stderr)
            if (i + 1) % 200 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f'  {i + 1}/{len(pairs)} ({rate:.1f} files/sec)')

    elapsed = time.time() - t0
    rate = len(pairs) / elapsed if elapsed > 0 else 0
    print()
    print(f'Done in {elapsed:.1f}s ({rate:.1f} files/sec). '
          f'ok={counts["ok"]}, already={counts["already"]}, '
          f'skipped={counts["skip"]}, errors={counts["err"]}')


if __name__ == '__main__':
    main()
