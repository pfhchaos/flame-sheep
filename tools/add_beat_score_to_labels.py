#!/usr/bin/env python3
"""Add a continuous `beat_score` array to existing BeatNet label .npz files.

The original .npz files have `labels` = (T, 3) soft probabilities for
(downbeat, beat, non-beat). The continuous reformulation of training
needs a single-channel target: beat-or-downbeat probability per frame,
which is just `1 - labels[:, 2]`.

This script post-processes each existing file in-place (atomic rename).
Fast path: copy the existing zip, then `zipfile.append` only the new
`beat_score.npy` entry — avoids decompressing and recompressing the mel
spectrogram (which is the bulk of each file). Parallelized across cores.

Usage:
    python tools/add_beat_score_to_labels.py ~/datasets/beat-labels/
    python tools/add_beat_score_to_labels.py ~/datasets/beat-labels/ -j 16
"""
from __future__ import annotations

import argparse
import io
import multiprocessing as mp
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np


def append_array_to_npz(path: Path, name: str, array: np.ndarray) -> None:
    """Append one array to an existing .npz without re-compressing the
    existing entries. Atomic via tmp-file + rename.

    Roughly N× faster than load+savez_compressed when the existing
    arrays are large (e.g. mel spectrograms) and the new array is small.
    """
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    npy_bytes = buf.getvalue()

    tmp = Path(str(path) + '.tmp')
    shutil.copy2(path, tmp)
    try:
        with zipfile.ZipFile(tmp, 'a', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f'{name}.npy', npy_bytes)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _process_one(path: Path) -> tuple[str, str]:
    try:
        with np.load(path) as d:
            if 'labels' not in d.files:
                return ('skip', f'{path.name}: no labels')
            if 'beat_score' in d.files:
                return ('already', '')
            labels = d['labels']
        beat_score = (1.0 - labels[:, 2]).astype(np.float32)
        append_array_to_npz(path, 'beat_score', beat_score)
        return ('ok', '')
    except (KeyError, OSError, ValueError, zipfile.BadZipFile) as e:
        return ('err', f'{path.name}: {e}')


def main():
    parser = argparse.ArgumentParser(
        description='Add `beat_score` array to existing label .npz files.')
    parser.add_argument('data_dir', type=Path,
                        help='Directory containing the .npz label files')
    parser.add_argument('-j', '--jobs', type=int, default=mp.cpu_count(),
                        help='Worker processes (default: CPU count)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print plan without writing')
    args = parser.parse_args()

    # Skip orphan .tmp files left by interrupted runs (old or new format).
    files = sorted(p for p in args.data_dir.glob('*.npz')
                   if not p.name.endswith('.tmp.npz')
                   and not p.name.endswith('.npz.tmp'))
    print(f'Found {len(files)} .npz files in {args.data_dir}')
    print(f'Workers: {args.jobs}')

    if args.dry_run:
        print('Dry run; exiting.')
        return

    t0 = time.time()
    counts = {'ok': 0, 'skip': 0, 'already': 0, 'err': 0}
    with mp.Pool(args.jobs) as pool:
        for i, (status, detail) in enumerate(
                pool.imap_unordered(_process_one, files, chunksize=8)):
            counts[status] += 1
            if status == 'err':
                print(f'  WARN: {detail}', file=sys.stderr)
            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f'  {i + 1}/{len(files)} processed '
                      f'({rate:.1f} files/sec)')

    elapsed = time.time() - t0
    rate = len(files) / elapsed if elapsed > 0 else 0.0
    print()
    print(f'Done in {elapsed:.1f}s ({rate:.1f} files/sec). '
          f'ok={counts["ok"]}, already_has={counts["already"]}, '
          f'skipped={counts["skip"]}, errors={counts["err"]}')


if __name__ == '__main__':
    main()
