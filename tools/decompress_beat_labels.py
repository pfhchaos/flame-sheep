#!/usr/bin/env python3
"""Convert compressed BeatNet label .npz files to uncompressed .npz.

Each uncompressed file is ~3-4× larger on disk but loads ~50× faster
because there's no DEFLATE step on read — the page cache + zip-entry
extraction is basically a memcpy. The trainer doesn't need any code
changes; just point --data_dir at the converted directory.

If the corpus is read repeatedly (multi-epoch training), the OS page
cache will hold the hot files in RAM and reads become free.

Usage:
    python tools/decompress_beat_labels.py \
        ~/datasets/beat-labels/ \
        ~/datasets/beat-labels-fast/

Disk cost: ~135 GB for the 7521-file corpus (vs ~38 GB compressed).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
import zipfile
from pathlib import Path

import numpy as np


def _is_uncompressed(path: Path) -> bool:
    """True iff path is a valid .npz with no DEFLATE entries."""
    try:
        with zipfile.ZipFile(path) as zf:
            return all(info.compress_type == zipfile.ZIP_STORED
                       for info in zf.infolist())
    except (zipfile.BadZipFile, OSError):
        return False


def _convert_one(args: tuple[Path, Path]) -> tuple[str, str]:
    src, dst = args
    if dst.exists() and _is_uncompressed(dst):
        return ('already', '')
    try:
        with np.load(src) as d:
            arrs = {k: np.array(d[k]) for k in d.files}
        tmp = Path(str(dst) + '.tmp')
        np.savez(tmp, **arrs)  # NB: savez, not savez_compressed
        # numpy may append .npz to the tmp path
        actual = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + '.npz')
        actual.replace(dst)
        return ('ok', '')
    except (KeyError, OSError, ValueError, zipfile.BadZipFile) as e:
        return ('err', f'{src.name}: {e}')


def main():
    parser = argparse.ArgumentParser(
        description='Decompress label .npz files for fast page-cache reads.')
    parser.add_argument('src_dir', type=Path)
    parser.add_argument('dst_dir', type=Path)
    parser.add_argument('-j', '--jobs', type=int, default=mp.cpu_count(),
                        help='Worker processes (default: CPU count)')
    args = parser.parse_args()

    args.dst_dir.mkdir(parents=True, exist_ok=True)

    src_files = sorted(p for p in args.src_dir.glob('*.npz')
                       if not p.name.endswith('.tmp.npz')
                       and not p.name.endswith('.npz.tmp'))
    pairs = [(f, args.dst_dir / f.name) for f in src_files]
    print(f'Converting {len(pairs)} files: {args.src_dir} -> {args.dst_dir}')
    print(f'Workers: {args.jobs}')

    t0 = time.time()
    counts = {'ok': 0, 'already': 0, 'err': 0}
    with mp.Pool(args.jobs) as pool:
        for i, (status, detail) in enumerate(
                pool.imap_unordered(_convert_one, pairs, chunksize=8)):
            counts[status] += 1
            if status == 'err':
                print(f'  WARN: {detail}', file=sys.stderr)
            if (i + 1) % 200 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f'  {i + 1}/{len(pairs)} ({rate:.1f} files/sec)')

    elapsed = time.time() - t0
    rate = len(pairs) / elapsed if elapsed > 0 else 0
    print()
    print(f'Done in {elapsed:.1f}s ({rate:.1f} files/sec). '
          f'ok={counts["ok"]}, already={counts["already"]}, '
          f'errors={counts["err"]}')


if __name__ == '__main__':
    main()
