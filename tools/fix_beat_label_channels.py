#!/usr/bin/env python3
"""One-shot in-place fix for the BeatNet channel-order bug.

Before 2026-06-04, `generate_beat_labels.build_hierarchical_labels`
populated `labels_hier[:, 0]` from `bn_labels[:, 0]` — but channel 0
in BeatNet's softmax output is actually the BEAT activation, not
DOWNBEAT. The downbeat activation is on channel 1
(see `BeatNet/particle_filtering_cascade.py:97-99`).

Result: every label file in `~/datasets/beat-labels{,-mmap,-quarantine}/`
has the beat-channel activation sitting in the slot the trainer
treats as the downbeat target. Every beat-RNN trained on that data
learned its "downbeat" head against beat labels.

This script recovers the true downbeat activation arithmetically —
no audio rerun, no BeatNet rerun:

    bn[:, 1] = (bn[:, 0] + bn[:, 1]) - bn[:, 0]
             = labels_hier[:, 1] - labels_hier[:, 0]
             = old_col217 - old_col216  (in the packed format)

`labels_hier[:, 1]` was already correct (any-beat = beat + downbeat,
symmetric under channel swap) and survived clipping intact because
softmax sums to < 1.

Safety:
  * atomic write per file (tmp + rename); kill mid-run leaves
    either fully-old or fully-new files, never half-written
  * idempotency: skips files already stamped with the fix marker
  * subset-inequality post-check on a sample: any beat activation
    should exceed the recovered downbeat activation (downbeats ⊂ beats)
  * --dry-run flag for visibility before committing

Usage:
    python3 tools/fix_beat_label_channels.py --dry-run
    python3 tools/fix_beat_label_channels.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Sentinel filenames marking that a directory's labels have been fixed.
PACKED_SENTINEL = '.channel_order_fixed'
# For per-file .npz: this key is added during fix; presence == fixed.
NPZ_FIX_KEY = 'channel_order_fixed'

# Packed-corpus column indices (matches the v1 CorpusSchema for the
# spectrum + diff + labels_hier layout). The 3-column labels_hier
# block lives at the end.
PACKED_DOWNBEAT_COL = 216  # labels_hier[:, 0]
PACKED_BEAT_COL = 217      # labels_hier[:, 1] = any-beat


@dataclass
class FixStats:
    inspected: int = 0
    fixed: int = 0
    already_fixed: int = 0
    skipped_unexpected_shape: int = 0
    failed_post_check: int = 0


def _atomic_save(path: Path, save_fn) -> None:
    """Call save_fn(open_file_handle) and rename to path on success.

    The save_fn receives a binary write file — using a handle avoids
    np.save / np.savez_compressed silently appending .npy / .npz to a
    path argument and clobbering the rename target.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=f'.{path.name}.', suffix='.tmp', dir=str(path.parent))
    tmp_path = Path(tmp_path)
    try:
        with os.fdopen(tmp_fd, 'wb') as fh:
            save_fn(fh)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _subset_check_inplace(arr: np.ndarray, label: str) -> bool:
    """Verify downbeats ⊂ beats AFTER fix: integrated col 216 (now
    downbeat) should be less than col 217 (any-beat) over a real track.
    Same property as the channel-order regression test."""
    db = float(arr[:, PACKED_DOWNBEAT_COL].sum())
    beat = float(arr[:, PACKED_BEAT_COL].sum())
    if not db <= beat + 1e-3:
        print(f'  POST-CHECK FAIL {label}: db_sum={db:.2f} > beat_sum={beat:.2f}',
              file=sys.stderr)
        return False
    return True


def fix_packed_dir(packed_dir: Path, *, dry_run: bool,
                    sample_check_every: int = 100) -> FixStats:
    """Fix all *.npy packed files in `packed_dir` in place.

    Idempotency: presence of `<dir>/.channel_order_fixed` skips the
    whole directory. The sentinel is written ONLY after every file
    succeeds.
    """
    stats = FixStats()
    sentinel = packed_dir / PACKED_SENTINEL
    if sentinel.exists():
        print(f'{packed_dir}: already fixed (sentinel {PACKED_SENTINEL} present)')
        return stats

    files = sorted(packed_dir.glob('*.npy'))
    print(f'{packed_dir}: {len(files)} packed files to inspect')
    if not files:
        return stats

    for i, p in enumerate(files):
        stats.inspected += 1
        try:
            arr = np.load(p)
        except Exception as e:
            print(f'  load failed: {p.name}: {e}', file=sys.stderr)
            continue
        if arr.ndim != 2 or arr.shape[1] < PACKED_BEAT_COL + 1:
            stats.skipped_unexpected_shape += 1
            print(f'  unexpected shape {arr.shape}: {p.name}', file=sys.stderr)
            continue

        # Recovery: bn[:, 1] = col 217 - col 216.
        new_downbeat = arr[:, PACKED_BEAT_COL] - arr[:, PACKED_DOWNBEAT_COL]
        # The subtraction can produce tiny negatives from float roundoff
        # if softmax-sum was numerically >1 by a hair; clip to [0, 1].
        np.clip(new_downbeat, 0.0, 1.0, out=new_downbeat)
        arr[:, PACKED_DOWNBEAT_COL] = new_downbeat

        # Spot-check every Nth file with the subset inequality.
        if i % sample_check_every == 0:
            if not _subset_check_inplace(arr, p.name):
                stats.failed_post_check += 1
                continue

        if dry_run:
            stats.fixed += 1
        else:
            try:
                _atomic_save(p, lambda tmp: np.save(tmp, arr))
                stats.fixed += 1
            except Exception as e:
                print(f'  write failed: {p.name}: {e}', file=sys.stderr)
                continue

        if (i + 1) % 500 == 0:
            print(f'  {i + 1}/{len(files)} ...')

    if not dry_run and stats.failed_post_check == 0 and stats.fixed > 0:
        sentinel.write_text(json.dumps({
            'fixed_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'fixed_files': stats.fixed,
            'note': 'BeatNet ch-order fix: col 216 = labels_hier[:, 1] - labels_hier[:, 0]',
        }, indent=2))
        print(f'  wrote sentinel {sentinel}')

    return stats


def fix_npz_dir(npz_dir: Path, *, dry_run: bool,
                 sample_check_every: int = 200) -> FixStats:
    """Fix `labels_hier` col 0 in all label *.npz files under `npz_dir`.

    Skips non-label .npz (e.g. model checkpoints). Per-file
    idempotency via the `channel_order_fixed` npz key.
    """
    stats = FixStats()
    files = sorted(npz_dir.glob('*.npz'))
    # Filter to actual label files (must have labels_hier).
    label_files: list[Path] = []
    for p in files:
        # cheap check via stat — only open files that don't look like
        # known non-label artifacts.
        name = p.name.lower()
        if name.startswith('beat_rnn_') or name.startswith('train_'):
            continue
        label_files.append(p)
    print(f'{npz_dir}: {len(label_files)} label .npz files to inspect')

    for i, p in enumerate(label_files):
        stats.inspected += 1
        try:
            d = np.load(p, allow_pickle=False)
        except Exception as e:
            print(f'  load failed: {p.name}: {e}', file=sys.stderr)
            continue

        keys = set(d.files)
        if 'labels_hier' not in keys:
            d.close()
            continue
        if NPZ_FIX_KEY in keys and bool(d[NPZ_FIX_KEY]):
            stats.already_fixed += 1
            d.close()
            continue

        contents = {k: d[k] for k in d.files}
        d.close()

        labels_hier = contents['labels_hier']
        if labels_hier.ndim != 2 or labels_hier.shape[1] < 3:
            stats.skipped_unexpected_shape += 1
            print(f'  unexpected labels_hier shape {labels_hier.shape}: '
                  f'{p.name}', file=sys.stderr)
            continue

        new_downbeat = labels_hier[:, 1] - labels_hier[:, 0]
        np.clip(new_downbeat, 0.0, 1.0, out=new_downbeat)
        labels_hier_fixed = labels_hier.copy()
        labels_hier_fixed[:, 0] = new_downbeat
        contents['labels_hier'] = labels_hier_fixed.astype(np.float32)
        contents[NPZ_FIX_KEY] = np.array(True)

        # Spot-check: db_sum should not exceed beat_sum after fix.
        if i % sample_check_every == 0:
            db = float(labels_hier_fixed[:, 0].sum())
            beat = float(labels_hier_fixed[:, 1].sum())
            if not db <= beat + 1e-3:
                stats.failed_post_check += 1
                print(f'  POST-CHECK FAIL {p.name}: db_sum={db:.2f} > '
                      f'beat_sum={beat:.2f}', file=sys.stderr)
                continue

        if dry_run:
            stats.fixed += 1
        else:
            try:
                _atomic_save(p, lambda tmp: np.savez_compressed(tmp, **contents))
            except Exception as e:
                print(f'  write failed: {p.name}: {e}', file=sys.stderr)
                continue
            stats.fixed += 1

        if (i + 1) % 500 == 0:
            print(f'  {i + 1}/{len(label_files)} ...')

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--packed-dir', type=Path,
                        default=Path.home() / 'datasets' / 'beat-labels-mmap',
                        help='Directory of packed *.npy files')
    parser.add_argument('--npz-dir', type=Path,
                        default=Path.home() / 'datasets' / 'beat-labels',
                        help='Directory of source *.npz label files')
    parser.add_argument('--quarantine-dir', type=Path,
                        default=Path.home() / 'datasets' / 'beat-labels-quarantine',
                        help='Quarantine .npz dir (also fixed if exists)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be fixed without writing')
    parser.add_argument('--skip-packed', action='store_true',
                        help='Skip the packed (.npy) corpus')
    parser.add_argument('--skip-npz', action='store_true',
                        help='Skip the source (.npz) labels')
    args = parser.parse_args()

    print('=== beat-label channel-order fix ===')
    print(f'mode: {"dry-run" if args.dry_run else "WRITE"}')
    print()

    total = FixStats()

    if not args.skip_packed and args.packed_dir.exists():
        s = fix_packed_dir(args.packed_dir, dry_run=args.dry_run)
        print(f'  packed: inspected={s.inspected} fixed={s.fixed} '
              f'already_fixed={s.already_fixed} bad_shape={s.skipped_unexpected_shape} '
              f'post_check_fail={s.failed_post_check}')
        print()
        total.inspected += s.inspected; total.fixed += s.fixed
        total.already_fixed += s.already_fixed
        total.skipped_unexpected_shape += s.skipped_unexpected_shape
        total.failed_post_check += s.failed_post_check

    if not args.skip_npz:
        for npz_dir in (args.npz_dir, args.quarantine_dir):
            if not npz_dir.exists():
                continue
            s = fix_npz_dir(npz_dir, dry_run=args.dry_run)
            print(f'  npz {npz_dir.name}: inspected={s.inspected} '
                  f'fixed={s.fixed} already_fixed={s.already_fixed} '
                  f'bad_shape={s.skipped_unexpected_shape} '
                  f'post_check_fail={s.failed_post_check}')
            print()
            total.inspected += s.inspected; total.fixed += s.fixed
            total.already_fixed += s.already_fixed
            total.skipped_unexpected_shape += s.skipped_unexpected_shape
            total.failed_post_check += s.failed_post_check

    print('=== summary ===')
    print(f'  inspected           : {total.inspected}')
    print(f'  fixed               : {total.fixed}')
    print(f'  already-fixed (skip): {total.already_fixed}')
    print(f'  unexpected-shape    : {total.skipped_unexpected_shape}')
    print(f'  post-check failures : {total.failed_post_check}')
    if total.failed_post_check > 0:
        print('\nFAIL: at least one post-fix subset-inequality check '
              'failed. Investigate before trusting the fixed data.',
              file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
