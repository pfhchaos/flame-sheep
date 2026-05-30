#!/usr/bin/env python3
"""Refresh checked-in eval baselines for flame_sheep_audio.

Run after an intentional algorithm change to capture the new behavior
as the reference. The regression test in tests/eval/ compares against
these JSONs and fails on drift > tolerance, so refreshing without
review can hide real regressions — only do it when you understand
WHY the numbers moved.

Usage:
    # Refresh all registered categories
    tools/refresh_eval_baselines.py

    # Refresh one category
    tools/refresh_eval_baselines.py band_routing

    # Refresh + attach a note describing the change
    tools/refresh_eval_baselines.py band_routing \\
        --note "switched percentile from 90 to 99"

    # Dry run (print the new baseline, don't write it)
    tools/refresh_eval_baselines.py --dry-run band_routing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Project imports
_PROJECT = Path(__file__).resolve().parent.parent
_AUDIO_SRC = _PROJECT / 'flame_sheep_audio' / 'src'
for p in (_PROJECT, _AUDIO_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from flame_sheep_audio.eval.common import (
    REGISTRY, Baseline, save_baseline, load_baseline, make_metadata,
)


def _refresh_one(category, *, note: str, dry_run: bool) -> None:
    print(f'\n=== Refreshing baseline: {category.name} ===')
    metrics, per_stim = category.run()

    new_baseline = Baseline(
        name=category.name,
        metrics=metrics,
        per_stimulus=per_stim,
        metadata=make_metadata(refresh_note=note),
    )

    # Show the diff against any existing baseline so the operator can
    # eyeball the change before committing.
    existing = load_baseline(category.name)
    if existing is None:
        print('  (no prior baseline; this is the first capture)')
    else:
        print('  metric drift vs prior baseline:')
        all_keys = sorted(set(existing.metrics) | set(metrics))
        for k in all_keys:
            old = existing.metrics.get(k)
            new = metrics.get(k)
            if old is None:
                print(f'    NEW  {k:40s}                 → {new:.4f}')
            elif new is None:
                print(f'    GONE {k:40s} {old:.4f} →')
            else:
                delta = new - old
                pct = 100.0 * delta / max(abs(old), 1e-9)
                print(f'         {k:40s} {old:.4f} → {new:.4f}  '
                      f'({pct:+.1f}%)')

    if dry_run:
        print('\n  dry-run: NOT writing baseline')
        return
    save_baseline(new_baseline)
    print(f'  wrote {category.name}.json')


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Refresh eval baselines after an intentional change')
    parser.add_argument('category', nargs='?',
                        help='Specific category to refresh (default: all)')
    parser.add_argument('--note', default='',
                        help='Short reason for the refresh — stored in '
                             'baseline metadata for the commit reader')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would change without writing')
    parser.add_argument('--list', action='store_true',
                        help='List registered categories and exit')
    args = parser.parse_args()

    if args.list:
        for c in REGISTRY:
            print(c.name)
        return 0

    if args.category:
        match = [c for c in REGISTRY if c.name == args.category]
        if not match:
            available = ', '.join(c.name for c in REGISTRY)
            print(f'Unknown category {args.category!r}. '
                  f'Available: {available}', file=sys.stderr)
            return 2
        targets = match
    else:
        targets = REGISTRY

    for c in targets:
        _refresh_one(c, note=args.note, dry_run=args.dry_run)
    return 0


if __name__ == '__main__':
    sys.exit(main())
