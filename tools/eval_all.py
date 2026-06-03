#!/usr/bin/env python3
"""Run the full eval scorecard and append results to evals/results.jsonl.

Each registered eval (in evals/builtin.py) gets called and its metrics
get recorded as one row. Includes git commit + dirty flag + free-form
notes so the JSONL grows alongside the project history.

Usage:
    # Run everything and append a row
    python tools/eval_all.py --notes "v2 + threshold tuning"

    # Just print the last few rows as a markdown table
    python tools/eval_all.py --report

    # Skip slow evals (tempo evals can take 30+ min)
    python tools/eval_all.py --skip-slow

    # Pick specific evals
    python tools/eval_all.py --only tempo.musdb tempo.osu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Trigger registration of all built-in evals.
from evals import builtin  # noqa: F401
from evals import REGISTRY, run, run_report


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--notes', default='',
                        help='Free-form notes recorded with this run')
    parser.add_argument('--only', nargs='+', default=None,
                        help='Run only these named evals (default: all)')
    parser.add_argument('--skip-slow', action='store_true',
                        help='Skip evals marked slow (saves ~30+ min)')
    parser.add_argument('--list', action='store_true',
                        help='List registered evals and exit')
    parser.add_argument('--report', action='store_true',
                        help='Print recent runs as a table and exit')
    parser.add_argument('--report-n', type=int, default=5,
                        help='How many recent runs to include in --report')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run evals but do not append to results.jsonl')
    args = parser.parse_args()

    if args.list:
        for name, spec in REGISTRY.items():
            tags = []
            if spec.slow:
                tags.append('slow')
            if spec.requires:
                tags.append('needs: ' + ', '.join(spec.requires))
            tag_str = f'  [{", ".join(tags)}]' if tags else ''
            print(f'  {name:30s}  {spec.description}{tag_str}')
        return

    if args.report:
        print(run_report(last_n=args.report_n))
        return

    run(notes=args.notes, only=args.only,
        skip_slow=args.skip_slow, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
