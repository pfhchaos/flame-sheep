#!/usr/bin/env python3
"""Pre-deployment gate: a new beat detector must beat the current
percentile baseline on osu before getting switched into production.

Exit code 0 = passes gate (better than or equal to baseline within
tolerance), non-zero = fails gate.

Use as a pre-deploy check:

    # Before changing audio.toml to point at new weights, gate them.
    python tools/check_beat_gate.py \\
        --detector beat_rnn_multidepth \\
        --weights flame_sheep/data/beat_rnn_v3.npz \\
    && cp flame_sheep/data/beat_rnn_v3.npz flame_sheep/data/beat_rnn_multidepth.npz

    # Or as `git push` hook against the scorecard.

The baseline is the most recent recorded percentile F1@70ms in
evals/results.jsonl. If no percentile row exists, falls back to a
hardcoded floor (0.50 — chosen as "anything less than this is
demonstrably worse than the floor we already have").

Rationale: 2026-06-03 we discovered the deployed v2 multidepth model
had been silently running for ~3 days at F1=0.398 while the unlearned
percentile detector hits 0.488. The "trained model worse than
baseline" regression shipped because nothing checked. This gate is
the structural fix — same shape as the test-count floor in the
silent-skip prevention, applied to model quality.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


_REPO = Path(__file__).resolve().parent.parent
_BEAT_TOOL = _REPO / 'tools' / 'eval_beat_detection.py'
_OSU_ROOT = Path.home() / '.cache' / 'flame-sheep' / 'corpus' / 'osu'
_RESULTS = _REPO / 'evals' / 'results.jsonl'
_HARDCODED_FLOOR = 0.50  # absolute minimum F1@70ms for any deployable detector


def _get_baseline_f1() -> tuple[float, str]:
    """Returns (baseline_f1, source_label).

    Looks for the most recent recorded percentile F1@70ms (pooled) in
    the scorecard. Falls back to _HARDCODED_FLOOR if no row found.
    """
    if not _RESULTS.exists():
        return _HARDCODED_FLOOR, f'hardcoded floor ({_RESULTS} not found)'
    import json
    latest = None
    with _RESULTS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            f1 = entry.get('metrics', {}).get('beat.osu.percentile.pooled.f1@70ms')
            if f1 is not None:
                latest = (float(f1), entry.get('timestamp', '?'))
    if latest is None:
        return _HARDCODED_FLOOR, f'hardcoded floor (no percentile row in scorecard)'
    return latest[0], f'recorded percentile baseline @ {latest[1]}'


def _run_detector_eval(detector: str, weights: Path | None) -> float:
    """Returns pooled F1@70ms for the detector on osu. Raises on
    eval failure."""
    if not _OSU_ROOT.exists():
        raise SystemExit(f'osu corpus missing: {_OSU_ROOT}')
    cmd = [sys.executable, str(_BEAT_TOOL),
           '--corpus', str(_OSU_ROOT),
           '--detector', detector]
    if weights is not None:
        cmd += ['--weights', str(weights)]
    print(f'Running {detector} eval on osu corpus...')
    print(f'  $ {" ".join(cmd)}')
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(result.stderr[-1000:], file=sys.stderr)
        raise SystemExit(f'eval exited {result.returncode}')
    # Parse pooled F1@70ms — same format as the scorecard's parser.
    import re
    pat = re.compile(
        r'=== Aggregate \(length-weighted.*?===\s*'
        r'F1@25ms:.*?'
        r'F1@50ms:.*?'
        r'F1@70ms:\s*([\d.]+)',
        re.DOTALL)
    m = pat.search(result.stdout)
    if not m:
        # Try mean fallback
        pat2 = re.compile(r'^\s*F1@70ms:\s*([\d.]+)', re.MULTILINE)
        m = pat2.search(result.stdout)
    if not m:
        print(result.stdout, file=sys.stderr)
        raise SystemExit('could not parse F1@70ms from eval output')
    return float(m.group(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--detector', required=True,
                        choices=['beat_rnn', 'beat_rnn_multidepth', 'beatnet'],
                        help='Detector to gate (current_system is the baseline, not gated)')
    parser.add_argument('--weights', type=Path,
                        help='Weights file (required for beat_rnn variants)')
    parser.add_argument('--tolerance', type=float, default=0.02,
                        help='How far below baseline F1 to allow (default 0.02 '
                             '= 2pp absolute). Use 0 for strict beat-or-equal.')
    parser.add_argument('--baseline-f1', type=float, default=None,
                        help='Override the auto-detected baseline. Use to '
                             'compare against an explicit value.')
    args = parser.parse_args()

    if args.baseline_f1 is not None:
        baseline, source = args.baseline_f1, 'CLI override'
    else:
        baseline, source = _get_baseline_f1()

    print(f'Baseline F1@70ms: {baseline:.3f}  ({source})')
    print(f'Tolerance:        {args.tolerance:.3f}')
    print(f'Floor for pass:   {baseline - args.tolerance:.3f}')
    print()

    f1 = _run_detector_eval(args.detector, args.weights)
    print()
    print(f'Result: {args.detector} F1@70ms = {f1:.3f}')

    if f1 >= baseline - args.tolerance:
        delta = f1 - baseline
        sign = '+' if delta >= 0 else '−'
        print(f'  ✓ PASS  ({sign}{abs(delta):.3f} vs baseline)')
        return 0
    delta = baseline - f1
    print(f'  ✗ FAIL  (−{delta:.3f} below baseline, beyond tolerance of {args.tolerance:.3f})')
    print()
    print('This detector underperforms the percentile baseline. Do NOT')
    print('deploy. Investigate before retraining or shipping.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
