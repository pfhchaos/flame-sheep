#!/usr/bin/env python3
"""Pre-deployment gate: a new beat detector must beat the current
percentile baseline on every corpus we have baselines for.

Exit code 0 = passes gate (better than or equal to baseline within
tolerance on EVERY corpus), non-zero = fails gate.

Use as a pre-deploy check:

    # Before changing audio.toml to point at new weights, gate them.
    python tools/check_beat_gate.py \\
        --detector beat_rnn_multidepth \\
        --weights flame_sheep/data/beat_rnn_v3.npz \\
    && cp flame_sheep/data/beat_rnn_v3.npz flame_sheep/data/beat_rnn_multidepth.npz

The baselines are the most recent recorded `beat.<corpus>.percentile.
pooled.f1@70ms` values in `evals/results.jsonl` — one per corpus.

osu is mandatory: a missing osu baseline falls back to a hardcoded
floor (0.50, "anything less than this is demonstrably worse than the
floor we already have").

gtzan is opt-in: a missing gtzan baseline means the corpus hasn't
landed yet, and the gate falls back to osu-only behavior. Once a
gtzan baseline exists in results.jsonl, EVERY future deploy must
clear both bars — a detector that improves osu by 0.05 but
regresses gtzan by 0.10 will fail.

Rationale: a single number masks asymmetric regressions, per the
val_acc-bias / signal-balance-audits feedback. The cross-corpus
gate is the structural defense against overfit-to-osu-conventions.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


_REPO = Path(__file__).resolve().parent.parent
_BEAT_TOOL = _REPO / 'tools' / 'eval_beat_detection.py'
_OSU_ROOT = Path.home() / '.cache' / 'flame-sheep' / 'corpus' / 'osu'
_GTZAN_ROOT = Path.home() / 'datasets' / 'GTZAN'
_RESULTS = _REPO / 'evals' / 'results.jsonl'
_HARDCODED_FLOOR = 0.50  # absolute minimum F1@70ms for any deployable detector

# Per-corpus configuration. The key is the user-visible corpus kind
# (also used for log lines); the value is the data we need to invoke
# the eval and look up the baseline metric.
_CORPORA = {
    'osu':   {'root': _OSU_ROOT,   'metric': 'beat.osu.percentile.pooled.f1@70ms'},
    'gtzan': {'root': _GTZAN_ROOT, 'metric': 'beat.gtzan.percentile.pooled.f1@70ms'},
}


def _latest_metric(metric_key: str) -> tuple[float | None, str]:
    """Return (value, source_label) for the most recent recorded
    value of `metric_key` in results.jsonl, or (None, why) if absent."""
    if not _RESULTS.exists():
        return None, f'{_RESULTS} not found'
    import json
    latest: tuple[float, str] | None = None
    with _RESULTS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            v = entry.get('metrics', {}).get(metric_key)
            if v is not None:
                latest = (float(v), entry.get('timestamp', '?'))
    if latest is None:
        return None, 'no row in scorecard'
    return latest[0], f'recorded @ {latest[1]}'


def _run_detector_eval(detector: str, weights: Path | None,
                        corpus_kind: str, corpus_root: Path) -> float:
    """Returns pooled F1@70ms for the detector on the named corpus."""
    if not corpus_root.exists():
        raise SystemExit(f'{corpus_kind} corpus missing: {corpus_root}')
    cmd = [sys.executable, str(_BEAT_TOOL),
           '--corpus-kind', corpus_kind,
           '--corpus', str(corpus_root),
           '--detector', detector]
    if weights is not None:
        cmd += ['--weights', str(weights)]
    print(f'Running {detector} eval on {corpus_kind} corpus...')
    print(f'  $ {" ".join(cmd)}')
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(result.stderr[-1000:], file=sys.stderr)
        raise SystemExit(f'eval exited {result.returncode}')
    # Parse pooled F1@70ms — same format as the scorecard's parser.
    pat = re.compile(
        r'=== .*?Aggregate \(length-weighted.*?===\s*'
        r'F1@25ms:.*?'
        r'F1@50ms:.*?'
        r'F1@70ms:\s*([\d.]+)',
        re.DOTALL)
    m = pat.search(result.stdout)
    if not m:
        pat2 = re.compile(r'^\s*F1@70ms:\s*([\d.]+)', re.MULTILINE)
        m = pat2.search(result.stdout)
    if not m:
        print(result.stdout, file=sys.stderr)
        raise SystemExit(f'could not parse F1@70ms from {corpus_kind} eval output')
    return float(m.group(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--detector', required=True,
                        choices=['beat_rnn', 'beat_rnn_multidepth', 'beatnet', 'madmom'],
                        help='Detector to gate (current_system is the baseline, not gated)')
    parser.add_argument('--weights', type=Path,
                        help='Weights file (required for beat_rnn variants)')
    parser.add_argument('--tolerance', type=float, default=0.02,
                        help='How far below baseline F1 to allow (default 0.02 '
                             '= 2pp absolute). Use 0 for strict beat-or-equal.')
    parser.add_argument('--baseline-osu', type=float, default=None,
                        help='Override auto-detected osu baseline.')
    parser.add_argument('--baseline-gtzan', type=float, default=None,
                        help='Override auto-detected gtzan baseline. Pass to '
                             'force gtzan check even with no scorecard row.')
    parser.add_argument('--require-gtzan', action='store_true',
                        help='Fail if gtzan baseline is missing rather than '
                             'falling back to osu-only.')
    args = parser.parse_args()

    # Resolve baselines per corpus.
    baselines: dict[str, tuple[float, str]] = {}

    # osu — mandatory, falls back to hardcoded floor.
    if args.baseline_osu is not None:
        baselines['osu'] = (args.baseline_osu, 'CLI override')
    else:
        v, src = _latest_metric(_CORPORA['osu']['metric'])
        if v is None:
            baselines['osu'] = (_HARDCODED_FLOOR, f'hardcoded floor ({src})')
        else:
            baselines['osu'] = (v, src)

    # gtzan — optional, opted in once a row exists or when overridden.
    if args.baseline_gtzan is not None:
        baselines['gtzan'] = (args.baseline_gtzan, 'CLI override')
    else:
        v, src = _latest_metric(_CORPORA['gtzan']['metric'])
        if v is not None:
            baselines['gtzan'] = (v, src)
        elif args.require_gtzan:
            print(f'--require-gtzan: no gtzan baseline ({src})', file=sys.stderr)
            return 4

    print('Baselines:')
    for corpus, (val, src) in baselines.items():
        print(f'  {corpus:6s} F1@70ms: {val:.3f}  ({src})')
    print(f'Tolerance: {args.tolerance:.3f}')
    print()

    # Run candidate on each corpus where we have a baseline.
    results: dict[str, float] = {}
    for corpus in baselines:
        cfg = _CORPORA[corpus]
        if not cfg['root'].exists():
            if corpus == 'gtzan':
                # Baseline exists but corpus disk doesn't — surface clearly
                # so it's never silently treated as a pass.
                print(f'  ✗ FAIL  gtzan baseline exists but {cfg["root"]} '
                      f'is missing; cannot evaluate', file=sys.stderr)
                return 5
            raise SystemExit(f'{corpus} corpus missing: {cfg["root"]}')
        f1 = _run_detector_eval(args.detector, args.weights,
                                  corpus_kind=corpus, corpus_root=cfg['root'])
        results[corpus] = f1
        print()

    print('Results:')
    failures: list[str] = []
    for corpus, f1 in results.items():
        baseline, _ = baselines[corpus]
        delta = f1 - baseline
        sign = '+' if delta >= 0 else '−'
        floor = baseline - args.tolerance
        status = '✓' if f1 >= floor else '✗'
        if f1 < floor:
            failures.append(
                f'{corpus}: {f1:.3f} < {floor:.3f} (baseline {baseline:.3f} '
                f'- tol {args.tolerance:.3f})')
        print(f'  {status} {corpus:6s} {args.detector} F1@70ms = {f1:.3f}  '
              f'({sign}{abs(delta):.3f} vs baseline)')
    print()

    if not failures:
        print('PASS — meets baseline on every gated corpus.')
        return 0

    print('FAIL — regresses on:', file=sys.stderr)
    for f in failures:
        print(f'  - {f}', file=sys.stderr)
    print(file=sys.stderr)
    print('Do NOT deploy. Investigate before retraining or shipping.',
          file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
