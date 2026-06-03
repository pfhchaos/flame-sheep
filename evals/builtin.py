"""Built-in eval registrations.

Wraps existing eval scripts (currently tools/eval_tempo_reconcile.py)
as registered functions that return structured metrics. New evals
get registered here so they show up in `python tools/eval_all.py`.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from .registry import register


_REPO = Path(__file__).resolve().parents[1]
_TEMPO_TOOL = _REPO / 'tools' / 'eval_tempo_reconcile.py'
_BEAT_TOOL = _REPO / 'tools' / 'eval_beat_detection.py'
_MUSDB_ROOT = Path.home() / 'MUSDB18' / 'MUSDB18-7'
_OSU_ROOT = Path.home() / '.cache' / 'flame-sheep' / 'corpus' / 'osu'
_MULTIDEPTH_WEIGHTS = _REPO / 'flame_sheep' / 'data' / 'beat_rnn_multidepth.npz'


def _parse_tempo_report(log: str) -> dict[str, float]:
    """Parse the markdown-ish report printed by eval_tempo_reconcile.

    Looks for the lines:
        baseline       8/38 (  21%)        17/38 (  45%)
        A: override    2/38 (   5%)        5/38 (  13%)
        ...
    and emits four metrics per strategy: exact_count, exact_pct,
    octave_count, octave_pct.
    """
    metrics: dict[str, float] = {}
    # Match the strategy table rows. Strategy label can have a colon
    # (e.g. "A: override") so use a non-greedy match up to the digits.
    pat = re.compile(
        r'^\s*(?P<label>[A-Za-z][A-Za-z:\s-]*?)\s+'
        r'(?P<ex_n>\d+)/(?P<ex_d>\d+)\s+\(\s*(?P<ex_pct>\d+)%\)\s+'
        r'(?P<oc_n>\d+)/(?P<oc_d>\d+)\s+\(\s*(?P<oc_pct>\d+)%\)',
        re.MULTILINE)
    label_map = {
        'baseline': 'baseline',
        'A: override': 'a_override',
        'B: shift': 'b_shift',
        'C: octave-only': 'c_octave_only',
    }
    for m in pat.finditer(log):
        label = m.group('label').strip()
        key = label_map.get(label)
        if not key:
            continue
        metrics[f'{key}.exact_pct'] = float(m.group('ex_pct'))
        metrics[f'{key}.octave_pct'] = float(m.group('oc_pct'))
        metrics[f'{key}.exact_n'] = float(m.group('ex_n'))
        metrics[f'{key}.exact_d'] = float(m.group('ex_d'))
    return metrics


def _run_tempo_eval(corpus: str, root: Path,
                    limit: int | None = None) -> dict[str, float]:
    if not root.exists():
        return {'__context__': {'skipped': f'corpus root missing: {root}'}}
    cmd = [sys.executable, str(_TEMPO_TOOL),
           '--corpus', corpus, '--root', str(root)]
    if limit:
        cmd += ['--limit', str(limit)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return {'__context__': {
            'failed': f'eval exited {result.returncode}',
            'stderr_tail': result.stderr[-500:],
        }}
    return _parse_tempo_report(result.stdout)


@register('tempo.osu',
          description='Tempo accuracy + reconciliation strategies on osu corpus (~38 tracks)',
          slow=True,
          requires=('osu_corpus',))
def eval_tempo_osu() -> dict[str, float]:
    return _run_tempo_eval('osu', _OSU_ROOT)


@register('tempo.musdb',
          description='Tempo accuracy + reconciliation strategies on MUSDB18 (144 tracks)',
          slow=True,
          requires=('musdb18',))
def eval_tempo_musdb() -> dict[str, float]:
    return _run_tempo_eval('musdb', _MUSDB_ROOT)


# ---------------------------------------------------------------------------
# Beat detection F1 — what does the detector ACTUALLY hit on osu ground truth?
# This is the metric tempo reconciliation needed to be ≥ tracker quality to
# work. Track here so any change to thresholds, weights, or detection logic
# shows its effect immediately.
# ---------------------------------------------------------------------------

def _parse_beat_report(log: str) -> dict[str, float]:
    """Parse the F1/precision/recall lines printed by eval_beat_detection.

    Two summaries are emitted (per-track mean, length-weighted pooled).
    We capture both — the pooled version is what should drive
    comparisons since it weights each second of audio equally.

    Expected formats:
        === Aggregate (mean over tracks) ===
          F1@70ms: 0.574 ± 0.064  (precision=0.493, recall=0.689)
        === Aggregate (length-weighted, pooled over corpus) ===
          F1@70ms: 0.580
    """
    metrics: dict[str, float] = {}
    mode: str | None = None
    pat_mean = re.compile(
        r'^\s*F1@(?P<win>\d+)ms:\s*(?P<f1>[\d.]+)\s*±\s*[\d.]+\s+'
        r'\(precision=(?P<p>[\d.]+),\s*recall=(?P<r>[\d.]+)\)')
    pat_pooled = re.compile(
        r'^\s*F1@(?P<win>\d+)ms:\s*(?P<f1>[\d.]+)\s*$')
    pat_cemgil_mean = re.compile(r'^\s*cemgil:\s*(?P<v>[\d.]+)\s*±')
    pat_cemgil_pooled = re.compile(r'^\s*cemgil:\s*(?P<v>[\d.]+)\s*$')

    for line in log.splitlines():
        if 'Aggregate (mean' in line:
            mode = 'mean'
        elif 'Aggregate (length-weighted' in line:
            mode = 'pooled'
        elif mode == 'mean':
            m = pat_mean.match(line)
            if m:
                w = m.group('win')
                metrics[f'mean.f1@{w}ms'] = float(m.group('f1'))
                metrics[f'mean.precision@{w}ms'] = float(m.group('p'))
                metrics[f'mean.recall@{w}ms'] = float(m.group('r'))
                continue
            m = pat_cemgil_mean.match(line)
            if m:
                metrics['mean.cemgil'] = float(m.group('v'))
        elif mode == 'pooled':
            m = pat_pooled.match(line)
            if m:
                w = m.group('win')
                metrics[f'pooled.f1@{w}ms'] = float(m.group('f1'))
                continue
            m = pat_cemgil_pooled.match(line)
            if m:
                metrics['pooled.cemgil'] = float(m.group('v'))
    return metrics


def _run_beat_eval(detector: str, *, tracks: int | None = None,
                    weights: Path | None = None) -> dict[str, float]:
    if not _OSU_ROOT.exists():
        return {'__context__': {'skipped': f'osu corpus missing: {_OSU_ROOT}'}}
    cmd = [sys.executable, str(_BEAT_TOOL),
           '--corpus', str(_OSU_ROOT),
           '--detector', detector]
    if tracks is not None:
        cmd += ['--tracks', str(tracks)]
    if weights is not None:
        cmd += ['--weights', str(weights)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return {'__context__': {
            'failed': f'eval exited {result.returncode}',
            'stderr_tail': result.stderr[-500:],
        }}
    return _parse_beat_report(result.stdout)


@register('beat.osu.multidepth',
          description='Beat detection F1 against osu ground truth — v2 multidepth at current thresholds',
          slow=True,
          requires=('osu_corpus', 'multidepth_weights'))
def eval_beat_multidepth() -> dict[str, float]:
    if not _MULTIDEPTH_WEIGHTS.exists():
        return {'__context__': {
            'skipped': f'weights missing: {_MULTIDEPTH_WEIGHTS}'}}
    return _run_beat_eval('beat_rnn_multidepth', weights=_MULTIDEPTH_WEIGHTS)


@register('beat.osu.percentile',
          description='Beat detection F1 against osu ground truth — CurrentSystemDetector (percentile/flux, no learning)',
          slow=True,
          requires=('osu_corpus',))
def eval_beat_percentile() -> dict[str, float]:
    return _run_beat_eval('current_system')


@register('beat.osu.beatnet',
          description='Beat detection F1 against osu ground truth — BeatNet oracle baseline',
          slow=True,
          requires=('osu_corpus', 'beatnet'))
def eval_beat_beatnet() -> dict[str, float]:
    return _run_beat_eval('beatnet')
