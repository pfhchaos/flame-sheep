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

from .registry import register, meta_eval


_REPO = Path(__file__).resolve().parents[1]
_TEMPO_TOOL = _REPO / 'tools' / 'eval_tempo_reconcile.py'
_BEAT_TOOL = _REPO / 'tools' / 'eval_beat_detection.py'
_TEMPO_EST_TOOL = _REPO / 'tools' / 'eval_tempo_estimators.py'
_MUSDB_ROOT = Path.home() / 'MUSDB18' / 'MUSDB18-7'
_OSU_ROOT = Path.home() / '.cache' / 'flame-sheep' / 'corpus' / 'osu'
_GTZAN_ROOT = Path.home() / 'datasets' / 'GTZAN'
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
    # Latency table patterns — emitted by tools/eval_beat_detection.py
    # `_print_latency`. Two sections: per-track totals + per-component.
    pat_latency_total = re.compile(
        r'^\s*total_sec:\s+mean=(?P<mean>[\d.]+)\s+'
        r'p50=(?P<p50>[\d.]+)\s+p95=(?P<p95>[\d.]+)\s+p99=(?P<p99>[\d.]+)')
    pat_latency_rtf = re.compile(
        r'^\s*realtime_factor:\s+mean=(?P<mean>[\d.]+)\s+p95=(?P<p95>[\d.]+)')
    pat_latency_comp = re.compile(
        r'^\s*(?P<comp>\w+)_sec:\s+mean=(?P<mean>[\d.]+)\s+p95=(?P<p95>[\d.]+)')

    for line in log.splitlines():
        if 'Aggregate (mean' in line:
            mode = 'mean'
        elif 'Aggregate (length-weighted' in line:
            mode = 'pooled'
        elif 'Latency (per-track, total)' in line:
            mode = 'latency_total'
        elif 'Latency (per-track, components)' in line:
            mode = 'latency_comp'
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
        elif mode == 'latency_total':
            m = pat_latency_total.match(line)
            if m:
                metrics['latency.total.mean_sec'] = float(m.group('mean'))
                metrics['latency.total.p50_sec'] = float(m.group('p50'))
                metrics['latency.total.p95_sec'] = float(m.group('p95'))
                metrics['latency.total.p99_sec'] = float(m.group('p99'))
                continue
            m = pat_latency_rtf.match(line)
            if m:
                metrics['latency.realtime_factor.mean'] = float(m.group('mean'))
                metrics['latency.realtime_factor.p95'] = float(m.group('p95'))
        elif mode == 'latency_comp':
            m = pat_latency_comp.match(line)
            if m:
                comp = m.group('comp')
                if comp == 'total':
                    continue  # already captured above
                metrics[f'latency.{comp}.mean_sec'] = float(m.group('mean'))
                metrics[f'latency.{comp}.p95_sec'] = float(m.group('p95'))
    return metrics


def _run_beat_eval(detector: str, *,
                    corpus_kind: str = 'osu',
                    corpus_root: Path | None = None,
                    kind: str = 'beat',
                    tracks: int | None = None,
                    weights: Path | None = None) -> dict[str, float]:
    """Invoke tools/eval_beat_detection.py and parse its aggregate report.

    corpus_kind:  'osu' | 'gtzan'         — which annotation system
    corpus_root:  defaults per corpus_kind if None
    kind:         'beat' | 'downbeat'      — which output to score
    """
    if corpus_root is None:
        corpus_root = _OSU_ROOT if corpus_kind == 'osu' else _GTZAN_ROOT
    if not corpus_root.exists():
        return {'__context__': {
            'skipped': f'{corpus_kind} corpus missing: {corpus_root}'}}
    cmd = [sys.executable, str(_BEAT_TOOL),
           '--corpus-kind', corpus_kind,
           '--corpus', str(corpus_root),
           '--kind', kind,
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


@register('beat.osu.madmom',
          description='Beat detection F1 against osu ground truth — madmom RNN+DBN oracle baseline',
          slow=True,
          requires=('osu_corpus', 'madmom'))
def eval_beat_madmom() -> dict[str, float]:
    return _run_beat_eval('madmom')


@register('beat.osu.beatnet_streaming',
          description='Beat detection F1 against osu ground truth — BeatNet in per-hop streaming mode (post 2026-06-06 indexing fix)',
          slow=True,
          requires=('osu_corpus', 'beatnet'))
def eval_beat_beatnet_streaming() -> dict[str, float]:
    return _run_beat_eval('beatnet_streaming')


@register('beat.osu.beatnet_sliding',
          description='Beat detection F1 against osu ground truth — BeatNet with sliding-STFT streaming features (deploy-candidate)',
          slow=True,
          requires=('osu_corpus', 'beatnet'))
def eval_beat_beatnet_sliding() -> dict[str, float]:
    return _run_beat_eval('beatnet_sliding')


@register('beat.osu.beatnet_lite',
          description='Beat detection F1 against osu ground truth — pure-numpy BeatNet (no torch/madmom; ~2MB install footprint)',
          slow=True,
          requires=('osu_corpus',))
def eval_beat_beatnet_lite() -> dict[str, float]:
    return _run_beat_eval('beatnet_lite')


# ---------------------------------------------------------------------------
# GTZAN-rhythm corpus (Marchand/Fresnel/Peeters 2015) — hand-annotated
# beats + downbeats for 999 GTZAN tracks (reggae.00086 missing upstream,
# jazz.00054 has corrupt audio, both auto-skipped by the loader).
#
# Cross-corpus consistency check: any detector that overfits to osu's
# mapper-derived annotation conventions will diverge on gtzan's
# hand-labels. The consistency meta-eval surfaces that divergence.
#
# tracks=9999 forces "all available" (the runner caps at len(handles)),
# in contrast to the osu evals which use the 10-track default. Stable
# baselines need the full corpus; 10 random gtzan tracks would be too
# sparse to compare detectors meaningfully.
# ---------------------------------------------------------------------------

@register('beat.gtzan.percentile',
          description='Beat detection F1 against gtzan ground truth — CurrentSystemDetector (deployed)',
          slow=True,
          requires=('gtzan_corpus',))
def eval_beat_gtzan_percentile() -> dict[str, float]:
    return _run_beat_eval('current_system', corpus_kind='gtzan', tracks=9999)


@register('beat.gtzan.beatnet',
          description='Beat detection F1 against gtzan ground truth — BeatNet oracle baseline',
          slow=True,
          requires=('gtzan_corpus', 'beatnet'))
def eval_beat_gtzan_beatnet() -> dict[str, float]:
    return _run_beat_eval('beatnet', corpus_kind='gtzan', tracks=9999)


@register('beat.gtzan.madmom',
          description='Beat detection F1 against gtzan ground truth — madmom RNN+DBN oracle baseline',
          slow=True,
          requires=('gtzan_corpus', 'madmom'))
def eval_beat_gtzan_madmom() -> dict[str, float]:
    return _run_beat_eval('madmom', corpus_kind='gtzan', tracks=9999)


@register('beat.gtzan.beatnet_streaming',
          description='Beat detection F1 against gtzan ground truth — BeatNet in per-hop streaming mode (post 2026-06-06 indexing fix)',
          slow=True,
          requires=('gtzan_corpus', 'beatnet'))
def eval_beat_gtzan_beatnet_streaming() -> dict[str, float]:
    return _run_beat_eval('beatnet_streaming', corpus_kind='gtzan', tracks=9999)


@register('beat.gtzan.beatnet_sliding',
          description='Beat detection F1 against gtzan ground truth — BeatNet with sliding-STFT streaming features (deploy-candidate)',
          slow=True,
          requires=('gtzan_corpus', 'beatnet'))
def eval_beat_gtzan_beatnet_sliding() -> dict[str, float]:
    return _run_beat_eval('beatnet_sliding', corpus_kind='gtzan', tracks=9999)


@register('beat.gtzan.beatnet_lite',
          description='Beat detection F1 against gtzan ground truth — pure-numpy BeatNet (no torch/madmom; ~2MB install footprint)',
          slow=True,
          requires=('gtzan_corpus',))
def eval_beat_gtzan_beatnet_lite() -> dict[str, float]:
    return _run_beat_eval('beatnet_lite', corpus_kind='gtzan', tracks=9999)


@register('downbeat.gtzan.beatnet',
          description='Downbeat detection F1 against gtzan ground truth — BeatNet oracle baseline',
          slow=True,
          requires=('gtzan_corpus', 'beatnet'))
def eval_downbeat_gtzan_beatnet() -> dict[str, float]:
    return _run_beat_eval('beatnet', corpus_kind='gtzan',
                            kind='downbeat', tracks=9999)


@register('downbeat.gtzan.madmom',
          description='Downbeat detection F1 against gtzan ground truth — madmom RNN+DBN oracle baseline',
          slow=True,
          requires=('gtzan_corpus', 'madmom'))
def eval_downbeat_gtzan_madmom() -> dict[str, float]:
    return _run_beat_eval('madmom', corpus_kind='gtzan',
                            kind='downbeat', tracks=9999)


# ---------------------------------------------------------------------------
# Cross-corpus consistency meta-evals — derive aggregate stats from the
# per-corpus F1 metrics each detector contributes.
#
# Why mean + min + spread + n_corpora and not just one number:
#   * mean hides asymmetric regressions (osu +0.05, gtzan -0.10 averages
#     to a wash);
#   * min is the floor that gates deployment per `tools/check_beat_gate.py`;
#   * spread = max - min surfaces overfitting to one corpus's annotation
#     conventions — a healthy detector should be flat-ish across corpora,
#     and a widening spread is the early warning before the gate flips.
# ---------------------------------------------------------------------------

_CONSISTENCY_KEY = 'pooled.f1@70ms'
_CONSISTENCY_CORPORA = ('osu', 'gtzan')


def _consistency_for_detector(latest: dict[str, float], kind: str,
                                detector: str) -> dict[str, float]:
    """Common shape: pull `<kind>.<corpus>.<detector>.<KEY>` for each
    corpus where it exists, emit mean/min/spread/n_corpora."""
    values: list[float] = []
    for corpus in _CONSISTENCY_CORPORA:
        key = f'{kind}.{corpus}.{detector}.{_CONSISTENCY_KEY}'
        v = latest.get(key)
        if v is not None:
            values.append(float(v))
    if not values:
        return {'__context__': {
            'skipped': f'no per-corpus {kind} F1 available for {detector}'}}
    return {
        'mean': sum(values) / len(values),
        'min': min(values),
        'spread': max(values) - min(values),
        'n_corpora': float(len(values)),
    }


@meta_eval('beat.consistency.percentile',
           description='Cross-corpus consistency of percentile beat F1 (osu, gtzan)')
def meta_beat_consistency_percentile(latest: dict[str, float]) -> dict[str, float]:
    return _consistency_for_detector(latest, 'beat', 'percentile')


@meta_eval('beat.consistency.beatnet',
           description='Cross-corpus consistency of BeatNet beat F1 (osu, gtzan)')
def meta_beat_consistency_beatnet(latest: dict[str, float]) -> dict[str, float]:
    return _consistency_for_detector(latest, 'beat', 'beatnet')


@meta_eval('beat.consistency.madmom',
           description='Cross-corpus consistency of madmom beat F1 (osu, gtzan)')
def meta_beat_consistency_madmom(latest: dict[str, float]) -> dict[str, float]:
    return _consistency_for_detector(latest, 'beat', 'madmom')


@meta_eval('beat.consistency.beatnet_streaming',
           description='Cross-corpus consistency of streaming BeatNet beat F1 (osu, gtzan)')
def meta_beat_consistency_beatnet_streaming(latest: dict[str, float]) -> dict[str, float]:
    return _consistency_for_detector(latest, 'beat', 'beatnet_streaming')


@meta_eval('beat.consistency.beatnet_sliding',
           description='Cross-corpus consistency of sliding-STFT BeatNet beat F1 (osu, gtzan)')
def meta_beat_consistency_beatnet_sliding(latest: dict[str, float]) -> dict[str, float]:
    return _consistency_for_detector(latest, 'beat', 'beatnet_sliding')


@meta_eval('beat.consistency.beatnet_lite',
           description='Cross-corpus consistency of pure-numpy BeatNet beat F1 (osu, gtzan)')
def meta_beat_consistency_beatnet_lite(latest: dict[str, float]) -> dict[str, float]:
    return _consistency_for_detector(latest, 'beat', 'beatnet_lite')


# ---------------------------------------------------------------------------
# Cross-corpus LATENCY consistency (§7) — parallels the F1 meta-evals.
#
# A detector that's fast on gtzan (30 s tracks) and slow on osu (3 min
# tracks) exposes scaling characteristics: large per-track overhead
# favors longer tracks (lower per-second cost), super-linear scaling
# favors shorter tracks. Spread surfaces both.
#
# Metric: latency.<corpus>.<detector>.latency.realtime_factor.mean.
# That's the "what fraction of audio duration does inference take" —
# directly comparable across corpora.
# ---------------------------------------------------------------------------

_LATENCY_CONSISTENCY_KEY = 'latency.realtime_factor.mean'


def _latency_consistency_for_detector(latest: dict[str, float], kind: str,
                                        detector: str) -> dict[str, float]:
    """Same shape as `_consistency_for_detector` but reads
    realtime-factor instead of F1."""
    values: list[float] = []
    for corpus in _CONSISTENCY_CORPORA:
        key = f'{kind}.{corpus}.{detector}.{_LATENCY_CONSISTENCY_KEY}'
        v = latest.get(key)
        if v is not None:
            values.append(float(v))
    if not values:
        return {'__context__': {
            'skipped': f'no per-corpus {kind} latency available for {detector}'}}
    return {
        'mean': sum(values) / len(values),
        'min': min(values),
        'max': max(values),
        'spread': max(values) - min(values),
        'n_corpora': float(len(values)),
    }


@meta_eval('beat.latency_consistency.percentile',
           description='Cross-corpus latency consistency of percentile beat detection')
def meta_latency_consistency_percentile(latest: dict[str, float]) -> dict[str, float]:
    return _latency_consistency_for_detector(latest, 'beat', 'percentile')


@meta_eval('beat.latency_consistency.beatnet',
           description='Cross-corpus latency consistency of BeatNet beat detection')
def meta_latency_consistency_beatnet(latest: dict[str, float]) -> dict[str, float]:
    return _latency_consistency_for_detector(latest, 'beat', 'beatnet')


@meta_eval('beat.latency_consistency.madmom',
           description='Cross-corpus latency consistency of madmom beat detection')
def meta_latency_consistency_madmom(latest: dict[str, float]) -> dict[str, float]:
    return _latency_consistency_for_detector(latest, 'beat', 'madmom')


@meta_eval('beat.latency_consistency.beatnet_streaming',
           description='Cross-corpus latency consistency of streaming BeatNet beat detection')
def meta_latency_consistency_beatnet_streaming(latest: dict[str, float]) -> dict[str, float]:
    return _latency_consistency_for_detector(latest, 'beat', 'beatnet_streaming')


@meta_eval('beat.latency_consistency.beatnet_sliding',
           description='Cross-corpus latency consistency of sliding-STFT BeatNet beat detection')
def meta_latency_consistency_beatnet_sliding(latest: dict[str, float]) -> dict[str, float]:
    return _latency_consistency_for_detector(latest, 'beat', 'beatnet_sliding')


@meta_eval('beat.latency_consistency.beatnet_lite',
           description='Cross-corpus latency consistency of pure-numpy BeatNet beat detection')
def meta_latency_consistency_beatnet_lite(latest: dict[str, float]) -> dict[str, float]:
    return _latency_consistency_for_detector(latest, 'beat', 'beatnet_lite')


# ---------------------------------------------------------------------------
# Tempo estimator scorecard — unified interface for tempo trackers.
#
# Mirrors the beat-detection registrations: one eval per (corpus,
# estimator) cell, with cross-corpus consistency meta-evals.
#
# Output format from tools/eval_tempo_estimators.py:
#     === Aggregate (mean over tracks) ===
#       exact:    62/100 (62.0%)
#       oe1:      83/100 (83.0%)
#       mae_bpm:  1.42 (over 62 correct-octave tracks)
#     === Latency (across tracks) ===
#       total.mean_sec:     2.354
#       ...
#       realtime_factor.mean: 0.078
# ---------------------------------------------------------------------------


def _parse_tempo_estimator_report(log: str) -> dict[str, float]:
    """Parse `eval_tempo_estimators.py` output. Returns flat metrics
    dict keyed by `<accuracy|latency>.<name>` so registrations can
    surface a stable scorecard column."""
    metrics: dict[str, float] = {}
    mode: str | None = None
    pat_count_pct = re.compile(
        r'^\s*(?P<name>exact|oe1):\s*(?P<n>\d+)/(?P<d>\d+)\s+\((?P<pct>[\d.]+)%\)')
    pat_mae = re.compile(r'^\s*mae_bpm:\s*(?P<v>[\d.]+)')
    pat_kv = re.compile(r'^\s*(?P<key>[a-zA-Z0-9_.]+):\s*(?P<v>[\d.]+)')

    for line in log.splitlines():
        if 'Aggregate (mean' in line:
            mode = 'accuracy'
        elif 'Latency (across' in line:
            mode = 'latency'
        elif mode == 'accuracy':
            m = pat_count_pct.match(line)
            if m:
                metrics[f'accuracy.{m.group("name")}_pct'] = float(m.group('pct'))
                metrics[f'accuracy.{m.group("name")}_n'] = float(m.group('n'))
                metrics[f'accuracy.{m.group("name")}_d'] = float(m.group('d'))
                continue
            m = pat_mae.match(line)
            if m:
                metrics['accuracy.mae_bpm'] = float(m.group('v'))
        elif mode == 'latency':
            m = pat_kv.match(line)
            if m:
                metrics[f'latency.{m.group("key")}'] = float(m.group('v'))
    return metrics


def _run_tempo_estimator_eval(estimator: str, *,
                                corpus_kind: str = 'gtzan',
                                corpus_root: Path | None = None,
                                tracks: int | None = None) -> dict[str, float]:
    """Invoke eval_tempo_estimators.py and parse its output."""
    if corpus_root is None:
        corpus_root = _GTZAN_ROOT if corpus_kind == 'gtzan' else _OSU_ROOT
    if not corpus_root.exists():
        return {'__context__': {
            'skipped': f'{corpus_kind} corpus missing: {corpus_root}'}}
    cmd = [sys.executable, str(_TEMPO_EST_TOOL),
           '--estimator', estimator,
           '--corpus-kind', corpus_kind,
           '--corpus-root', str(corpus_root)]
    if tracks is not None:
        cmd += ['--tracks', str(tracks)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return {'__context__': {
            'failed': f'eval exited {result.returncode}',
            'stderr_tail': result.stderr[-500:],
        }}
    return _parse_tempo_estimator_report(result.stdout)


# Per-(corpus, estimator) registrations. Mirrors the beat layout.
# gtzan uses tracks=9999 for full-corpus runs; osu is small (~38), runs
# all tracks by default.

@register('tempo.gtzan.acf',
          description='ACF tempo estimator on gtzan — production streaming tracker',
          slow=True, requires=('gtzan_corpus',))
def eval_tempo_gtzan_acf() -> dict[str, float]:
    return _run_tempo_estimator_eval('acf', corpus_kind='gtzan', tracks=9999)


@register('tempo.gtzan.btrack',
          description='BTrack tempo estimator on gtzan — production streaming tracker',
          slow=True, requires=('gtzan_corpus', 'btrack'))
def eval_tempo_gtzan_btrack() -> dict[str, float]:
    return _run_tempo_estimator_eval('btrack', corpus_kind='gtzan', tracks=9999)


@register('tempo.gtzan.beatnet_pf',
          description='BeatNet particle-filter tempo on gtzan — deployed when detector=beatnet_lite',
          slow=True, requires=('gtzan_corpus', 'beatnet'))
def eval_tempo_gtzan_beatnet_pf() -> dict[str, float]:
    return _run_tempo_estimator_eval('beatnet_pf', corpus_kind='gtzan', tracks=9999)


@register('tempo.gtzan.madmom_comb',
          description='madmom comb-filter tempo on gtzan — offline oracle',
          slow=True, requires=('gtzan_corpus', 'madmom'))
def eval_tempo_gtzan_madmom_comb() -> dict[str, float]:
    return _run_tempo_estimator_eval('madmom_comb', corpus_kind='gtzan', tracks=9999)


@register('tempo.gtzan.madmom_acf',
          description='madmom ACF tempo on gtzan — offline oracle',
          slow=True, requires=('gtzan_corpus', 'madmom'))
def eval_tempo_gtzan_madmom_acf() -> dict[str, float]:
    return _run_tempo_estimator_eval('madmom_acf', corpus_kind='gtzan', tracks=9999)


@register('tempo.gtzan.madmom_dbn',
          description='madmom DBN tempo on gtzan — offline oracle',
          slow=True, requires=('gtzan_corpus', 'madmom'))
def eval_tempo_gtzan_madmom_dbn() -> dict[str, float]:
    return _run_tempo_estimator_eval('madmom_dbn', corpus_kind='gtzan', tracks=9999)


# osu — uses tool defaults (all available tracks, ~38).

@register('tempo.osu.acf',
          description='ACF tempo estimator on osu — production streaming tracker',
          slow=True, requires=('osu_corpus',))
def eval_tempo_osu_acf() -> dict[str, float]:
    return _run_tempo_estimator_eval('acf', corpus_kind='osu')


@register('tempo.osu.btrack',
          description='BTrack tempo estimator on osu — production streaming tracker',
          slow=True, requires=('osu_corpus', 'btrack'))
def eval_tempo_osu_btrack() -> dict[str, float]:
    return _run_tempo_estimator_eval('btrack', corpus_kind='osu')


@register('tempo.osu.beatnet_pf',
          description='BeatNet particle-filter tempo on osu — deployed when detector=beatnet_lite',
          slow=True, requires=('osu_corpus', 'beatnet'))
def eval_tempo_osu_beatnet_pf() -> dict[str, float]:
    return _run_tempo_estimator_eval('beatnet_pf', corpus_kind='osu')


@register('tempo.osu.madmom_comb',
          description='madmom comb-filter tempo on osu — offline oracle',
          slow=True, requires=('osu_corpus', 'madmom'))
def eval_tempo_osu_madmom_comb() -> dict[str, float]:
    return _run_tempo_estimator_eval('madmom_comb', corpus_kind='osu')


@register('tempo.osu.madmom_acf',
          description='madmom ACF tempo on osu — offline oracle',
          slow=True, requires=('osu_corpus', 'madmom'))
def eval_tempo_osu_madmom_acf() -> dict[str, float]:
    return _run_tempo_estimator_eval('madmom_acf', corpus_kind='osu')


@register('tempo.osu.madmom_dbn',
          description='madmom DBN tempo on osu — offline oracle',
          slow=True, requires=('osu_corpus', 'madmom'))
def eval_tempo_osu_madmom_dbn() -> dict[str, float]:
    return _run_tempo_estimator_eval('madmom_dbn', corpus_kind='osu')


# Cross-corpus consistency meta-evals — the load-bearing comparison.
# Per `PLAN_TEMPO_METER.md` priority order: octave accuracy is what
# matters most for the wallpaper (catastrophic if wrong), then exact,
# then jitter. Surface both axes in the consistency view.

_TEMPO_CONSISTENCY_KEY_EXACT = 'accuracy.exact_pct'
_TEMPO_CONSISTENCY_KEY_OCTAVE = 'accuracy.oe1_pct'


def _tempo_consistency_for_estimator(latest: dict[str, float],
                                       estimator: str,
                                       metric_key: str) -> dict[str, float]:
    values: list[float] = []
    for corpus in _CONSISTENCY_CORPORA:
        key = f'tempo.{corpus}.{estimator}.{metric_key}'
        v = latest.get(key)
        if v is not None:
            values.append(float(v))
    if not values:
        return {'__context__': {
            'skipped': f'no per-corpus {metric_key} for {estimator}'}}
    return {
        'mean': sum(values) / len(values),
        'min': min(values),
        'max': max(values),
        'spread': max(values) - min(values),
        'n_corpora': float(len(values)),
    }


for _est in ('acf', 'btrack', 'beatnet_pf',
             'madmom_comb', 'madmom_acf', 'madmom_dbn'):
    # Bind via default-arg trick so the lambda captures the current
    # estimator name, not the last one in the loop.
    def _exact_consistency(latest, _est=_est):
        return _tempo_consistency_for_estimator(
            latest, _est, _TEMPO_CONSISTENCY_KEY_EXACT)
    _exact_consistency.__name__ = f'meta_tempo_exact_consistency_{_est}'
    meta_eval(f'tempo.consistency.exact.{_est}',
              description=f'Cross-corpus exact-tempo accuracy consistency for {_est}')(
        _exact_consistency)

    def _octave_consistency(latest, _est=_est):
        return _tempo_consistency_for_estimator(
            latest, _est, _TEMPO_CONSISTENCY_KEY_OCTAVE)
    _octave_consistency.__name__ = f'meta_tempo_octave_consistency_{_est}'
    meta_eval(f'tempo.consistency.octave.{_est}',
              description=f'Cross-corpus octave-tolerant tempo accuracy consistency for {_est}')(
        _octave_consistency)

