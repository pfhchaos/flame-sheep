"""Tests for the cross-corpus consistency meta-eval.

This file lives under tests/unified_evals/ (not tests/evals/) because a
top-level `evals` package exists at repo root — naming the test dir
`evals` would shadow it on sys.path. See feedback_silent_skip_anti_pattern.
"""
from __future__ import annotations

import pytest

# Triggers builtin registration of all evals.
from evals import builtin  # noqa: F401
from evals import REGISTRY, meta_eval, register
from evals.builtin import _consistency_for_detector
from evals.runner import run


# --- Pure-logic tests for the consistency reducer ---

def test_consistency_two_corpora():
    latest = {
        'beat.osu.percentile.pooled.f1@70ms': 0.488,
        'beat.gtzan.percentile.pooled.f1@70ms': 0.420,
        # Irrelevant noise to make sure we don't pick it up.
        'beat.osu.percentile.mean.f1@50ms': 0.999,
        'tempo.osu.baseline.exact_pct': 21.0,
    }
    out = _consistency_for_detector(latest, 'beat', 'percentile')
    assert out['n_corpora'] == 2.0
    assert out['mean'] == pytest.approx((0.488 + 0.420) / 2)
    assert out['min'] == pytest.approx(0.420)
    assert out['spread'] == pytest.approx(0.488 - 0.420)


def test_consistency_one_corpus_falls_through():
    # Only osu has a value — meta-eval should still produce something
    # so downstream consumers can fall back, but spread is 0.
    latest = {'beat.osu.percentile.pooled.f1@70ms': 0.488}
    out = _consistency_for_detector(latest, 'beat', 'percentile')
    assert out['n_corpora'] == 1.0
    assert out['mean'] == pytest.approx(0.488)
    assert out['min'] == pytest.approx(0.488)
    assert out['spread'] == pytest.approx(0.0)


def test_consistency_no_corpora_reports_skipped():
    out = _consistency_for_detector({}, 'beat', 'percentile')
    assert '__context__' in out
    assert 'skipped' in out['__context__']


def test_consistency_filters_by_kind():
    # Downbeat metric should NOT contribute to a beat consistency check.
    latest = {
        'downbeat.osu.percentile.pooled.f1@70ms': 0.30,
        'beat.gtzan.percentile.pooled.f1@70ms': 0.40,
    }
    out = _consistency_for_detector(latest, 'beat', 'percentile')
    # Only gtzan beat contributes — osu downbeat is for a different kind.
    assert out['n_corpora'] == 1.0
    assert out['mean'] == pytest.approx(0.40)


# --- Integration: meta-eval sees in-progress run metrics ---

def test_meta_eval_sees_current_run_metrics(tmp_path, monkeypatch):
    """A meta-eval scheduled in the same run as its inputs should see
    today's freshly computed values, not just historical ones."""
    saved = dict(REGISTRY)
    REGISTRY.clear()
    # Pretend results.jsonl is empty so we only see today's metrics.
    monkeypatch.setattr('evals.runner.latest_metric_values', lambda: {})

    @register('beat.fakecorp.fake',
              description='synthetic regular eval')
    def _regular():
        return {'pooled.f1@70ms': 0.7}

    @meta_eval('beat.consistency.fake',
                description='meta sees regular')
    def _meta(latest):
        v = latest.get('beat.fakecorp.fake.pooled.f1@70ms')
        return {'echoed': v if v is not None else -1.0}

    try:
        entry = run(notes='meta test', dry_run=True)
        assert entry.metrics['beat.consistency.fake.echoed'] == 0.7
    finally:
        REGISTRY.clear()
        REGISTRY.update(saved)


# --- Smoke test: the actual registered consistency evals don't crash. ---

def test_registered_consistency_evals_are_callable():
    fake_latest = {
        'beat.osu.percentile.pooled.f1@70ms': 0.5,
        'beat.gtzan.percentile.pooled.f1@70ms': 0.4,
        'beat.osu.beatnet.pooled.f1@70ms': 0.6,
        'beat.osu.madmom.pooled.f1@70ms': 0.85,
    }
    for name in ('beat.consistency.percentile',
                 'beat.consistency.beatnet',
                 'beat.consistency.madmom'):
        spec = REGISTRY[name]
        assert spec.is_meta
        result = spec.fn(fake_latest)
        assert isinstance(result, dict)


# --- Latency-consistency meta-eval (§7) ---

def test_latency_consistency_two_corpora():
    from evals.builtin import _latency_consistency_for_detector
    latest = {
        'beat.osu.percentile.latency.realtime_factor.mean': 0.40,
        'beat.gtzan.percentile.latency.realtime_factor.mean': 0.07,
        # Noise — should not be picked up.
        'beat.osu.percentile.pooled.f1@70ms': 0.488,
    }
    out = _latency_consistency_for_detector(latest, 'beat', 'percentile')
    assert out['n_corpora'] == 2.0
    assert out['min'] == pytest.approx(0.07)
    assert out['max'] == pytest.approx(0.40)
    assert out['spread'] == pytest.approx(0.33)
    assert out['mean'] == pytest.approx((0.40 + 0.07) / 2)


def test_latency_consistency_skipped_when_no_corpora():
    from evals.builtin import _latency_consistency_for_detector
    out = _latency_consistency_for_detector({}, 'beat', 'madmom')
    assert '__context__' in out and 'skipped' in out['__context__']


def test_registered_latency_meta_evals():
    for name in ('beat.latency_consistency.percentile',
                 'beat.latency_consistency.beatnet',
                 'beat.latency_consistency.madmom'):
        spec = REGISTRY[name]
        assert spec.is_meta


# --- Parser extension: latency table rows --

def test_parse_latency_table():
    """Assert `_parse_beat_report` extracts the §7 latency rows."""
    from evals.builtin import _parse_beat_report
    sample = '''
=== Beat Aggregate (mean over tracks) ===
  F1@25ms: 0.300 ± 0.020  (precision=0.250, recall=0.400)
  F1@50ms: 0.400 ± 0.020  (precision=0.300, recall=0.500)
  F1@70ms: 0.500 ± 0.020  (precision=0.350, recall=0.600)
  cemgil:   0.420 ± 0.030

=== Beat Aggregate (length-weighted, pooled over corpus) ===
  F1@25ms: 0.310
  F1@50ms: 0.410
  F1@70ms: 0.510
  cemgil:   0.430

=== Beat Latency (per-track, total) ===
  total_sec:         mean=2.171  p50=2.162  p95=2.199  p99=2.202
  realtime_factor:   mean=0.072  p95=0.073

=== Beat Latency (per-track, components) ===
  cqt_sec: mean=1.0239  p95=1.0338  n=3
  csd_transform_sec: mean=0.1061  p95=0.1082  n=3
  peak_pick_sec: mean=1.0379  p95=1.0537  n=3
'''
    m = _parse_beat_report(sample)
    assert m['latency.total.mean_sec'] == 2.171
    assert m['latency.total.p99_sec'] == 2.202
    assert m['latency.realtime_factor.mean'] == 0.072
    assert m['latency.realtime_factor.p95'] == 0.073
    assert m['latency.cqt.mean_sec'] == 1.0239
    assert m['latency.peak_pick.p95_sec'] == 1.0537
    # Existing F1 keys still present.
    assert m['pooled.f1@70ms'] == 0.510
    assert m['mean.precision@25ms'] == 0.250
