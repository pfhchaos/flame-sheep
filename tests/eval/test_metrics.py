"""Tests for the beat-eval metrics module."""
from __future__ import annotations

import numpy as np
import pytest

from flame_sheep.eval.metrics import f_measure, cemgil, evaluate


def test_perfect_match_scores_one():
    beats = np.arange(0.0, 5.0, 0.5)
    r = f_measure(beats, beats, tolerance=0.07)
    assert r['f1'] == pytest.approx(1.0)
    assert r['precision'] == pytest.approx(1.0)
    assert r['recall'] == pytest.approx(1.0)
    assert r['tp'] == len(beats)
    assert r['fp'] == 0
    assert r['fn'] == 0


def test_all_predictions_miss():
    pred = np.array([0.0, 1.0, 2.0])
    true = np.array([0.5, 1.5, 2.5])
    r = f_measure(pred, true, tolerance=0.07)
    assert r['f1'] == 0.0
    assert r['tp'] == 0
    assert r['fp'] == 3
    assert r['fn'] == 3


def test_predictions_within_tolerance():
    """Predictions within ±70ms should match; outside should miss."""
    true = np.array([0.0, 1.0, 2.0])
    pred = np.array([0.05, 1.06, 2.08])  # 50ms, 60ms, 80ms offsets
    r = f_measure(pred, true, tolerance=0.07)
    assert r['tp'] == 2  # first two match (50, 60ms ≤ 70); third (80ms) doesn't
    assert r['fp'] == 1
    assert r['fn'] == 1


def test_greedy_one_to_one():
    """One prediction can't match two true beats. Two predictions can't
    both match the same true beat."""
    true = np.array([0.0, 1.0])
    pred = np.array([0.01, 0.02])  # both within tolerance of true[0]
    r = f_measure(pred, true, tolerance=0.07)
    # First prediction matches true[0]; second has no remaining true beat
    # within tolerance.
    assert r['tp'] == 1
    assert r['fp'] == 1
    assert r['fn'] == 1


def test_empty_inputs():
    r = f_measure(np.array([]), np.array([]), tolerance=0.07)
    assert r['f1'] == 1.0  # vacuously perfect

    r = f_measure(np.array([1.0]), np.array([]), tolerance=0.07)
    assert r['f1'] == 0.0
    assert r['fp'] == 1

    r = f_measure(np.array([]), np.array([1.0]), tolerance=0.07)
    assert r['f1'] == 0.0
    assert r['fn'] == 1


def test_cemgil_perfect_is_one():
    beats = np.arange(0.0, 5.0, 0.5)
    assert cemgil(beats, beats, sigma=0.04) == pytest.approx(1.0)


def test_cemgil_smooth_in_drift():
    """Small drifts cost less than gross misses, and cost grows
    monotonically with the drift size."""
    true = np.arange(0.0, 5.0, 0.5)
    perfect = cemgil(true, true, sigma=0.04)
    drift_10ms = cemgil(true + 0.01, true, sigma=0.04)
    drift_30ms = cemgil(true + 0.03, true, sigma=0.04)
    drift_100ms = cemgil(true + 0.10, true, sigma=0.04)
    assert perfect > drift_10ms > drift_30ms > drift_100ms
    assert drift_100ms < 0.1


def test_cemgil_penalizes_overprediction():
    """Predicting beats at every true location AND extra spurious ones
    should score lower than just the true locations."""
    true = np.arange(0.0, 5.0, 0.5)
    pred_clean = true.copy()
    pred_noisy = np.concatenate([true, np.arange(0.25, 5.0, 0.5)])
    assert cemgil(pred_clean, true) > cemgil(pred_noisy, true)


def test_evaluate_returns_flat_dict():
    true = np.arange(0.0, 5.0, 0.5)
    pred = true + 0.02  # 20 ms drift — matches @25ms, @50ms, @70ms
    out = evaluate(pred, true, tolerances_ms=(25, 50, 70))
    assert out['n_pred'] == len(pred)
    assert out['n_true'] == len(true)
    assert out['f1_25ms'] == pytest.approx(1.0)
    assert out['f1_50ms'] == pytest.approx(1.0)
    assert out['f1_70ms'] == pytest.approx(1.0)
    assert 0.0 <= out['cemgil'] <= 1.0


def test_evaluate_tolerance_sensitivity():
    """At a 40ms drift, F1@25ms should fail but F1@50ms and F1@70ms succeed."""
    true = np.arange(0.0, 5.0, 0.5)
    pred = true + 0.04
    out = evaluate(pred, true, tolerances_ms=(25, 50, 70))
    assert out['f1_25ms'] == 0.0  # 40 > 25
    assert out['f1_50ms'] == pytest.approx(1.0)
    assert out['f1_70ms'] == pytest.approx(1.0)
