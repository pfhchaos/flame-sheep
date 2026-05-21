"""Beat-detection evaluation metrics.

We can't use madmom (broken on Python 3.10+ — uses pre-3.10 collections
imports) and don't want a new pip dep just for F1. The math is short:
- f_measure: greedy nearest-unused matching with a hard tolerance window
- cemgil: Gaussian-weighted match; smooth in the tolerance dimension

Greedy vs Hungarian matching diverges only when two true beats are
within `tolerance` of each other AND a single prediction sits between
them; for music with beats spaced ≥150 ms (>400 BPM) and tolerance
≤70 ms, this is essentially never. Live with greedy.
"""
from __future__ import annotations

import numpy as np


def f_measure(pred: np.ndarray, true: np.ndarray,
              tolerance: float = 0.07) -> dict:
    """F-measure with ±`tolerance` seconds, greedy nearest-unused match.

    Returns: f1, precision, recall, tp, fp, fn (Python int/float).
    """
    pred = np.sort(np.asarray(pred, dtype=np.float64))
    true = np.sort(np.asarray(true, dtype=np.float64))

    n_pred = len(pred)
    n_true = len(true)
    if n_pred == 0 and n_true == 0:
        return {'f1': 1.0, 'precision': 1.0, 'recall': 1.0,
                'tp': 0, 'fp': 0, 'fn': 0}

    used_true = np.zeros(n_true, dtype=bool)
    tp = 0
    for p in pred:
        if n_true == 0:
            break
        diffs = np.abs(true - p)
        diffs[used_true] = np.inf
        idx = int(np.argmin(diffs))
        if diffs[idx] <= tolerance:
            used_true[idx] = True
            tp += 1

    fp = n_pred - tp
    fn = n_true - tp
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_true if n_true else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {'f1': f1, 'precision': precision, 'recall': recall,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}


def cemgil(pred: np.ndarray, true: np.ndarray,
           sigma: float = 0.04) -> float:
    """Gaussian-weighted match score. For each true beat, finds the
    nearest prediction and contributes `exp(-Δt² / (2σ²))`. Normalized
    by (n_pred + n_true) / 2 — a perfect match scores 1.0; over-
    prediction (lots of FPs) is penalized.

    sigma=40 ms is the MIREX convention. Smooth in the tolerance
    dimension, unlike hard-windowed F1 — useful when small drifts are
    worth less penalty than gross misses.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    n_pred, n_true = len(pred), len(true)
    if n_pred == 0 and n_true == 0:
        return 1.0
    if n_pred == 0 or n_true == 0:
        return 0.0
    score = 0.0
    for t in true:
        diff_min = np.min(np.abs(pred - t))
        score += np.exp(-(diff_min ** 2) / (2.0 * sigma ** 2))
    return float(score / ((n_pred + n_true) / 2.0))


def evaluate(pred: np.ndarray, true: np.ndarray,
             tolerances_ms: tuple[int, ...] = (25, 50, 70)) -> dict:
    """Run F1 at each requested tolerance + cemgil. Returns a flat dict.

    Keys: 'f1_25ms', 'precision_25ms', ... 'cemgil', 'n_pred', 'n_true'.
    """
    out: dict = {'n_pred': int(len(pred)), 'n_true': int(len(true))}
    for ms in tolerances_ms:
        sub = f_measure(pred, true, tolerance=ms / 1000.0)
        suffix = f'_{ms}ms'
        for k, v in sub.items():
            out[k + suffix] = v
    out['cemgil'] = cemgil(pred, true)
    return out
