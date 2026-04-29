"""Evaluation metrics for beat detection, tempo, and HPSS quality."""

from __future__ import annotations

import numpy as np


def onset_precision_recall(est: list[float], gt: list[float],
                           window: float = 0.1) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 for onset detection.

    Args:
        est: estimated onset times (seconds)
        gt: ground truth onset times (seconds)
        window: matching tolerance in seconds (default 100ms)

    Returns:
        (precision, recall, f1)
    """
    if not est or not gt:
        if not est and not gt:
            return 1.0, 1.0, 1.0
        if not est:
            return 1.0, 0.0, 0.0
        return 0.0, 1.0, 0.0

    est_arr = np.array(sorted(est))
    gt_arr = np.array(sorted(gt))

    # For each estimated onset, find nearest ground truth
    gt_matched = set()
    tp = 0
    for e in est_arr:
        dists = np.abs(gt_arr - e)
        best_idx = int(np.argmin(dists))
        if dists[best_idx] <= window and best_idx not in gt_matched:
            tp += 1
            gt_matched.add(best_idx)

    precision = tp / len(est_arr) if est_arr.size > 0 else 0.0
    recall = tp / len(gt_arr) if gt_arr.size > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def tempo_accuracy(est_bpm: float, ref_bpm: float,
                   tolerance: float = 0.04) -> dict[str, bool | float]:
    """Evaluate tempo estimation accuracy.

    Args:
        est_bpm: estimated BPM
        ref_bpm: reference/ground truth BPM
        tolerance: relative tolerance (default ±4%)

    Returns:
        dict with keys: exact, octave, error, octave_ratio
    """
    if ref_bpm <= 0:
        return {'exact': False, 'octave': False, 'error': float('inf'), 'octave_ratio': 0.0}

    error = abs(est_bpm - ref_bpm) / ref_bpm
    exact = error <= tolerance

    # Check octave matches (2x, 0.5x, 3x, 1/3x)
    octave = exact
    best_ratio = 1.0
    for ratio in [0.5, 2.0, 1.0/3, 3.0]:
        octave_error = abs(est_bpm - ref_bpm * ratio) / (ref_bpm * ratio)
        if octave_error <= tolerance:
            octave = True
            best_ratio = ratio
            break

    return {
        'exact': exact,
        'octave': octave,
        'error': error,
        'octave_ratio': best_ratio if not exact else 1.0,
    }


def hpss_similarity(our_harmonic: np.ndarray, our_percussive: np.ndarray,
                    ref_harmonic: np.ndarray, ref_percussive: np.ndarray) -> dict[str, float]:
    """Compare our H/P separation against a reference (e.g., librosa HPSS).

    All inputs should be spectrograms of shape (n_bins, n_frames) or
    masks of the same shape.

    Returns:
        dict with harmonic_sim, percussive_sim, mean_sim (cosine similarities)
    """
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        a_flat = a.flatten().astype(np.float64)
        b_flat = b.flatten().astype(np.float64)
        dot = np.dot(a_flat, b_flat)
        na = np.linalg.norm(a_flat)
        nb = np.linalg.norm(b_flat)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(dot / (na * nb))

    h_sim = _cosine_sim(our_harmonic, ref_harmonic)
    p_sim = _cosine_sim(our_percussive, ref_percussive)

    return {
        'harmonic_sim': h_sim,
        'percussive_sim': p_sim,
        'mean_sim': (h_sim + p_sim) / 2,
    }
