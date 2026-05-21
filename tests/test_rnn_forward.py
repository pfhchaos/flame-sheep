"""Tests for the CPU RNN forward pass + BeatRNNDetector wrapper."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flame_sheep.eval.rnn_forward import (
    sigmoid, linear_forward, gru_forward_step, unpack_weights,
    forward_sequence, peak_pick,
)


def test_sigmoid_matches_definition():
    x = np.array([-1e3, -1.0, 0.0, 1.0, 1e3], dtype=np.float32)
    expected = 1.0 / (1.0 + np.exp(-x.astype(np.float64)))
    np.testing.assert_allclose(sigmoid(x), expected, atol=1e-6)


def test_linear_forward_with_and_without_relu():
    W = np.array([[1.0, -1.0]], dtype=np.float32)
    b = np.array([0.0, 0.5], dtype=np.float32)
    x = np.array([[2.0]], dtype=np.float32)
    y = linear_forward(x, W, b, relu=False)
    np.testing.assert_allclose(y, [[2.0, -1.5]])
    y_relu = linear_forward(x, W, b, relu=True)
    np.testing.assert_allclose(y_relu, [[2.0, 0.0]])  # -1.5 → 0


def test_gru_step_zero_hidden_zero_input():
    # GRU step with zero input + zero hidden + zero bias must produce
    # zero hidden out (the tanh(0)=0, z=sigmoid(0)=0.5, h_hat=0 case).
    H, I = 4, 3
    W = np.zeros((3, I, H), dtype=np.float32)
    U = np.zeros((3, H, H), dtype=np.float32)
    b = np.zeros((6, H), dtype=np.float32)
    x = np.zeros(I, dtype=np.float32)
    h = np.zeros(H, dtype=np.float32)
    h_new = gru_forward_step(x, h, W, U, b, H, I)
    np.testing.assert_allclose(h_new, np.zeros(H))


def test_unpack_weights_correct_total():
    """Total params for known geometry must match the flat array length."""
    I, P, H, C = 216, 32, 48, 1
    # linear_in: I*P + P, GRU: 3*P*H + 3*H*H + 6*H, linear_out: H*C + C
    total = I * P + P + 3 * P * H + 3 * H * H + 6 * H + H * C + C
    flat = np.arange(total, dtype=np.float32)
    parts = unpack_weights(flat, I, P, H, C)
    assert parts[0].shape == (I, P)
    assert parts[2].shape == (3, P, H)
    assert parts[5].shape == (H, C)


def test_unpack_weights_rejects_wrong_length():
    # Wrong size — fails either with our explicit message or with the
    # earlier numpy reshape error; both are "incompatible weights".
    with pytest.raises(ValueError):
        unpack_weights(np.zeros(100, dtype=np.float32),
                       input_size=216, proj_size=32, hidden_size=48,
                       n_classes=1)


def test_forward_sequence_zero_input_zero_weights():
    """With all-zero weights+biases, the network produces zero logits
    regardless of input. Useful as a soft-correctness sanity check."""
    I, P, H, C = 4, 3, 5, 1
    n = (I * P + P + 3 * P * H + 3 * H * H + 6 * H + H * C + C)
    flat = np.zeros(n, dtype=np.float32)
    weights = unpack_weights(flat, I, P, H, C)
    features = np.random.randn(10, I).astype(np.float32)
    logits = forward_sequence(features, weights, hidden_size=H, proj_size=P)
    assert logits.shape == (10, C)
    np.testing.assert_allclose(logits, 0.0)


def test_peak_pick_simple():
    """Three clear peaks at known indices, none in refractory range."""
    scores = np.array([0, 0, 1.0, 0, 0, 0, 0.8, 0, 0, 0, 0, 0.6, 0],
                       dtype=np.float32)
    peaks = peak_pick(scores, threshold=0.3, min_distance=2)
    np.testing.assert_array_equal(peaks, [2, 6, 11])


def test_peak_pick_refractory_suppresses():
    """Two adjacent local maxima within min_distance: keep only the first."""
    scores = np.array([0, 0, 1.0, 0.9, 0, 0, 0], dtype=np.float32)
    peaks = peak_pick(scores, threshold=0.3, min_distance=5)
    np.testing.assert_array_equal(peaks, [2])


def test_peak_pick_respects_threshold():
    scores = np.array([0, 0, 0.2, 0, 0, 0.5, 0], dtype=np.float32)
    peaks = peak_pick(scores, threshold=0.3, min_distance=1)
    np.testing.assert_array_equal(peaks, [5])  # 0.2 below threshold


# -- BeatRNNDetector smoke (only runs if a checkpoint is on disk) ----------

CHECKPOINT_CANDIDATES = [
    Path.home() / 'datasets' / 'beat-labels-mmap' / 'beat_rnn_continuous.npz',
    Path.home() / 'datasets' / 'beat-labels' / 'beat_rnn_continuous.npz',
]


def _checkpoint_or_skip() -> Path:
    for p in CHECKPOINT_CANDIDATES:
        if p.exists():
            return p
    pytest.skip('No beat_rnn checkpoint available')


def test_beat_rnn_detector_loads_and_runs():
    """End-to-end: instantiate the detector on a real checkpoint, feed
    synthetic audio, get a result of the right shape. Doesn't assert
    F1 quality — just that the wiring runs."""
    from flame_sheep.eval.detectors import BeatRNNDetector
    ckpt = _checkpoint_or_skip()
    d = BeatRNNDetector(ckpt)
    assert d.name == 'beat_rnn'
    assert d.version.startswith('beat_rnn_')

    # 5 seconds of pink-ish noise → some peaks should fire (just sanity).
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(48000 * 5) * 0.1).astype(np.float32)
    beats = d.detect(audio, sr=48000)
    assert beats.ndim == 1
    assert beats.dtype.kind == 'f'
    assert np.all(np.diff(beats) >= 0)
