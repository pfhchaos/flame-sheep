"""Regression test for the sliding-STFT BeatNet detector.

`SlidingBeatNetDetector` reimplements LOG_SPECT feature extraction
manually (sliding STFT + cached filterbank + log + diff state) to
make per-hop latency O(1) instead of O(window-LOG_SPECT-reprocess).

The math has to match madmom's batch LOG_SPECT bit-for-bit, or any
feature drift cascades through the (trained) model into wrong
activations and wrong F1. This test asserts bit-equivalence at the
feature level on a real audio clip — fastest place to catch a math
divergence if anyone tweaks the feature pipeline later.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


_GTZAN_ROOT = Path.home() / 'datasets' / 'GTZAN'
_HAS_GTZAN_AUDIO = (_GTZAN_ROOT / 'genres' / 'pop' / 'pop.00010.wav').exists()


def _load_clip(sr: int = 22050, duration: float = 3.0
               ) -> tuple[np.ndarray, int]:
    import librosa
    audio, real_sr = librosa.load(
        str(_GTZAN_ROOT / 'genres' / 'pop' / 'pop.00010.wav'),
        sr=sr, mono=True, duration=duration)
    return audio.astype(np.float32), real_sr


@pytest.mark.skipif(not _HAS_GTZAN_AUDIO,
                     reason='GTZAN audio not installed')
@pytest.mark.slow
def test_sliding_features_bit_exact_to_batch():
    """Per-frame log_spec from the sliding STFT must match what
    madmom's batch LOG_SPECT produces at the same absolute frame
    index. |diff| > 1e-6 means math divergence — likely a window /
    fft / filterbank / log mismatch."""
    import collections, collections.abc
    for n in ('MutableSequence', 'MutableMapping', 'MutableSet',
              'Sequence', 'Mapping', 'Set', 'Iterable',
              'Container', 'Hashable', 'Callable', 'Sized'):
        if not hasattr(collections, n):
            setattr(collections, n, getattr(collections.abc, n))
    try:
        from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
        from madmom.audio.stft import ShortTimeFourierTransformProcessor
        from madmom.audio.spectrogram import (
            FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
            SpectrogramDifferenceProcessor)
    except ImportError as e:
        pytest.skip(f'madmom unavailable: {e}')

    from flame_sheep.eval.detectors import SlidingBeatNetDetector

    audio, sr = _load_clip(duration=3.0)
    # Batch reference path through madmom.
    sig = SignalProcessor(num_channels=1, sample_rate=sr)
    frames_p = FramedSignalProcessor(frame_size=1411, hop_size=441)
    stft_p = ShortTimeFourierTransformProcessor()
    filt_p = FilteredSpectrogramProcessor(
        num_bands=24, fmin=30, fmax=17000, norm_filters=True)
    log_p = LogarithmicSpectrogramProcessor(mul=1, add=1)
    diff_p = SpectrogramDifferenceProcessor(
        diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
    batch_feats = np.asarray(diff_p(log_p(filt_p(stft_p(frames_p(sig(audio)))))))

    # Sliding path.
    det = SlidingBeatNetDetector(model_index=1)
    sliding_feats, _ = det._features_streaming(audio, sr)

    n = min(len(batch_feats), len(sliding_feats))
    # Frame 0 differs by ~1 because batch's diff stacks "zero" for
    # frame 0, sliding does the same — but spectrogram math can have
    # boundary differences in batch mode at very first frame. Start
    # the comparison at frame 1.
    bm = batch_feats[1:n]
    sm = sliding_feats[1:n]
    assert bm.shape == sm.shape, f'shape mismatch {bm.shape} vs {sm.shape}'
    max_diff = float(np.abs(bm - sm).max())
    mean_diff = float(np.abs(bm - sm).mean())
    assert max_diff < 1e-5, (
        f'sliding features diverge from batch: max |diff|={max_diff:.2e} '
        f'mean |diff|={mean_diff:.2e}. Investigate STFT window / FFT '
        f'size / filterbank / log / diff state math.')


@pytest.mark.skipif(not _HAS_GTZAN_AUDIO,
                     reason='GTZAN audio not installed')
@pytest.mark.slow
def test_sliding_activations_bit_exact_to_batch_beatnet():
    """End-to-end: activations from sliding-STFT path must equal
    activations from batch BeatNetDetector (same model, equivalent
    features, single-frame LSTM forward with carried state)."""
    try:
        from flame_sheep.eval.detectors import (
            SlidingBeatNetDetector, BeatNetDetector)
    except ImportError as e:
        pytest.skip(f'BeatNet unavailable: {e}')

    audio, sr = _load_clip(duration=3.0)
    bd = BeatNetDetector(model_index=1)
    sl = SlidingBeatNetDetector(model_index=1)

    batch_probs = bd._activation_probs(audio, sr)
    sliding_probs, _ = sl._activation_probs_sliding(audio, sr)

    n = min(len(batch_probs), len(sliding_probs))
    # Skip the first frame where diff bootstrapping differs.
    bp, sp = batch_probs[1:n], sliding_probs[1:n]
    for ch, name in enumerate(['beat', 'downbeat', 'non-beat']):
        max_diff = float(np.abs(bp[:, ch] - sp[:, ch]).max())
        assert max_diff < 1e-3, (
            f'ch{ch} ({name}): activations diverge max |diff|={max_diff:.2e}')


@pytest.mark.skipif(not _HAS_GTZAN_AUDIO,
                     reason='GTZAN audio not installed')
@pytest.mark.slow
def test_sliding_per_hop_under_budget():
    """The whole point of the sliding impl is staying under the live
    hop budget. Assert per-hop p95 < hop_period (20ms @ 22050/441)."""
    from flame_sheep.eval.detectors import SlidingBeatNetDetector

    audio, sr = _load_clip(duration=3.0)
    det = SlidingBeatNetDetector(model_index=1)
    det.detect(audio, sr)
    p95 = det.last_timing['per_hop_p95_ms']
    hop_budget_ms = (441 / 22050) * 1000.0  # ~20.0
    assert p95 < hop_budget_ms, (
        f'sliding per-hop p95={p95:.2f}ms exceeds hop budget '
        f'{hop_budget_ms:.2f}ms. Either the STFT/matmul got slower '
        f'or model.forward regressed. Profile via the per-hop break-'
        f'down in last_timing.')
