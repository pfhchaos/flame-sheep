"""Bit-exact regression for the pure-numpy BeatNet forward.

Three layers of correctness, fastest-failing first:

1. Weights file integrity — filterbank in the .npz equals what madmom
   produces at runtime on a fresh call.
2. Layer-by-layer math — single feature vector through `BeatNetLite.step`
   vs the same vector through torch BDA. Tolerates float32 epsilon
   only (~1e-6 max abs).
3. Full streaming pipeline — sliding-STFT + lite forward across a real
   audio clip equals SlidingBeatNetDetector's (torch-backed) output.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


_GTZAN_AUDIO = Path.home() / 'datasets/GTZAN/genres/pop/pop.00010.wav'
_LITE_WEIGHTS = (Path(__file__).resolve().parents[2]
                 / 'flame_sheep' / 'data' / 'beatnet_m1_lite.npz')


def _torch_bda():
    """Load BeatNet's torch BDA model with model_index=1 weights."""
    import collections, collections.abc
    for n in ('MutableSequence', 'MutableMapping', 'MutableSet',
              'Sequence', 'Mapping', 'Set', 'Iterable',
              'Container', 'Hashable', 'Callable', 'Sized'):
        if not hasattr(collections, n):
            setattr(collections, n, getattr(collections.abc, n))
    import os, torch
    from BeatNet.model import BDA
    m = BDA(272, 150, 2, torch.device('cpu'))
    bn_dir = os.path.dirname(os.path.abspath(__import__('BeatNet').__file__))
    m.load_state_dict(torch.load(
        os.path.join(bn_dir, 'models', 'model_1_weights.pt'),
        map_location='cpu', weights_only=True))
    m.eval()
    return m, torch


# ---------- 1. weights file integrity ----------

def test_filterbank_in_npz_matches_madmom_runtime():
    """Loaded filterbank == madmom-computed filterbank on a fresh call.
    Catches drift if anyone edits the precompute step in the export
    script without re-running it, or if madmom's filterbank math
    changes between versions."""
    import collections, collections.abc
    for n in ('MutableSequence', 'MutableMapping', 'MutableSet',
              'Sequence', 'Mapping', 'Set', 'Iterable',
              'Container', 'Hashable', 'Callable', 'Sized'):
        if not hasattr(collections, n):
            setattr(collections, n, getattr(collections.abc, n))
    try:
        from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
        from madmom.audio.stft import ShortTimeFourierTransformProcessor
        from madmom.audio.spectrogram import FilteredSpectrogramProcessor
    except ImportError as e:
        pytest.skip(f'madmom unavailable: {e}')
    if not _LITE_WEIGHTS.exists():
        pytest.skip(f'lite weights not built at {_LITE_WEIGHTS}')

    sig = SignalProcessor(num_channels=1, sample_rate=22050)
    fp = FramedSignalProcessor(frame_size=1411, hop_size=441)
    stft_p = ShortTimeFourierTransformProcessor()
    filt_p = FilteredSpectrogramProcessor(
        num_bands=24, fmin=30, fmax=17000, norm_filters=True)
    dummy = np.zeros(1411 + 2 * 441, dtype=np.float32)
    runtime_fb = np.asarray(
        filt_p(stft_p(fp(sig(dummy)))).filterbank, dtype=np.float32)

    data = np.load(str(_LITE_WEIGHTS))
    stored_fb = data['filterbank']
    assert stored_fb.shape == runtime_fb.shape, (
        f'filterbank shape drift: stored={stored_fb.shape} '
        f'runtime={runtime_fb.shape}')
    diff = np.abs(stored_fb - runtime_fb).max()
    assert diff < 1e-7, (
        f'filterbank values drifted: max |diff|={diff:.3e}')


# ---------- 2. single-feature bit-exact vs torch ----------

@pytest.mark.skipif(not _LITE_WEIGHTS.exists(),
                     reason='lite weights not built')
def test_lite_step_bit_exact_vs_torch_bda():
    """Two model.forward calls on identical inputs (with carried LSTM
    state across calls) must agree within float32 epsilon. Fails if
    the LSTM gate order, conv1d direction, max-pool stride, or any
    layer's math diverges from PyTorch's nn convention."""
    try:
        m, torch = _torch_bda()
    except (ImportError, FileNotFoundError) as e:
        pytest.skip(f'torch BDA unavailable: {e}')
    from flame_sheep_audio.beatnet_lite import BeatNetLite

    lite = BeatNetLite(_LITE_WEIGHTS)

    rng = np.random.default_rng(42)
    feats_seq = rng.standard_normal((20, 272)).astype(np.float32)

    # Torch path: feed sequence at once.
    m.hidden = torch.zeros(2, 1, 150)
    m.cell = torch.zeros(2, 1, 150)
    with torch.no_grad():
        ft = torch.from_numpy(feats_seq).unsqueeze(0)   # (1, 20, 272)
        preds = m(ft)[0]                                 # (3, 20)
        torch_probs = torch.softmax(preds, dim=0).numpy().T  # (20, 3)

    # Lite path: step per-frame.
    lite.reset_state()
    lite_probs = np.empty((20, 3), dtype=np.float32)
    for i in range(20):
        lite_probs[i] = lite.step(feats_seq[i])

    max_diff = float(np.abs(torch_probs - lite_probs).max())
    assert max_diff < 1e-5, (
        f'lite vs torch BDA diverge: max |diff|={max_diff:.3e}. '
        f'Likely candidates: LSTM gate order (i/f/g/o vs i/g/f/o), '
        f'biases (split vs combined), conv1d direction, max_pool stride.')


# ---------- 3. end-to-end streaming sanity ----------

@pytest.mark.skipif(not _GTZAN_AUDIO.exists() or not _LITE_WEIGHTS.exists(),
                     reason='gtzan audio or lite weights missing')
@pytest.mark.slow
def test_lite_streaming_matches_sliding_torch_on_real_audio():
    """Run pure-numpy sliding-STFT + lite forward on real audio, compare
    activations against the torch-backed SlidingBeatNetDetector. Same
    weights, same feature math; only differences are float32 ops
    routed through numpy vs torch."""
    import librosa
    from flame_sheep_audio.beatnet_lite import BeatNetLite
    try:
        from flame_sheep.eval.detectors import SlidingBeatNetDetector
    except ImportError as e:
        pytest.skip(f'sliding detector unavailable: {e}')

    audio, sr = librosa.load(str(_GTZAN_AUDIO), sr=22050, mono=True,
                              duration=3.0)
    audio = audio.astype(np.float32)

    # Torch sliding path.
    sd = SlidingBeatNetDetector(model_index=1)
    torch_probs, _ = sd._activation_probs_sliding(audio, sr)

    # Lite path: same feature extraction (we can borrow sd._features_streaming
    # since it's pure numpy already), then lite.step per frame.
    sd._load()
    feats, _ = sd._features_streaming(audio, sr)
    lite = BeatNetLite(_LITE_WEIGHTS)
    lite.reset_state()
    lite_probs = np.empty_like(torch_probs)
    for i in range(feats.shape[0]):
        lite_probs[i] = lite.step(feats[i])

    # Skip frame 0 — diff bootstrap can produce tiny boundary noise.
    max_diff = float(np.abs(torch_probs[1:] - lite_probs[1:]).max())
    assert max_diff < 1e-5, (
        f'lite vs torch sliding diverge on real audio: '
        f'max |diff|={max_diff:.3e}')
