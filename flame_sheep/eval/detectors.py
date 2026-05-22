"""Beat-detector implementations for the eval harness.

Contract: `BeatDetector.detect(audio, sr) -> np.ndarray of beat times
in seconds`. Implementations may need their own preprocessing
(resampling, mono conversion); the harness just hands them raw audio.

The `version` property is part of the contract — used as a cache key
so retrained checkpoints automatically invalidate cached eval results.
"""
from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

# Ensure the in-tree flame_sheep_audio package is importable even when
# the parent project's editable install hasn't picked it up.
_AUDIO_SRC = (Path(__file__).resolve().parents[2]
              / 'flame_sheep_audio' / 'src')
if _AUDIO_SRC.exists() and str(_AUDIO_SRC) not in sys.path:
    sys.path.insert(0, str(_AUDIO_SRC))


class BeatDetector(ABC):
    """Abstract base. Subclasses must set `name` and `version` and
    implement `detect`."""

    name: str = ''
    version: str = ''

    @abstractmethod
    def detect(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Return beat times in seconds. Mono float audio in [-1, 1]."""


def _ensure_mono_float32(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        # Mix down to mono. Assume (channels, samples) or (samples, channels).
        audio = audio.mean(axis=0 if audio.shape[0] < audio.shape[1] else 1)
    return audio.astype(np.float32, copy=False)


def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    # Lazy import — scipy is heavy.
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sr_in, sr_out)
    return resample_poly(audio, sr_out // g, sr_in // g).astype(np.float32)


class CurrentSystemDetector(BeatDetector):
    """Wraps the in-tree PercentileBeatDetector for offline eval.

    Matches the live wallpaper's audio pipeline at construction time
    (same SpectrumEngine fallback chain, same detector class + config).
    Streams the audio through one hop at a time and records every
    frame at which any non-`song_start` BeatEvent is emitted.
    """

    name = 'current_system'

    def __init__(self, percentile: float = 99.0):
        # Import inside __init__ so the package path tweak above is in
        # effect by the time we actually use the imports.
        from flame_sheep_audio.beat_detector import PercentileBeatDetector
        from flame_sheep_audio._constants import (
            SAMPLE_RATE, HOP_SIZE, FFT_SIZE,
        )
        self._SAMPLE_RATE = SAMPLE_RATE
        self._HOP_SIZE = HOP_SIZE
        self._FFT_SIZE = FFT_SIZE
        self._detector_cls = PercentileBeatDetector
        self._percentile = percentile

        # Version: bake the engine + detector identity into a stable hash so
        # any code change to either invalidates the eval cache.
        import hashlib
        from flame_sheep_audio import beat_detector as _bd
        from flame_sheep_audio import _spectrum as _spec
        src = (Path(_bd.__file__).read_text()
               + Path(_spec.__file__).read_text())
        self.version = (
            'current_system_' + hashlib.sha256(src.encode()).hexdigest()[:12]
        )

    def _make_engine(self):
        # Same single-engine choice as the live daemon: CQT.
        from flame_sheep_audio._cqt_engine import CqtEngine
        return CqtEngine()

    def detect(self, audio: np.ndarray, sr: int) -> np.ndarray:
        audio = _ensure_mono_float32(audio)
        audio = _resample(audio, sr, self._SAMPLE_RATE)

        engine = self._make_engine()
        freqs = getattr(engine, 'bin_centers', None)
        detector = self._detector_cls(percentile=self._percentile,
                                      freqs=freqs)

        hop = self._HOP_SIZE
        n_hops = len(audio) // hop
        beat_frames: list[int] = []
        last_frame = -1
        for i in range(n_hops):
            start = i * hop
            frame = engine.push_hop(audio[start:start + hop])
            events = detector.detect(frame)
            # Any non-`song_start` event counts as a beat. Dedupe across
            # bands within the same frame (don't double-count when low
            # and mid fire on the same kick).
            for e in events:
                if e.kind == 'song_start':
                    continue
                if i != last_frame:
                    beat_frames.append(i)
                    last_frame = i
                break

        return np.asarray(beat_frames, dtype=np.float64) * hop / self._SAMPLE_RATE


# ============================================================================
# BeatRNNDetector — wraps a trained beat-RNN checkpoint, CPU forward.
# ============================================================================

# Native audio rate of the training pipeline (matches generate_beat_labels.py).
_BEATRNN_SR = 48000
_BEATRNN_HOP = 512
_BEATRNN_FPS = _BEATRNN_SR / _BEATRNN_HOP   # ~93.75


class BeatRNNDetector(BeatDetector):
    """Loads a beat-RNN checkpoint, runs CPU forward, peak-picks the
    continuous activation. Compatible with the continuous-reformulation
    (n_classes=1, sigmoid) and the legacy 3-class (n_classes=3, softmax,
    `kind='beat'` corresponds to channel index 1 by training convention).

    No GPU needed — uses numpy. Slow on long files (a few seconds of
    walltime per minute of audio) but doesn't contend with the wallpaper.
    """

    name = 'beat_rnn'

    def __init__(self, weights_path,
                 hidden_size: int = 48, proj_size: int = 32,
                 input_size: int = 216, n_classes: int | None = None,
                 peak_threshold: float = 0.3, min_distance_frames: int = 5):
        import hashlib
        self._weights_path = Path(weights_path)
        if not self._weights_path.exists():
            raise FileNotFoundError(self._weights_path)
        self._hidden = hidden_size
        self._proj = proj_size
        self._input = input_size
        self._threshold = peak_threshold
        self._min_distance = min_distance_frames

        # Load + infer n_classes if not specified by trying the supported
        # head sizes. Continuous-reformulation models are n_classes=1.
        from .rnn_forward import load_weights_npz, unpack_weights
        self._flat = load_weights_npz(self._weights_path)
        if n_classes is None:
            for c in (1, 3):
                try:
                    unpack_weights(self._flat, input_size, proj_size,
                                   hidden_size, c)
                    n_classes = c
                    break
                except ValueError:
                    continue
            if n_classes is None:
                raise ValueError(
                    f'Could not infer n_classes for {self._weights_path} '
                    f'({len(self._flat)} params, hidden={hidden_size}, '
                    f'proj={proj_size}, input={input_size}).')
        self._n_classes = n_classes
        self._weights = unpack_weights(
            self._flat, input_size, proj_size, hidden_size, n_classes)

        # version = checkpoint content hash; ensures eval cache invalidates
        # when a fresh checkpoint is saved with the same path.
        content_hash = hashlib.sha256(
            self._weights_path.read_bytes()).hexdigest()[:16]
        self.version = f'beat_rnn_h{hidden_size}_c{n_classes}_{content_hash}'

    @staticmethod
    def _compute_features(audio: np.ndarray, sr: int) -> np.ndarray:
        """Return (T, 216) features matching generate_beat_labels.py:
        log-CQT (108 bins) + half-wave-rectified first difference (108)."""
        import librosa
        if sr != _BEATRNN_SR:
            audio = _resample(audio, sr, _BEATRNN_SR)
        cqt = librosa.cqt(audio, sr=_BEATRNN_SR, hop_length=_BEATRNN_HOP,
                          n_bins=108, bins_per_octave=12,
                          fmin=librosa.note_to_hz('C1'))
        mag = np.abs(cqt).T.astype(np.float32)
        mag = np.log1p(mag * 10.0)
        diff = np.zeros_like(mag)
        diff[1:] = np.maximum(0.0, mag[1:] - mag[:-1])
        return np.concatenate([mag, diff], axis=1)

    def detect(self, audio: np.ndarray, sr: int) -> np.ndarray:
        from .rnn_forward import forward_sequence, peak_pick, sigmoid
        audio = _ensure_mono_float32(audio)
        features = self._compute_features(audio, sr)  # (T, 216)
        logits = forward_sequence(features, self._weights,
                                   self._hidden, self._proj)  # (T, n_classes)
        if self._n_classes == 1:
            scores = sigmoid(logits[:, 0])
        else:
            # 3-class: BeatNet convention is (downbeat, beat, non-beat).
            # 'beat-or-downbeat' = 1 - non_beat = sum of first two channels'
            # softmax probs.
            ex = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = ex / ex.sum(axis=1, keepdims=True)
            scores = (probs[:, 0] + probs[:, 1]).astype(np.float32)
        peaks = peak_pick(scores, threshold=self._threshold,
                           min_distance=self._min_distance)
        return peaks.astype(np.float64) / _BEATRNN_FPS


# ============================================================================
# BeatNetDetector — wraps the BeatNet teacher. Heavy deps (torch + BeatNet
# + madmom monkey-patch); lazy-imported so this module loads without them.
# ============================================================================

class BeatNetDetector(BeatDetector):
    """Runs BeatNet on audio, peak-picks the beat channel.

    Requires: BeatNet, torch, madmom (with collections monkey-patch).
    Lazy-imports so the eval harness still loads when those aren't
    installed; raises a clear ImportError on detect() if they're missing.
    """

    name = 'beatnet'

    def __init__(self, model_index: int = 1,
                 peak_threshold: float = 0.3, min_distance_frames: int = 3):
        self._model_index = model_index
        self._threshold = peak_threshold
        self._min_distance = min_distance_frames
        # Defer the heavy import + model load until detect() so constructor
        # is cheap and the eval-harness CLI can `--help` without the
        # BeatNet stack present.
        self._model = None
        self._proc = None
        # Static version — BeatNet's published weights are immutable per
        # model_index. If a custom finetune is ever swapped in, the file
        # hash can replace this; for now, model index is enough.
        self.version = f'beatnet_m{model_index}'

    def _load(self):
        if self._model is not None:
            return
        # madmom uses pre-3.10 collections imports; patch before importing.
        import collections
        import collections.abc
        for name in ('MutableSequence', 'MutableMapping', 'MutableSet',
                     'Sequence', 'Mapping', 'Set', 'Iterable',
                     'Container', 'Hashable', 'Callable', 'Sized'):
            if not hasattr(collections, name):
                setattr(collections, name, getattr(collections.abc, name))

        import os
        try:
            import torch
            from BeatNet.model import BDA
            from BeatNet.log_spect import LOG_SPECT
        except ImportError as e:
            raise ImportError(
                f'BeatNetDetector needs `BeatNet` + `torch` installed: {e}'
            ) from e

        bn_dir = os.path.dirname(os.path.abspath(
            __import__('BeatNet').__file__))
        sample_rate = 22050
        hop = int(20 * 0.001 * sample_rate)  # 441 = 20 ms
        win = int(64 * 0.001 * sample_rate)  # 1411 = 64 ms

        self._proc = LOG_SPECT(sample_rate=sample_rate, win_length=win,
                                hop_size=hop, n_bands=[24, 24, 24])

        model_paths = {
            1: os.path.join(bn_dir, 'models', 'model_1_weights.pt'),
            2: os.path.join(bn_dir, 'models', 'model_2_weights.pt'),
            3: os.path.join(bn_dir, 'models', 'model_3_weights.pt'),
        }
        model = BDA(272, 150, 2, torch.device('cpu'))
        model.load_state_dict(
            torch.load(model_paths[self._model_index],
                       map_location='cpu', weights_only=True))
        model.eval()
        self._model = model
        self._torch = torch
        self._fps = 50.0  # BeatNet's native frame rate

    def detect(self, audio: np.ndarray, sr: int) -> np.ndarray:
        self._load()
        audio = _ensure_mono_float32(audio)
        audio = _resample(audio, sr, 22050)
        torch = self._torch
        with torch.no_grad():
            feats = self._proc.process_audio(audio).T
            feats = torch.from_numpy(feats).unsqueeze(0)
            self._model.hidden = torch.zeros(2, 1, self._model.dim_hd)
            self._model.cell = torch.zeros(2, 1, self._model.dim_hd)
            preds = self._model(feats)[0]                 # (3, T)
            probs = torch.softmax(preds, dim=0).numpy().T  # (T, 3)
        # Channel order from generate_beat_labels.py docstring:
        # 0 = downbeat, 1 = beat, 2 = non-beat. Beat-or-downbeat = 1 - col2.
        from .rnn_forward import peak_pick
        scores = (1.0 - probs[:, 2]).astype(np.float32)
        peaks = peak_pick(scores, threshold=self._threshold,
                           min_distance=self._min_distance)
        return peaks.astype(np.float64) / self._fps
