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
    implement `detect`.

    Subclasses MAY override `detect_downbeats` to also return downbeat
    times. The default returns None, which the eval harness treats as
    "this detector is beats-only" — the downbeat eval is skipped
    rather than scored as 0.

    Subclasses MUST populate `last_timing` after each `detect*()` call
    with a dict mapping component name → seconds. The component
    'total' is mandatory and should equal the sum of the other
    components modulo overhead. Detectors that can't decompose
    (e.g. black-box library wrappers) emit `{'total': ...}` only.
    """

    name: str = ''
    version: str = ''

    def __init__(self) -> None:
        # Populated by detect()/detect_downbeats(). Empty before first call.
        self.last_timing: dict[str, float] = {}

    @abstractmethod
    def detect(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Return beat times in seconds. Mono float audio in [-1, 1]."""

    def detect_downbeats(self, audio: np.ndarray, sr: int) -> np.ndarray | None:
        """Return downbeat times in seconds, or None if not supported."""
        return None


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
    """Wraps the in-tree PercentileBeatDetector for offline eval —
    **matched to the live daemon's pipeline**.

    The daemon applies a Complex Spectral Difference transform to each
    raw CQT frame BEFORE feeding it to PercentileBeatDetector (see
    `flame_sheep_audio/processor.py`: `csd_frame = self._csd(raw_frame);
    self._detector.detect(csd_frame)`). Before 2026-06-03 this wrapper
    skipped the CSD step and measured F1 against raw flux, producing
    a ~0.06 artifact gap vs the deployed pipeline.

    Streams audio through one hop at a time, transforms each frame
    via CSD, then records every frame at which any non-`song_start`
    BeatEvent fires.
    """

    name = 'current_system'

    def __init__(self, percentile: float = 99.0):
        super().__init__()
        # Import inside __init__ so the package path tweak above is in
        # effect by the time we actually use the imports.
        from flame_sheep_audio.beat_detector import PercentileBeatDetector
        from flame_sheep_audio import SAMPLE_RATE, HOP_SIZE, FFT_SIZE
        self._SAMPLE_RATE = SAMPLE_RATE
        self._HOP_SIZE = HOP_SIZE
        self._FFT_SIZE = FFT_SIZE
        self._detector_cls = PercentileBeatDetector
        self._percentile = percentile

        # Version: bake the engine + detector + CSD transform identity
        # into a stable hash so any code change to any of them
        # invalidates the eval cache.
        import hashlib
        from flame_sheep_audio import beat_detector as _bd
        from flame_sheep_audio import _spectrum as _spec
        from flame_sheep_audio import hpss as _hpss
        src = (Path(_bd.__file__).read_text()
               + Path(_spec.__file__).read_text()
               + Path(_hpss.__file__).read_text())
        self.version = (
            'current_system_csd_' + hashlib.sha256(src.encode()).hexdigest()[:12]
        )

    def _make_engine(self):
        # Same single-engine choice as the live daemon: CQT.
        from flame_sheep_audio import CqtEngine
        return CqtEngine()

    def detect(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import time
        from flame_sheep_audio.hpss import ComplexSpectralDiffTransform

        audio = _ensure_mono_float32(audio)
        audio = _resample(audio, sr, self._SAMPLE_RATE)

        engine = self._make_engine()
        csd = ComplexSpectralDiffTransform()
        freqs = getattr(engine, 'bin_centers', None)
        detector = self._detector_cls(percentile=self._percentile,
                                      freqs=freqs)

        hop = self._HOP_SIZE
        n_hops = len(audio) // hop
        beat_frames: list[int] = []
        last_frame = -1
        cqt_time = 0.0
        csd_time = 0.0
        peak_time = 0.0
        t0 = time.monotonic()
        for i in range(n_hops):
            start = i * hop
            tc0 = time.monotonic()
            raw_frame = engine.push_hop(audio[start:start + hop])
            tc1 = time.monotonic()
            csd_frame = csd(raw_frame)
            tc2 = time.monotonic()
            events = detector.detect(csd_frame)
            tc3 = time.monotonic()
            cqt_time += tc1 - tc0
            csd_time += tc2 - tc1
            peak_time += tc3 - tc2
            for e in events:
                if e.kind == 'song_start':
                    continue
                if i != last_frame:
                    beat_frames.append(i)
                    last_frame = i
                break
        total = time.monotonic() - t0

        self.last_timing = {
            'cqt': cqt_time,
            'csd_transform': csd_time,
            'peak_pick': peak_time,
            'total': total,
        }

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
        super().__init__()
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
        import time
        from .rnn_forward import forward_sequence, peak_pick, sigmoid
        t0 = time.monotonic()
        audio = _ensure_mono_float32(audio)
        tf0 = time.monotonic()
        features = self._compute_features(audio, sr)  # (T, 216)
        tf1 = time.monotonic()
        logits = forward_sequence(features, self._weights,
                                   self._hidden, self._proj)
        tf2 = time.monotonic()
        if self._n_classes == 1:
            scores = sigmoid(logits[:, 0])
        else:
            ex = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = ex / ex.sum(axis=1, keepdims=True)
            scores = (probs[:, 0] + probs[:, 1]).astype(np.float32)
        peaks = peak_pick(scores, threshold=self._threshold,
                           min_distance=self._min_distance)
        tf3 = time.monotonic()
        self.last_timing = {
            'features': tf1 - tf0,
            'gru_forward': tf2 - tf1,
            'peak_pick': tf3 - tf2,
            'total': tf3 - t0,
        }
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
        super().__init__()
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

    def _activation_probs(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Returns (T, 3) softmax-normalized probs.
        Channel order (BeatNet convention): 0=beat, 1=downbeat, 2=non-beat.
        Verified against `particle_filtering_cascade.py:97-99` in the
        upstream package; previously documented backwards across this
        codebase. See tests/eval/test_beatnet_channels.py for a
        regression check.
        """
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
        return probs

    def detect(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import time
        t0 = time.monotonic()
        probs = self._activation_probs(audio, sr)
        # Beat-or-downbeat = 1 - non_beat.
        from .rnn_forward import peak_pick
        scores = (1.0 - probs[:, 2]).astype(np.float32)
        peaks = peak_pick(scores, threshold=self._threshold,
                           min_distance=self._min_distance)
        total = time.monotonic() - t0
        # BeatNet is a wrapped library — we measure end-to-end only.
        self.last_timing = {'total': total}
        return peaks.astype(np.float64) / self._fps

    def detect_downbeats(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import time
        t0 = time.monotonic()
        probs = self._activation_probs(audio, sr)
        # Downbeats happen at ~1/3-1/4 the rate of beats, so a slightly
        # larger min-distance reduces double-fires on adjacent frames.
        from .rnn_forward import peak_pick
        scores = probs[:, 1].astype(np.float32)
        peaks = peak_pick(scores, threshold=self._threshold,
                           min_distance=max(self._min_distance * 2, 6))
        total = time.monotonic() - t0
        self.last_timing = {'total': total}
        return peaks.astype(np.float64) / self._fps


# ============================================================================
# StreamingBeatNetDetector — same model as BeatNetDetector but fed per-hop
# instead of as a batch over the whole audio. Reproduces upstream's
# `activation_extractor_realtime` shape (5-hop sliding window per call) so
# the per-hop latency we measure is what the live wallpaper would see if
# we deployed BeatNet inside the daemon.
#
# Why this matters: the batch BeatNet eval said RTF 0.018 (~55× realtime).
# But that processes the whole track in one neural-net forward pass. The
# wallpaper needs hop-by-hop streaming. If per-hop latency exceeds the
# audio hop period (~20ms at 22050 Hz / 441 samples), BeatNet can't keep
# up live regardless of how fast its batch form is.
#
# We deliberately match upstream's redundant 5-hop reprocessing so the
# latency reflects what you'd pay shipping today's BeatNet as-is.
# Optimizing the feature extraction is a separate question.
# ============================================================================

class StreamingBeatNetDetector(BeatDetector):
    """Per-hop BeatNet feature extraction + model forward + peak-pick.

    Matches BeatNet's published `realtime` mode shape (each hop slides
    a 5-hop window through `LOG_SPECT`, takes the last-frame feature,
    calls model). Post-processing is the same peak-pick used in the
    batch BeatNetDetector — keeps batch vs streaming F1 comparisons
    apples-to-apples; the only difference is HOW features get fed.

    Doesn't use BeatNet's particle filter cascade — numpy 2 removed
    `np.in1d` which the upstream PF depends on. The PF is also not
    apples-to-apples vs our batch detector, so dropping it isolates
    the streaming question.
    """

    name = 'beatnet_streaming'

    def __init__(self, model_index: int = 1,
                 peak_threshold: float = 0.3, min_distance_frames: int = 3):
        super().__init__()
        self._model_index = model_index
        self._threshold = peak_threshold
        self._min_distance = min_distance_frames
        self._sr = 22050
        # Match upstream LOG_SPECT timing: 20 ms hop, 64 ms window.
        self._hop = int(0.020 * self._sr)   # 441 samples
        self._win = int(0.064 * self._sr)   # 1411 samples
        self._fps = 50.0
        self._model = None
        self._proc = None
        self.version = f'beatnet_streaming_m{model_index}'

    def _load(self):
        if self._model is not None:
            return
        import collections, collections.abc
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
                f'StreamingBeatNetDetector needs `BeatNet` + `torch`: {e}'
            ) from e
        bn_dir = os.path.dirname(os.path.abspath(
            __import__('BeatNet').__file__))
        self._proc = LOG_SPECT(sample_rate=self._sr,
                                win_length=self._win,
                                hop_size=self._hop,
                                n_bands=[24, 24, 24])
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

    # Streaming requires enough preceding audio that LOG_SPECT's
    # centered framing reaches the target frame without future-padding.
    # Empirically `n_history >= 4` gives bit-exact match against the
    # full-audio batch feature at the same absolute frame index;
    # 8 buys safety margin against any centering subtlety we're missing.
    _STREAM_HISTORY_HOPS = 8

    def _activation_probs_streaming(self, audio: np.ndarray, sr: int
                                       ) -> tuple[np.ndarray, list[float]]:
        """Iterate hop-by-hop. Returns (probs (T, 3), per_hop_latencies_s).

        Per hop: keep `_STREAM_HISTORY_HOPS` hops of audio context behind
        the target frame, run LOG_SPECT on that buffer, take the
        target frame's feature (index = history depth), run
        model.forward. Equivalent per-frame to the batch detector's
        features, verified bit-exact in tests/eval/test_streaming_*.

        The history depth is required because LOG_SPECT's STFT
        framing centers frames and looks past the buffer end; taking
        the literal last frame yields a frame at a different absolute
        time than the target counter. Taking `wf[history]` aligns
        with `full_audio_features[counter]`.
        """
        import time
        self._load()
        audio = _ensure_mono_float32(audio)
        audio = _resample(audio, sr, self._sr)
        torch = self._torch

        # Reset hidden state per-track. Streaming inside a track
        # carries hidden state automatically across model() calls,
        # since the model holds it as self.hidden / self.cell.
        self._model.hidden = torch.zeros(2, 1, self._model.dim_hd)
        self._model.cell = torch.zeros(2, 1, self._model.dim_hd)

        n_hops = len(audio) // self._hop
        history = self._STREAM_HISTORY_HOPS
        probs_frames: list[np.ndarray] = []
        per_hop_latencies: list[float] = []
        for counter in range(n_hops):
            t_hop = time.monotonic()
            if counter < history:
                # Warmup — emit zero probability so frame indexing
                # stays aligned with the batch eval. Costs `history`
                # frames of detection at the start of each track;
                # cheap vs the alignment bugs of computing partial
                # histories.
                probs_frames.append(np.array([0.0, 0.0, 1.0], dtype=np.float32))
            else:
                start = self._hop * (counter - history)
                end = self._hop * counter + self._win
                end = min(end, len(audio))
                window = audio[start:end]
                with torch.no_grad():
                    wf = self._proc.process_audio(window).T
                    # Target frame: index = history depth into the
                    # window. The frames after that are LOG_SPECT's
                    # future-padded continuation, which we don't use
                    # in streaming.
                    target_idx = min(history, wf.shape[0] - 1)
                    feats = wf[target_idx]
                    feats = torch.from_numpy(feats).unsqueeze(0).unsqueeze(0)
                    pred = self._model(feats)[0]            # (3, 1)
                    probs = torch.softmax(pred, dim=0).numpy().flatten()
                probs_frames.append(probs.astype(np.float32))
            per_hop_latencies.append(time.monotonic() - t_hop)

        return np.stack(probs_frames, axis=0), per_hop_latencies

    def detect(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import time
        t0 = time.monotonic()
        probs, per_hop = self._activation_probs_streaming(audio, sr)
        from .rnn_forward import peak_pick
        scores = (1.0 - probs[:, 2]).astype(np.float32)
        peaks = peak_pick(scores, threshold=self._threshold,
                           min_distance=self._min_distance)
        total = time.monotonic() - t0
        arr = np.asarray(per_hop, dtype=np.float64)
        self.last_timing = {
            'per_hop_mean_ms': float(np.mean(arr)) * 1000.0,
            'per_hop_p95_ms': float(np.percentile(arr, 95)) * 1000.0,
            'per_hop_p99_ms': float(np.percentile(arr, 99)) * 1000.0,
            'per_hop_max_ms': float(np.max(arr)) * 1000.0,
            'total': total,
        }
        return peaks.astype(np.float64) / self._fps

    def detect_downbeats(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import time
        t0 = time.monotonic()
        probs, per_hop = self._activation_probs_streaming(audio, sr)
        from .rnn_forward import peak_pick
        scores = probs[:, 1].astype(np.float32)
        peaks = peak_pick(scores, threshold=self._threshold,
                           min_distance=max(self._min_distance * 2, 6))
        total = time.monotonic() - t0
        arr = np.asarray(per_hop, dtype=np.float64)
        self.last_timing = {
            'per_hop_mean_ms': float(np.mean(arr)) * 1000.0,
            'per_hop_p95_ms': float(np.percentile(arr, 95)) * 1000.0,
            'per_hop_p99_ms': float(np.percentile(arr, 99)) * 1000.0,
            'per_hop_max_ms': float(np.max(arr)) * 1000.0,
            'total': total,
        }
        return peaks.astype(np.float64) / self._fps


# ============================================================================
# SlidingBeatNetDetector — same BeatNet model, fed by a manual streaming
# feature pipeline. Per hop: one centered STFT frame + cached filterbank
# matmul + log1p + diff-against-prev. Bit-exact to madmom LOG_SPECT on a
# per-frame basis (verified in tests). Skips the redundant LOG_SPECT
# windowed-reprocess pattern of StreamingBeatNetDetector.
#
# This is the impl that would actually deploy on a live wallpaper.
# StreamingBeatNetDetector is retained as a correctness-reference for the
# scorecard; both should produce equivalent F1, sliding should be much
# faster per-hop.
# ============================================================================

class SlidingBeatNetDetector(BeatDetector):
    """Per-hop sliding STFT → cached filterbank → log → diff-state →
    BeatNet model. Designed for live-wallpaper deployment: per-hop
    work is O(FFT + matmul + model.forward), no LOG_SPECT reprocess.
    """

    name = 'beatnet_sliding'

    def __init__(self, model_index: int = 1,
                 peak_threshold: float = 0.3, min_distance_frames: int = 3):
        super().__init__()
        self._model_index = model_index
        self._threshold = peak_threshold
        self._min_distance = min_distance_frames
        self._sr = 22050
        self._hop = int(0.020 * self._sr)   # 441
        self._win = int(0.064 * self._sr)   # 1411
        self._fps = 50.0
        # Lazy-init heavy state.
        self._model = None
        self._torch = None
        self._filterbank = None      # (n_fft_bins, n_filterbank_bins)
        self._hanning = None         # (win,)
        self.version = f'beatnet_sliding_m{model_index}'

    def _load(self):
        if self._model is not None:
            return
        import collections, collections.abc
        for name in ('MutableSequence', 'MutableMapping', 'MutableSet',
                     'Sequence', 'Mapping', 'Set', 'Iterable',
                     'Container', 'Hashable', 'Callable', 'Sized'):
            if not hasattr(collections, name):
                setattr(collections, name, getattr(collections.abc, name))
        import os
        try:
            import torch
            from BeatNet.model import BDA
            from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
            from madmom.audio.stft import ShortTimeFourierTransformProcessor
            from madmom.audio.spectrogram import (
                FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor)
        except ImportError as e:
            raise ImportError(
                f'SlidingBeatNetDetector needs `BeatNet` + `torch` + '
                f'`madmom`: {e}') from e
        bn_dir = os.path.dirname(os.path.abspath(
            __import__('BeatNet').__file__))
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

        # Bootstrap the filterbank: run madmom's pipeline once on a
        # tiny dummy signal so we can extract the matrix. The
        # filterbank is fully determined by (num_bands=24, fmin=30,
        # fmax=17000, norm_filters=True), so this only needs to
        # happen once per process.
        sig = SignalProcessor(num_channels=1, sample_rate=self._sr)
        frames_p = FramedSignalProcessor(frame_size=self._win, hop_size=self._hop)
        stft_p = ShortTimeFourierTransformProcessor()
        filt_p = FilteredSpectrogramProcessor(
            num_bands=24, fmin=30, fmax=17000, norm_filters=True)
        dummy = np.zeros(self._win + 2 * self._hop, dtype=np.float32)
        fs = filt_p(stft_p(frames_p(sig(dummy))))
        self._filterbank = np.asarray(fs.filterbank, dtype=np.float32)
        self._hanning = np.hanning(self._win).astype(np.float32)

    def _features_streaming(self, audio: np.ndarray, sr: int
                              ) -> tuple[np.ndarray, list[float]]:
        """Compute (T, 272) features hop-by-hop using a sliding STFT.

        Returns (features, per-frame compute times).
        Matches madmom LOG_SPECT bit-exact per frame (verified in
        tests/eval/test_sliding_beatnet.py).
        """
        import time
        self._load()
        audio = _ensure_mono_float32(audio)
        audio = _resample(audio, sr, self._sr)
        hop = self._hop
        win = self._win
        half_win = win // 2

        n_hops = len(audio) // hop
        feats = np.empty((n_hops, self._filterbank.shape[1] * 2), dtype=np.float32)
        prev_log_spec = None
        per_hop_times: list[float] = []

        for c in range(n_hops):
            t_hop = time.monotonic()
            # Centered framing: ref = c*hop, slice [ref - win/2, ref + win - win/2).
            ref = c * hop
            start = ref - half_win
            stop = start + win
            if start < 0 or stop > len(audio):
                frame = np.zeros(win, dtype=np.float32)
                a_start = max(0, start)
                a_stop = min(len(audio), stop)
                if a_stop > a_start:
                    frame[a_start - start: a_stop - start] = audio[a_start:a_stop]
            else:
                frame = audio[start:stop]

            windowed = frame * self._hanning
            fft_mag = np.abs(np.fft.rfft(windowed, n=win))[:-1]   # drop nyquist
            log_spec = np.log10(1.0 + fft_mag @ self._filterbank)
            if prev_log_spec is None:
                diff = np.zeros_like(log_spec)
            else:
                diff = np.maximum(log_spec - prev_log_spec, 0.0)
            feats[c, :log_spec.size] = log_spec
            feats[c, log_spec.size:] = diff
            prev_log_spec = log_spec
            per_hop_times.append(time.monotonic() - t_hop)

        return feats, per_hop_times

    def _activation_probs_sliding(self, audio: np.ndarray, sr: int
                                     ) -> tuple[np.ndarray, list[float]]:
        """Run sliding features + model.forward per-hop with carried
        LSTM state. Returns (probs (T, 3), per-hop total times)."""
        import time
        feats, feat_times = self._features_streaming(audio, sr)
        torch = self._torch
        self._model.hidden = torch.zeros(2, 1, self._model.dim_hd)
        self._model.cell = torch.zeros(2, 1, self._model.dim_hd)
        probs_frames = np.empty((feats.shape[0], 3), dtype=np.float32)
        per_hop_total: list[float] = []
        for c in range(feats.shape[0]):
            t_hop = time.monotonic()
            with torch.no_grad():
                ft = torch.from_numpy(feats[c]).unsqueeze(0).unsqueeze(0)
                pred = self._model(ft)[0]
                probs_frames[c] = torch.softmax(pred, dim=0).numpy().flatten()
            per_hop_total.append(feat_times[c] + (time.monotonic() - t_hop))
        return probs_frames, per_hop_total

    def detect(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import time
        t0 = time.monotonic()
        probs, per_hop = self._activation_probs_sliding(audio, sr)
        from .rnn_forward import peak_pick
        scores = (1.0 - probs[:, 2]).astype(np.float32)
        peaks = peak_pick(scores, threshold=self._threshold,
                           min_distance=self._min_distance)
        total = time.monotonic() - t0
        arr = np.asarray(per_hop, dtype=np.float64)
        self.last_timing = {
            'per_hop_mean_ms': float(np.mean(arr)) * 1000.0,
            'per_hop_p95_ms': float(np.percentile(arr, 95)) * 1000.0,
            'per_hop_p99_ms': float(np.percentile(arr, 99)) * 1000.0,
            'per_hop_max_ms': float(np.max(arr)) * 1000.0,
            'total': total,
        }
        return peaks.astype(np.float64) / self._fps

    def detect_downbeats(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import time
        t0 = time.monotonic()
        probs, per_hop = self._activation_probs_sliding(audio, sr)
        from .rnn_forward import peak_pick
        scores = probs[:, 1].astype(np.float32)
        peaks = peak_pick(scores, threshold=self._threshold,
                           min_distance=max(self._min_distance * 2, 6))
        total = time.monotonic() - t0
        arr = np.asarray(per_hop, dtype=np.float64)
        self.last_timing = {
            'per_hop_mean_ms': float(np.mean(arr)) * 1000.0,
            'per_hop_p95_ms': float(np.percentile(arr, 95)) * 1000.0,
            'per_hop_p99_ms': float(np.percentile(arr, 99)) * 1000.0,
            'per_hop_max_ms': float(np.max(arr)) * 1000.0,
            'total': total,
        }
        return peaks.astype(np.float64) / self._fps


# ============================================================================
# LiteBeatNetDetector — pure-numpy port of BeatNet's BDA model + the sliding
# STFT feature pipeline. Zero torch, zero madmom in the runtime — just
# numpy (+ scipy.signal.resample_poly via the shared _resample helper).
#
# Install footprint vs SlidingBeatNetDetector (which is torch + madmom):
#   torch + madmom: ~500 MB+
#   numpy + the lite .npz: ~2 MB (1.5 MB weights + 0.4 MB filterbank)
#
# Designed for the live wallpaper's r/unixporn install story — see
# `feedback_flame_sheep_audience_scope.md`. The .npz files in
# `flame_sheep/data/beatnet_m{1,2,3}_lite.npz` ship with the package;
# regenerate with `tools/export_beatnet_lite_weights.py` if BeatNet's
# upstream model weights ever change.
# ============================================================================

class LiteBeatNetDetector(BeatDetector):
    """Pure-numpy BeatNet streaming detector. Bit-exact to
    SlidingBeatNetDetector (which is torch-backed) on every frame —
    asserted in tests/eval/test_beatnet_lite.py.

    Per-hop work: sliding STFT + cached filterbank matmul + log + diff
    state + manual conv1d + ReLU + max_pool + linear + 2-layer LSTM
    + linear + softmax. All numpy.
    """

    name = 'beatnet_lite'

    def __init__(self, model_index: int = 1,
                 peak_threshold: float = 0.3, min_distance_frames: int = 3,
                 weights_path: Path | None = None):
        super().__init__()
        self._model_index = model_index
        self._threshold = peak_threshold
        self._min_distance = min_distance_frames
        if weights_path is None:
            weights_path = (Path(__file__).resolve().parents[1]
                            / 'data' / f'beatnet_m{model_index}_lite.npz')
        self._weights_path = Path(weights_path)
        if not self._weights_path.exists():
            raise FileNotFoundError(
                f'lite weights file missing: {self._weights_path}. '
                f'Regenerate via tools/export_beatnet_lite_weights.py.')
        self._lite = None    # lazy-loaded BeatNetLite
        # Sample-rate / hop / win read from the .npz at load.
        self._sr = 22050
        self._hop = 441
        self._win = 1411
        self._fps = 50.0
        self.version = f'beatnet_lite_m{model_index}'

    def _load(self):
        if self._lite is not None:
            return
        from .beatnet_lite import BeatNetLite
        self._lite = BeatNetLite(self._weights_path)
        # Pull canonical rates from the weights metadata so this stays
        # honest if the export script ever ships a different config.
        self._sr = self._lite.sample_rate
        self._hop = self._lite.hop
        self._win = self._lite.win

    def _features_and_probs(self, audio: np.ndarray, sr: int
                              ) -> tuple[np.ndarray, list[float]]:
        """Compute (probs (T, 3), per-hop total times) end-to-end."""
        import time
        self._load()
        audio = _ensure_mono_float32(audio)
        audio = _resample(audio, sr, self._sr)
        hop = self._hop
        win = self._win
        half_win = win // 2
        filterbank = self._lite.filterbank
        hanning = self._lite.hanning_window

        self._lite.reset_state()
        n_hops = len(audio) // hop
        probs_frames = np.empty((n_hops, 3), dtype=np.float32)
        per_hop_times: list[float] = []
        prev_log_spec = None

        for c in range(n_hops):
            t_hop = time.monotonic()
            ref = c * hop
            start = ref - half_win
            stop = start + win
            if start < 0 or stop > len(audio):
                frame = np.zeros(win, dtype=np.float32)
                a_start = max(0, start)
                a_stop = min(len(audio), stop)
                if a_stop > a_start:
                    frame[a_start - start: a_stop - start] = audio[a_start:a_stop]
            else:
                frame = audio[start:stop]
            windowed = frame * hanning
            fft_mag = np.abs(np.fft.rfft(windowed, n=win))[:-1]
            log_spec = np.log10(1.0 + fft_mag @ filterbank).astype(np.float32)
            if prev_log_spec is None:
                diff = np.zeros_like(log_spec)
            else:
                diff = np.maximum(log_spec - prev_log_spec, 0.0)
            features = np.concatenate([log_spec, diff])   # (272,)
            probs_frames[c] = self._lite.step(features)
            prev_log_spec = log_spec
            per_hop_times.append(time.monotonic() - t_hop)

        return probs_frames, per_hop_times

    def detect(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import time
        t0 = time.monotonic()
        probs, per_hop = self._features_and_probs(audio, sr)
        from .rnn_forward import peak_pick
        scores = (1.0 - probs[:, 2]).astype(np.float32)
        peaks = peak_pick(scores, threshold=self._threshold,
                           min_distance=self._min_distance)
        total = time.monotonic() - t0
        arr = np.asarray(per_hop, dtype=np.float64)
        self.last_timing = {
            'per_hop_mean_ms': float(np.mean(arr)) * 1000.0,
            'per_hop_p95_ms': float(np.percentile(arr, 95)) * 1000.0,
            'per_hop_p99_ms': float(np.percentile(arr, 99)) * 1000.0,
            'per_hop_max_ms': float(np.max(arr)) * 1000.0,
            'total': total,
        }
        return peaks.astype(np.float64) / self._fps

    def detect_downbeats(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import time
        t0 = time.monotonic()
        probs, per_hop = self._features_and_probs(audio, sr)
        from .rnn_forward import peak_pick
        scores = probs[:, 1].astype(np.float32)
        peaks = peak_pick(scores, threshold=self._threshold,
                           min_distance=max(self._min_distance * 2, 6))
        total = time.monotonic() - t0
        arr = np.asarray(per_hop, dtype=np.float64)
        self.last_timing = {
            'per_hop_mean_ms': float(np.mean(arr)) * 1000.0,
            'per_hop_p95_ms': float(np.percentile(arr, 95)) * 1000.0,
            'per_hop_p99_ms': float(np.percentile(arr, 99)) * 1000.0,
            'per_hop_max_ms': float(np.max(arr)) * 1000.0,
            'total': total,
        }
        return peaks.astype(np.float64) / self._fps


# ============================================================================
# MultiDepthBeatRNNDetector — wraps the production runtime detector so
# the eval measures EXACTLY what the wallpaper sees, including the
# 770× CQT scale correction, the per-head threshold cascade, the
# refractory peak-picker, and the lookahead buffer.
#
# The legacy BeatRNNDetector above handles single-arch checkpoints with
# its own numpy forward pass. For multidepth we go through the same
# `load_beat_rnn` factory the daemon uses — both because reproducing
# the forward logic here would drift, and because we want the eval
# to surface bugs in the runtime path, not just in the algorithm.
# ============================================================================

class MultiDepthBeatRNNDetector(BeatDetector):
    """Production multidepth beat-RNN, exercised via the runtime detector.

    `kinds` selects which event kinds count as 'beats' for F1 scoring:
        ('low',)         → downbeats only
        ('low', 'mid')   → all beats (downbeat + beat heads)   ← default
        ('low','mid','high') → any onset

    The default matches what we use for tempo reconciliation: the
    'beat-level' event stream (cascade tier 1 + 2). For pure beat F1
    against osu beat times, ('low', 'mid') is the right thing.
    """

    name = 'beat_rnn_multidepth'

    def __init__(self, weights_path,
                 downbeat_threshold: float = 0.80,
                 beat_threshold: float = 0.55,
                 onset_threshold: float = 0.15,
                 min_distance_frames: int = 9,
                 lookahead_frames: int = 9,
                 kinds: tuple[str, ...] = ('low', 'mid')):
        super().__init__()
        import hashlib
        self._weights_path = Path(weights_path)
        if not self._weights_path.exists():
            raise FileNotFoundError(self._weights_path)
        self._downbeat_th = float(downbeat_threshold)
        self._beat_th = float(beat_threshold)
        self._onset_th = float(onset_threshold)
        self._min_distance = int(min_distance_frames)
        self._lookahead = int(lookahead_frames)
        self._kinds = frozenset(kinds)

        # Version = checkpoint content hash + threshold tuple.
        # Cache invalidates when weights change OR when we change the
        # per-head thresholds (since those are part of "what produced
        # these beats").
        content_hash = hashlib.sha256(
            self._weights_path.read_bytes()).hexdigest()[:12]
        thr_tag = f'd{int(self._downbeat_th*100):02d}b{int(self._beat_th*100):02d}o{int(self._onset_th*100):02d}'
        kind_tag = ''.join(sorted(self._kinds))
        self.version = f'mdrnn_{content_hash}_{thr_tag}_k{kind_tag}'

    def detect(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import time
        from flame_sheep_audio import SAMPLE_RATE, HOP_SIZE
        from flame_sheep_audio._cqt_engine import CqtEngine
        from flame_sheep_audio.beat_rnn import load_beat_rnn

        audio = _ensure_mono_float32(audio)
        if sr != SAMPLE_RATE:
            audio = _resample(audio, sr, SAMPLE_RATE)
        cqt = CqtEngine()
        detector = load_beat_rnn(
            str(self._weights_path),
            threshold=0.3,
            downbeat_threshold=self._downbeat_th,
            beat_threshold=self._beat_th,
            onset_threshold=self._onset_th,
            min_peak_distance_frames=self._min_distance,
            lookahead_frames=self._lookahead,
        )

        hop_dur = HOP_SIZE / SAMPLE_RATE
        t = 0.0
        beat_times: list[float] = []
        t0 = time.monotonic()
        for pos in range(0, len(audio) - HOP_SIZE, HOP_SIZE):
            chunk = audio[pos:pos + HOP_SIZE]
            frame = cqt.push_hop(chunk)
            events = detector.detect(frame)
            for ev in events:
                if ev.kind in self._kinds:
                    beat_times.append(t)
            t += hop_dur
        self.last_timing = {'total': time.monotonic() - t0}

        return np.asarray(beat_times, dtype=np.float64)


# ============================================================================
# MadmomDetector — wraps madmom's RNNBeatProcessor + DBNBeatTrackingProcessor.
# Second reference-quality oracle alongside BeatNet. Per alter ego's earlier
# critique: madmom is the architectural family our home-grown trained model
# has failed to replicate three times (CRNN-style RNN beat activations +
# HMM/DBN decoder over the activation curve). Useful as both a known-good
# ceiling and as a sanity check against BeatNet (independent implementations,
# so disagreement is informative).
#
# Madmom 0.17.dev0+ is required for Python 3.13 compatibility (the 0.16.1
# PyPI release uses removed numpy.float aliases). Installed via:
#   pip install "git+https://github.com/CPJKU/madmom.git@main"
# ============================================================================

class MadmomDetector(BeatDetector):
    """Beat tracking via madmom's RNN + DBN HMM-style decoder.

    Pipeline:
      * `RNNBeatProcessor` runs the bundled CNN/RNN ensemble over the
        audio, producing per-frame beat activation (~100 fps).
      * `DBNBeatTrackingProcessor` decodes the activation curve via a
        dynamic Bayesian network whose state space encodes
        beat-period × beat-phase. Output is beat times in seconds.

    Both processors load pre-trained weights bundled with madmom.

    Version pins the madmom package version so a future upgrade
    invalidates the eval cache automatically.
    """

    name = 'madmom'

    def __init__(self, fps: int = 100,
                 min_bpm: float = 55.0, max_bpm: float = 215.0):
        super().__init__()
        self._fps = fps
        self._min_bpm = min_bpm
        self._max_bpm = max_bpm
        # Lazy-init the processors — they load model weights and we
        # don't want that hit at import time.
        self._rnn = None
        self._dbn = None
        self._db_rnn = None
        self._db_dbn = None
        # Version: madmom package version + processor parameters.
        # Anything below changes → cache invalidates.
        import madmom
        self.version = (
            f'madmom_{madmom.__version__}_fps{fps}_'
            f'bpm{int(min_bpm)}-{int(max_bpm)}')

    def _load(self):
        if self._rnn is not None:
            return
        from madmom.features.beats import (
            RNNBeatProcessor, DBNBeatTrackingProcessor)
        self._rnn = RNNBeatProcessor()
        self._dbn = DBNBeatTrackingProcessor(
            min_bpm=self._min_bpm, max_bpm=self._max_bpm, fps=self._fps)

    def _load_downbeat(self):
        if self._db_rnn is not None:
            return
        from madmom.features.downbeats import (
            RNNDownBeatProcessor, DBNDownBeatTrackingProcessor)
        self._db_rnn = RNNDownBeatProcessor()
        # `beats_per_bar` is the meter grid the DBN considers; (3, 4)
        # covers waltz + common time and is madmom's standard default.
        self._db_dbn = DBNDownBeatTrackingProcessor(
            beats_per_bar=[3, 4],
            min_bpm=self._min_bpm, max_bpm=self._max_bpm, fps=self._fps)

    def detect(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import time
        self._load()
        audio = _ensure_mono_float32(audio)
        # madmom's RNN processor expects 44100 Hz internally; resample
        # if needed. The processor accepts a (samples, sample_rate)
        # input via the .Signal interface, but the direct call path
        # also accepts a numpy array — when we pass audio it assumes
        # 44100, so resample if the input was anything else.
        t0 = time.monotonic()
        if sr != 44100:
            audio = _resample(audio, sr, 44100)
        tr0 = time.monotonic()
        activation = self._rnn(audio)        # (T,) beat probabilities
        tr1 = time.monotonic()
        beat_times = self._dbn(activation)   # (N,) beat times in seconds
        tr2 = time.monotonic()
        self.last_timing = {
            'rnn': tr1 - tr0,
            'dbn': tr2 - tr1,
            'total': tr2 - t0,
        }
        return np.asarray(beat_times, dtype=np.float64)

    def detect_downbeats(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import time
        self._load_downbeat()
        audio = _ensure_mono_float32(audio)
        t0 = time.monotonic()
        if sr != 44100:
            audio = _resample(audio, sr, 44100)
        tr0 = time.monotonic()
        activation = self._db_rnn(audio)
        tr1 = time.monotonic()
        decoded = self._db_dbn(activation)
        tr2 = time.monotonic()
        self.last_timing = {
            'rnn': tr1 - tr0,
            'dbn': tr2 - tr1,
            'total': tr2 - t0,
        }
        if decoded.size == 0:
            return np.asarray([], dtype=np.float64)
        downbeat_times = decoded[decoded[:, 1] == 1.0, 0]
        return np.asarray(downbeat_times, dtype=np.float64)


# ============================================================================
# BeatNetLivePFDetector — drives the PRODUCTION daemon beat path.
#
# This is the only detector in this module that scores the *shipping* beat
# times. Every other BeatNet variant above (batch / streaming / sliding /
# lite) is a parallel reimplementation that bypasses the production particle
# filter; the tempo-estimator `BeatNetPFTempoEstimator` drives the real PF
# but scores BPM, not beat times. This closes that gap.
#
# Pipeline mirrors the daemon exactly:
#   BeatNetLiveDetector(use_particle_filter=True)  (beat_detector_beatnet.py)
#     → feed_audio_hop(hop@48k)  resample→22050, sliding-STFT, lite LSTM
#     → activations → BeatNet particle_filter_cascade (fast PF via _pf_fast)
#     → det._pf.path rows [time_seconds, kind]  (kind 1=downbeat, 2=beat)
#
# Predicted beat TIMES are read from `det._pf.path`, NOT from the BeatEvents
# returned by detect() — those carry no timestamp. The PF path time base is
# `offset + counter * (1/fps)`; with fps=50 that's 20 ms per processed
# BeatNet frame, measured from the start of the stream fed to that PF.
#
# REGIMES (how a detector instance is reused across the eval's per-track
# detect() calls — each regime is a different fidelity question):
#   fresh             — new BeatNetLiveDetector per track. Clean-slate PF
#                       each time; no cross-track contamination. The
#                       conservative default and the only one safe to run at
#                       large N while the PF has a per-track particle leak.
#   persistent_reset  — one detector reused across tracks, reset_bands()
#                       between tracks. Faithful to the daemon handling a
#                       playlist of distinct songs (reset on song change).
#   persistent_noreset— one detector reused, NEVER reset. The continuous-
#                       audio regime; also the one that exposes the PF
#                       particle/path leak (path grows unbounded across the
#                       whole corpus). Times are de-offset per track so
#                       scoring still aligns to each track's ground truth.
# ============================================================================

_LIVE_PF_REGIMES = ('fresh', 'persistent_reset', 'persistent_noreset')


class BeatNetLivePFDetector(BeatDetector):
    """Production daemon beat path, scored for beat-time accuracy.

    Constructs `flame_sheep_audio.beat_detector_beatnet.BeatNetLiveDetector`
    with the particle filter enabled and streams audio through it hop-by-hop
    exactly as the daemon does (and as `BeatNetPFTempoEstimator` does for
    BPM). Predicted beat/downbeat times are read from the PF's `path`.

    `regime` selects instance reuse across tracks — see module comment.
    """

    name = 'beatnet_live_pf'

    def __init__(self, regime: str = 'fresh',
                 model_index: int = 1,
                 offset_ms: float = 0.0):
        super().__init__()
        if regime not in _LIVE_PF_REGIMES:
            raise ValueError(
                f'regime must be one of {_LIVE_PF_REGIMES}, got {regime!r}')
        self._regime = regime
        self._model_index = int(model_index)
        # Constant latency correction (ms) applied to predicted times at
        # SCORING time (see eval_beat_detection). Stored here only so it can
        # be baked into the cache namespace — the production PF beats lead
        # GTZAN ground truth by ~65 ms, so matrix cells score with +65.
        self.offset_ms = float(offset_ms)
        # Resampling mode the production PF will run in (read from the same
        # env var the PF reads at install time). This CHANGES the predicted
        # beat path, so it MUST be part of the cache key — otherwise the
        # file-hash below is identical across modes (the mode lives in the
        # environment, not the source) and the three variants would collide.
        import os as _os
        # Production PF is systematic-only now (resample toggle collapsed);
        # kept as a stable cache-key label, no longer env-selectable.
        self._resample_mode = 'systematic'
        # Reused detector for the persistent regimes; None for fresh.
        self._persistent_det = None

        from flame_sheep_audio import SAMPLE_RATE, HOP_SIZE
        self._SAMPLE_RATE = SAMPLE_RATE
        self._HOP_SIZE = HOP_SIZE

        # Version: hash the two source files that define the production beat
        # path — the adapter + the fast PF install. Any edit to the PF
        # (the planned later step) or the adapter gets a fresh cache
        # namespace, so before/after numbers never collide. Regime is baked
        # in too so the three variants cache independently.
        import hashlib
        from flame_sheep_audio import beat_detector_beatnet as _adapter_mod
        from flame_sheep_audio import _pf_fast as _pf_mod
        src = (Path(_adapter_mod.__file__).read_text()
               + Path(_pf_mod.__file__).read_text())
        h = hashlib.sha256(src.encode()).hexdigest()[:12]
        # name is the cache namespace alongside version; keep a stable base
        # name but encode the regime + resample mode there so caches don't
        # cross-pollute (mode changes the predicted path; see above).
        self.name = f'beatnet_live_pf_{regime}_{self._resample_mode}'
        # Version carries mode AND offset so a given (mode, offset) matrix
        # cell is fully isolated in the cache. Offset is applied post-hoc at
        # scoring (predictions themselves are offset-independent), so it is
        # cache-neutral in practice, but it is baked in per the matrix spec
        # to guarantee cells can never cross-pollute.
        off_tag = f'off{int(round(self.offset_ms))}'
        self.version = (f'live_pf_{regime}_{self._resample_mode}_'
                        f'{off_tag}_m{model_index}_{h}')

    def _make_detector(self):
        from flame_sheep_audio.beat_detector_beatnet import BeatNetLiveDetector
        return BeatNetLiveDetector(model_index=self._model_index,
                                   use_particle_filter=True)

    def _acquire_detector(self):
        """Return the detector to use for this track, honoring the regime."""
        if self._regime == 'fresh':
            return self._make_detector()
        # persistent_* — build once, then reuse.
        if self._persistent_det is None:
            self._persistent_det = self._make_detector()
        elif self._regime == 'persistent_reset':
            # Daemon song-change semantics: clear LSTM + buffers + rebuild PF.
            self._persistent_det.reset_bands()
        # persistent_noreset: reuse as-is, no reset (leak/continuous regime).
        return self._persistent_det

    def _stream(self, audio: np.ndarray, sr: int, kind_code: int
                ) -> np.ndarray:
        """Stream one track through the production detector and return the
        predicted times (seconds, track-relative) for the requested PF
        `kind_code` (1=downbeat, 2=beat)."""
        import time
        audio = _ensure_mono_float32(audio)
        audio = _resample(audio, sr, self._SAMPLE_RATE)

        t0 = time.monotonic()
        det = self._acquire_detector()
        setup = time.monotonic() - t0

        pf = det._pf
        # Time/position of the PF *before* this track so persistent_noreset
        # (shared, ever-growing path) can be sliced + de-offset to this
        # track. For fresh / reset the PF is brand-new (counter == -1,
        # path == [[0,0]]), so these reduce to 0 / 1 and are no-ops.
        counter_before = int(pf.counter)
        path_len_before = len(pf.path)
        t_offset = (counter_before + 1) * float(pf.T)

        hop = self._HOP_SIZE
        n = len(audio)
        pos = 0
        t_stream = time.monotonic()
        while pos + hop <= n:
            det.feed_audio_hop(audio[pos:pos + hop])
            det.detect(None)   # drain pending events (unused; keeps buffer bounded)
            pos += hop
        stream = time.monotonic() - t_stream

        path = np.asarray(pf.path, dtype=np.float64)
        new_rows = path[path_len_before:] if path.shape[0] > path_len_before \
            else path[:0]
        if new_rows.size:
            sel = new_rows[new_rows[:, 1].astype(int) == kind_code]
            times = sel[:, 0] - t_offset
            # Clamp any tiny negative (frame-center) artifacts to 0.
            times = times[times >= -1e-6]
            times = np.clip(times, 0.0, None)
        else:
            times = np.asarray([], dtype=np.float64)

        self.last_timing = {
            'setup': setup,
            'stream': stream,
            'total': setup + stream,
        }
        return np.asarray(np.sort(times), dtype=np.float64)

    def detect(self, audio: np.ndarray, sr: int) -> np.ndarray:
        return self._stream(audio, sr, kind_code=2)

    def detect_downbeats(self, audio: np.ndarray, sr: int) -> np.ndarray:
        return self._stream(audio, sr, kind_code=1)
