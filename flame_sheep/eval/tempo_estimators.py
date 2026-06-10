"""Tempo-estimator implementations for the eval harness.

Mirrors the BeatDetector pattern in flame_sheep/eval/detectors.py:
each estimator takes (audio, sr) and returns a single global BPM
estimate. `last_timing` is populated after each `estimate()` with a
dict of component → seconds; `total` is mandatory, per-component
breakdowns are optional. Black-box wrappers (madmom) emit `{'total':}`
only.

Why this exists separately from `flame_sheep_audio.tempo.TempoTrackerBase`:
the runtime tracker is causal / streaming (feed one frame at a time);
many published tempo algorithms are offline-batch (madmom, librosa).
A common eval interface lets us swap them all behind the same
scorecard and compare numbers directly.
"""
from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

# Match detectors.py: make in-tree flame_sheep_audio importable.
_AUDIO_SRC = (Path(__file__).resolve().parents[2]
              / 'flame_sheep_audio' / 'src')
if _AUDIO_SRC.exists() and str(_AUDIO_SRC) not in sys.path:
    sys.path.insert(0, str(_AUDIO_SRC))


class TempoEstimator(ABC):
    """Abstract base. Subclasses must set `name` and `version` and
    implement `estimate`.

    Subclasses MUST populate `last_timing` after each `estimate()` call
    with a dict mapping component name → seconds. The component
    'total' is mandatory and should equal the sum of the other
    components modulo overhead. Estimators that can't decompose
    (e.g. black-box library wrappers) emit `{'total': ...}` only.
    """

    name: str = ''
    version: str = ''

    def __init__(self) -> None:
        self.last_timing: dict[str, float] = {}

    @abstractmethod
    def estimate(self, audio: np.ndarray, sr: int) -> float:
        """Return a single BPM estimate. Mono float audio in [-1, 1]."""


def _ensure_mono_float32(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=0 if audio.shape[0] < audio.shape[1] else 1)
    return audio.astype(np.float32, copy=False)


def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sr_in, sr_out)
    return resample_poly(audio, sr_out // g, sr_in // g).astype(np.float32)


# ---------------------------------------------------------------------------
# In-tree ACF tracker wrapper
# ---------------------------------------------------------------------------

class AcfTempoEstimator(TempoEstimator):
    """Streams audio through CQT → flux → AutocorrelationTempoTracker.

    Mirrors the production audio pipeline (processor.py + tempo_acf.py)
    so the eval measures the actual deployed algorithm, not a parallel
    implementation. Per feedback_reuse_production_code.md.
    """

    name = 'acf'

    def __init__(self) -> None:
        super().__init__()
        # Lazy imports so loader code doesn't pay the cost at import time
        # and the path tweak above is honored.
        from flame_sheep_audio import SAMPLE_RATE, HOP_SIZE
        self._SAMPLE_RATE = SAMPLE_RATE
        self._HOP_SIZE = HOP_SIZE

        import hashlib
        from flame_sheep_audio import tempo_acf as _acf_mod
        from flame_sheep_audio import tempo as _tempo_mod
        src = (Path(_acf_mod.__file__).read_text()
               + Path(_tempo_mod.__file__).read_text())
        self.version = (
            'acf_' + hashlib.sha256(src.encode()).hexdigest()[:12])

    def estimate(self, audio: np.ndarray, sr: int) -> float:
        from flame_sheep_audio._cqt_engine import CqtEngine
        from flame_sheep_audio.tempo_acf import AutocorrelationTempoTracker

        audio = _ensure_mono_float32(audio)
        audio = _resample(audio, sr, self._SAMPLE_RATE)

        t0 = time.perf_counter()
        engine = CqtEngine()
        hop_dur = self._HOP_SIZE / self._SAMPLE_RATE
        tracker = AutocorrelationTempoTracker(hop_duration=hop_dur)
        setup_elapsed = time.perf_counter() - t0

        # Stream at HOP_SIZE cadence — matches the threaded production path.
        t_stream = time.perf_counter()
        pos = 0
        n = len(audio)
        while pos < n:
            chunk = audio[pos:pos + self._HOP_SIZE]
            if len(chunk) < self._HOP_SIZE:
                chunk = np.pad(chunk, (0, self._HOP_SIZE - len(chunk)))
            frame = engine.push_hop(chunk)
            tracker.feed(frame.onset_strength)
            pos += self._HOP_SIZE
        stream_elapsed = time.perf_counter() - t_stream

        # Surface the tracker's own per-component timing alongside the
        # streaming wall-clock. 'total' is the eval wall-clock; the
        # tracker-internal breakdown is reported under its own keys.
        tracker_timing = tracker.last_timing
        self.last_timing = {
            'setup': setup_elapsed,
            'stream': stream_elapsed,
            'tracker_feed': tracker_timing.get('feed', 0.0),
            'tracker_update': tracker_timing.get('update', 0.0),
            'total': setup_elapsed + stream_elapsed,
        }
        return float(tracker.effective_bpm)


# ---------------------------------------------------------------------------
# BTrack — production tempo tracker, fed raw audio hop-by-hop
# ---------------------------------------------------------------------------

class BTrackTempoEstimator(TempoEstimator):
    """Streams audio through `BTrackTempoTracker` at HOP_SIZE cadence,
    then reads `effective_bpm` at end-of-track.

    BTrack is the daemon's tempo source for any detector OTHER than
    BeatNet-PF (which uses its own particle filter for tempo). Running
    it here under the same eval interface gives us a head-to-head with
    ACF and the offline madmom estimators on the same corpus.

    Native lib (libBTrack); raises ImportError at instantiation if the
    pybind11 wrapper isn't installed.
    """

    name = 'btrack'

    def __init__(self) -> None:
        super().__init__()
        from flame_sheep_audio import SAMPLE_RATE, HOP_SIZE
        self._SAMPLE_RATE = SAMPLE_RATE
        self._HOP_SIZE = HOP_SIZE

        import hashlib
        from flame_sheep_audio import tempo_btrack as _bt_mod
        src = Path(_bt_mod.__file__).read_text()
        self.version = 'btrack_' + hashlib.sha256(src.encode()).hexdigest()[:12]

    def estimate(self, audio: np.ndarray, sr: int) -> float:
        from flame_sheep_audio.tempo_btrack import BTrackTempoTracker

        audio = _ensure_mono_float32(audio)
        audio = _resample(audio, sr, self._SAMPLE_RATE)

        t0 = time.perf_counter()
        tracker = BTrackTempoTracker(hop_size=self._HOP_SIZE,
                                       sample_rate=self._SAMPLE_RATE)
        setup_elapsed = time.perf_counter() - t0

        t_stream = time.perf_counter()
        pos = 0
        n = len(audio)
        while pos + self._HOP_SIZE <= n:
            chunk = audio[pos:pos + self._HOP_SIZE]
            tracker.feed_audio(chunk)
            pos += self._HOP_SIZE
        stream_elapsed = time.perf_counter() - t_stream

        self.last_timing = {
            'setup': setup_elapsed,
            'stream': stream_elapsed,
            'total': setup_elapsed + stream_elapsed,
        }
        return float(tracker.effective_bpm)


# ---------------------------------------------------------------------------
# BeatNet PF — particle filter's tempo state space, read at end-of-track
# ---------------------------------------------------------------------------

class BeatNetPFTempoEstimator(TempoEstimator):
    """Streams audio through `BeatNetLiveDetector` (PF enabled), reads
    `current_tempo()` at end-of-track.

    Mirrors the production daemon path when `cfg.detector.kind ==
    'beatnet_lite'`: lite-model activations feed the vectorized PF
    cascade, the PF tracks tempo as part of its (position-in-bar,
    beat-period) state space. The read at the end picks the median
    particle's beat period → BPM.

    Unlike `BeatNetTempoTracker` (which EMA-smooths the read for the
    daemon), this eval reads the raw median directly — we want to
    measure the PF's accuracy, not the post-smoothing accuracy. The
    EMA's job is jitter reduction, not accuracy.
    """

    name = 'beatnet_pf'

    def __init__(self) -> None:
        super().__init__()
        from flame_sheep_audio import SAMPLE_RATE, HOP_SIZE
        self._SAMPLE_RATE = SAMPLE_RATE
        self._HOP_SIZE = HOP_SIZE

        # Version: lite weights hash + PF cascade source hash so any
        # change to either invalidates the cache.
        import hashlib
        from flame_sheep.eval import beatnet_lite as _lite_mod
        from flame_sheep_audio import beat_detector_beatnet as _adapter_mod
        from flame_sheep_audio import _pf_fast as _pf_mod
        src = (Path(_lite_mod.__file__).read_text()
               + Path(_adapter_mod.__file__).read_text()
               + Path(_pf_mod.__file__).read_text())
        self.version = ('beatnet_pf_'
                        + hashlib.sha256(src.encode()).hexdigest()[:12])

    def estimate(self, audio: np.ndarray, sr: int) -> float:
        from flame_sheep_audio.beat_detector_beatnet import BeatNetLiveDetector

        audio = _ensure_mono_float32(audio)
        audio = _resample(audio, sr, self._SAMPLE_RATE)

        t0 = time.perf_counter()
        det = BeatNetLiveDetector(use_particle_filter=True)
        setup_elapsed = time.perf_counter() - t0

        t_stream = time.perf_counter()
        pos = 0
        n = len(audio)
        while pos + self._HOP_SIZE <= n:
            chunk = audio[pos:pos + self._HOP_SIZE]
            det.feed_audio_hop(chunk)
            det.detect(None)
            pos += self._HOP_SIZE
        stream_elapsed = time.perf_counter() - t_stream

        snap = det.current_tempo()
        bpm = float(snap['bpm']) if snap else 0.0
        confidence = float(snap['confidence']) if snap else 0.0

        self.last_timing = {
            'setup': setup_elapsed,
            'stream': stream_elapsed,
            # Confidence isn't latency but it's a useful per-track signal
            # for downstream consumers (low-confidence reads should be
            # treated as untrustworthy). Stuffed into timing so the
            # cache layer carries it without a separate field.
            'pf_confidence': confidence,
            'total': setup_elapsed + stream_elapsed,
        }
        return bpm


# ---------------------------------------------------------------------------
# madmom — tested with three decoder methods (comb / acf / dbn)
# ---------------------------------------------------------------------------

class MadmomTempoEstimator(TempoEstimator):
    """Wraps madmom.features.tempo.TempoEstimationProcessor.

    Pipeline: RNNBeatProcessor produces beat activations, then the
    TempoEstimationProcessor decodes them into a tempo histogram.
    Method names map to different decoders:
      - 'comb': comb-filter bank (default)
      - 'acf':  autocorrelation on the activation
      - 'dbn':  DBN decoder (same family as madmom's beat tracker)

    Returns the strongest BPM from the histogram. madmom is
    offline — the whole audio array is processed at once.
    """

    def __init__(self, method: str = 'comb', fps: int = 100,
                 min_bpm: float = 40.0, max_bpm: float = 250.0) -> None:
        super().__init__()
        self._method = method
        self._fps = fps
        self._min_bpm = min_bpm
        self._max_bpm = max_bpm
        self._rnn = None
        self._proc = None

        import madmom
        self.name = f'madmom_tempo_{method}'
        self.version = (
            f'madmom_{madmom.__version__}_{method}'
            f'_fps{fps}_bpm{int(min_bpm)}-{int(max_bpm)}')

    def _load(self) -> None:
        if self._rnn is not None:
            return
        from madmom.features.beats import RNNBeatProcessor
        from madmom.features.tempo import TempoEstimationProcessor
        self._rnn = RNNBeatProcessor()
        self._proc = TempoEstimationProcessor(
            method=self._method, min_bpm=self._min_bpm,
            max_bpm=self._max_bpm, fps=self._fps)

    def estimate(self, audio: np.ndarray, sr: int) -> float:
        self._load()
        audio = _ensure_mono_float32(audio)
        # madmom RNN is trained on 44100 Hz
        audio = _resample(audio, sr, 44100)

        t_rnn = time.perf_counter()
        activations = self._rnn(audio)
        rnn_elapsed = time.perf_counter() - t_rnn

        t_dec = time.perf_counter()
        tempi = self._proc(activations)
        dec_elapsed = time.perf_counter() - t_dec

        if len(tempi) == 0:
            bpm = 0.0
        else:
            # tempi is (N, 2): [(bpm, strength), ...] sorted by strength desc
            bpm = float(tempi[0][0])

        self.last_timing = {
            'rnn': rnn_elapsed,
            'decode': dec_elapsed,
            'total': rnn_elapsed + dec_elapsed,
        }
        return bpm
