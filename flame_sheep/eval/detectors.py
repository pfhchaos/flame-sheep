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
        # Same fallback ladder as flame_sheep_audio.processor.AudioProcessor:
        # prefer CQT (if librosa+numba available), else OctaveBank, else FFT.
        try:
            from flame_sheep_audio._cqt_engine import CqtEngine
            return CqtEngine()
        except ImportError:
            try:
                from flame_sheep_audio._octave_bank import OctaveBankEngine
                return OctaveBankEngine()
            except ImportError:
                from flame_sheep_audio._spectrum import SpectrumEngine
                return SpectrumEngine()

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
