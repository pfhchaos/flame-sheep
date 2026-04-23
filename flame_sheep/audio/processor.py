"""
Audio capture, FFT, and beat detection.

Constants, types, and adaptive band logic live in the audio/ subpackage.
This module re-exports them for backward compatibility and contains the
AudioProcessor and SyntheticAudioProcessor classes.
"""

import logging

log = logging.getLogger(__name__)


import threading
import time
import numpy as np
import sounddevice as sd
from collections import deque
from scipy.signal import windows

# Import from sibling modules within audio package
from ._constants import (
    SAMPLE_RATE, DEFAULT_DEVICE, BLOCK_SIZE, FFT_SIZE, N_BINS, HISTORY_LEN,
    HOP_SIZE, FREQS,
)
from ._types import BeatEvent, AudioSnapshot
from ._spectrum import SpectrumEngine, SpectrumFrame
from .beat_detector import FluxBeatDetector
from .energy import EnergyAnalyzer
from .source import PipeWireSource, FeedSource
from ._bands import (
    AdaptiveBand, make_mask, make_weights, a_weight_curve, A_WEIGHTS,
    ALLOWED_RANGES, DEFAULT_RANGES,
    ADAPT_ALPHA, ADAPT_FAST_ALPHA, ADAPT_INTERVAL, ADAPT_ANCHOR,
    SECTION_THRESHOLD, FAST_ADAPT_FRAMES,
)

# Backward-compat aliases for underscore-prefixed names used by existing code
_FREQS = FREQS
_make_mask = make_mask
_make_weights = make_weights
_A_WEIGHTS = A_WEIGHTS
_ALLOWED_RANGES = ALLOWED_RANGES
_DEFAULT_RANGES = DEFAULT_RANGES
_ADAPT_ALPHA = ADAPT_ALPHA
_ADAPT_FAST_ALPHA = ADAPT_FAST_ALPHA
_ADAPT_INTERVAL = ADAPT_INTERVAL
_ADAPT_ANCHOR = ADAPT_ANCHOR
_SECTION_THRESHOLD = SECTION_THRESHOLD
_FAST_ADAPT_FRAMES = FAST_ADAPT_FRAMES


class AudioProcessor:
    """
    Orchestrates audio analysis: signal source -> spectrum -> detection + energy.

    Two modes determined by source type:
      - Threaded (PipeWireSource): audio analysis runs in a daemon thread at
        HOP_SIZE cadence. Render thread calls drain() to get accumulated state.
      - Synchronous (FeedSource): process() works as before for deterministic tests.
    """

    def __init__(self, device: str | int | None = None, adaptive: bool = False,
                 sharpness: bool = True, source=None):
        self._source = source or PipeWireSource(device=device)
        self._spectrum_engine = SpectrumEngine()
        self._detector = FluxBeatDetector(adaptive=adaptive, sharpness=sharpness)
        self._energy = EnergyAnalyzer()

        # Auto-detect: FeedSource is synchronous, everything else is threaded
        self._threaded = not isinstance(self._source, FeedSource)

        # Shared state (lock-protected, read by drain(), written by audio thread or process())
        self._lock     = threading.Lock()
        self._spectrum = np.zeros(N_BINS, dtype=np.float32)
        self._waveform = np.zeros(FFT_SIZE, dtype=np.float32)
        self._rms      = 0.0
        self._centroid = 1000.0
        self._centroid_delta = 0.0
        self._centroid_rms = 0.0
        self._percussiveness = 0.5
        self._band_rms = {'kick': 0.0, 'snare': 0.0, 'clap': 0.0, 'hihat': 0.0}
        self._pending_events: list[BeatEvent] = []

        # Thread state
        self._thread: threading.Thread | None = None
        self._running = False

    def reset_bands(self):
        """Reset adaptive bands to defaults. Call on song change."""
        self._detector.reset_bands()

    def start(self):
        self._source.start()
        if self._threaded:
            self._running = True
            self._thread = threading.Thread(
                target=self._audio_loop, daemon=True, name='audio-analysis')
            self._thread.start()
            log.info('Audio thread started (hop=%d, %.1fms)', HOP_SIZE,
                     HOP_SIZE / SAMPLE_RATE * 1000)

    def stop(self):
        self._running = False
        self._source.stop()  # unblocks read_hop via stop_event
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def feed(self, pcm: np.ndarray):
        """Feed PCM samples into the source buffer."""
        self._source.feed(pcm)

    # ------------------------------------------------------------------
    # Threaded mode: audio loop + drain
    # ------------------------------------------------------------------

    def _audio_loop(self):
        """Runs in daemon thread. Reads hops, analyses, publishes."""
        while self._running:
            hop = self._source.read_hop(HOP_SIZE)
            if hop is None:
                break  # source stopped

            frame = self._spectrum_engine.push_hop(hop)
            self._energy.update(frame.magnitude, frame.flux)
            events = self._detector.detect(frame)

            with self._lock:
                self._pending_events.extend(events)
                self._spectrum[:] = frame.magnitude
                self._waveform[:] = frame.waveform
                self._rms = self._energy.rms
                self._centroid = self._energy.centroid
                self._centroid_delta = self._energy.centroid_delta
                self._centroid_rms = self._energy.centroid_rms
                self._percussiveness = self._energy.percussiveness
                self._band_rms = self._energy.band_rms

    def drain(self) -> AudioSnapshot:
        """Atomically read and clear accumulated audio state.

        In threaded mode: returns events accumulated since last drain.
        In sync mode: calls process() once, wraps result in AudioSnapshot.
        """
        if not self._threaded:
            events = self.process()
            return AudioSnapshot(
                events=events,
                spectrum=self._spectrum.copy(),
                rms=self._rms,
                waveform=self._waveform.copy(),
                centroid=self._centroid,
                centroid_delta=self._centroid_delta,
                centroid_rms=self._centroid_rms,
                percussiveness=self._percussiveness,
                band_rms=self._band_rms.copy(),
            )

        with self._lock:
            snap = AudioSnapshot(
                events=self._pending_events,
                spectrum=self._spectrum.copy(),
                rms=self._rms,
                waveform=self._waveform.copy(),
                centroid=self._centroid,
                centroid_delta=self._centroid_delta,
                centroid_rms=self._centroid_rms,
                percussiveness=self._percussiveness,
                band_rms=self._band_rms.copy(),
            )
            self._pending_events = []
            return snap

    # ------------------------------------------------------------------
    # Synchronous mode: process (for FeedSource / tests)
    # ------------------------------------------------------------------

    def process(self) -> list[BeatEvent]:
        """Run one analysis frame synchronously. Returns beat events.

        For FeedSource/test use. In threaded mode, use drain() instead.
        """
        pcm = self._source.read()
        if pcm is None:
            return []

        frame = self._spectrum_engine.compute(pcm)
        self._energy.update(frame.magnitude, frame.flux)

        with self._lock:
            self._spectrum[:] = frame.magnitude
            self._waveform[:] = frame.waveform
            self._rms = self._energy.rms
            self._centroid = self._energy.centroid
            self._centroid_delta = self._energy.centroid_delta
            self._centroid_rms = self._energy.centroid_rms
            self._percussiveness = self._energy.percussiveness

        return self._detector.detect(frame)

    @property
    def spectrum(self) -> np.ndarray:
        """Latest FFT magnitude spectrum, N_BINS long. For GPU texture upload."""
        with self._lock:
            return self._spectrum.copy()

    @property
    def waveform(self) -> np.ndarray:
        """Latest raw PCM window. For GPU texture upload if desired."""
        with self._lock:
            return self._waveform.copy()

    @property
    def rms(self) -> float:
        """Smoothed broadband RMS of the input signal (0..1 range for typical audio)."""
        with self._lock:
            return self._rms


class SyntheticAudioProcessor:
    """
    Drop-in replacement for AudioProcessor that generates a predictable
    metronome signal instead of capturing real audio.

    Useful for visual/integration testing: you know exactly when beats
    should fire and can judge whether the app responds correctly.

    Pattern (all timings in seconds, relative to start()):
      kick  — every `kick_interval`  seconds  (default 0.5s = 120 bpm)
      snare — every `snare_interval` seconds  (default 1.0s, on the 2 and 4)
      hihat — every `hihat_interval` seconds  (default 0.25s = 8th notes)

    A fake spectrum is synthesised so the audio visualiser on the GPU
    (tonemap.frag pulse effect) still animates.
    """

    def __init__(
        self,
        kick_interval:  float = 0.5,
        snare_interval: float = 1.0,
        hihat_interval: float = 0.25,
        bpm_label:      str   = '120 bpm',
        clock=None,
    ):
        self.kick_interval  = kick_interval
        self.snare_interval = snare_interval
        self.hihat_interval = hihat_interval
        self.bpm_label      = bpm_label
        self._clock         = clock  # callable returning seconds, or None for perf_counter

        self._start_time: float | None = None
        self._last: dict[str, float]   = {'kick': -1.0, 'snare': -1.0, 'clap': -1.0, 'hihat': -1.0}
        self._spectrum = np.zeros(N_BINS, dtype=np.float32)
        self._rms      = 0.5  # synthetic audio is "always playing"

    def start(self):
        if self._clock is not None:
            self._start_time = self._clock()
        else:
            import time
            self._start_time = time.perf_counter()
        log.info(f'[synthetic audio] {self.bpm_label}  '
              f'kick={self.kick_interval:.2f}s  '
              f'snare={self.snare_interval:.2f}s  '
              f'hihat={self.hihat_interval:.2f}s')

    def stop(self):
        pass  # nothing to close

    def process(self) -> list[BeatEvent]:
        if self._start_time is None:
            return []

        if self._clock is not None:
            now = self._clock() - self._start_time
        else:
            import time
            now = time.perf_counter() - self._start_time
        events  = []
        self._spectrum[:] = 0.0

        for band, interval in [
            ('kick',  self.kick_interval),
            ('snare', self.snare_interval),
            ('hihat', self.hihat_interval),
        ]:
            # Fire when we cross a beat boundary since last call
            beat_num_now  = int(now / interval)
            beat_num_last = int(self._last[band] / interval) if self._last[band] >= 0 else -1
            if beat_num_now > beat_num_last:
                events.append(BeatEvent(kind=band, energy=1.0))
                self._last[band] = now
                # Inject energy into the matching spectrum region so the
                # GPU pulse effect fires visually too
                freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
                if band == 'kick':
                    mask = (freqs >= 50) & (freqs < 100)
                elif band == 'snare':
                    mask = (freqs >= 150) & (freqs < 800)
                else:
                    mask = freqs >= 8000
                self._spectrum[mask] = 1.0

        return events

    def drain(self) -> AudioSnapshot:
        """Wrap process() into AudioSnapshot for uniform API with AudioProcessor."""
        events = self.process()
        return AudioSnapshot(
            events=events,
            spectrum=self._spectrum.copy(),
            rms=self._rms,
            waveform=np.zeros(FFT_SIZE, dtype=np.float32),
        )

    @property
    def spectrum(self) -> np.ndarray:
        return self._spectrum.copy()

    @property
    def rms(self) -> float:
        return self._rms


def list_monitor_devices() -> list[dict]:
    """Helper: list available input devices, highlighting monitor sinks."""
    devices = []
    for i, d in enumerate(sd.query_devices()):
        if d['max_input_channels'] > 0:
            devices.append({'index': i, 'name': d['name'], 'device': d})
    return devices
