"""
Audio capture, FFT, and beat detection.

Captures from PipeWire monitor sink (whatever is playing) via sounddevice
(which uses PortAudio, transparently bridged by PipeWire's PA compat layer).

Pipeline:
  sounddevice InputStream (callback) → circular PCM buffer
  → per-frame: Hann-windowed FFT → magnitude spectrum
  → multi-band onset detection → beat events

Beat events emitted:
  'kick'  — bass onset  (20-150 Hz)
  'snare' — mid onset   (150-800 Hz)
  'hihat' — treble onset (8kHz+)
"""

import threading
import numpy as np
import sounddevice as sd
from collections import deque
from scipy.signal import windows
from dataclasses import dataclass


SAMPLE_RATE   = 44100
BLOCK_SIZE    = 1024   # frames per callback
FFT_SIZE      = 2048   # FFT window (zero-padded if > BLOCK_SIZE)
N_BINS        = FFT_SIZE // 2 + 1
HISTORY_LEN   = 43     # ~0.5s of frames at 60fps for local average


@dataclass
class BeatEvent:
    kind: str       # 'kick' | 'snare' | 'hihat'
    energy: float   # normalized 0..1, how strong the onset was


class AudioProcessor:
    """
    Owns the sounddevice stream and does all audio analysis.
    Thread-safe: stream callback writes, main thread reads.
    """

    def __init__(self, device: str | int | None = None):
        self._lock      = threading.Lock()
        self._buffer    = deque(maxlen=FFT_SIZE)  # circular PCM buffer (mono, float32)
        self._spectrum  = np.zeros(N_BINS,  dtype=np.float32)
        self._waveform  = np.zeros(FFT_SIZE, dtype=np.float32)
        self._events: list[BeatEvent] = []

        # Per-band energy history for onset detection
        self._history = {
            'kick':  deque(maxlen=HISTORY_LEN),
            'snare': deque(maxlen=HISTORY_LEN),
            'hihat': deque(maxlen=HISTORY_LEN),
        }

        self._window = windows.hann(FFT_SIZE, sym=False).astype(np.float32)

        # Frequency bin ranges for each band
        freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
        self._bands = {
            'kick':  (freqs >= 50)   & (freqs <  100),
            'snare': (freqs >= 150)  & (freqs <  800),
            'hihat': (freqs >= 8000),
        }

        # Per-band cooldown: frame counter since last onset
        self._cooldown_frames = {'kick': 0, 'snare': 0, 'hihat': 0}
        self._frame_count     = {'kick': 0, 'snare': 0, 'hihat': 0}

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            dtype='float32',
            device=device,
            callback=self._callback,
        )

    def start(self):
        self._stream.start()

    def stop(self):
        self._stream.stop()
        self._stream.close()

    def _callback(self, indata: np.ndarray, frames: int, time, status):
        """Called by sounddevice on audio thread. Must be fast."""
        mono = indata[:, 0]
        with self._lock:
            self._buffer.extend(mono)

    def process(self) -> list[BeatEvent]:
        """
        Call once per frame from the main/render thread.
        Runs FFT on latest buffer, detects onsets, returns beat events.
        Clears event list after returning.
        """
        with self._lock:
            if len(self._buffer) < FFT_SIZE:
                return []
            pcm = np.array(self._buffer, dtype=np.float32)

        windowed  = pcm * self._window
        spectrum  = (np.abs(np.fft.rfft(windowed)) / FFT_SIZE).astype(np.float32)

        with self._lock:
            self._spectrum[:] = spectrum
            self._waveform[:] = pcm

        events = self._detect_onsets(spectrum)
        return events

    def _detect_onsets(self, spectrum: np.ndarray) -> list[BeatEvent]:
        """
        Simple local-average onset detection per band.
        An onset fires when current energy exceeds local mean by threshold
        AND the band is not in cooldown (frame counter since last onset).
        """
        events = []
        THRESHOLD  = 2.5   # current must be > 2.5x local average
        # Minimum absolute energy floor — ignore noise below this level
        MIN_ENERGY = {'kick': 0.02, 'snare': 0.01, 'hihat': 0.005}
        # Don't fire same band twice within N frames
        COOLDOWN   = 20

        for band, mask in self._bands.items():
            energy = float(spectrum[mask].mean())
            hist   = self._history[band]

            self._frame_count[band] += 1
            in_cooldown = (self._frame_count[band] - self._cooldown_frames[band]) < COOLDOWN

            if len(hist) >= 10 and energy > MIN_ENERGY[band] and not in_cooldown:
                local_avg = float(np.mean(hist))
                # If local average is near-zero (e.g. silence warmup), treat any
                # energy above floor as an onset at max strength.
                if local_avg < MIN_ENERGY[band]:
                    events.append(BeatEvent(kind=band, energy=1.0))
                    self._cooldown_frames[band] = self._frame_count[band]
                elif energy > local_avg * THRESHOLD:
                    normalized = min(1.0, (energy / local_avg - THRESHOLD) / THRESHOLD)
                    events.append(BeatEvent(kind=band, energy=normalized))
                    self._cooldown_frames[band] = self._frame_count[band]

            hist.append(energy)

        return events

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


def list_monitor_devices() -> list[dict]:
    """Helper: list available input devices, highlighting monitor sinks."""
    devices = []
    for i, d in enumerate(sd.query_devices()):
        if d['max_input_channels'] > 0:
            devices.append({'index': i, 'name': d['name'], 'device': d})
    return devices
