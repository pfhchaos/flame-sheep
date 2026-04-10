"""
Audio capture, FFT, and beat detection.

Captures from PipeWire monitor sink (whatever is playing) via sounddevice
(which uses PortAudio, transparently bridged by PipeWire's PA compat layer).

Pipeline:
  sounddevice InputStream (callback) → circular PCM buffer
  → per-frame: Hann-windowed FFT → magnitude spectrum
  → RMS normalize spectrum (volume-invariant)
  → per-band spectral flux (frame-to-frame change, not absolute energy)
  → flux history → local average flux
  → onset = flux > THRESHOLD * local_avg_flux  (self-calibrating)

Using flux rather than absolute energy means:
  - Volume changes don't affect sensitivity (a quiet song and a loud song
    with the same beat structure produce the same flux pattern)
  - Works on any audio content, not just music
  - No hand-tuned absolute energy floors needed

Beat events emitted:
  'kick'  — bass flux onset   (50-100 Hz)
  'snare' — mid flux onset    (300-1000 Hz, confirmed by 1k-3kHz)
  'hihat' — treble flux onset (8kHz+)

Also exposes:
  .rms    — current broadband RMS (0..1), for autonomous drift when quiet
"""

import threading
import numpy as np
import sounddevice as sd
from collections import deque
from scipy.signal import windows
from dataclasses import dataclass


SAMPLE_RATE        = 48000
# Stable PipeWire monitor source name — use this instead of a device index
# which can change between sessions as PipeWire adds/removes nodes.
DEFAULT_DEVICE     = 'Companion Speaker Analog Surround 5.1'
BLOCK_SIZE    = 1024   # frames per callback
FFT_SIZE      = 2048   # FFT window (zero-padded if > BLOCK_SIZE)
N_BINS        = FFT_SIZE // 2 + 1
HISTORY_LEN   = 20     # ~0.33s of frames at 60fps — shorter = less averaging of sustained bass


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
        self._rms       = 0.0

        self._window = windows.hann(FFT_SIZE, sym=False).astype(np.float32)

        # Frequency bin ranges
        freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
        self._bands = {
            'kick':  (freqs >= 50)   & (freqs <  100),
            'snare': (freqs >= 300)  & (freqs <  1000),
            'hihat': (freqs >= 8000),
        }
        self._snare_confirm = (freqs >= 1000) & (freqs < 3000)
        self._all_bins      = np.ones(N_BINS, dtype=bool)  # for broadband RMS

        # Spectral flux history per band (flux = positive energy change vs prev frame)
        # Longer history than before — flux variance is lower so we need more context
        self._prev_spectrum: np.ndarray | None = None
        self._flux_history = {
            'kick':           deque(maxlen=HISTORY_LEN),
            'snare':          deque(maxlen=HISTORY_LEN),
            'hihat':          deque(maxlen=HISTORY_LEN),
            '_snare_confirm': deque(maxlen=HISTORY_LEN),
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
        Runs FFT, computes spectral flux, detects onsets.
        """
        with self._lock:
            if len(self._buffer) < FFT_SIZE:
                return []
            pcm = np.array(self._buffer, dtype=np.float32)

        windowed = pcm * self._window
        spectrum = (np.abs(np.fft.rfft(windowed)) / FFT_SIZE).astype(np.float32)

        # Broadband RMS of the raw PCM window, smoothed with a leaky integrator
        rms = float(np.sqrt(np.mean(pcm ** 2)))

        with self._lock:
            self._spectrum[:] = spectrum
            self._waveform[:] = pcm
            self._rms = self._rms * 0.9 + rms * 0.1

        events = self._detect_onsets(spectrum)
        return events

    def _detect_onsets(self, spectrum: np.ndarray) -> list[BeatEvent]:
        """
        Spectral flux onset detection.

        Flux for a band = mean of positive (spectrum - prev_spectrum) differences.
        This measures *increase* in energy, not absolute level — so it is
        inherently volume-normalised and self-adapting to any genre or content.

        An onset fires when flux > THRESHOLD * local_avg_flux AND the band
        is not in cooldown.
        """
        if self._prev_spectrum is None:
            self._prev_spectrum = spectrum.copy()
            return []

        # Half-wave rectified flux: only count increases, ignore decreases
        diff = spectrum - self._prev_spectrum
        flux = np.maximum(diff, 0.0)  # shape: (N_BINS,)
        self._prev_spectrum = spectrum.copy()

        events = []
        THRESHOLD = 1.5   # flux must exceed 1.5x local average flux to fire
        COOLDOWN  = 12    # frames of silence after an onset
        # Minimum flux floor — gates out truly silent signals.
        # Expressed as a fraction of typical flux: anything below this is
        # treated as silence regardless of ratio. Kept very small so it
        # only blocks DC offsets / numerical noise, not quiet audio.
        MIN_FLUX  = 1e-7

        for band, mask in self._bands.items():
            band_flux = float(flux[mask].mean())
            hist      = self._flux_history[band]

            self._frame_count[band] += 1
            in_cooldown = (self._frame_count[band] - self._cooldown_frames[band]) < COOLDOWN

            if len(hist) >= 10 and band_flux > MIN_FLUX and not in_cooldown:
                local_avg = float(np.mean(hist))

                # Snare: corroborate with 1k-3kHz flux to reject kick harmonics
                if band == 'snare':
                    confirm_flux = float(flux[self._snare_confirm].mean())
                    confirm_hist = self._flux_history['_snare_confirm']
                    self._flux_history['_snare_confirm'].append(confirm_flux)
                    confirm_avg  = float(np.mean(confirm_hist)) if len(confirm_hist) >= 5 else 0
                    if confirm_avg > 0 and confirm_flux < confirm_avg * THRESHOLD:
                        hist.append(band_flux)
                        continue

                if local_avg < MIN_FLUX:
                    # No flux history yet — first onset after silence
                    events.append(BeatEvent(kind=band, energy=1.0))
                    self._cooldown_frames[band] = self._frame_count[band]
                elif band_flux > local_avg * THRESHOLD:
                    normalized = min(1.0, (band_flux / local_avg - THRESHOLD) / THRESHOLD)
                    events.append(BeatEvent(kind=band, energy=normalized))
                    self._cooldown_frames[band] = self._frame_count[band]

            hist.append(band_flux)

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
    ):
        self.kick_interval  = kick_interval
        self.snare_interval = snare_interval
        self.hihat_interval = hihat_interval
        self.bpm_label      = bpm_label

        self._start_time: float | None = None
        self._last: dict[str, float]   = {'kick': -1.0, 'snare': -1.0, 'hihat': -1.0}
        self._spectrum = np.zeros(N_BINS, dtype=np.float32)
        self._rms      = 0.5  # synthetic audio is "always playing"

    def start(self):
        import time
        self._start_time = time.perf_counter()
        print(f'[synthetic audio] {self.bpm_label}  '
              f'kick={self.kick_interval:.2f}s  '
              f'snare={self.snare_interval:.2f}s  '
              f'hihat={self.hihat_interval:.2f}s')

    def stop(self):
        pass  # nothing to close

    def process(self) -> list[BeatEvent]:
        import time
        if self._start_time is None:
            return []

        now     = time.perf_counter() - self._start_time
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
