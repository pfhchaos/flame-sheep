"""
Tests for audio processing and beat detection.
Uses synthetic audio — no hardware or PipeWire required.
"""

import numpy as np
import pytest
from flame_sheep.audio import AudioProcessor, SAMPLE_RATE, FFT_SIZE, N_BINS


def make_sine(freq: float, duration_samples: int, amplitude: float = 0.5) -> np.ndarray:
    """Generate a pure sine wave at given frequency."""
    t = np.arange(duration_samples) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_impulse(duration_samples: int, amplitude: float = 1.0) -> np.ndarray:
    """Single-sample impulse — broadband onset."""
    sig = np.zeros(duration_samples, dtype=np.float32)
    sig[duration_samples // 2] = amplitude
    return sig


def make_silence(duration_samples: int) -> np.ndarray:
    return np.zeros(duration_samples, dtype=np.float32)


def feed_audio(processor: AudioProcessor, signal: np.ndarray):
    """Feed signal directly into processor buffer, bypassing sounddevice."""
    with processor._lock:
        processor._buffer.extend(signal)


def make_processor() -> AudioProcessor:
    """Create an AudioProcessor without starting the audio stream."""
    proc = AudioProcessor.__new__(AudioProcessor)
    import threading
    from collections import deque
    from scipy.signal import windows as scipy_windows
    proc._lock     = threading.Lock()
    proc._buffer   = deque(maxlen=FFT_SIZE)
    proc._spectrum = np.zeros(N_BINS, dtype=np.float32)
    proc._waveform = np.zeros(FFT_SIZE, dtype=np.float32)
    proc._events   = []
    proc._history  = {
        'kick':  deque(maxlen=43),
        'snare': deque(maxlen=43),
        'hihat': deque(maxlen=43),
    }
    proc._cooldown_frames = {'kick': 0, 'snare': 0, 'hihat': 0}
    proc._frame_count     = {'kick': 0, 'snare': 0, 'hihat': 0}
    proc._window = scipy_windows.hann(FFT_SIZE, sym=False).astype(np.float32)
    freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
    proc._bands = {
        'kick':  (freqs >= 50)  & (freqs <  100),
        'snare': (freqs >= 150) & (freqs <  800),
        'hihat': (freqs >= 8000),
    }
    return proc


# ----------------------------------------------------------------
# FFT / spectrum
# ----------------------------------------------------------------

class TestSpectrum:

    def test_silence_produces_low_spectrum(self):
        proc = make_processor()
        feed_audio(proc, make_silence(FFT_SIZE))
        proc.process()
        assert proc.spectrum.max() < 1.0

    def test_sine_peaks_at_correct_bin(self):
        """A 100Hz sine should produce a peak in the kick band."""
        proc = make_processor()
        feed_audio(proc, make_sine(100, FFT_SIZE * 4))
        proc.process()
        freqs   = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
        kick_mask = (freqs >= 20) & (freqs < 150)
        snare_mask = (freqs >= 150) & (freqs < 800)
        kick_energy  = proc.spectrum[kick_mask].max()
        snare_energy = proc.spectrum[snare_mask].max()
        assert kick_energy > snare_energy * 5, \
            f"100Hz sine should dominate kick band: kick={kick_energy:.1f} snare={snare_energy:.1f}"

    def test_high_freq_sine_peaks_in_hihat(self):
        """A 10kHz sine should produce a peak in the hihat band."""
        proc = make_processor()
        feed_audio(proc, make_sine(10000, FFT_SIZE * 4))
        proc.process()
        freqs      = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
        hihat_mask = freqs >= 8000
        kick_mask  = (freqs >= 20) & (freqs < 150)
        hihat_energy = proc.spectrum[hihat_mask].max()
        kick_energy  = proc.spectrum[kick_mask].max()
        assert hihat_energy > kick_energy * 5, \
            f"10kHz sine should dominate hihat band"

    def test_spectrum_length(self):
        proc = make_processor()
        feed_audio(proc, make_silence(FFT_SIZE))
        proc.process()
        assert len(proc.spectrum) == N_BINS


# ----------------------------------------------------------------
# Beat detection
# ----------------------------------------------------------------

class TestBeatDetection:

    def _warm_up(self, proc, signal):
        """Feed enough history that onset detection has a baseline."""
        for _ in range(50):
            feed_audio(proc, signal)
            proc.process()

    def test_silence_no_beats(self):
        proc = make_processor()
        self._warm_up(proc, make_silence(FFT_SIZE))
        events = proc.process()
        assert len(events) == 0, f"silence should produce no beats, got {events}"

    def test_sustained_tone_no_repeated_beats(self):
        """Sustained tone fires once then cooldown prevents re-firing."""
        proc  = make_processor()
        tone  = make_sine(80, FFT_SIZE, amplitude=0.8)
        self._warm_up(proc, make_silence(FFT_SIZE))
        # First hit on the tone
        feed_audio(proc, tone)
        first = proc.process()
        # Subsequent frames with same tone should not re-fire
        subsequent_hits = 0
        for _ in range(10):
            feed_audio(proc, tone)
            events = proc.process()
            subsequent_hits += len(events)
        assert subsequent_hits == 0, \
            f"sustained tone should not re-trigger after cooldown, got {subsequent_hits} hits"

    def test_kick_frequency_triggers_kick_band(self):
        """Strong 80Hz onset should trigger kick, not snare or hihat."""
        proc      = make_processor()
        silence   = make_silence(FFT_SIZE)
        kick_tone = make_sine(80, FFT_SIZE, amplitude=0.9)
        self._warm_up(proc, silence)
        feed_audio(proc, kick_tone)
        events = proc.process()
        kinds  = [e.kind for e in events]
        assert 'kick' in kinds, f"80Hz onset should trigger kick, got {kinds}"
        assert 'hihat' not in kinds, f"80Hz should not trigger hihat, got {kinds}"

    def test_beat_energy_normalized(self):
        """Beat energy should be in 0..1 range."""
        proc      = make_processor()
        silence   = make_silence(FFT_SIZE)
        kick_tone = make_sine(80, FFT_SIZE, amplitude=0.9)
        self._warm_up(proc, silence)
        feed_audio(proc, kick_tone)
        events = proc.process()
        for e in events:
            assert 0.0 <= e.energy <= 1.0, f"energy {e.energy} out of range"

    def test_minimum_energy_floor_blocks_noise(self):
        """Very quiet signal below energy floor should not trigger beats."""
        proc  = make_processor()
        # Generate a very quiet kick-frequency signal
        quiet = make_sine(80, FFT_SIZE, amplitude=0.001)
        self._warm_up(proc, quiet)
        feed_audio(proc, quiet * 3)  # 3x but still tiny
        events = proc.process()
        assert len(events) == 0, \
            f"sub-floor signal should not trigger beats, got {events}"
