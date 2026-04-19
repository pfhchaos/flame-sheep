"""
Tests for audio processing and beat detection.
Uses synthetic audio — no hardware or PipeWire required.
"""

import numpy as np
import pytest
from flame_sheep.audio import SAMPLE_RATE, FFT_SIZE, N_BINS

from .conftest import make_processor, make_sine, make_silence, feed_audio


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
        feed_audio(proc, tone)
        first = proc.process()
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

    def test_silence_no_flux(self):
        """Constant signal (zero flux) should never trigger beats."""
        proc = make_processor()
        tone = make_sine(80, FFT_SIZE, amplitude=0.5)
        self._warm_up(proc, tone)
        for _ in range(10):
            feed_audio(proc, tone)
            events = proc.process()
            assert len(events) == 0, \
                f"constant tone should have zero flux, got {events}"
