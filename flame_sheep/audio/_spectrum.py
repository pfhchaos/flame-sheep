"""Spectrum computation engine — FFT, windowing, spectral flux."""

import numpy as np
from dataclasses import dataclass
from scipy.signal import windows

from ._constants import FFT_SIZE, N_BINS, SAMPLE_RATE


@dataclass
class SpectrumFrame:
    """Output of SpectrumEngine — computed once, consumed by all analyzers."""
    magnitude: np.ndarray    # (N_BINS,) float32, FFT magnitude spectrum
    flux: np.ndarray         # (N_BINS,) float32, half-wave rectified spectral flux
    waveform: np.ndarray     # (FFT_SIZE,) float32, raw PCM window


class SpectrumEngine:
    """Computes FFT magnitude spectrum and spectral flux from raw PCM.

    Call compute() once per audio frame. The resulting SpectrumFrame is
    shared by all downstream analyzers (beat detector, energy analyzer, etc.)
    so the FFT is only computed once.
    """

    def __init__(self):
        self._window = windows.hann(FFT_SIZE, sym=False).astype(np.float32)
        self._prev_spectrum: np.ndarray | None = None

    def compute(self, pcm: np.ndarray) -> SpectrumFrame:
        """Compute spectrum and flux from a PCM window.

        Args:
            pcm: float32 array of FFT_SIZE samples.

        Returns:
            SpectrumFrame with magnitude, flux, and waveform.
        """
        windowed = pcm * self._window
        magnitude = (np.abs(np.fft.rfft(windowed)) / FFT_SIZE).astype(np.float32)

        if self._prev_spectrum is not None:
            flux = np.maximum(magnitude - self._prev_spectrum, 0.0).astype(np.float32)
        else:
            flux = np.zeros(N_BINS, dtype=np.float32)

        self._prev_spectrum = magnitude.copy()

        return SpectrumFrame(
            magnitude=magnitude,
            flux=flux,
            waveform=pcm.copy(),
        )

    def reset(self):
        """Clear previous spectrum state."""
        self._prev_spectrum = None
