"""Energy analyzer — sub-bass RMS tracking for visceral loudness."""

import numpy as np

from ._constants import N_BINS
from ._bands import make_mask


class EnergyAnalyzer:
    """Track sub-bass energy as a proxy for perceived physical loudness.

    Low frequencies (20-200Hz) drive the "feel it in your chest" sensation.
    Uses EMA smoothing for frame-to-frame stability.
    """

    def __init__(self, lo: float = 20.0, hi: float = 200.0, alpha: float = 0.9):
        self._bins = make_mask(lo, hi)
        self._alpha = alpha
        self._rms = 0.0

    def update(self, spectrum: np.ndarray) -> float:
        """Update RMS from spectrum magnitude. Returns smoothed RMS."""
        raw_rms = float(np.sqrt(np.mean(spectrum[self._bins] ** 2)))
        self._rms = self._alpha * self._rms + (1 - self._alpha) * raw_rms
        return self._rms

    @property
    def rms(self) -> float:
        return self._rms
