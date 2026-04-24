"""Per-bin magnitude variance tracking for harmonic/percussive separation.

Tracks rolling variance of FFT magnitude per frequency bin. Bins with
low temporal variance contain sustained harmonic content (vocals, pads).
Bins with high variance contain transient percussive content (kicks, snares).

This is a lightweight alternative to full HPSS — no spectrogram buffering,
O(N_BINS) per frame, zero latency.
"""

import numpy as np

from ._constants import N_BINS


class MagnitudeStability:
    """Per-bin magnitude stability via EMA variance tracking.

    High stability = sustained/harmonic (magnitude barely changes).
    Low stability = transient/percussive (sharp onset, quick decay).
    """

    def __init__(self, alpha: float = 0.95):
        self._alpha = alpha
        self._mag_ema = np.zeros(N_BINS, dtype=np.float32)
        self._mag_var = np.zeros(N_BINS, dtype=np.float32)

    def update(self, magnitude: np.ndarray):
        """Update per-bin variance from new magnitude frame."""
        diff = magnitude - self._mag_ema
        self._mag_ema = self._alpha * self._mag_ema + (1 - self._alpha) * magnitude
        self._mag_var = self._alpha * self._mag_var + (1 - self._alpha) * diff * diff

    def band_stability(self, mask: np.ndarray) -> float:
        """Stability score for a frequency band.

        Args:
            mask: boolean array (N_BINS,) selecting bins in the band.

        Returns:
            0.0 = highly transient (percussive), 1.0 = perfectly stable (harmonic).
        """
        if not mask.any():
            return 1.0
        band_var = self._mag_var[mask].mean()
        band_mag = self._mag_ema[mask].mean()
        if band_mag < 1e-10:
            return 1.0  # silence is "stable"
        cv = float(np.sqrt(band_var) / (band_mag + 1e-10))
        return max(0.0, min(1.0, 1.0 - cv))

    def reset(self):
        """Clear all state."""
        self._mag_ema[:] = 0.0
        self._mag_var[:] = 0.0
