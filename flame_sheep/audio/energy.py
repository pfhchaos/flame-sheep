"""Energy analyzer — sub-bass RMS, spectral centroid, percussiveness."""

import numpy as np

from ._constants import N_BINS, FREQS, SAMPLE_RATE
from ._bands import make_mask, A_WEIGHTS


class EnergyAnalyzer:
    """Track audio energy features for visual response.

    Provides:
      - rms: sub-bass energy (20-200Hz) — visceral loudness
      - centroid: spectral center of mass — bright vs dark
      - centroid_rms: energy around the centroid — how present the dominant voice is
      - percussiveness: flux/magnitude ratio — drums vs sustain
    """

    def __init__(self, lo: float = 20.0, hi: float = 200.0, alpha: float = 0.9):
        self._bins = make_mask(lo, hi)
        self._alpha = alpha
        self._rms = 0.0

        # Per-band RMS (matches beat detection bands)
        self._band_masks = {
            'kick':  make_mask(50, 100),
            'snare': make_mask(300, 1000),
            'clap':  make_mask(1000, 8000),
            'hihat': make_mask(8000, SAMPLE_RATE / 2),
        }
        self._band_rms = {k: 0.0 for k in self._band_masks}
        self._band_alpha = 0.9

        # Centroid tracking
        self._centroid = 1000.0  # Hz, start at a reasonable default
        self._prev_centroid = 1000.0
        self._centroid_rms = 0.0
        self._centroid_alpha = 0.85  # faster than RMS — follow tonal changes

        # Percussiveness tracking
        self._percussiveness = 0.5
        self._perc_alpha = 0.92  # smooth but responsive

    def update(self, spectrum: np.ndarray, flux: np.ndarray | None = None) -> float:
        """Update all features from spectrum magnitude and flux.

        Args:
            spectrum: FFT magnitude (N_BINS,)
            flux: spectral flux (N_BINS,), optional — needed for percussiveness

        Returns:
            Smoothed sub-bass RMS.
        """
        # Sub-bass RMS
        raw_rms = float(np.sqrt(np.mean(spectrum[self._bins] ** 2)))
        self._rms = self._alpha * self._rms + (1 - self._alpha) * raw_rms

        # Per-band RMS
        for band, mask in self._band_masks.items():
            if mask.any():
                raw = float(np.sqrt(np.mean(spectrum[mask] ** 2)))
                self._band_rms[band] = (self._band_alpha * self._band_rms[band]
                                        + (1 - self._band_alpha) * raw)

        # Spectral centroid (A-weighted for perceptual accuracy)
        weighted_spec = spectrum * A_WEIGHTS
        mag_sum = weighted_spec.sum()
        if mag_sum > 1e-10:
            raw_centroid = float(np.sum(FREQS * weighted_spec) / mag_sum)
            self._prev_centroid = self._centroid
            self._centroid = (self._centroid_alpha * self._centroid
                              + (1 - self._centroid_alpha) * raw_centroid)

            # RMS around centroid (±1 octave)
            lo_c = self._centroid / 2
            hi_c = self._centroid * 2
            centroid_mask = (FREQS >= lo_c) & (FREQS <= hi_c)
            if centroid_mask.any():
                raw_c_rms = float(np.sqrt(np.mean(spectrum[centroid_mask] ** 2)))
                self._centroid_rms = (self._centroid_alpha * self._centroid_rms
                                      + (1 - self._centroid_alpha) * raw_c_rms)

        # Percussiveness: flux / magnitude ratio
        if flux is not None and mag_sum > 1e-10:
            raw_perc = float(flux.sum() / mag_sum)
            self._percussiveness = (self._perc_alpha * self._percussiveness
                                     + (1 - self._perc_alpha) * raw_perc)

        return self._rms

    @property
    def rms(self) -> float:
        return self._rms

    @property
    def centroid(self) -> float:
        """Spectral centroid in Hz — where the energy center of mass is."""
        return self._centroid

    @property
    def centroid_delta(self) -> float:
        """Absolute change in centroid since last frame (Hz)."""
        return abs(self._centroid - self._prev_centroid)

    @property
    def centroid_rms(self) -> float:
        """RMS energy around the centroid (±1 octave)."""
        return self._centroid_rms

    @property
    def band_rms(self) -> dict[str, float]:
        """Per-band RMS: kick (50-100Hz), snare (300-1kHz), clap (1-8kHz), hihat (8kHz+)."""
        return dict(self._band_rms)

    @property
    def percussiveness(self) -> float:
        """Flux/magnitude ratio — high = drums/transients, low = sustained tonal."""
        return self._percussiveness
