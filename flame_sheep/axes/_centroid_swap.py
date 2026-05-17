"""Centroid swap detector — triggers genome swap on mel centroid shift."""

from __future__ import annotations

import numpy as np

from flame_sheep.config import cfg
from flame_sheep_audio import SAMPLE_RATE
from flame_sheep_audio.response import MelCentroid, Delta


class CentroidSwap:
    """Tracks mel-space centroid delta and signals when a swap should trigger.

    Returns True from tick() when centroid delta exceeds threshold during
    low onset density. Does not mutate morph state — caller decides what to do.
    """

    def __init__(self) -> None:
        self._mel_centroid: MelCentroid | None = None
        self._mel_delta = Delta()

    def tick(self, spectrum: np.ndarray, low_density: float,
             morph_t: float) -> bool:
        """Update centroid tracking. Returns True if swap should trigger."""
        n_bins = len(spectrum)
        if n_bins == 0:
            return False

        # Lazy init / reinit on spectrum size change
        if self._mel_centroid is None or self._mel_centroid.n_bins != n_bins:
            freqs = np.linspace(0, SAMPLE_RATE / 2, n_bins)
            self._mel_centroid = MelCentroid(freqs)

        mel_centroid = self._mel_centroid.compute(spectrum)
        mel_delta = self._mel_delta.update(mel_centroid)

        return (
            low_density < cfg.genome.centroid_swap_density_gate
            and mel_delta > cfg.genome.centroid_swap_threshold
            and morph_t > 0.3
        )
