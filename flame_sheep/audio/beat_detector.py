"""Spectral flux beat detector — onset detection from FFT magnitude changes."""

import numpy as np
from collections import deque

from ._constants import SAMPLE_RATE, N_BINS, HISTORY_LEN, FREQS, FFT_SIZE, HOP_SIZE
from ._types import BeatEvent
from ._spectrum import SpectrumFrame
from ._bands import (
    AdaptiveBand, make_mask, BAND_MASKS,
    ADAPT_ALPHA, ADAPT_FAST_ALPHA, ADAPT_INTERVAL, ADAPT_ANCHOR,
    SECTION_THRESHOLD, FAST_ADAPT_FRAMES,
)


class FluxBeatDetector:
    """Detect beat onsets from spectral flux.

    Computes per-band mean flux (weighted or boolean-masked), compares
    to a rolling average, and fires a BeatEvent when flux exceeds the
    threshold. Supports adaptive band tracking.

    Usage:
        detector = FluxBeatDetector(adaptive=True)
        events = detector.detect(spectrum_frame)
    """

    # Detection constants
    THRESHOLD = 1.5       # flux must exceed this × local average to fire
    KICK_THRESHOLD = 3.5  # higher threshold for kick (2-bin band has high variance)
    COOLDOWN  = 12    # audio frames between onsets (~128ms at HOP_SIZE=512)
    KICK_COOLDOWN = 10 # shorter for kick to allow fast patterns (~107ms)
    STABILITY_SCALING = 2.0  # how much stability raises the kick threshold
    MIN_FLUX  = 1e-7  # gates out DC/numerical noise
    SHARPNESS = 3.0   # min flux ratio (current vs pre-attack) for snare/hihat
    SHARPNESS_LOOKBACK = FFT_SIZE // HOP_SIZE  # span the full overlap attack ramp

    def __init__(self, adaptive: bool = False, sharpness: bool = True,
                 stability=None):
        self._adaptive = adaptive
        self._sharpness = sharpness
        self._stability = stability  # MagnitudeStability reference (optional)

        # Static band masks (from shared definitions)
        self._bands = {k: BAND_MASKS[k] for k in ('kick', 'snare', 'clap', 'hihat')}
        self._snare_confirm = make_mask(1000, 3000)

        # Adaptive bands
        if adaptive:
            self._adaptive_bands = {
                'kick':           AdaptiveBand('kick'),
                'snare':          AdaptiveBand('snare'),
                'clap':           AdaptiveBand('clap'),
                'hihat':          AdaptiveBand('hihat'),
                '_snare_confirm': AdaptiveBand('_snare_confirm'),
            }
            self._adapt_frame = 0
            self._adapt_alpha = ADAPT_ALPHA
            self._fast_adapt_remaining = 0

        # Per-band flux history
        self._flux_history = {
            'kick':           deque(maxlen=HISTORY_LEN),
            'snare':          deque(maxlen=HISTORY_LEN),
            'clap':           deque(maxlen=HISTORY_LEN),
            'hihat':          deque(maxlen=HISTORY_LEN),
            '_snare_confirm': deque(maxlen=HISTORY_LEN),
        }

        # Per-band cooldown
        self._cooldown_frames = {'kick': 0, 'snare': 0, 'clap': 0, 'hihat': 0}
        self._frame_count     = {'kick': 0, 'snare': 0, 'clap': 0, 'hihat': 0}

        # Per-band recent flux (for attack sharpness lookback)
        self._recent_flux = {
            band: deque(maxlen=self.SHARPNESS_LOOKBACK + 1)
            for band in ('kick', 'snare', 'clap', 'hihat')
        }

    @property
    def adaptive_bands(self):
        """Access adaptive band state (for tests/inspection)."""
        return self._adaptive_bands if self._adaptive else None

    def reset_bands(self):
        """Reset adaptive bands to defaults. Call on song change."""
        if self._adaptive:
            for ab in self._adaptive_bands.values():
                ab.reset()
            self._adapt_alpha = ADAPT_ALPHA
            self._fast_adapt_remaining = 0

    def detect(self, frame: SpectrumFrame) -> list[BeatEvent]:
        """Detect beat onsets from a spectrum frame.

        Args:
            frame: SpectrumFrame with pre-computed flux.

        Returns:
            List of BeatEvent for detected onsets.
        """
        flux = frame.flux

        if self._adaptive:
            self._update_adaptive_bands(flux)

        events = []

        for band in self._bands:
            band_flux = self._band_flux(flux, band)
            hist = self._flux_history[band]
            recent = self._recent_flux[band]
            recent.append(band_flux)

            self._frame_count[band] += 1
            cd = self.KICK_COOLDOWN if band == 'kick' else self.COOLDOWN
            in_cooldown = (self._frame_count[band]
                           - self._cooldown_frames[band]) < cd

            if len(hist) >= 10 and band_flux > self.MIN_FLUX and not in_cooldown:
                local_avg = float(np.mean(hist))

                # Attack sharpness gate for snare/hihat:
                # compare to pre-attack baseline (median of lookback window)
                # to span the overlap attack ramp from 75% overlapping windows
                if (self._sharpness and band in ('kick', 'snare', 'clap', 'hihat')
                        and len(recent) > self.SHARPNESS_LOOKBACK):
                    pre_attack = float(np.median(list(recent)[:-1]))
                    if pre_attack > self.MIN_FLUX and band_flux / pre_attack < self.SHARPNESS:
                        hist.append(band_flux)
                        continue

                # Snare: corroborate with confirmation band
                if band == 'snare':
                    confirm_flux = self._confirm_flux(flux)
                    confirm_hist = self._flux_history['_snare_confirm']
                    self._flux_history['_snare_confirm'].append(confirm_flux)
                    confirm_avg = (float(np.mean(confirm_hist))
                                   if len(confirm_hist) >= 5 else 0)
                    if (confirm_avg > 0
                            and confirm_flux < confirm_avg * self.THRESHOLD):
                        hist.append(band_flux)
                        continue

                thresh = self.KICK_THRESHOLD if band == 'kick' else self.THRESHOLD
                # Scale threshold by band stability: stable (harmonic)
                # bands need a larger flux spike to fire
                if self._stability is not None:
                    stab = self._stability.band_stability(self._bands[band])
                    thresh *= (1.0 + stab * self.STABILITY_SCALING)
                if local_avg < self.MIN_FLUX:
                    events.append(BeatEvent(kind=band, energy=1.0))
                    self._cooldown_frames[band] = self._frame_count[band]
                elif band_flux > local_avg * thresh:
                    normalized = min(1.0,
                        (band_flux / local_avg - thresh) / thresh)
                    events.append(BeatEvent(kind=band, energy=normalized))
                    self._cooldown_frames[band] = self._frame_count[band]

            hist.append(band_flux)

        return events

    def _band_flux(self, flux: np.ndarray, band: str) -> float:
        """Compute weighted mean flux for a band."""
        if self._adaptive:
            w = self._adaptive_bands[band].weights
            s = w.sum()
            return float(np.dot(flux, w) / s) if s > 0 else 0.0
        else:
            mask = self._bands[band]
            return float(flux[mask].mean()) if mask.any() else 0.0

    def _confirm_flux(self, flux: np.ndarray) -> float:
        """Compute snare confirmation band flux."""
        if self._adaptive:
            w = self._adaptive_bands['_snare_confirm'].weights
            s = w.sum()
            return float(np.dot(flux, w) / s) if s > 0 else 0.0
        else:
            return float(flux[self._snare_confirm].mean())

    def _update_adaptive_bands(self, flux: np.ndarray):
        """EMA update of per-bin flux accumulators + periodic weight recompute."""
        alpha = self._adapt_alpha

        for ab in self._adaptive_bands.values():
            band_flux = flux * ab.allowed_mask
            ab.flux_accum = alpha * ab.flux_accum + (1 - alpha) * band_flux

        if self._fast_adapt_remaining > 0:
            self._fast_adapt_remaining -= 1
            if self._fast_adapt_remaining == 0:
                self._adapt_alpha = ADAPT_ALPHA

        self._adapt_frame += 1
        if self._adapt_frame < ADAPT_INTERVAL:
            return
        self._adapt_frame = 0

        for name in ('kick', 'snare', 'clap', 'hihat'):
            ab = self._adaptive_bands[name]
            raw = ab.flux_accum * ab.allowed_mask
            peak = raw.max()
            old_weights = ab.weights.copy()

            if peak > 1e-7:
                normalized = raw / peak
                ab.weights = ((1 - ADAPT_ANCHOR) * normalized
                              + ADAPT_ANCHOR * ab.default_weights)
                s = ab.weights.sum()
                if s > 0:
                    ab.weights /= s
            else:
                ab.weights = ab.default_weights.copy()

            shift = np.sum(np.abs(ab.weights - old_weights))
            if shift > SECTION_THRESHOLD:
                self._adapt_alpha = ADAPT_FAST_ALPHA
                self._fast_adapt_remaining = FAST_ADAPT_FRAMES

        # Snare confirm: derive from snare band's centroid
        snare_ab = self._adaptive_bands['snare']
        snare_centroid = np.average(FREQS, weights=snare_ab.weights + 1e-10)
        confirm_center = snare_centroid * 2.5
        confirm_sigma = confirm_center * 0.5
        confirm_ab = self._adaptive_bands['_snare_confirm']
        confirm_raw = np.exp(
            -0.5 * ((FREQS - confirm_center) / (confirm_sigma + 1e-10)) ** 2)
        confirm_raw *= confirm_ab.allowed_mask
        s = confirm_raw.sum()
        if s > 0:
            confirm_ab.weights = (confirm_raw / s).astype(np.float32)
        else:
            confirm_ab.weights = confirm_ab.default_weights.copy()
