"""Adaptive spectral band definitions and utilities."""

import numpy as np
from ._constants import SAMPLE_RATE, N_BINS, FREQS


# Allowed ranges: hard clamps beyond which a band can never drift
ALLOWED_RANGES = {
    'kick':           (25, 150),
    'snare':          (150, 2000),
    'clap':           (800, 8000),
    'hihat':          (5000, SAMPLE_RATE / 2),
    '_snare_confirm': (800, 5000),
}

# Default (static) ranges: what the bands start at and anchor toward
DEFAULT_RANGES = {
    'kick':           (50, 100),
    'snare':          (300, 1000),
    'clap':           (1000, 8000),
    'hihat':          (8000, SAMPLE_RATE / 2),
    '_snare_confirm': (1000, 3000),
}

# All analysis bands including sub-bass (shared by all analyzers)
BAND_RANGES = {
    'subbass': (20, 200),
    'kick':    (50, 100),
    'snare':   (300, 1000),
    'clap':    (1000, 8000),
    'hihat':   (8000, SAMPLE_RATE / 2),
}

# Detection bands (subset used by beat detector + onset density)
DETECTION_BANDS = ('kick', 'snare', 'clap', 'hihat')

# Adaptation constants
ADAPT_ALPHA        = 0.98   # EMA decay per frame (~0.6s half-life at 60fps)
ADAPT_FAST_ALPHA   = 0.90   # faster decay during section changes
ADAPT_INTERVAL     = 30     # recompute weights every N frames (~0.5s)
ADAPT_ANCHOR       = 0.3    # fraction of default weights to blend in (anchoring)
SECTION_THRESHOLD  = 0.3    # weight shift that triggers fast-adapt mode
FAST_ADAPT_FRAMES  = 60     # how long fast-adapt lasts after section change


def make_mask(lo: float, hi: float) -> np.ndarray:
    """Boolean mask selecting FFT bins in [lo, hi) Hz."""
    return (FREQS >= lo) & (FREQS < hi)


def make_weights(lo: float, hi: float) -> np.ndarray:
    """Uniform weights within [lo, hi), zero outside, normalized to sum=1."""
    w = np.where((FREQS >= lo) & (FREQS < hi), 1.0, 0.0).astype(np.float32)
    s = w.sum()
    if s > 0:
        w /= s
    return w


def a_weight_curve(freqs: np.ndarray) -> np.ndarray:
    """A-weighting curve: perceptual loudness weights per frequency bin.

    Based on IEC 61672:2003. Returns linear gain (not dB) normalized
    so 1kHz = 1.0.
    """
    f2 = np.maximum(freqs, 1e-6) ** 2
    num = 12194**2 * f2**2
    den = ((f2 + 20.6**2)
           * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2))
           * (f2 + 12194**2))
    ra = num / (den + 1e-20)
    f1k = 1000.0**2
    num_1k = 12194**2 * f1k**2
    den_1k = ((f1k + 20.6**2)
              * np.sqrt((f1k + 107.7**2) * (f1k + 737.9**2))
              * (f1k + 12194**2))
    ra_1k = num_1k / den_1k
    return (ra / ra_1k).astype(np.float32)


# Precomputed A-weighting for our FFT bins
A_WEIGHTS = a_weight_curve(FREQS)

# Precomputed masks for all analysis bands (must be after make_mask definition)
BAND_MASKS = {name: make_mask(*rng) for name, rng in BAND_RANGES.items()}


class AdaptiveBand:
    """Soft-weighted frequency band that adapts to onset flux distribution."""

    __slots__ = ('weights', 'flux_accum', 'allowed_mask', 'default_weights',
                 'name')

    def __init__(self, name: str):
        self.name = name
        lo_a, hi_a = ALLOWED_RANGES[name]
        lo_d, hi_d = DEFAULT_RANGES[name]
        self.allowed_mask    = make_mask(lo_a, hi_a)
        self.default_weights = make_weights(lo_d, hi_d)
        self.weights         = self.default_weights.copy()
        self.flux_accum      = np.zeros(N_BINS, dtype=np.float32)

    def reset(self):
        self.weights = self.default_weights.copy()
        self.flux_accum[:] = 0.0
