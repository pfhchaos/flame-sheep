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
    'kick':           (30, 200),
    'snare':          (200, 1000),
    'clap':           (1000, 8000),
    'hihat':          (8000, SAMPLE_RATE / 2),
    '_snare_confirm': (1000, 3000),
}

# All analysis bands including sub-bass (shared by all analyzers)
BAND_RANGES = {
    'subbass': (20, 200),
    'kick':    (30, 200),
    'snare':   (200, 1000),
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


class SpringBand:
    """Frequency band that drifts toward percussive energy via spring physics.

    Center frequency is pulled toward the stability-weighted flux centroid
    within the allowed range, pulled back to default by an anchor spring,
    and repelled by neighboring bands. Width can also adapt.

    Update at ~10/s (every ADAPT_INTERVAL frames), not every frame.
    """

    __slots__ = ('name', 'center', 'width', 'default_center', 'default_width',
                 'lo_allowed', 'hi_allowed', 'mask', '_flux_ema')

    def __init__(self, name: str):
        self.name = name
        lo_a, hi_a = ALLOWED_RANGES[name]
        lo_d, hi_d = DEFAULT_RANGES[name]
        self.lo_allowed = lo_a
        self.hi_allowed = hi_a
        self.default_center = (lo_d + hi_d) / 2.0
        self.default_width = (hi_d - lo_d) / 2.0
        self.center = self.default_center
        self.width = self.default_width
        self.mask = make_mask(lo_d, hi_d)
        self._flux_ema = np.zeros(N_BINS, dtype=np.float32)

    def update_flux_ema(self, flux: np.ndarray, stability: np.ndarray,
                        alpha: float = 0.95):
        """Accumulate stability-weighted flux (percussive energy only)."""
        # Weight flux by (1-stability) so transient bins dominate
        percussive_flux = flux * (1.0 - stability)
        allowed = percussive_flux * self._allowed_mask_f()
        self._flux_ema = alpha * self._flux_ema + (1 - alpha) * allowed

    def flux_centroid(self) -> float:
        """Centroid of accumulated flux within allowed range."""
        total = self._flux_ema.sum()
        if total < 1e-10:
            return self.default_center
        return float(np.dot(FREQS, self._flux_ema) / total)

    def apply_forces(self, anchor_k: float, flux_k: float,
                     neighbors: list['SpringBand'], repulsion_k: float):
        """Update center position from spring forces.

        Args:
            anchor_k: spring constant pulling toward default center
            flux_k: spring constant pulling toward flux centroid
            neighbors: adjacent bands for repulsion
            repulsion_k: repulsive force strength
        """
        # Force 1: anchor spring — pull toward default
        f_anchor = anchor_k * (self.default_center - self.center)

        # Force 2: flux pull — pull toward percussive energy centroid
        target = self.flux_centroid()
        f_flux = flux_k * (target - self.center)

        # Force 3: repulsion from neighbors
        f_repulsion = 0.0
        for nb in neighbors:
            dist = self.center - nb.center
            if abs(dist) < 1e-6:
                dist = 1.0  # avoid division by zero
            # Repulsive force inversely proportional to distance
            f_repulsion += repulsion_k / dist

        # Apply net force (overdamped — no velocity, just position update)
        self.center += f_anchor + f_flux + f_repulsion

        # Clamp to allowed range
        half = self.width
        self.center = max(self.lo_allowed + half,
                          min(self.hi_allowed - half, self.center))

        # Rebuild mask from new center/width
        lo = max(self.lo_allowed, self.center - half)
        hi = min(self.hi_allowed, self.center + half)
        self.mask = make_mask(lo, hi)

    def _allowed_mask_f(self) -> np.ndarray:
        """Float mask for allowed frequency range."""
        return ((FREQS >= self.lo_allowed) & (FREQS < self.hi_allowed)).astype(np.float32)

    def reset(self):
        self.center = self.default_center
        self.width = self.default_width
        self.mask = make_mask(self.default_center - self.default_width,
                              self.default_center + self.default_width)
        self._flux_ema[:] = 0.0


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
