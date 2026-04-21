"""
Shared test fixtures and helpers for flame-sheep tests.
"""

import numpy as np
import pytest

from flame_sheep.audio import AudioProcessor, SAMPLE_RATE, FFT_SIZE, N_BINS, HISTORY_LEN
from flame_sheep.audio.source import FeedSource
from flame_sheep.genome import Genome, Transform, NUM_VARIATIONS, _random_palette


# -------------------------------------------------------------------
# Audio processor factories
# -------------------------------------------------------------------

def make_processor(adaptive: bool = False, sharpness: bool = True) -> AudioProcessor:
    """Create an AudioProcessor with a FeedSource (no audio hardware)."""
    return AudioProcessor(source=FeedSource(), adaptive=adaptive, sharpness=sharpness)


# -------------------------------------------------------------------
# Trivial genome factory (instant, no viability checks)
# -------------------------------------------------------------------

def trivial_genome(seed: int = 0) -> Genome:
    """Instant genome — no viability check, no rejection sampling.
    A simple Sierpinski triangle variant that's always viable."""
    rng = np.random.default_rng(seed)
    g = Genome()
    for _ in range(3):
        t = Transform()
        scale = rng.uniform(0.3, 0.5)
        t.affine = np.array([scale, 0, rng.uniform(-0.5, 0.5),
                             0, scale, rng.uniform(-0.5, 0.5)], dtype=np.float32)
        t.variations = np.zeros(NUM_VARIATIONS, dtype=np.float32)
        t.variations[0] = 1.0  # linear only
        t.color = float(rng.uniform(0, 1))
        t.weight = 1.0
        g.transforms.append(t)
    g.palette = _random_palette(rng)
    g.zoom = 1.0
    g.rotation = 0.0
    g.center = np.zeros(2, dtype=np.float32)
    return g


# -------------------------------------------------------------------
# Fake clock for deterministic timing
# -------------------------------------------------------------------

class FakeClock:
    """Injectable clock for deterministic testing. Advances manually."""
    def __init__(self, start: float = 1000.0):
        self.t = start
    def __call__(self) -> float:
        return self.t
    def advance(self, dt: float):
        self.t += dt


# -------------------------------------------------------------------
# PCM helpers
# -------------------------------------------------------------------

def make_sine(freq: float, duration_samples: int, amplitude: float = 0.5) -> np.ndarray:
    """Generate a pure sine wave at given frequency."""
    t = np.arange(duration_samples) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_silence(duration_samples: int) -> np.ndarray:
    return np.zeros(duration_samples, dtype=np.float32)


def make_impulse(duration_samples: int, amplitude: float = 1.0) -> np.ndarray:
    """Single-sample impulse — broadband onset."""
    sig = np.zeros(duration_samples, dtype=np.float32)
    sig[duration_samples // 2] = amplitude
    return sig


def feed_audio(processor: AudioProcessor, signal: np.ndarray):
    """Feed signal into processor buffer."""
    processor.feed(signal)
