"""
Shared test helpers for flame-sheep visualizer tests.

Separated from conftest.py to avoid import collisions when
flame_sheep_audio/tests/ is also on sys.path.
"""

import numpy as np

from flame_sheep.genome import Genome, Transform, NUM_VARIATIONS, _random_palette


def trivial_genome(seed: int = 0) -> Genome:
    """Instant genome -- no viability check, no rejection sampling.
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


class FakeClock:
    """Injectable clock for deterministic testing. Advances manually."""
    def __init__(self, start: float = 1000.0):
        self.t = start
    def __call__(self) -> float:
        return self.t
    def advance(self, dt: float):
        self.t += dt
