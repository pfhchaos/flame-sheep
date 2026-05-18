"""Detail axis — maps audio energy to chaos game iteration count.

Uses the slow-envelope harmonic RMS from the audio engine so that
sustained musical energy drives detail, not individual hits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flame_sheep_audio import AudioState, HOP_SIZE, SAMPLE_RATE
from flame_sheep_audio.response import AsymmetricEnvelope
from flame_sheep.role_mapper import RoleMapper, ENERGY
from flame_sheep.renderer import LIVE_ITER_MIN, LIVE_ITER_MAX

if TYPE_CHECKING:
    from flame_sheep.core import FlameSheepCore

HOP_TIME = HOP_SIZE / SAMPLE_RATE


class DetailAxis:
    """Energy -> iterations. Quiet=sparse, loud=dense and detailed.

    Uses an asymmetric envelope (fast attack, slow release) on raw centroid
    RMS so sustained musical energy drives detail, not individual hits.
    """

    def __init__(self, role: RoleMapper,
                 min_iters: int = LIVE_ITER_MIN,
                 max_iters: int = LIVE_ITER_MAX,
                 rms_scale: float = 0.23) -> None:
        self.enabled = True
        self._role = role
        self.min_iters = min_iters
        self.max_iters = max_iters
        self.rms_scale = rms_scale
        self.iterations = min_iters
        self._envelope = AsymmetricEnvelope(attack=0.1, release=2.0, hop_time=HOP_TIME)

    def tick(self, audio: AudioState, dt: float, clock: float) -> None:
        energy = self._envelope.update(audio.centroid_rms)
        # Floor: below rms_floor, stay at min_iters. Scale from floor to scale.
        rms_floor = 0.05
        scale_range = max(self.rms_scale - rms_floor, 0.01)
        t = max(energy - rms_floor, 0.0) / scale_range
        t = min(t, 1.0) ** 2.0
        self.iterations = int(self.min_iters + t * (self.max_iters - self.min_iters))

    def contribute(self, frame: FlameSheepCore.FrameState) -> None:
        frame.iterations = self.iterations
