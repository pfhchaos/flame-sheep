"""Detail axis — maps audio energy to chaos game iteration count.

Uses the slow-envelope harmonic RMS from the audio engine so that
sustained musical energy drives detail, not individual hits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flame_sheep_audio import AudioState, HOP_SIZE, SAMPLE_RATE
from flame_sheep_audio.response import AsymmetricEnvelope
from flame_sheep.role_mapper import RoleMapper, ENERGY

if TYPE_CHECKING:
    from flame_sheep.main import FlameSheepCore

HOP_TIME = HOP_SIZE / SAMPLE_RATE


class DetailAxis:
    """Energy -> iterations. Quiet=sparse, loud=dense and detailed.

    Uses an asymmetric envelope (fast attack, slow release) on raw centroid
    RMS so sustained musical energy drives detail, not individual hits.
    """

    def __init__(self, role: RoleMapper, min_iters: int = 100, max_iters: int = 500,
                 rms_scale: float = 0.07) -> None:
        self.enabled = True
        self._role = role
        self.min_iters = min_iters
        self.max_iters = max_iters
        self.rms_scale = rms_scale
        self.iterations = min_iters
        self._envelope = AsymmetricEnvelope(attack=0.1, release=2.0, hop_time=HOP_TIME)

    def tick(self, audio: AudioState, dt: float, clock: float) -> None:
        energy = self._envelope.update(audio.centroid_rms)
        t = min(energy / self.rms_scale, 1.0) ** 0.5
        self.iterations = int(self.min_iters + t * (self.max_iters - self.min_iters))

    def contribute(self, frame: FlameSheepCore.FrameState) -> None:
        frame.iterations = self.iterations
