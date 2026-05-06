"""Detail axis — maps audio energy to chaos game iteration count.

Uses the slow-envelope harmonic RMS from the audio engine so that
sustained musical energy drives detail, not individual hits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flame_sheep_audio import AudioState
from flame_sheep.role_mapper import RoleMapper, ENERGY

if TYPE_CHECKING:
    from flame_sheep.main import FlameSheepCore


class DetailAxis:
    """Energy -> iterations. Quiet=sparse, loud=dense and detailed."""

    def __init__(self, role: RoleMapper, min_iters: int = 100, max_iters: int = 500,
                 rms_scale: float = 0.07) -> None:
        self.enabled = True
        self._role = role
        self.min_iters = min_iters
        self.max_iters = max_iters
        self.rms_scale = rms_scale
        self.iterations = min_iters

    def tick(self, audio: AudioState, dt: float, clock: float) -> None:
        energy = audio.centroid_harmonic_rms
        t = min(energy / self.rms_scale, 1.0) ** 0.5
        self.iterations = int(self.min_iters + t * (self.max_iters - self.min_iters))

    def contribute(self, frame: FlameSheepCore.FrameState) -> None:
        frame.iterations = self.iterations
