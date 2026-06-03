"""Zoom axis — downbeat (kick) events drive a zoom pulse that decays over time."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

log = logging.getLogger(__name__)

from flame_sheep_audio import AudioState
from flame_sheep.config import cfg
from flame_sheep.runtime.role_mapper import RoleMapper, DOWNBEAT

if TYPE_CHECKING:
    from flame_sheep.runtime import FlameSheepCore


class ZoomAxis:
    """Downbeat -> zoom pulse. Decays back to baseline between events."""

    @property
    def ZOOM_BOOST_MAX(self) -> float: return cfg.zoom.boost_max
    @property
    def ZOOM_DECAY(self) -> float: return cfg.zoom.decay

    def __init__(self, role: RoleMapper) -> None:
        self.enabled = True
        self._role = role
        self.zoom_boost = 0.0

    def tick(self, audio: AudioState, dt: float, clock: float) -> None:
        self.zoom_boost *= self.ZOOM_DECAY
        if self.zoom_boost < 0.001:
            self.zoom_boost = 0.0

        for event in audio.events:
            if event.kind == self._role.band_for_role(DOWNBEAT):
                self.zoom_boost = min(
                    self.ZOOM_BOOST_MAX,
                    self.zoom_boost + event.energy * 0.15)
                log.debug(f'[downbeat] energy={event.energy:.2f}  '
                          f'zoom={self.zoom_boost:.3f}')

    def contribute(self, frame: FlameSheepCore.FrameState) -> None:
        # Scale boost inversely with zoom so the visual effect is consistent
        # regardless of how zoomed in/out the fractal is.
        # At zoom=1.0, boost is applied as-is. At zoom=0.25, boost is 4x stronger.
        scale = 1.0 / max(frame.genome.zoom, 0.1)
        frame.genome.zoom *= (1.0 + self.zoom_boost * scale)
