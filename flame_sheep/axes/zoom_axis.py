"""Zoom axis — hihat/clap events drive a zoom pulse that decays over time."""

import logging

log = logging.getLogger(__name__)

from flame_sheep.audio._types import AudioState
from flame_sheep.config import cfg


class ZoomAxis:
    """Hihat/clap -> zoom pulse. Decays back to baseline between events.

    At high density, decay speeds up and pulses shrink — producing a
    buzzy shimmer instead of discrete throbs.
    """

    @property
    def ZOOM_BOOST_MAX(self): return cfg.zoom.boost_max
    @property
    def ZOOM_DECAY(self): return cfg.zoom.decay
    @property
    def DENSITY_DAMPING(self): return cfg.zoom.density_damping
    @property
    def DENSITY_DECAY_SCALE(self): return cfg.zoom.density_decay_scale

    def __init__(self):
        self.enabled = True
        self.zoom_boost = 0.0

    def tick(self, audio: AudioState, dt: float, clock: float) -> None:
        # Faster decay at high density = rapid vibration
        combined_density = (audio.bands['hihat'].onset_density
                            + audio.bands['clap'].onset_density)
        decay = self.ZOOM_DECAY ** (1.0 + combined_density * self.DENSITY_DECAY_SCALE)
        self.zoom_boost *= decay
        if self.zoom_boost < 0.001:
            self.zoom_boost = 0.0

        # Pulse magnitude scales down with density
        density_scale = 1.0 / (1.0 + combined_density * self.DENSITY_DAMPING)

        for event in audio.events:
            if event.kind in ('hihat', 'clap'):
                self.zoom_boost = min(
                    self.ZOOM_BOOST_MAX,
                    self.zoom_boost + event.energy * 0.15 * density_scale)
                log.debug(f'[{event.kind}] energy={event.energy:.2f}  '
                          f'zoom={self.zoom_boost:.3f}')

    def contribute(self, frame) -> None:
        frame.genome.zoom *= (1.0 + self.zoom_boost)
