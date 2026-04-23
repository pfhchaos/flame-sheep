"""Zoom axis — hihat/clap events drive a zoom pulse that decays over time."""

from flame_sheep.audio._types import AudioState


class ZoomAxis:
    """Hihat/clap -> zoom pulse. Decays back to baseline between events."""

    ZOOM_BOOST_MAX = 0.3
    ZOOM_DECAY     = 0.95

    def __init__(self):
        self.enabled = True
        self.zoom_boost = 0.0

    def tick(self, audio: AudioState, dt: float, clock: float) -> None:
        # Decay
        self.zoom_boost *= self.ZOOM_DECAY
        if self.zoom_boost < 0.001:
            self.zoom_boost = 0.0

        # Hihat/clap events add boost
        for event in audio.events:
            if event.kind in ('hihat', 'clap'):
                self.zoom_boost = min(
                    self.ZOOM_BOOST_MAX,
                    self.zoom_boost + event.energy * 0.15)

    def contribute(self, frame) -> None:
        frame.genome.zoom *= (1.0 + self.zoom_boost)
