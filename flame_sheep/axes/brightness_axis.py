"""Brightness axis — maps audio energy to display gamma.

Uses a slow-attack / fast-release envelope so that sustained tonal
energy (vocals, strings, wall-of-bass) drives brightness, while
isolated percussive hits decay before they can ramp it up.
"""

from flame_sheep.audio._types import AudioState
from flame_sheep.config import cfg


class BrightnessAxis:
    """Energy -> brightness. Quiet=ghostly, loud=vivid."""

    def __init__(self, floor: float = 0.7, ceiling: float = 12.0,
                 rms_scale: float = 0.01):
        self.enabled = True
        self.floor = floor
        self.ceiling = ceiling
        self.rms_scale = rms_scale
        self.brightness = floor
        self._envelope = 0.0

    def tick(self, audio: AudioState, dt: float, clock: float) -> None:
        energy = max(audio.bands['subbass'].harmonic_rms,
                     audio.centroid_harmonic_rms)
        # Asymmetric envelope: slow attack, faster release
        if energy > self._envelope:
            alpha = cfg.intensity.attack_alpha
        else:
            alpha = cfg.intensity.release_alpha
        self._envelope = alpha * self._envelope + (1 - alpha) * energy

        t = min(self._envelope / self.rms_scale, 1.0) ** 0.5
        self.brightness = self.floor + t * (self.ceiling - self.floor)

    def contribute(self, frame) -> None:
        frame.brightness = self.brightness
