"""Brightness axis — maps audio energy to display gamma."""

from flame_sheep.audio._types import AudioState


class BrightnessAxis:
    """Energy -> brightness. Quiet=ghostly, loud=vivid.

    Uses centroid RMS (energy around the dominant frequency) so that
    vocals, guitars, and other tonal content drive brightness, not
    just bass energy.
    """

    def __init__(self, floor: float = 0.7, ceiling: float = 12.0,
                 rms_scale: float = 0.01):
        self.enabled = True
        self.floor = floor
        self.ceiling = ceiling
        self.rms_scale = rms_scale
        self.brightness = floor

    def tick(self, audio: AudioState, dt: float, clock: float) -> None:
        energy = max(audio.bands['subbass'].harmonic_rms,
                     audio.centroid_harmonic_rms)
        t = min(energy / self.rms_scale, 1.0) ** 0.5
        self.brightness = self.floor + t * (self.ceiling - self.floor)

    def contribute(self, frame) -> None:
        frame.brightness = self.brightness
