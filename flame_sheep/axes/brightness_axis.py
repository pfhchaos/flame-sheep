"""Brightness axis — maps bass energy to display gamma."""

from flame_sheep.audio._types import BeatEvent


class BrightnessAxis:
    """Energy → brightness. Quiet=ghostly, loud=vivid."""

    def __init__(self, floor: float = 0.7, ceiling: float = 12.0,
                 rms_scale: float = 0.01):
        self.enabled = True
        self.floor = floor
        self.ceiling = ceiling
        self.rms_scale = rms_scale
        self.brightness = floor

    def tick(self, events: list[BeatEvent], rms: float,
             dt: float, clock: float) -> None:
        t = min(rms / self.rms_scale, 1.0) ** 0.5
        self.brightness = self.floor + t * (self.ceiling - self.floor)

    def contribute(self, frame) -> None:
        frame.brightness = self.brightness
