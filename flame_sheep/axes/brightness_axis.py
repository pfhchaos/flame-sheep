"""Brightness axis — maps audio energy to display gamma.

Uses the slow-envelope harmonic RMS from the audio engine so that
sustained tonal energy (vocals, strings, wall-of-bass) drives
brightness, while isolated percussive hits decay before they can
ramp it up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flame_sheep_audio import AudioState
from flame_sheep.role_mapper import RoleMapper, ENERGY

if TYPE_CHECKING:
    from flame_sheep.main import FlameSheepCore


class BrightnessAxis:
    """Energy -> brightness. Quiet=ghostly, loud=vivid."""

    def __init__(self, role: RoleMapper, floor: float = 0.7, ceiling: float = 12.0,
                 rms_scale: float = 0.01) -> None:
        self.enabled = True
        self._role = role
        self.floor = floor
        self.ceiling = ceiling
        self.rms_scale = rms_scale
        self.brightness = floor

    def tick(self, audio: AudioState, dt: float, clock: float) -> None:
        energy = max(self._role.band_state(audio, ENERGY).slow_harmonic_rms,
                     audio.centroid_harmonic_rms)
        t = min(energy / self.rms_scale, 1.0) ** 0.5
        self.brightness = self.floor + t * (self.ceiling - self.floor)

    def contribute(self, frame: FlameSheepCore.FrameState) -> None:
        frame.brightness = self.brightness
