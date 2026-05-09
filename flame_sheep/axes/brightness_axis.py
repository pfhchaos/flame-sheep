"""Brightness axis — maps audio energy to display gamma.

Uses the slow-envelope harmonic RMS from the audio engine so that
sustained tonal energy (vocals, strings, wall-of-bass) drives
brightness, while isolated percussive hits decay before they can
ramp it up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flame_sheep_audio import AudioState, HOP_SIZE, SAMPLE_RATE
from flame_sheep_audio.response import AsymmetricEnvelope
from flame_sheep.role_mapper import RoleMapper, ENERGY

if TYPE_CHECKING:
    from flame_sheep.main import FlameSheepCore

HOP_TIME = HOP_SIZE / SAMPLE_RATE


class BrightnessAxis:
    """Energy -> brightness. Quiet=ghostly, loud=vivid.

    Uses an asymmetric envelope (fast attack, slow release) on raw centroid
    RMS so sustained tonal energy drives brightness while transients decay.
    """

    def __init__(self, role: RoleMapper, floor: float = 0.7, ceiling: float = 12.0,
                 rms_scale: float = 0.23, noise_floor: float = 0.05) -> None:
        self.enabled = True
        self._role = role
        self.floor = floor
        self.ceiling = ceiling
        self.rms_scale = rms_scale
        self.noise_floor = noise_floor
        self.brightness = floor
        self._envelope = AsymmetricEnvelope(attack=0.1, release=2.0, hop_time=HOP_TIME)

    def tick(self, audio: AudioState, dt: float, clock: float) -> None:
        raw = self._envelope.update(audio.centroid_rms)
        energy = max(raw - self.noise_floor, 0.0)
        t = min(energy / self.rms_scale, 1.0) ** 2.0
        self.brightness = self.floor + t * (self.ceiling - self.floor)

    def contribute(self, frame: FlameSheepCore.FrameState) -> None:
        frame.brightness = self.brightness
