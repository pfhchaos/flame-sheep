"""Rotation driver — phase accumulation with beat boost envelope."""

from __future__ import annotations

import numpy as np

from flame_sheep.config import cfg
from flame_sheep_audio import HOP_SIZE, SAMPLE_RATE
from flame_sheep_audio.response import AsymmetricEnvelope


TWO_PI = 2.0 * np.pi


class RotationDriver:
    """Accumulates rotation phase with optional beat-driven boost.

    Independent output — no coupling to morph state or genomes.
    """

    def __init__(self) -> None:
        _hop_time = HOP_SIZE / SAMPLE_RATE
        self._boost_envelope = AsymmetricEnvelope(
            attack=0.001, release=1.0, hop_time=_hop_time, unit='beats')
        self._phase = 0.0

    @property
    def phase(self) -> float:
        return self._phase

    @phase.setter
    def phase(self, value: float) -> None:
        self._phase = value

    def tick(self, bpm: float, break_damping: float) -> None:
        """Advance rotation phase by one frame."""
        boost = self._boost_envelope.update(0.0, bpm=bpm)
        speed = cfg.genome.rotation_speed + boost * cfg.genome.rotation_beat_boost
        self._phase += speed * break_damping
        if self._phase >= TWO_PI:
            self._phase -= TWO_PI

    def boost(self, energy: float, bpm: float) -> None:
        """Apply beat boost impulse."""
        self._boost_envelope.update(energy, bpm=bpm)
