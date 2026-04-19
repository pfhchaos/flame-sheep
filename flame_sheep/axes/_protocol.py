"""Visual axis protocol — interface for composable beat-reactive state machines."""

from typing import Protocol
from flame_sheep.audio._types import BeatEvent


class VisualAxis(Protocol):
    """A visual axis maps audio events/energy to visual parameters.

    Each axis:
    - Has an `enabled` flag (skip tick/contribute when False)
    - `tick()` advances internal state from events + energy
    - `contribute()` writes results into the shared FrameState
    """
    enabled: bool

    def tick(self, events: list[BeatEvent], rms: float,
             dt: float, clock: float) -> None: ...

    def contribute(self, frame) -> None: ...
