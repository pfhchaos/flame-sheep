"""Visual axis protocol — interface for composable beat-reactive state machines."""

from typing import Protocol
from flame_sheep_audio import AudioState


class VisualAxis(Protocol):
    """A visual axis maps audio events/energy to visual parameters.

    Each axis:
    - Has an `enabled` flag (skip tick/contribute when False)
    - `tick()` advances internal state from audio state
    - `contribute()` writes results into the shared FrameState

    Axes should ignore event kinds they don't recognize in audio.events.
    """
    enabled: bool

    def tick(self, audio: AudioState, dt: float, clock: float) -> None: ...

    def contribute(self, frame) -> None: ...
