"""MorphCycle — core morph state machine.

Two-and-a-half state cycle:
  MORPHING → SWAP_READY (beat mode only) → DWELL → MORPHING

Owns morph_t, morph_state, dwell_start. Exposes mutation methods so
other components can trigger transitions without touching state directly.
"""

from __future__ import annotations

import enum
import logging

from flame_sheep.config import cfg

log = logging.getLogger(__name__)


class MorphState(enum.Enum):
    MORPHING = 'morphing'
    SWAP_READY = 'swap_ready'
    DWELL = 'dwell'


class MorphCycle:
    """Pure state machine for morph progression."""

    def __init__(self) -> None:
        self.state = MorphState.DWELL
        self.t = 0.0
        self.dwell_start: float = 0.0

    @property
    def morphing(self) -> bool:
        return self.state == MorphState.MORPHING

    @property
    def in_dwell(self) -> bool:
        return self.state == MorphState.DWELL

    @property
    def in_swap_ready(self) -> bool:
        return self.state == MorphState.SWAP_READY

    def advance(self) -> None:
        """Advance morph_t by morph_speed. Call each tick during MORPHING."""
        self.t = min(1.0, self.t + cfg.drift.morph_speed)

    def is_complete(self) -> bool:
        """True if morph_t >= 1.0."""
        return self.t >= 1.0

    def enter_swap_ready(self) -> None:
        """MORPHING → SWAP_READY (beat mode: wait for kick to commit)."""
        self.state = MorphState.SWAP_READY
        log.debug('[morph] complete, waiting for beat to swap')

    def commit_swap(self, clock: float) -> None:
        """SWAP_READY/MORPHING → DWELL. Resets morph_t."""
        self.state = MorphState.DWELL
        self.t = 0.0
        self.dwell_start = clock

    def release_dwell(self) -> None:
        """DWELL → MORPHING."""
        self.state = MorphState.MORPHING

    def start_morph(self) -> None:
        """Reset to MORPHING with t=0. For loop switches and forced morphs."""
        self.state = MorphState.MORPHING
        self.t = 0.0

    def nudge(self, amount: float) -> None:
        """Nudge morph_t forward (e.g., by beat energy). Clamps to 1.0."""
        if self.morphing:
            self.t = min(1.0, self.t + amount)
