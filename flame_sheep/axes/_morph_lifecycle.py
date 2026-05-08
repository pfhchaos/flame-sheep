"""Morph lifecycle state machine for genome transitions.

States:
  DWELL     — dwelling on current genome, waiting for dwell duration to elapse
  READY     — dwell complete, waiting for a real beat to trigger morph
  MORPHING  — crossfading from current to target genome over morph duration

Transitions:
  DWELL → READY:    when elapsed time >= dwell_beats * 60/bpm
  READY → MORPHING: when on_beat() is called (real beat detected)
  MORPHING → DWELL: when elapsed time >= morph_beats * 60/bpm (genome swaps)
"""

from __future__ import annotations

import enum
import logging

log = logging.getLogger(__name__)


class MorphState(enum.Enum):
    DWELL = 'dwell'
    READY = 'ready'
    MORPHING = 'morphing'


class MorphLifecycle:
    """Tempo-timed dwell, beat-triggered morph, smooth per-frame crossfade."""

    __slots__ = ('state', 'morph_t', '_dwell_start', '_morph_start',
                 '_dwell_beats', '_morph_beats', '_initialized')

    def __init__(self, clock: float | None = None,
                 dwell_beats: int = 16, morph_beats: int = 4) -> None:
        self.state = MorphState.DWELL
        self.morph_t = 0.0
        self._dwell_start = clock if clock is not None else 0.0
        self._morph_start = 0.0
        self._dwell_beats = dwell_beats
        self._morph_beats = morph_beats
        self._initialized = clock is not None

    def tick(self, clock: float, bpm: float) -> bool:
        """Advance the lifecycle. Returns True if morph just completed.

        Call every frame. Handles DWELL → READY (time-based) and
        morph_t advancement during MORPHING.
        """
        bpm = max(bpm, 1.0)

        if not self._initialized:
            self._dwell_start = clock
            self._initialized = True

        if self.state == MorphState.DWELL:
            dwell_duration = self._dwell_beats * 60.0 / bpm
            if clock - self._dwell_start >= dwell_duration:
                self.state = MorphState.READY
                log.debug(f"[lifecycle] DWELL → READY after {dwell_duration:.1f}s")
            return False

        if self.state == MorphState.MORPHING:
            morph_duration = self._morph_beats * 60.0 / bpm
            elapsed = clock - self._morph_start
            self.morph_t = min(1.0, elapsed / max(morph_duration, 0.01))

            if self.morph_t >= 1.0:
                self.morph_t = 0.0
                self.state = MorphState.DWELL
                self._dwell_start = clock
                log.debug("[lifecycle] MORPHING → DWELL (complete)")
                return True  # morph completed — caller should swap genomes

        return False

    def on_beat(self, clock: float) -> bool:
        """Signal a beat event. Returns True if morph just started.

        Only triggers morph start from READY state.
        """
        if self.state == MorphState.READY:
            self.state = MorphState.MORPHING
            self._morph_start = clock
            log.debug(f"[lifecycle] READY → MORPHING (beat at {clock:.2f}s, "
                     f"dwell was {clock - self._dwell_start:.1f}s)")
            return True
        return False

    def reset(self, clock: float | None = None) -> None:
        """Reset to DWELL — used on song start, handoff, etc.

        If clock is None, dwell start will be set from the next tick() call.
        """
        self.state = MorphState.DWELL
        self.morph_t = 0.0
        if clock is not None:
            self._dwell_start = clock
            self._initialized = True
        else:
            self._initialized = False

    @property
    def is_morphing(self) -> bool:
        return self.state == MorphState.MORPHING

    @property
    def is_ready(self) -> bool:
        return self.state == MorphState.READY
