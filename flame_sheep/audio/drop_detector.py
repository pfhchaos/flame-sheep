"""Break detector — detects sustained quiet (energy dropout).

A "break" is a sustained period where energy drops well below average —
the moment in a song where the beat drops out. Unlike the old drop detector
which fired on kick return, this exposes a continuous `breaking` state
that drives exponential morph slowdown in the visual axes.

Detection requires:
  - Energy (centroid_rms) below DROP_ENERGY_RATIO × running average
  - Sustained for QUIET_THRESHOLD_FRAMES
  - At least MIN_KICKS_BEFORE_DROP kicks have occurred (not song start)
  - Not in drift mode (sustained silence with no music)
  - Cooldown period after a break ends
"""

import logging

from ._types import BeatEvent

log = logging.getLogger(__name__)


class DropDetector:
    """Detect breaks from energy features.

    Call tick() each frame. Read .breaking to check if a break is active.
    """

    QUIET_THRESHOLD_FRAMES = 60    # ~1s of quiet before break activates
    MIN_KICKS_BEFORE_DROP  = 8     # ignore song start
    BREAK_COOLDOWN         = 15.0  # seconds after a break ends before next
    DROP_ENERGY_RATIO      = 0.15  # centroid_rms must drop below this × average

    def __init__(self):
        self._quiet_frames = 0
        self._total_kicks = 0
        self._cooldown = 0.0
        self._centroid_rms_avg = 0.0
        self._warmup_frames = 0
        self.breaking = False

    def reset(self):
        """Reset state — call on song change."""
        self._quiet_frames = 0
        self._total_kicks = 0
        self._cooldown = 0.0
        self._centroid_rms_avg = 0.0
        self.breaking = False

    def detect(self, events: list[BeatEvent], centroid_rms: float,
               bpm: float, drifting: bool, dt: float) -> None:
        """Update break state. Read .breaking for current state.

        Args:
            events: beat events this frame (checks for kicks)
            centroid_rms: current energy around spectral centroid
            bpm: current tempo estimate (0 if unknown)
            drifting: True if drift mode is active (no music)
            dt: frame time delta
        """
        self._centroid_rms_avg = 0.97 * self._centroid_rms_avg + 0.03 * centroid_rms
        self._warmup_frames += 1

        if self._cooldown > 0:
            self._cooldown -= dt

        if self._warmup_frames < 300:
            return

        is_quiet = (centroid_rms < self._centroid_rms_avg * self.DROP_ENERGY_RATIO
                    and self._centroid_rms_avg > 1e-6)

        has_kick = any(e.kind == 'kick' for e in events)

        if has_kick:
            self._total_kicks += 1
            # Break ends on kick return
            if self.breaking:
                log.info(f'[BREAK END] after {self._quiet_frames} quiet frames')
                self._cooldown = self.BREAK_COOLDOWN
                self.breaking = False
            self._quiet_frames = 0
        elif is_quiet:
            self._quiet_frames += 1
        else:
            # Energy recovered without a kick — break ends
            if self.breaking:
                self._cooldown = self.BREAK_COOLDOWN
                self.breaking = False
            self._quiet_frames = 0

        # Activate break when quiet long enough
        if (not self.breaking
                and self._quiet_frames >= self.QUIET_THRESHOLD_FRAMES
                and self._total_kicks > self.MIN_KICKS_BEFORE_DROP
                and self._cooldown <= 0
                and not drifting):
            self.breaking = True
            log.info(f'[BREAK] centroid_rms={centroid_rms:.4f} '
                     f'avg={self._centroid_rms_avg:.4f}')
