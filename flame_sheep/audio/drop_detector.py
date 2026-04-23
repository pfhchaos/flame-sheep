"""Drop detector — detects silence-to-onset transitions (drops/breaks).

A "drop" is a dramatic onset after sustained quiet — the moment in a song
where everything comes back in after a break. Emits BeatEvent(kind='drop').

Detection requires:
  - No kicks for QUIET_THRESHOLD_FRAMES
  - Energy (centroid_rms) below DROP_ENERGY_RATIO × running average
  - At least MIN_KICKS_BEFORE_DROP kicks have occurred (not song start)
  - Not in drift mode (sustained silence with no music)
  - Cooldown period between drops
"""

import logging

from ._types import BeatEvent

log = logging.getLogger(__name__)


class DropDetector:
    """Detect drops from beat events + energy features.

    Call detect() each frame with the current events and energy state.
    Returns a BeatEvent(kind='drop') if a drop is detected.
    """

    QUIET_THRESHOLD_FRAMES = 120   # ~2s of quiet before a drop can fire
    MIN_KICKS_BEFORE_DROP  = 8     # ignore song start
    DROP_COOLDOWN          = 15.0  # seconds between drops
    DROP_ENERGY_RATIO      = 0.15  # centroid_rms must drop below this × average
    DROP_FREEZE_BARS       = 2     # how many bars the drop "lasts" (for consumers)
    DROP_FREEZE_FALLBACK   = 2.0   # seconds, used when BPM is unknown

    def __init__(self):
        self._quiet_frames = 0
        self._total_kicks = 0
        self._cooldown = 0.0
        self._centroid_rms_avg = 0.0
        self._warmup_frames = 0

    def reset(self):
        """Reset state — call on song change."""
        self._quiet_frames = 0
        self._total_kicks = 0
        self._cooldown = 0.0
        self._centroid_rms_avg = 0.0

    def detect(self, events: list[BeatEvent], centroid_rms: float,
               bpm: float, drifting: bool, dt: float) -> BeatEvent | None:
        """Check for a drop. Returns BeatEvent(kind='drop') or None.

        Args:
            events: beat events this frame (checks for kicks)
            centroid_rms: current energy around spectral centroid
            bpm: current tempo estimate (0 if unknown)
            drifting: True if drift axis is active (no music)
            dt: frame time delta
        """
        # Track centroid RMS average (needs warmup before meaningful)
        self._centroid_rms_avg = 0.97 * self._centroid_rms_avg + 0.03 * centroid_rms
        self._warmup_frames += 1

        # Tick cooldown
        if self._cooldown > 0:
            self._cooldown -= dt

        # Need ~5s of audio history before drop detection is reliable
        if self._warmup_frames < 300:
            return None

        # Is it quiet? No kicks AND energy well below average
        is_quiet = (centroid_rms < self._centroid_rms_avg * self.DROP_ENERGY_RATIO
                    and self._centroid_rms_avg > 1e-6)

        has_kick = any(e.kind == 'kick' for e in events)

        if has_kick:
            self._total_kicks += 1

            # Check for drop
            if (self._quiet_frames >= self.QUIET_THRESHOLD_FRAMES
                    and self._total_kicks > self.MIN_KICKS_BEFORE_DROP
                    and self._cooldown <= 0
                    and not drifting):
                # Compute freeze duration from tempo
                if bpm > 0:
                    bar_duration = 4 * 60.0 / bpm
                    freeze = bar_duration * self.DROP_FREEZE_BARS
                else:
                    freeze = self.DROP_FREEZE_FALLBACK

                self._cooldown = self.DROP_COOLDOWN
                quiet = self._quiet_frames
                self._quiet_frames = 0

                log.info(f'[DROP] after {quiet} quiet frames — '
                         f'freeze {freeze:.1f}s'
                         + (f' ({self.DROP_FREEZE_BARS} bars @ {bpm:.0f} BPM)'
                            if bpm > 0 else ' (no tempo)'))

                return BeatEvent(kind='drop', energy=freeze)

            self._quiet_frames = 0
        elif is_quiet:
            self._quiet_frames += 1
        else:
            self._quiet_frames = 0

        return None
