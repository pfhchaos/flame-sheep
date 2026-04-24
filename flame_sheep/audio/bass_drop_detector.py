"""Bass drop detector — detects sub-bass dropout followed by kick return.

Complements the EDM DropDetector which requires full-silence breaks.
This detector catches pop/rock-style drops where kick and bass disappear
but vocals and mids continue — the sub-bass vanishes while centroid_rms
stays high.

Detection requires:
  - Sub-bass RMS (20-200Hz) below 10% of running average for ~1.5s
  - A kick event to trigger the drop (the "return" moment)
  - At least MIN_KICKS_BEFORE_DROP kicks have occurred
  - Not in drift mode
  - Cooldown period between drops
"""

import logging

from ._types import BeatEvent

log = logging.getLogger(__name__)


class BassDropDetector:
    """Detect drops from sub-bass energy dropout + kick return.

    Call detect() each frame with sub-bass RMS and beat events.
    Returns a BeatEvent(kind='drop') if a bass drop is detected.
    """

    QUIET_THRESHOLD_FRAMES = 90    # ~1.5s of bass quiet (shorter than EDM's 2s)
    MIN_KICKS_BEFORE_DROP  = 8     # ignore song start
    DROP_COOLDOWN          = 15.0  # seconds between drops
    SUBBASS_DROP_RATIO     = 0.10  # sub-bass must drop below 10% of average
    DROP_FREEZE_BARS       = 2
    DROP_FREEZE_FALLBACK   = 2.0

    def __init__(self):
        self._quiet_frames = 0
        self._total_kicks = 0
        self._cooldown = 0.0
        self._subbass_avg = 0.0
        self._warmup_frames = 0

    def reset(self):
        """Reset state — call on song change."""
        self._quiet_frames = 0
        self._total_kicks = 0
        self._cooldown = 0.0
        self._subbass_avg = 0.0

    def detect(self, events: list[BeatEvent], subbass_rms: float,
               bpm: float, drifting: bool, dt: float) -> BeatEvent | None:
        """Check for a bass drop. Returns BeatEvent(kind='drop') or None.

        Args:
            events: beat events this frame
            subbass_rms: current 20-200Hz RMS energy (snap.rms)
            bpm: current tempo estimate (0 if unknown)
            drifting: True if drift mode is active
            dt: frame time delta
        """
        self._subbass_avg = 0.97 * self._subbass_avg + 0.03 * subbass_rms
        self._warmup_frames += 1

        if self._cooldown > 0:
            self._cooldown -= dt

        if self._warmup_frames < 300:
            return None

        # Is sub-bass quiet?
        is_bass_quiet = (subbass_rms < self._subbass_avg * self.SUBBASS_DROP_RATIO
                         and self._subbass_avg > 1e-6)

        has_kick = any(e.kind == 'kick' for e in events)

        if has_kick:
            self._total_kicks += 1

            if (self._quiet_frames >= self.QUIET_THRESHOLD_FRAMES
                    and self._total_kicks > self.MIN_KICKS_BEFORE_DROP
                    and self._cooldown <= 0
                    and not drifting):
                if bpm > 0:
                    bar_duration = 4 * 60.0 / bpm
                    freeze = bar_duration * self.DROP_FREEZE_BARS
                else:
                    freeze = self.DROP_FREEZE_FALLBACK

                self._cooldown = self.DROP_COOLDOWN
                quiet = self._quiet_frames
                self._quiet_frames = 0

                log.info(f'[BASS DROP] after {quiet} quiet frames — '
                         f'freeze {freeze:.1f}s'
                         + (f' ({self.DROP_FREEZE_BARS} bars @ {bpm:.0f} BPM)'
                            if bpm > 0 else ' (no tempo)'))

                return BeatEvent(kind='drop', energy=freeze)

            self._quiet_frames = 0
        elif is_bass_quiet:
            self._quiet_frames += 1
        else:
            self._quiet_frames = 0

        return None
