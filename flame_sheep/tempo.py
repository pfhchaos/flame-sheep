"""
Tempo detection via onset IOI histogram.

Algorithm:
  1. Collect kick onset timestamps in a rolling window
  2. Compute inter-onset intervals (IOIs)
  3. Histogram IOIs to find dominant periodicity → BPM estimate
  4. Track confidence via grid alignment testing

The BPM estimate and confidence are published for visualization use
(e.g. drop freeze duration, morph speed calibration).
No gating is performed — all events pass through unconditionally.

External tempo hints seed the estimate directly.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


# Tempo range
MIN_BPM = 60
MAX_BPM = 400

# IOI estimation
MIN_ONSETS_FOR_ESTIMATE = 8
ESTIMATE_EVERY_N_ONSETS = 4
IOI_WINDOW = 32
IOI_HIST_BINS = 200
IOI_MIN = 60.0 / MAX_BPM
IOI_MAX = 60.0 / MIN_BPM

# Grid matching (for confidence, not gating)
PHASE_TOLERANCE = 0.12

# Confidence dynamics
CONFIDENCE_UP = 0.08
CONFIDENCE_DOWN = 0.06
LOCK_THRESHOLD = 0.5
UNLOCK_THRESHOLD = 0.2
INITIAL_CONFIDENCE = 0.3


@dataclass
class TempoState:
    """Current tempo tracker state, exposed for debugging/display."""
    bpm: float
    confidence: float
    phase: float
    locked: bool
    last_beat: float


class TempoTracker:
    """IOI-based tempo tracker.

    Collects kick onset timestamps, estimates BPM from inter-onset
    interval histograms, and tracks confidence via grid alignment.
    Does NOT gate events — all onsets pass through.
    """

    def __init__(self):
        self._kick_times: deque[float] = deque(maxlen=IOI_WINDOW)
        self._onset_count = 0

        self._bpm = 0.0
        self._beat_period = 0.0
        self._confidence = 0.0
        self._locked = False
        self._last_beat_time = 0.0
        self._has_hypothesis = False

        self._hint_bpm: float | None = None
        self._saturated = False  # True when kicks are faster than MAX_BPM

    def reset(self):
        """Clear all state — call on song change."""
        self._kick_times.clear()
        self._onset_count = 0
        self._bpm = 0.0
        self._beat_period = 0.0
        self._confidence = 0.0
        self._locked = False
        self._last_beat_time = 0.0
        self._has_hypothesis = False
        self._hint_bpm = None
        self._saturated = False

    def hint_tempo(self, bpm: float):
        """Provide a tempo hint — seeds hypothesis directly."""
        if MIN_BPM <= bpm <= MAX_BPM:
            self._hint_bpm = bpm
            self._set_hypothesis(bpm, INITIAL_CONFIDENCE + 0.2)

    def song_started(self):
        """Signal new song — resets state."""
        self.reset()

    def process_onset(self, kind: str, timestamp: float):
        """Process an onset event for tempo estimation.

        Only kick onsets contribute to BPM estimation.
        All onsets are tested against the grid for confidence tracking.
        """
        if kind == 'kick':
            self._kick_times.append(timestamp)
            self._onset_count += 1

            # Check for tempo saturation: many kicks but intervals too short
            if len(self._kick_times) >= 4:
                recent = sorted(self._kick_times)[-4:]
                median_ioi = sorted(recent[i+1] - recent[i] for i in range(3))[1]
                self._saturated = median_ioi < IOI_MIN and len(self._kick_times) >= MIN_ONSETS_FOR_ESTIMATE

            # Periodically estimate BPM from IOIs
            if (self._onset_count >= MIN_ONSETS_FOR_ESTIMATE
                    and self._onset_count % ESTIMATE_EVERY_N_ONSETS == 0):
                bpm = self._estimate_bpm_from_ioi()
                if bpm > 0:
                    if not self._has_hypothesis:
                        self._set_hypothesis(bpm, INITIAL_CONFIDENCE)
                    elif not self._locked:
                        self._bpm = bpm
                        self._beat_period = 60.0 / bpm

        if not self._has_hypothesis:
            return

        # Test onset against grid, update confidence
        on_grid = self._is_on_grid(timestamp)

        if on_grid:
            self._confidence = min(1.0, self._confidence + CONFIDENCE_UP)
            if self._beat_period > 0:
                elapsed = timestamp - self._last_beat_time
                beats = round(elapsed / self._beat_period)
                if beats > 0:
                    self._last_beat_time += beats * self._beat_period
        else:
            self._confidence = max(0.0, self._confidence - CONFIDENCE_DOWN)

        # Lock/unlock
        was_locked = self._locked
        if self._confidence >= LOCK_THRESHOLD:
            self._locked = True
        elif self._confidence < UNLOCK_THRESHOLD:
            if self._locked or self._has_hypothesis:
                self._has_hypothesis = False
                log.debug(f'hypothesis dropped (confidence={self._confidence:.2f})')
            self._locked = False

        if self._locked and not was_locked:
            log.info(f'tempo locked: {self._bpm:.1f} BPM '
                     f'(confidence={self._confidence:.2f})')
        elif was_locked and not self._locked:
            log.info(f'tempo unlocked (confidence={self._confidence:.2f})')

    def _set_hypothesis(self, bpm: float, confidence: float):
        """Set a new tempo hypothesis with optimal phase alignment."""
        self._bpm = bpm
        self._beat_period = 60.0 / bpm
        self._confidence = confidence
        self._has_hypothesis = True

        recent = sorted(self._kick_times)[-16:]
        if len(recent) < 2:
            self._last_beat_time = recent[-1] if recent else 0.0
            return

        best_anchor = recent[-1]
        best_count = 0
        divisors = [1, 2] if bpm < 180 else [1]
        for candidate in recent:
            count = 0
            for t in recent:
                elapsed = t - candidate
                for divisor in divisors:
                    period = self._beat_period / divisor
                    if period < IOI_MIN:
                        continue
                    phase = (elapsed / period) % 1.0
                    if phase < PHASE_TOLERANCE or phase > (1.0 - PHASE_TOLERANCE):
                        count += 1
                        break
            if count > best_count:
                best_count = count
                best_anchor = candidate
        self._last_beat_time = best_anchor
        log.debug(f'hypothesis: {bpm:.1f} BPM (confidence={confidence:.2f})')

    def _estimate_bpm_from_ioi(self) -> float:
        """Estimate BPM from inter-onset intervals of recent kicks."""
        times = sorted(self._kick_times)
        if len(times) < MIN_ONSETS_FOR_ESTIMATE:
            return 0.0

        iois = []
        for i in range(len(times)):
            for j in range(i + 1, min(i + 4, len(times))):
                ioi = times[j] - times[i]
                if IOI_MIN <= ioi <= IOI_MAX:
                    iois.append(ioi)
                # Symmetric octave folding — add both half and double
                half = ioi / 2
                if IOI_MIN <= half <= IOI_MAX:
                    iois.append(half)
                double = ioi * 2
                if IOI_MIN <= double <= IOI_MAX:
                    iois.append(double)

        if len(iois) < 4:
            return 0.0

        ioi_arr = np.array(iois)
        counts, edges = np.histogram(ioi_arr, bins=IOI_HIST_BINS,
                                     range=(IOI_MIN, IOI_MAX))

        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        smoothed = np.convolve(counts, kernel, mode='same')

        peak_bin = np.argmax(smoothed)
        peak_ioi = (edges[peak_bin] + edges[peak_bin + 1]) / 2

        if smoothed[peak_bin] < len(iois) * 0.1:
            return 0.0

        bpm = 60.0 / peak_ioi

        # Prefer lower octave: if BPM > 200, check for half-BPM peak
        if bpm > 200:
            half_ioi = peak_ioi * 2
            if IOI_MIN <= half_ioi <= IOI_MAX:
                bin_width = (IOI_MAX - IOI_MIN) / IOI_HIST_BINS
                half_bin = int((half_ioi - IOI_MIN) / bin_width)
                half_bin = min(max(half_bin, 2), IOI_HIST_BINS - 3)
                half_peak = float(max(smoothed[half_bin - 2 : half_bin + 3]))
                if half_peak > smoothed[peak_bin] * 0.4:
                    bpm = bpm / 2

        if MIN_BPM <= bpm <= MAX_BPM:
            return bpm
        return 0.0

    def _is_on_grid(self, timestamp: float) -> bool:
        """Check if timestamp aligns with any grid subdivision (1x, 2x)."""
        if self._beat_period <= 0:
            return True

        elapsed = timestamp - self._last_beat_time
        # Above 180 BPM, only check primary grid — divisor 2 would
        # reinforce a double-time hypothesis by accepting half-time onsets
        divisors = [1, 2] if self._bpm < 180 else [1]
        for divisor in divisors:
            period = self._beat_period / divisor
            if period < 0.1:
                continue
            phase = (elapsed / period) % 1.0
            if phase < PHASE_TOLERANCE or phase > (1.0 - PHASE_TOLERANCE):
                return True

        return False

    @property
    def state(self) -> TempoState:
        phase = 0.0
        if self._beat_period > 0 and self._last_beat_time > 0:
            now = time.perf_counter()
            elapsed = now - self._last_beat_time
            phase = (elapsed / self._beat_period) % 1.0
        return TempoState(
            bpm=self._bpm,
            confidence=self._confidence,
            phase=phase,
            locked=self._locked,
            last_beat=self._last_beat_time,
        )

    @property
    def locked(self) -> bool:
        return self._locked

    @property
    def bpm(self) -> float:
        return self._bpm

    @property
    def effective_bpm(self) -> float:
        """BPM blended with default based on confidence.

        When confident: uses measured BPM.
        When lost: drifts toward default (120).
        When saturated (fast content beyond MAX_BPM): uses MAX_BPM.
        """
        from .config import cfg
        if self._saturated:
            return float(cfg.tempo.max_bpm)
        default = cfg.tempo.default_bpm
        return default * (1 - self._confidence) + self._bpm * self._confidence

    @property
    def saturated(self) -> bool:
        """True when onset rate exceeds MAX_BPM tracking range."""
        return self._saturated

    @property
    def confidence(self) -> float:
        return self._confidence
