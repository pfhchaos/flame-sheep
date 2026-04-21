"""
Tempo detection via hypothesis testing.

Algorithm:
  1. Collect inter-onset intervals (IOIs) to build an initial tempo estimate
  2. Once enough IOIs exist, pick the histogram peak as a BPM hypothesis
  3. Each subsequent onset is tested against the grid:
     - on-grid → confidence rises
     - off-grid → confidence falls
  4. Above GATE_THRESHOLD confidence, kick events are gated to the grid
  5. Below UNLOCK_THRESHOLD, re-estimate from IOI histogram

External tempo hints (from music player) seed the hypothesis directly.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


# Tempo range
MIN_BPM = 60
MAX_BPM = 300

# IOI collection
IOI_HISTORY = 128       # intervals to keep for histogram
MIN_IOIS_TO_ESTIMATE = 12  # need this many before first estimate

# Grid matching
PHASE_TOLERANCE = 0.10  # fraction of beat period to count as "on grid"

# Confidence dynamics
CONFIDENCE_UP = 0.10    # per on-grid onset
CONFIDENCE_DOWN = 0.08  # per off-grid onset
GATE_THRESHOLD = 0.5    # start gating kicks above this
UNLOCK_THRESHOLD = 0.2  # re-estimate below this
INITIAL_CONFIDENCE = 0.3  # starting confidence when hypothesis is set


@dataclass
class TempoState:
    """Current tempo tracker state, exposed for debugging/display."""
    bpm: float
    confidence: float
    phase: float
    locked: bool
    last_beat: float


class TempoTracker:
    """
    Hypothesis-testing tempo tracker.

    Estimates BPM from IOI histogram, then validates each onset against
    the grid. Confidence rises/falls based on grid alignment, and gating
    engages when confidence is high enough.
    """

    def __init__(self):
        # Per-kind onset times (avoids cross-instrument IOI noise)
        self._onset_times_by_kind: dict[str, deque[float]] = {
            'kick': deque(maxlen=IOI_HISTORY + 1),
            'snare': deque(maxlen=IOI_HISTORY + 1),
            'hihat': deque(maxlen=IOI_HISTORY + 1),
        }
        self._ioi_history: deque[float] = deque(maxlen=IOI_HISTORY)

        # Hypothesis state
        self._bpm = 0.0
        self._beat_period = 0.0
        self._confidence = 0.0
        self._locked = False  # True when confidence >= GATE_THRESHOLD
        self._last_beat_time = 0.0
        self._has_hypothesis = False

        # External hints
        self._hint_bpm: float | None = None
        self._trusted_source = False

    def reset(self):
        """Clear all state — call on song change."""
        for q in self._onset_times_by_kind.values():
            q.clear()
        self._ioi_history.clear()
        self._bpm = 0.0
        self._beat_period = 0.0
        self._confidence = 0.0
        self._locked = False
        self._last_beat_time = 0.0
        self._has_hypothesis = False
        self._hint_bpm = None
        self._trusted_source = False

    def hint_tempo(self, bpm: float):
        """Provide a tempo hint — seeds hypothesis directly."""
        if MIN_BPM <= bpm <= MAX_BPM:
            self._hint_bpm = bpm
            self._trusted_source = True
            self._set_hypothesis(bpm, INITIAL_CONFIDENCE + 0.2)

    def song_started(self):
        """Signal new song — resets and enables trusted mode."""
        self.reset()
        self._trusted_source = True

    def process_onset(self, kind: str, timestamp: float) -> bool:
        """
        Process an onset event. Returns True if it should be kept.

        When we have a confident tempo hypothesis, kicks are gated to
        the grid. Snare/hihat always pass (they're used for estimation).
        """
        # Record IOI
        kind_times = self._onset_times_by_kind.get(kind)
        if kind_times is not None and kind_times:
            ioi = timestamp - kind_times[-1]
            min_ioi = 60.0 / MAX_BPM
            max_ioi = 60.0 / MIN_BPM
            if min_ioi <= ioi <= max_ioi:
                self._ioi_history.append(ioi)
        if kind_times is not None:
            kind_times.append(timestamp)

        # Phase 1: build hypothesis from IOI histogram
        if not self._has_hypothesis:
            if len(self._ioi_history) >= MIN_IOIS_TO_ESTIMATE:
                bpm = self._estimate_bpm()
                if bpm > 0:
                    self._set_hypothesis(bpm, INITIAL_CONFIDENCE)
            # Pass everything while learning
            return True

        # Phase 2: test onset against grid, update confidence
        on_grid = self._is_on_grid(timestamp)

        if on_grid:
            self._confidence = min(1.0, self._confidence + CONFIDENCE_UP)
            # Advance last_beat_time to nearest beat before this onset
            # Prevents phase drift without re-anchoring to noisy onsets
            if self._beat_period > 0:
                elapsed = timestamp - self._last_beat_time
                beats = round(elapsed / self._beat_period)
                if beats > 0:
                    self._last_beat_time += beats * self._beat_period
        else:
            self._confidence = max(0.0, self._confidence - CONFIDENCE_DOWN)

        # Lock/unlock
        was_locked = self._locked
        if self._confidence >= GATE_THRESHOLD:
            self._locked = True
        elif self._confidence < UNLOCK_THRESHOLD:
            self._locked = False
            # Re-estimate from recent IOIs
            self._has_hypothesis = False
            log.debug(f'hypothesis dropped (confidence={self._confidence:.2f})')

        if self._locked and not was_locked:
            log.info(f'tempo locked: {self._bpm:.1f} BPM '
                     f'(confidence={self._confidence:.2f})')
        elif was_locked and not self._locked:
            log.info(f'tempo unlocked (confidence={self._confidence:.2f})')

        # Gating: only gate kicks, and only when locked
        # Use subdivisions for gating (more permissive than confidence check)
        if self._locked and kind == 'kick':
            return self._is_on_grid(timestamp)

        return True

    def _set_hypothesis(self, bpm: float, confidence: float):
        """Set a new tempo hypothesis with optimal phase alignment."""
        self._bpm = bpm
        self._beat_period = 60.0 / bpm
        self._confidence = confidence
        self._has_hypothesis = True

        # Find phase offset that maximizes grid alignment with recent onsets
        all_times = []
        for kind_times in self._onset_times_by_kind.values():
            all_times.extend(kind_times)
        all_times.sort()

        if len(all_times) < 2:
            self._last_beat_time = all_times[-1] if all_times else 0.0
        else:
            # Try each recent onset as a phase anchor, count on-grid hits
            recent = all_times[-min(len(all_times), 32):]
            best_anchor = recent[-1]
            best_count = 0
            for candidate in recent:
                count = 0
                for t in recent:
                    elapsed = t - candidate
                    for divisor in [1, 2]:
                        period = self._beat_period / divisor
                        if period < 60.0 / MAX_BPM:
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

    def _estimate_bpm(self) -> float:
        """Estimate BPM from IOI histogram with harmonic folding."""
        iois = np.array(self._ioi_history)

        bins = np.linspace(0.15, 1.1, 192)  # ~5ms bins for better resolution
        hist, edges = np.histogram(iois, bins=bins)
        combined = hist.astype(float)

        # Apply hint as prior
        if self._hint_bpm is not None:
            bin_width = edges[1] - edges[0]
            hint_period = 60.0 / self._hint_bpm
            hint_bin = round((hint_period - edges[0]) / bin_width)
            for offset in range(-3, 4):
                idx = hint_bin + offset
                if 0 <= idx < len(combined):
                    combined[idx] *= 2.0

        if combined.max() == 0:
            return 0.0

        peak_bin = np.argmax(combined)
        peak_period = (edges[peak_bin] + edges[peak_bin + 1]) / 2
        return 60.0 / peak_period

    def _is_on_beat(self, timestamp: float) -> bool:
        """Check if timestamp aligns with the base beat period only."""
        if self._beat_period <= 0:
            return True
        elapsed = timestamp - self._last_beat_time
        phase = (elapsed / self._beat_period) % 1.0
        return phase < PHASE_TOLERANCE or phase > (1.0 - PHASE_TOLERANCE)

    def _is_on_grid(self, timestamp: float) -> bool:
        """Check if timestamp aligns with any grid subdivision (1x, 2x)."""
        if self._beat_period <= 0:
            return True

        elapsed = timestamp - self._last_beat_time
        for divisor in [1, 2]:
            period = self._beat_period / divisor
            if period < 60.0 / MAX_BPM:
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
    def confidence(self) -> float:
        return self._confidence
