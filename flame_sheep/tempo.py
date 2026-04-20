"""
Tempo detection and rhythm coherence gating.

Analyzes onset timing to find a consistent tempo. When no rhythmic
pattern is detected (e.g., speech, ambient noise), beat events are
suppressed to avoid false positives.

Algorithm:
  1. Collect inter-onset intervals (IOIs) from all beat events
  2. Build a histogram of IOIs, quantized to ~5ms bins
  3. Find peaks in the histogram — these are candidate beat periods
  4. Use autocorrelation to find the dominant periodicity
  5. Track tempo hypothesis over time, require consistency to "lock"
  6. Once locked, predict beat times and only pass events near predictions

The key insight: real music has beats landing on a grid (120 BPM = 500ms).
Speech "beats" are aperiodic — the IOI histogram is flat, no peaks.
"""

import logging

log = logging.getLogger(__name__)


import time
from collections import deque
from dataclasses import dataclass
import numpy as np


# Tempo detection parameters
MIN_BPM = 60      # slowest tempo we'll detect
MAX_BPM = 300     # fastest tempo we'll detect (DnB hits can be very fast)
IOI_HISTORY = 48  # number of inter-onset intervals to keep
LOCK_THRESHOLD = 0.75  # confidence needed to "lock" tempo (raised from 0.6)
UNLOCK_THRESHOLD = 0.4 # confidence below which we "unlock"
PHASE_TOLERANCE = 0.15 # fraction of beat period — events within this are "on beat"
LOCK_STABILITY = 5     # must exceed threshold for this many consecutive onsets


@dataclass
class TempoState:
    """Current tempo tracker state, exposed for debugging/display."""
    bpm: float           # estimated tempo (0 if unknown)
    confidence: float    # 0..1, how sure we are
    phase: float         # 0..1, where we are in the current beat cycle
    locked: bool         # True if we have a stable tempo lock
    last_beat: float     # timestamp of last predicted beat


class TempoTracker:
    """
    Tracks tempo from onset events and gates them for rhythm coherence.
    
    Usage:
        tracker = TempoTracker()
        
        # In your beat detection loop:
        for event in raw_beat_events:
            filtered = tracker.process_onset(event.kind, time.perf_counter())
            if filtered:
                # This onset is rhythmically coherent, use it
                handle_beat(event)
        
        # Or check state:
        state = tracker.state
        if state.locked:
            log.info(f"Tempo: {state.bpm:.1f} BPM")
    """
    
    def __init__(self):
        self._onset_times: deque[float] = deque(maxlen=IOI_HISTORY + 1)
        self._onset_times_by_kind: dict[str, deque[float]] = {
            'kick': deque(maxlen=IOI_HISTORY + 1),
            'snare': deque(maxlen=IOI_HISTORY + 1),
            'hihat': deque(maxlen=IOI_HISTORY + 1),
        }
        self._ioi_history: deque[float] = deque(maxlen=IOI_HISTORY)
        
        # Current tempo hypothesis
        self._bpm = 0.0
        self._confidence = 0.0
        self._locked = False
        self._last_beat_time = 0.0
        self._beat_period = 0.0  # seconds per beat
        
        # Stability tracking: require sustained high confidence to lock
        self._high_confidence_streak = 0
        
        # External tempo hint (from music player metadata)
        self._hint_bpm: float | None = None
        
        # Bypass learning phase when we have external signal
        self._trusted_source = False
        
    def reset(self):
        """Clear all state — call on song change."""
        self._onset_times.clear()
        for q in self._onset_times_by_kind.values():
            q.clear()
        self._ioi_history.clear()
        self._bpm = 0.0
        self._confidence = 0.0
        self._locked = False
        self._last_beat_time = 0.0
        self._beat_period = 0.0
        self._high_confidence_streak = 0
        self._hint_bpm = None
        self._trusted_source = False
        
    def hint_tempo(self, bpm: float):
        """
        Provide a tempo hint from external source (e.g., music player metadata).
        This biases detection toward the hinted tempo and enables trusted mode.
        """
        if MIN_BPM <= bpm <= MAX_BPM:
            self._hint_bpm = bpm
            self._trusted_source = True
            
    def song_started(self):
        """
        Signal that a new song has started (from music player).
        Resets state and enables trusted mode to bypass learning phase.
        """
        self.reset()
        self._trusted_source = True
            
    def process_onset(self, kind: str, timestamp: float) -> bool:
        """
        Process an onset event and return whether it should be passed through.
        
        Args:
            kind: 'kick', 'snare', 'hihat'
            timestamp: time.perf_counter() when the onset was detected
            
        Returns:
            True if this onset is rhythmically coherent (or we're not locked),
            False if it should be suppressed as noise/speech.
        """
        # Record onset time — measure IOI per beat type so mixed drum hits
        # (kick then hihat 50ms later) don't produce tiny IOIs
        kind_times = self._onset_times_by_kind.get(kind)
        if kind_times is not None and kind_times:
            ioi = timestamp - kind_times[-1]
            min_ioi = 60.0 / MAX_BPM
            max_ioi = 60.0 / MIN_BPM
            if min_ioi <= ioi <= max_ioi:
                self._ioi_history.append(ioi)
        if kind_times is not None:
            kind_times.append(timestamp)
        self._onset_times.append(timestamp)
        
        # Update tempo estimate
        self._update_tempo()

        # Gating logic:
        # - If locked: only pass on-beat events
        # - If not locked but confident: pass events (building toward lock)
        # - If not locked and low confidence: suppress (likely speech/noise)
        
        if self._locked:
            return self._is_on_beat(timestamp)
        
        # Not locked yet - require minimum confidence to pass events
        # This prevents speech from triggering visuals while we're learning
        # Exception: if we have a trusted source (music player signal), pass events
        if self._trusted_source:
            return True
            
        MIN_PASS_CONFIDENCE = 0.2
        return self._confidence >= MIN_PASS_CONFIDENCE
        
    def _update_tempo(self):
        """Analyze IOI history to estimate tempo."""
        if len(self._ioi_history) < 8:
            # Not enough data yet
            self._confidence = 0.0
            return
            
        iois = np.array(self._ioi_history)
        
        # Build IOI histogram
        # Bin width ~10ms, range covers 60-200 BPM (0.3s - 1.0s)
        bins = np.linspace(0.15, 1.1, 96)  # 95 bins, ~10ms each (covers up to ~400 BPM)
        hist, edges = np.histogram(iois, bins=bins)
        
        # Use raw histogram for concentration calculation
        # (half/double intervals just confuse things)
        combined = hist.astype(float)
        
        # Apply tempo hint as a prior if available
        if self._hint_bpm is not None:
            hint_period = 60.0 / self._hint_bpm
            hint_bin = int((hint_period - 0.25) / 0.01)
            if 0 <= hint_bin < len(combined):
                # Boost bins near the hint
                for offset in range(-5, 6):
                    idx = hint_bin + offset
                    if 0 <= idx < len(combined):
                        combined[idx] *= 1.5
        
        # Find peak
        if combined.max() == 0:
            self._confidence = 0.0
            return
            
        peak_bin = np.argmax(combined)
        peak_period = (edges[peak_bin] + edges[peak_bin + 1]) / 2
        peak_count = combined[peak_bin]
        
        # Confidence based on multiple factors:
        # 1. Peak prominence (peak vs mean) - is there a clear winner?
        # 2. Peak concentration - are IOIs tightly clustered around peak?
        # 3. Total mass in peak region vs spread - rhythmic = concentrated
        
        total_count = combined.sum()
        if total_count == 0:
            self._confidence = 0.0
            return
            
        # Mass in peak region (+/- 2 bins, ~40ms tolerance)
        peak_region = slice(max(0, peak_bin - 2), min(len(combined), peak_bin + 3))
        peak_mass = combined[peak_region].sum()
        concentration = peak_mass / total_count  # what fraction is near peak?
        
        # Prominence: how much does peak stand out from background?
        mean_count = combined.mean()
        if mean_count > 0:
            prominence = (peak_count - mean_count) / (mean_count + 1)
        else:
            prominence = 0.0
            
        # Combined confidence: need both concentration AND prominence
        # Rhythmic music: high concentration (>0.5) AND high prominence (>2)
        # Speech: low concentration (<0.3) OR low prominence (<1)
        self._confidence = min(1.0, concentration * min(prominence / 2.0, 1.0))
            
        # Update tempo estimate
        new_bpm = 60.0 / peak_period
        
        # Smooth tempo changes when locked
        if self._locked and self._bpm > 0:
            # Only allow gradual drift
            max_drift = 2.0  # BPM per update
            delta = new_bpm - self._bpm
            delta = np.clip(delta, -max_drift, max_drift)
            self._bpm += delta * 0.3
        else:
            self._bpm = new_bpm
            
        self._beat_period = 60.0 / self._bpm if self._bpm > 0 else 0
        
        # Lock/unlock logic with stability requirement
        if self._confidence >= LOCK_THRESHOLD:
            self._high_confidence_streak += 1
        else:
            self._high_confidence_streak = 0
            
        if not self._locked and self._high_confidence_streak >= LOCK_STABILITY:
            self._locked = True
            self._last_beat_time = self._onset_times[-1] if self._onset_times else 0
            log.info(f'locked: {self._bpm:.1f} BPM (confidence={self._confidence:.2f}, streak={self._high_confidence_streak})')
        elif self._locked and self._confidence < UNLOCK_THRESHOLD:
            self._locked = False
            self._high_confidence_streak = 0
            log.info(f'unlocked (confidence={self._confidence:.2f})')
            
    def _is_on_beat(self, timestamp: float) -> bool:
        """Check if timestamp falls near a predicted beat."""
        if self._beat_period <= 0:
            return True
            
        # How far into the beat cycle are we?
        elapsed = timestamp - self._last_beat_time
        cycles = elapsed / self._beat_period
        phase = cycles % 1.0
        
        # Update last beat time if we've crossed a beat
        if cycles >= 1.0:
            beats_elapsed = int(cycles)
            self._last_beat_time += beats_elapsed * self._beat_period
            
        # Is the phase near 0 or 1 (i.e., near a beat)?
        # phase in [0, tolerance] or [1-tolerance, 1]
        near_beat = phase < PHASE_TOLERANCE or phase > (1.0 - PHASE_TOLERANCE)
        return near_beat
        
    @property
    def state(self) -> TempoState:
        """Current tempo tracker state."""
        # Calculate current phase
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
        """True if we have a stable tempo lock."""
        return self._locked
        
    @property
    def bpm(self) -> float:
        """Current tempo estimate in BPM (0 if unknown)."""
        return self._bpm
        
    @property
    def confidence(self) -> float:
        """Confidence in current tempo estimate (0..1)."""
        return self._confidence
