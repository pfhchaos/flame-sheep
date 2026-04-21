"""
Tempo detection via onset strength autocorrelation + hypothesis testing.

Algorithm:
  1. Accumulate spectral flux into a rolling onset strength signal
  2. Periodically autocorrelate to find dominant periodicity → BPM hypothesis
  3. Each onset event is tested against the grid:
     - on-grid → confidence rises
     - off-grid → confidence falls
  4. Above GATE_THRESHOLD, kick events are gated to the grid
  5. Below UNLOCK_THRESHOLD, re-estimate from autocorrelation

External tempo hints seed the hypothesis directly.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


# Tempo range
MIN_BPM = 60
MAX_BPM = 200

# Audio buffer for tempo estimation
AUDIO_BUFFER_SEC = 20   # seconds of audio to keep for tempo estimation
ESTIMATE_INTERVAL = 50  # re-estimate every N frames (~2.1s)
MIN_AUDIO_SEC = 4       # minimum audio before first estimate

# Grid matching
PHASE_TOLERANCE = 0.10  # fraction of beat period to count as "on grid"

# Confidence dynamics
CONFIDENCE_UP = 0.10    # per on-grid onset
CONFIDENCE_DOWN = 0.08  # per off-grid onset
GATE_THRESHOLD = 0.5    # start gating kicks above this
UNLOCK_THRESHOLD = 0.2  # re-estimate below this
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
    """
    Autocorrelation-based tempo tracker with hypothesis testing.

    Receives spectral flux each frame to build an onset strength signal.
    Periodically autocorrelates to estimate tempo. Onset events are then
    validated against the grid to build/decay confidence.
    """

    def __init__(self, sr: int = 48000, hop: int = 2048):
        self._sr = sr
        self._hop = hop

        # Rolling audio buffer for librosa tempo estimation
        self._audio_buffer: deque[np.ndarray] = deque(
            maxlen=int(AUDIO_BUFFER_SEC * sr / hop))
        self._frame_count = 0

        # Hypothesis state
        self._bpm = 0.0
        self._beat_period = 0.0
        self._confidence = 0.0
        self._locked = False
        self._last_beat_time = 0.0
        self._has_hypothesis = False

        # Per-kind onset times (for phase alignment)
        self._onset_times: deque[float] = deque(maxlen=256)

        # External hints
        self._hint_bpm: float | None = None
        self._trusted_source = False

    def reset(self):
        """Clear all state — call on song change."""
        self._audio_buffer.clear()
        self._frame_count = 0
        self._onset_times.clear()
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

    def feed_audio(self, pcm: np.ndarray):
        """Feed one frame of raw PCM audio.

        Call once per audio frame, before process_onset().
        """
        self._audio_buffer.append(pcm.copy())
        self._frame_count += 1

        min_frames = int(MIN_AUDIO_SEC * self._sr / self._hop)
        if (self._frame_count >= min_frames
                and self._frame_count % ESTIMATE_INTERVAL == 0):
            bpm = self._estimate_bpm()
            if bpm > 0:
                if not self._has_hypothesis:
                    self._set_hypothesis(bpm, INITIAL_CONFIDENCE)
                elif not self._locked:
                    self._bpm = bpm
                    self._beat_period = 60.0 / bpm

    def process_onset(self, kind: str, timestamp: float) -> bool:
        """
        Process an onset event. Returns True if it should be kept.

        Call after feed_flux() for the current frame.
        """
        self._onset_times.append(timestamp)

        if not self._has_hypothesis:
            return True

        # Test onset against grid, update confidence
        on_grid = self._is_on_grid(timestamp)

        if on_grid:
            self._confidence = min(1.0, self._confidence + CONFIDENCE_UP)
            # Advance last_beat_time to nearest beat
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
            if self._locked or self._has_hypothesis:
                self._has_hypothesis = False
                log.debug(f'hypothesis dropped (confidence={self._confidence:.2f})')
            self._locked = False

        if self._locked and not was_locked:
            log.info(f'tempo locked: {self._bpm:.1f} BPM '
                     f'(confidence={self._confidence:.2f})')
        elif was_locked and not self._locked:
            log.info(f'tempo unlocked (confidence={self._confidence:.2f})')

        # Gating: only gate kicks, and only when locked
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
        recent = list(self._onset_times)[-32:]
        if len(recent) < 2:
            self._last_beat_time = recent[-1] if recent else 0.0
            return

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
        """Estimate BPM from accumulated audio using librosa."""
        import librosa

        if len(self._audio_buffer) < 2:
            return 0.0

        # Concatenate audio buffer
        audio = np.concatenate(list(self._audio_buffer))

        tempo, _ = librosa.beat.beat_track(
            y=audio, sr=self._sr,
            start_bpm=self._hint_bpm or 120.0,
        )
        bpm = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)

        if MIN_BPM <= bpm <= MAX_BPM:
            return bpm
        return 0.0

    def _is_on_grid(self, timestamp: float) -> bool:
        """Check if timestamp aligns with any grid subdivision (1x, 2x)."""
        if self._beat_period <= 0:
            return True

        elapsed = timestamp - self._last_beat_time
        for divisor in [1, 2]:
            period = self._beat_period / divisor
            if period < 0.1:  # floor at 100ms (600 BPM) — always safe
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
