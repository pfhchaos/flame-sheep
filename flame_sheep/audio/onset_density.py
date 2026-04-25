"""Onset density tracker — per-band onset rate as a continuous feature.

Tracks how many onsets per second are arriving in each band.
At normal tempos (120 BPM), kick density is ~2/s.  At speedcore
(270+ BPM), it's 4-5/s.  Axes can use this to smoothly blend
behavior rather than counting individual kicks.

Also provides kick_density_delta — the accelerando signal.
Positive = speeding up, negative = slowing down.
"""

from collections import deque

from ..config import cfg


class OnsetDensityTracker:
    """Multi-band onset density tracker.

    Call process_onset() for each detected onset, update() each frame.
    Read density properties for current values.
    """

    @property
    def WINDOW(self): return cfg.density.window
    @property
    def DELTA_WINDOW(self): return cfg.density.delta_window
    BANDS = ('kick', 'snare', 'clap', 'hihat')

    @property
    def ALPHA(self): return cfg.density.alpha

    def __init__(self):
        self._times: dict[str, deque[float]] = {b: deque() for b in self.BANDS}
        self._density: dict[str, float] = {b: 0.0 for b in self.BANDS}
        # History for kick delta only (accelerando signal)
        self._kick_history: deque[tuple[float, float]] = deque()

    def reset(self):
        """Clear all state — call on song change."""
        for b in self.BANDS:
            self._times[b].clear()
            self._density[b] = 0.0
        self._kick_history.clear()

    def process_onset(self, kind: str, timestamp: float):
        """Record an onset event."""
        if kind in self._times:
            self._times[kind].append(timestamp)

    def update(self, now: float):
        """Recompute densities. Call once per audio frame."""
        cutoff = now - self.WINDOW
        for band in self.BANDS:
            times = self._times[band]
            while times and times[0] < cutoff:
                times.popleft()
            raw = len(times) / self.WINDOW
            self._density[band] = (self.ALPHA * self._density[band]
                                   + (1 - self.ALPHA) * raw)

        # Track kick density history for delta
        self._kick_history.append((now, self._density['kick']))
        delta_cutoff = now - self.DELTA_WINDOW
        while (self._kick_history
               and self._kick_history[0][0] < delta_cutoff):
            self._kick_history.popleft()

    @property
    def densities(self) -> dict[str, float]:
        """Per-band onset densities (onsets/second)."""
        return dict(self._density)

    @property
    def kick_density_delta(self) -> float:
        """Rate of change of kick density (accelerando signal)."""
        if len(self._kick_history) < 2:
            return 0.0
        return self._density['kick'] - self._kick_history[0][1]
