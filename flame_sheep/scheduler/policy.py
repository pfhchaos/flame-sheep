"""Policy: given a rolling window of busy samples, decide whether
batch work should RUN, SLOW (rate-limited), or PAUSE.

Rolling-max over the window (not mean) — we want to be pessimistic.
A 200ms spike to 95% should pause batch even if the surrounding
samples averaged 40%.

Tuning is intentionally simple: two thresholds, three states. If we
need finer control later, the place to add it is here without
changing the worker protocol.

Accepts any "signal source" with the duck-typed interface:
    .available  -> bool
    .sample()   -> (busy_fraction: float, window_s: float) | None
Currently used with either GpuLoadSampler (system-wide GPU%) or
WallpaperSignalReader (wallpaper's own frame-budget usage).
"""
from __future__ import annotations

import collections
import enum
import time
from typing import Protocol


class BatchState(enum.Enum):
    RUN = 'run'      # GPU has headroom; full pace
    SLOW = 'slow'    # GPU partly loaded; rate-limit
    PAUSE = 'pause'  # GPU near saturation; halt batch


class _SignalSource(Protocol):
    @property
    def available(self) -> bool: ...
    def sample(self) -> tuple[float, float] | None: ...


class Policy:
    """Polls a busy-fraction signal source at a fixed cadence,
    maintains a rolling window of values, and exposes a current
    BatchState.

    Tick from the main loop (or a scheduler thread): `policy.tick()`
    samples + recomputes state. Read `policy.state` to act on it.

    Sampling cadence and window length set at construction. Defaults
    tuned for ~250ms reaction time to load spikes — fast enough to
    pause batch before a render frame deadline blows up, slow enough
    not to flip-flop on every microsecond of jitter.

    If the source is unavailable, the policy stays in RUN — fail
    open. Without a signal we have no basis to throttle, and the
    user can still rely on nice/ionice to keep batch out of the way.
    """

    def __init__(self,
                  sampler: _SignalSource,
                  *,
                  sample_interval_s: float = 0.25,
                  window_samples: int = 5,
                  pause_threshold: float = 0.85,
                  slow_threshold: float = 0.55,
                  ):
        self.sampler = sampler
        self.sample_interval_s = sample_interval_s
        self.pause_threshold = pause_threshold
        self.slow_threshold = slow_threshold
        self._window: collections.deque[float] = collections.deque(
            maxlen=window_samples)
        self._last_sample_t = 0.0
        self._state = (BatchState.RUN if not sampler.available
                        else BatchState.PAUSE)  # start conservative
                                                  # until we have data
        self._last_busy: float = 0.0
        # State-duration accumulator — observability. The wallpaper
        # periodically calls drain_state_distribution() to log how
        # much time was spent in each state since last drain. Tells
        # us whether the policy is actually engaging or sleeping at
        # the wheel.
        self._state_durations: dict[BatchState, float] = {
            BatchState.RUN: 0.0, BatchState.SLOW: 0.0,
            BatchState.PAUSE: 0.0,
        }
        self._last_state_t: float = time.perf_counter()

    @property
    def state(self) -> BatchState:
        return self._state

    @property
    def last_busy_fraction(self) -> float:
        """Most recent busy fraction read (not the rolling max).
        Useful for telemetry/logging."""
        return self._last_busy

    @property
    def rolling_max(self) -> float:
        return max(self._window) if self._window else 0.0

    def _charge_state_duration(self, now: float) -> None:
        """Add (now - last_state_t) to the current state's duration.
        Called whenever state might change (before transition) and
        also on drain (so the trailing window is accounted for)."""
        delta = now - self._last_state_t
        if delta > 0:
            self._state_durations[self._state] += delta
        self._last_state_t = now

    def drain_state_distribution(self) -> dict[BatchState, float]:
        """Return state-duration dict (seconds) since last drain, and
        reset. For perf-log telemetry."""
        self._charge_state_duration(time.perf_counter())
        out = dict(self._state_durations)
        for k in self._state_durations:
            self._state_durations[k] = 0.0
        return out

    def tick(self) -> BatchState:
        """If sample_interval has elapsed since last sample, draw one
        and update state. Returns the current state. Safe to call as
        often as the caller likes — internal interval-gating means
        oversampling is cheap.
        """
        if not self.sampler.available:
            return self._state

        now = time.perf_counter()
        if now - self._last_sample_t < self.sample_interval_s:
            return self._state
        self._last_sample_t = now

        result = self.sampler.sample()
        if result is None:
            return self._state
        busy, _window_s = result
        self._last_busy = busy
        self._window.append(busy)

        # Need a few samples before trusting the rolling max; until
        # then stay PAUSE to be safe. This warms up in ~window_samples
        # × sample_interval (default ~1.25s).
        if len(self._window) < self._window.maxlen:
            return self._state

        rm = max(self._window)
        new_state = (BatchState.PAUSE if rm >= self.pause_threshold
                      else BatchState.SLOW if rm >= self.slow_threshold
                      else BatchState.RUN)
        if new_state is not self._state:
            # Charge time-in-old-state before flipping so each state's
            # duration counts only the interval it was actually current.
            self._charge_state_duration(now)
            self._state = new_state
        return self._state
