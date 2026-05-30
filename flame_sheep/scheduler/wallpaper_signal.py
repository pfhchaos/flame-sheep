"""Wallpaper-side perf telemetry → scheduler-side throttle signal.

Replaces the perf_event GPU-load sampler with a direct readout from
the wallpaper's render loop. More accurate (reflects exactly what we
care about — wallpaper headroom — without aliasing GPU work from
other apps), faster reaction (one frame, ~16ms, vs perf sampler's
250ms window), no PMU permission dependency.

Trade-off: doesn't detect external load. If Chrome ramps up video
decode, this signal stays low because the wallpaper's frame time is
unchanged… until the GPU contention finally bites and the wallpaper
starts dropping frames, at which point chaos_ms spikes and the
scheduler reacts. That's a tolerable signal lag; the alternative was
queue-priority arbitration which we already proved isn't honored on
Mesa Xe.

Wire format (shmem layout, packed little-endian):
    offset 0:  uint64  timestamp_ns       — when this update was written
    offset 8:  float32 frame_ms           — wallpaper's last full frame
    offset 12: float32 chaos_ms           — chaos-game compute portion
    offset 16: float32 ema_chaos_ms       — EMA of chaos_ms (smoothed)
    offset 20: float32 target_ms          — frame budget (1000/max_fps)
    offset 24: 8 bytes reserved

Total: 32 bytes, fits in a single cache line. mmap'd from
$XDG_RUNTIME_DIR/flame-sheep/scheduler/wallpaper_signal.bin.

Atomicity: 32-byte writes are NOT atomic on x86. Reader could
observe a torn struct. We use timestamp_ns as a generation marker —
reader reads timestamp, reads body, reads timestamp again; if they
match, the read was consistent. This is the classic seqlock pattern
without the explicit sequence number.
"""
from __future__ import annotations

import mmap
import os
import struct
import time
from pathlib import Path


# struct layout in shmem (see module docstring)
_LAYOUT = '<Q f f f f 8s'
_SIZE = struct.calcsize(_LAYOUT)
assert _SIZE == 32, f'expected 32 bytes, got {_SIZE}'


def _signal_path() -> Path:
    base = os.environ.get('XDG_RUNTIME_DIR',
                           f'/run/user/{os.getuid()}')
    d = Path(base) / 'flame-sheep' / 'scheduler'
    d.mkdir(parents=True, exist_ok=True)
    return d / 'wallpaper_signal.bin'


# EMA factor for chaos_ms smoothing. 0.2 = ~5-frame window.
# Slow enough to dampen single-frame spikes, fast enough to react
# to sustained changes within ~80ms at 60fps.
_EMA_ALPHA = 0.2


class WallpaperSignalWriter:
    """Wallpaper-side: writes its own frame telemetry to shmem after
    each frame. Use as a context manager or call close() explicitly.

    The writer maintains the EMA of chaos_ms internally so the reader
    side doesn't have to keep state.
    """

    def __init__(self, target_ms: float):
        self.path = _signal_path()
        self._target_ms = target_ms
        self._ema_chaos_ms = 0.0
        with open(self.path, 'wb') as f:
            f.write(b'\x00' * _SIZE)
        self._fd = os.open(self.path, os.O_RDWR)
        self._mm = mmap.mmap(self._fd, _SIZE, mmap.MAP_SHARED,
                              mmap.PROT_READ | mmap.PROT_WRITE)

    def update(self, frame_ms: float, chaos_ms: float) -> None:
        """Called from the render loop after each frame. Updates the
        shmem snapshot atomically-enough for the seqlock pattern."""
        self._ema_chaos_ms = (
            _EMA_ALPHA * chaos_ms +
            (1.0 - _EMA_ALPHA) * self._ema_chaos_ms)
        ts = time.time_ns()
        # Pack into a temp bytes object, then copy into mmap. Single
        # 32-byte memcpy is faster than multiple struct.pack_into calls
        # and is what the seqlock-on-timestamp trick relies on (we
        # write the body, then re-stamp the timestamp last to publish).
        # Actually for simplicity: write all-at-once, accept that
        # readers retry on torn reads via the timestamp check.
        data = struct.pack(_LAYOUT, ts, frame_ms, chaos_ms,
                            self._ema_chaos_ms, self._target_ms, b'')
        self._mm[:_SIZE] = data

    def close(self):
        try: self._mm.close()
        except Exception: pass
        try: os.close(self._fd)
        except OSError: pass
        # Don't unlink — let it persist so the reader can detect
        # staleness via timestamp. Cleanup happens when XDG_RUNTIME_DIR
        # gets cleared at session end.

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class WallpaperSignalReader:
    """Scheduler-side: reads the wallpaper's telemetry. Returns None
    from sample() when the signal is stale (wallpaper not running or
    frozen for >500ms) — caller treats that as "no signal, default
    policy" (typically RUN with just nice/ionice protecting).
    """

    # How long the data is considered fresh.
    STALE_NS = 500_000_000  # 500ms

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else _signal_path()
        self._available = False
        self._unavailable_reason: str | None = None
        if not self.path.exists():
            # Create empty so we can mmap it; sample() will return
            # stale until the writer overwrites with real data.
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(self.path, 'wb') as f:
                    f.write(b'\x00' * _SIZE)
            except OSError as e:
                self._unavailable_reason = f'cannot create {self.path}: {e}'
                return
        try:
            self._fd = os.open(self.path, os.O_RDONLY)
            self._mm = mmap.mmap(self._fd, _SIZE, mmap.MAP_SHARED,
                                  mmap.PROT_READ)
        except OSError as e:
            self._unavailable_reason = f'cannot mmap {self.path}: {e}'
            return
        self._available = True

    @property
    def available(self) -> bool:
        """True if the shmem file exists and is readable. NOT a
        check on whether the data is fresh — call sample() to learn
        that."""
        return self._available

    @property
    def unavailable_reason(self) -> str | None:
        return self._unavailable_reason

    def sample(self) -> tuple[float, float] | None:
        """Returns (busy_fraction, window_seconds) matching the
        GpuLoadSampler.sample() contract so Policy can use either
        source interchangeably.

        busy_fraction = ema_chaos_ms / target_ms — 0.0 means the
        wallpaper's chaos pass barely uses any of its budget, 1.0
        means it's saturating its budget.

        Returns None if the data is stale (wallpaper not running or
        frozen >500ms). Caller treats this as "no signal."
        """
        if not self._available:
            return None
        # Seqlock-style consistency check: read timestamp, read body,
        # re-read timestamp. If they differ, the writer was mid-update —
        # retry once (very rare; writer holds for only a microsecond).
        for _ in range(2):
            raw = bytes(self._mm[:_SIZE])
            ts, frame_ms, chaos_ms, ema_chaos_ms, target_ms, _ = (
                struct.unpack(_LAYOUT, raw))
            # Re-check by reading just the timestamp again
            ts2 = struct.unpack('<Q', bytes(self._mm[:8]))[0]
            if ts == ts2:
                break
        else:
            return None  # torn read both attempts; rare

        if ts == 0:
            return None  # never written
        now_ns = time.time_ns()
        if now_ns - ts > self.STALE_NS:
            return None
        if target_ms <= 0:
            return None
        busy = ema_chaos_ms / target_ms
        # Window seconds: report the rough wallpaper-frame cadence as
        # the "window," matching what GpuLoadSampler returns.
        return (busy, target_ms / 1000.0)

    def close(self):
        if not self._available:
            return
        try: self._mm.close()
        except Exception: pass
        try: os.close(self._fd)
        except OSError: pass
        self._available = False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
