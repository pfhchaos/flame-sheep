"""Read Intel Xe GPU engine-busy fraction via perf_event_open.

What this gives us:
    busy = (engine-active-ticks delta over window) /
            (engine-total-ticks delta over window)

This is the same signal intel_gpu_top reads (when it works on i915);
since the user is on Xe driver and intel_gpu_top is i915-only, we
bind perf_event_open via ctypes directly. No shell-out, no extra
binaries to install — the kernel exposes everything we need at
/sys/bus/event_source/devices/xe_*/.

Usage:
    sampler = GpuLoadSampler()
    busy_fraction = sampler.sample()  # 0.0..1.0 over the window since last call

The scheduler uses this to gate batch work: if the GPU is hot, pause
batch jobs and let the wallpaper renderer keep its frame budget.

This module is x86_64 + Linux + Xe-driver specific. Other platforms
need parallel implementations (i915 PMU, AMDGPU, NVIDIA, etc.) — out
of scope for the first cut.
"""
from __future__ import annotations

import ctypes
import errno
import os
import struct
import time
from ctypes import c_int32, c_uint8, c_uint16, c_uint32, c_uint64
from pathlib import Path

# perf_event_open syscall number on x86_64. Other arches differ;
# pull from <asm-generic/unistd.h> if you need to port.
_SYS_perf_event_open = 298

_XE_PMU_BASE = Path('/sys/bus/event_source/devices')

# Compute engine — what our chaos game compute pipelines run on.
# Render — what the tonemap+present graphics pipeline uses.
# Sample both; sum the active ticks across them to get a total
# "is the GPU's userspace-visible work pipeline busy" signal.
# Engine class IDs come from drivers/gpu/drm/xe/xe_hw_engine_types.h
# in the kernel — RENDER=0, COMPUTE=4.
_ENGINE_CLASS_RENDER = 0
_ENGINE_CLASS_COMPUTE = 4

# perf event format bit positions (from sysfs format/ entries):
#   event = config[0:11]
#   engine_instance = config[12:19]
#   engine_class = config[20:27]
# (We leave function/gt at 0.)
_EVENT_BITS_START = 0
_INSTANCE_BITS_START = 12
_CLASS_BITS_START = 20

# Event IDs (from sysfs events/ entries):
_EVENT_ACTIVE_TICKS = 0x02
_EVENT_TOTAL_TICKS = 0x03


# struct perf_event_attr, VER0 layout (64 bytes). Newer kernel
# versions extend this struct; we only fill VER0 fields and set
# attr.size accordingly so the kernel accepts our shorter version.
class _PerfEventAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ('type',                       c_uint32),  # 0
        ('size',                       c_uint32),  # 4
        ('config',                     c_uint64),  # 8
        ('sample_period_or_freq',      c_uint64),  # 16
        ('sample_type',                c_uint64),  # 24
        ('read_format',                c_uint64),  # 32
        ('flags',                      c_uint64),  # 40 (bitfield blob)
        ('wakeup_events_or_watermark', c_uint32),  # 48
        ('bp_type',                    c_uint32),  # 52
        ('config1',                    c_uint64),  # 56 — 64 bytes total
    ]


assert ctypes.sizeof(_PerfEventAttr) == 64, \
    f'expected 64 bytes, got {ctypes.sizeof(_PerfEventAttr)}'


_libc = ctypes.CDLL('libc.so.6', use_errno=True)
_libc.syscall.restype = ctypes.c_long


def _perf_event_open(attr: _PerfEventAttr, pid: int, cpu: int,
                       group_fd: int, flags: int) -> int:
    fd = _libc.syscall(_SYS_perf_event_open, ctypes.byref(attr),
                        pid, cpu, group_fd, flags)
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, f'perf_event_open: {os.strerror(err)}')
    return fd


def _read_counter(fd: int) -> int:
    """Read 8 bytes from the perf fd → u64 counter value."""
    data = os.read(fd, 8)
    return struct.unpack('<Q', data)[0]


def _discover_xe_pmu() -> tuple[int, int] | None:
    """Find the Xe PMU on this system. Returns (type, cpu) or None
    if no Xe device is bound to the perf interface.

    `type` is the perf event type ID (numeric). `cpu` is the CPU
    perf_event_open should bind to (Xe is uncore-like — one CPU
    per PMU). Both come from sysfs.
    """
    if not _XE_PMU_BASE.exists():
        return None
    for child in _XE_PMU_BASE.iterdir():
        if child.name.startswith('xe_'):
            try:
                t = int((child / 'type').read_text().strip())
                cpu = int((child / 'cpumask').read_text().strip()
                          .split(',')[0].split('-')[0])
                return (t, cpu)
            except (OSError, ValueError):
                continue
    return None


def _build_config(event: int, engine_class: int,
                    engine_instance: int = 0) -> int:
    """Pack the perf config u64 the way xe PMU expects: event in
    bits [0:11], engine_instance in [12:19], engine_class in
    [20:27]."""
    return ((event       << _EVENT_BITS_START)
             | (engine_instance << _INSTANCE_BITS_START)
             | (engine_class    << _CLASS_BITS_START))


class GpuLoadSampler:
    """Streaming sampler — open the four fds once, call sample()
    repeatedly to get the busy-fraction over the interval since the
    last call.

    Returns None if no Xe PMU was found (other GPU vendors / missing
    driver). Callers should treat None as "no information" — typically
    fall back to permissive scheduling rather than crash.
    """

    def __init__(self):
        self._fds: list[int] = []  # paired: [render_active, render_total,
                                    #          compute_active, compute_total]
        self._prev_active: int = 0
        self._prev_total: int = 0
        self._prev_t: float = 0.0
        self._available: bool = False
        self._unavailable_reason: str | None = None

        pmu = _discover_xe_pmu()
        if pmu is None:
            self._unavailable_reason = (
                'no Xe PMU found at /sys/bus/event_source/devices/xe_*/ '
                '(other GPU vendor, or driver not loaded)')
            return
        pmu_type, pmu_cpu = pmu

        try:
            for klass in (_ENGINE_CLASS_RENDER, _ENGINE_CLASS_COMPUTE):
                for event in (_EVENT_ACTIVE_TICKS, _EVENT_TOTAL_TICKS):
                    attr = _PerfEventAttr()
                    attr.type = pmu_type
                    attr.size = ctypes.sizeof(_PerfEventAttr)
                    attr.config = _build_config(event, klass)
                    # disabled=0 (start enabled) → flags stays 0
                    # pid=-1 cpu=N → system-wide on a specific CPU
                    # (uncore-style)
                    fd = _perf_event_open(attr, pid=-1, cpu=pmu_cpu,
                                            group_fd=-1, flags=0)
                    self._fds.append(fd)
        except PermissionError as e:
            # /proc/sys/kernel/perf_event_paranoid >= 2 blocks
            # unprivileged uncore reads. Scheduler should fall back to
            # permissive (no GPU-load gating) when this trips.
            #
            # Long-term fix is a CAP_PERFMON-bearing helper binary the
            # main process talks to via a socket; defer that until we
            # ship beyond this dev box. For now: degrade gracefully.
            for fd in self._fds:
                try: os.close(fd)
                except OSError: pass
            self._fds.clear()
            self._unavailable_reason = (
                f'perf_event_open denied ({e.strerror}). Either lower '
                f'/proc/sys/kernel/perf_event_paranoid (sudo sysctl '
                f'kernel.perf_event_paranoid=1) or run with CAP_PERFMON.')
            return
        except OSError as e:
            for fd in self._fds:
                try: os.close(fd)
                except OSError: pass
            self._fds.clear()
            self._unavailable_reason = f'perf_event_open failed: {e}'
            return

        # Prime the deltas
        self._prev_active = self._read_active()
        self._prev_total = self._read_total()
        self._prev_t = time.perf_counter()
        self._available = True

    def _read_active(self) -> int:
        return _read_counter(self._fds[0]) + _read_counter(self._fds[2])

    def _read_total(self) -> int:
        return _read_counter(self._fds[1]) + _read_counter(self._fds[3])

    @property
    def available(self) -> bool:
        """True if the Xe PMU was found and fds are open."""
        return self._available

    @property
    def unavailable_reason(self) -> str | None:
        """Diagnostic string when available is False; None otherwise."""
        return self._unavailable_reason

    def sample(self) -> tuple[float, float] | None:
        """Return (busy_fraction, window_seconds) since the previous
        call. None if no PMU.

        busy_fraction is active/total summed across render + compute
        engines, so it represents "userspace-visible GPU work" in
        aggregate. 0.0 = idle, 1.0 = both engines saturated. Can
        exceed 1.0 briefly if the kernel's tick accounting drifts —
        clamp at the consumer if you need a bounded scalar.
        """
        if not self._available:
            return None
        now = time.perf_counter()
        active = self._read_active()
        total = self._read_total()

        d_active = active - self._prev_active
        d_total = total - self._prev_total
        d_t = now - self._prev_t

        self._prev_active = active
        self._prev_total = total
        self._prev_t = now

        if d_total == 0:
            return (0.0, d_t)
        return (d_active / d_total, d_t)

    def close(self):
        for fd in self._fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()
        self._available = False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
