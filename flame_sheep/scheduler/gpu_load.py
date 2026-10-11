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

Busy-fraction alone is a misleading load signal: the card can read
100% busy clocked down at its floor as easily as 100% busy at boost
(the A770 here: floor ~600MHz vs boost 2400MHz — "busy but loafing"
vs "actually working" look identical on occupancy alone). sample()
is kept as-is (existing callers unpack a 2-tuple — see policy.py /
__main__.py), but GpuLoadSampler also exposes sample_telemetry(),
which adds the honest signals on top of the same busy-fraction:
    - actual GT clock vs its max, from sysfs
      (.../tile*/gt*/freq0/{act_freq,max_freq}), giving a
      throughput_fraction = busy * act_freq/max_freq.
    - board power draw, from the xe hwmon node (power*_input if the
      kernel exposes it, else integrated from the energy*_input
      µJ counter), giving a power_fraction = power_w / tdp_w.
These are plain sysfs/hwmon file reads — no GPU context, no churn —
so the driver-lockout hazard that applies to GPU-context code does
NOT apply here (see the flame-sheep skill's gpu-reality.md §3, §5).
Every new field degrades to None if its sysfs path is missing or
permission-denied, same contract as the PMU branch below.
"""
from __future__ import annotations

import ctypes
import errno
import os
import re
import struct
import time
from ctypes import c_int32, c_uint8, c_uint16, c_uint32, c_uint64
from dataclasses import dataclass
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


# --- Clock + power (sysfs/hwmon) -------------------------------------------
# Verified on-machine (this box: Arc A770, Xe driver, kernel 6.19-gentoo):
#   freq:  /sys/class/drm/card0/device/tile0/gt0/freq0/{act_freq,max_freq}
#   hwmon: /sys/class/drm/card0/device/hwmon/hwmon2/ (name == "xe"),
#          no power2_input on this kernel — only energy2_input (uJ counter,
#          label "pkg") and power2_max (= 190000000 uW, matching the A770's
#          published 190W board TDP exactly). So power is integrated from
#          the energy counter here, not read instantaneously.
# The tile/gt/hwmon index numbers are kernel-version-dependent (NOT
# hardcoded as tile0/gt0/freq0/hwmon2 below) — we glob for them.

_DRM_BASE = Path('/sys/class/drm')
_CARD_DIR_RE = re.compile(r'^card\d+$')  # excludes connector pseudo-dirs
                                          # like "card0-DP-1"

# Fallback board TDP (watts) for the A770 — used only if the hwmon
# power*_max cap sysfs node isn't readable. On this machine power2_max
# already reports 190000000 uW == this constant, so the constant is a
# backstop, not the primary source.
_A770_TDP_W = 190.0


def _discover_xe_drm_device() -> Path | None:
    """Find the DRM device dir for the xe-driven card
    (.../card*/device, e.g. /sys/class/drm/card0/device). Skips
    connector pseudo-entries (card0-DP-1 etc.) via _CARD_DIR_RE."""
    if not _DRM_BASE.exists():
        return None
    for child in sorted(_DRM_BASE.iterdir()):
        if not _CARD_DIR_RE.match(child.name):
            continue
        driver_link = child / 'device' / 'driver'
        try:
            if driver_link.resolve().name == 'xe':
                return child / 'device'
        except OSError:
            continue
    return None


def _discover_freq_dir(device: Path) -> Path | None:
    """Find tile*/gt*/freq0 under the device dir. Index numbers and
    even the tile/gt nesting are kernel-version-dependent — glob
    rather than hardcode tile0/gt0."""
    for freq_dir in sorted(device.glob('tile*/gt*/freq0')):
        if (freq_dir / 'act_freq').exists():
            return freq_dir
    return None


def _discover_xe_hwmon_dir(device: Path) -> Path | None:
    """Find device/hwmon/hwmon<N> whose 'name' file reads 'xe'."""
    hwmon_base = device / 'hwmon'
    if not hwmon_base.exists():
        return None
    for child in sorted(hwmon_base.iterdir()):
        try:
            if child.joinpath('name').read_text().strip() == 'xe':
                return child
        except OSError:
            continue
    return None


def _read_int_sysfs(path: Path) -> int | None:
    """Read one sysfs integer leaf. None on any missing/denied/
    malformed read — same soft-degrade contract as the PMU path."""
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


@dataclass
class GpuTelemetrySample:
    """Occupancy PLUS clock + power, so "100% busy" idling at the
    frequency floor can be told apart from "100% busy" at boost.

    Any field is None if its sysfs/PMU source was missing or
    permission-denied — callers should treat None as "no
    information" per-field, not fail the whole sample.
    """
    busy_fraction: float | None       # same signal as sample()[0]
    window_s: float                   # wall-clock seconds this sample covers
                                       # (0.0 if busy_fraction is None)
    act_freq_mhz: float | None        # freq0/act_freq — actual GT clock now
    max_freq_mhz: float | None        # freq0/max_freq (or rp0_freq fallback)
    throughput_fraction: float | None  # busy * act_freq/max_freq
    power_w: float | None             # instantaneous watts, direct or
                                       # energy-integrated (see module docstring)
    tdp_w: float | None               # hwmon power*_max cap, else _A770_TDP_W
    power_fraction: float | None      # power_w / tdp_w


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

        # Clock + power sysfs/hwmon sources — independent of the PMU
        # setup below (plain file paths, nothing to open/hold), so set
        # these up unconditionally before any PMU early-return. Any
        # path not found just leaves the corresponding attribute None,
        # and the read_*() helpers below treat None as "unavailable"
        # for that one field only.
        self._freq_dir: Path | None = None
        self._power_input_path: Path | None = None   # instantaneous, uW
        self._energy_input_path: Path | None = None  # cumulative counter, uJ
        self._power_max_uw: int | None = None         # hwmon cap -> tdp_w
        self._prev_energy_uj: int | None = None
        self._prev_energy_t: float = 0.0

        xe_device = _discover_xe_drm_device()
        if xe_device is not None:
            self._freq_dir = _discover_freq_dir(xe_device)
            hwmon_dir = _discover_xe_hwmon_dir(xe_device)
            if hwmon_dir is not None:
                power_inputs = sorted(hwmon_dir.glob('power[0-9]*_input'))
                if power_inputs:
                    self._power_input_path = power_inputs[0]
                energy_inputs = sorted(hwmon_dir.glob('energy[0-9]*_input'))
                if energy_inputs:
                    self._energy_input_path = energy_inputs[0]
                    self._prev_energy_uj = _read_int_sysfs(
                        self._energy_input_path)
                    self._prev_energy_t = time.perf_counter()
                power_maxes = sorted(hwmon_dir.glob('power[0-9]*_max'))
                if power_maxes:
                    self._power_max_uw = _read_int_sysfs(power_maxes[0])

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

    def _read_freq_mhz(self) -> tuple[float | None, float | None]:
        """(act_freq_mhz, max_freq_mhz) from sysfs. Either half may be
        None independently (missing node, or a kernel that only
        exposes one of the pair)."""
        if self._freq_dir is None:
            return (None, None)
        act = _read_int_sysfs(self._freq_dir / 'act_freq')
        mx = _read_int_sysfs(self._freq_dir / 'max_freq')
        if mx is None:
            # Some kernel layouts only expose the turbo cap as
            # rp0_freq rather than max_freq.
            mx = _read_int_sysfs(self._freq_dir / 'rp0_freq')
        return (float(act) if act is not None else None,
                float(mx) if mx is not None else None)

    def _read_power_w(self) -> float | None:
        """Instantaneous board power in watts.

        Prefers a direct power*_input reading (uW) if the kernel
        exposes one. Otherwise integrates the energy*_input (uJ,
        monotonic) counter across the interval since the previous
        call — this machine's kernel only exposes the latter (see
        module docstring). Returns None on the first energy-counter
        call (no prior sample to diff against yet), on a counter
        rollover (treated as "skip this sample," not a negative
        power), or if neither sysfs node exists.
        """
        if self._power_input_path is not None:
            uw = _read_int_sysfs(self._power_input_path)
            return (uw / 1e6) if uw is not None else None

        if self._energy_input_path is not None:
            now = time.perf_counter()
            uj = _read_int_sysfs(self._energy_input_path)
            prev_uj, prev_t = self._prev_energy_uj, self._prev_energy_t
            self._prev_energy_uj, self._prev_energy_t = uj, now
            if uj is None or prev_uj is None:
                return None
            d_t = now - prev_t
            d_uj = uj - prev_uj
            if d_t <= 0 or d_uj < 0:  # d_uj < 0 => counter rolled over
                return None
            return (d_uj / 1e6) / d_t

        return None

    def sample_telemetry(self) -> GpuTelemetrySample:
        """Full telemetry sample: busy-fraction (sample(), if the PMU
        is available) PLUS clock + power from independent sysfs/hwmon
        sources (read even when the PMU is unavailable). Cheap file
        reads — safe to call at the same cadence as sample(); see the
        module docstring for why this isn't GPU-context churn.

        Unlike sample(), this never returns None — individual fields
        are None when their source is missing/denied, so a caller
        logs a partial sample instead of losing the whole tick.
        """
        result = self.sample()
        busy, window_s = result if result is not None else (None, 0.0)

        act_freq_mhz, max_freq_mhz = self._read_freq_mhz()
        power_w = self._read_power_w()
        tdp_w = (self._power_max_uw / 1e6
                 if self._power_max_uw is not None else _A770_TDP_W)

        throughput_fraction = None
        if busy is not None and act_freq_mhz is not None and max_freq_mhz:
            throughput_fraction = busy * (act_freq_mhz / max_freq_mhz)

        power_fraction = None
        if power_w is not None and tdp_w:
            power_fraction = power_w / tdp_w

        return GpuTelemetrySample(
            busy_fraction=busy,
            window_s=window_s,
            act_freq_mhz=act_freq_mhz,
            max_freq_mhz=max_freq_mhz,
            throughput_fraction=throughput_fraction,
            power_w=power_w,
            tdp_w=tdp_w,
            power_fraction=power_fraction,
        )

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
