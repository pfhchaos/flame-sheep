"""GPU + genome telemetry sampler — Phase 1 (measurement only).

Logs, per sample over long real-usage periods, the data needed to answer
two questions offline:

  (a) Can a local Whisper STT job share the Intel Arc GPU with the
      wallpaper?  -> per-engine GPU busy% (compute / render / video) plus
      the wallpaper's own iter_count + genome cost features, so we can see
      how much compute headroom exists while the wallpaper runs.
  (b) Later: seed an adaptive iteration-budget controller. -> iter_count
      vs genome cost vs audio-reactivity vs realized GPU busy%.

This phase is MEASUREMENT ONLY. There is no controller here and nothing
in this module feeds back into the render loop.

Design constraints (see docs/telemetry-hookmap.md for the full map):
  - Runs on a BACKGROUND thread at a ~1-5s cadence. It must never stall
    or pace the render loop. The render loop's only interaction is a
    cheap, lock-guarded publish() call that stores references; all the
    work (GPU sampling, genome feature extraction, JSONL writes) happens
    on the sampler thread.
  - Fails soft. If the Xe PMU is unreadable, or the Vulkan VRAM probe
    errors, the affected fields are logged as null and sampling
    continues. This module must NEVER crash the wallpaper.

Output: JSONL, append-mode, to a configurable path. One header line
(record_type="header") captured once at startup, then one
record_type="sample" line per interval.

IMPORTANT ARCHITECTURE NOTE (verify on-machine):
  The LIVE wallpaper renders with OpenGL (moderngl / EGL), NOT Vulkan.
  Vulkan (wallpaper_ml.VkCompute) is only used by the offline ML
  trainers, in separate processes. There is therefore no live Vulkan
  device to reuse for the VRAM query. query_vram_once() creates its own
  short-lived Vulkan instance to read the memory budget. That means a
  SECOND graphics API context is briefly created inside the GL render
  process. The wallpaper_ml skill warns that the Mesa Xe KMD has crashed
  sway under some GPU-sharing scenarios (fork-based multiprocessing); an
  in-process second Vulkan instance is a different path, but it is
  UNTESTED here. See docs/telemetry-hookmap.md "VRAM query" for the
  subprocess-isolation fallback if this perturbs the live context.
"""

from __future__ import annotations

import ctypes
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flame_sheep_audio import AudioSnapshot

log = logging.getLogger(__name__)

# Default output path — mirrors the audio feature logger's convention
# (~/.local/share/flame-sheep/...), separate file so the two logs don't
# interleave schemas.
DEFAULT_TELEMETRY_FILE = Path(
    '~/.local/share/flame-sheep/telemetry.jsonl').expanduser()

# Default sampling cadence. Task asks for ~1-5s; 2s is a reasonable
# middle ground for "long real-usage periods" without bloating the log.
DEFAULT_INTERVAL_S = 2.0


# ---------------------------------------------------------------------------
# GPU per-engine busy sampler (Intel Arc, Xe driver, via perf_event_open)
# ---------------------------------------------------------------------------
#
# The project already reads the Xe PMU in flame_sheep/scheduler/gpu_load.py,
# but GpuLoadSampler SUMS render+compute into a single busy fraction. We need
# the engines broken out (compute vs render vs video) for the STT-sharing
# question, so we reuse that module's hard-won perf_event_open plumbing and
# add a per-engine read on top. Importing the private helpers avoids a
# parallel perf_event_open implementation that would drift from the original.
#
# BUG-CLASS NOTE: the fact that two callers now want per-engine Xe data is a
# hint that gpu_load.py should grow a first-class per-engine API and
# GpuLoadSampler should become a thin "sum of [render, compute]" wrapper over
# it. Left as a follow-up so this phase stays read-only on existing code.
try:
    from flame_sheep.scheduler.gpu_load import (
        _PerfEventAttr,
        _perf_event_open,
        _read_counter,
        _discover_xe_pmu,
        _build_config,
        _EVENT_ACTIVE_TICKS,
        _EVENT_TOTAL_TICKS,
    )
    _GPU_LOAD_IMPORT_ERROR: str | None = None
except Exception as e:  # pragma: no cover - platform/driver dependent
    _GPU_LOAD_IMPORT_ERROR = repr(e)

# Xe engine class IDs. RENDER=0 and COMPUTE=4 are confirmed by gpu_load.py
# (from the kernel's drivers/gpu/drm/xe/xe_hw_engine_types.h). The VIDEO_*
# ids below are INFERRED from the same header and MUST be verified on-machine
# (dump /sys/bus/event_source/devices/xe_*/events/ to confirm the class
# encoding, or cross-check against `intel_gpu_top`'s engine list if it ever
# runs). If a video engine class is wrong, that engine simply reports 0.0 /
# null; it will not break the other engines.
_ENGINE_CLASSES: dict[str, int] = {
    'render': 0,
    'video_decode': 2,    # INFERRED (XE_ENGINE_CLASS_VIDEO_DECODE)
    'video_enhance': 3,   # INFERRED (XE_ENGINE_CLASS_VIDEO_ENHANCE)
    'compute': 4,
}


@dataclass
class _EnginePair:
    """Open perf fds for one engine's active + total tick counters."""
    name: str
    active_fd: int
    total_fd: int
    prev_active: int = 0
    prev_total: int = 0


class PerEngineGpuSampler:
    """Per-engine Xe busy-fraction sampler.

    sample() returns {engine_name: busy_fraction_over_window} for every
    engine whose counters opened successfully, with busy_fraction in
    0.0..1.0 (may briefly exceed 1.0 on tick-accounting drift). Engines
    that could not be opened are simply absent from the dict; callers
    log null for them.

    Returns an empty dict (and .available == False) when there is no Xe
    PMU or perf_event_open is denied — same graceful-degradation contract
    as scheduler.gpu_load.GpuLoadSampler.
    """

    def __init__(self, engine_classes: dict[str, int] | None = None) -> None:
        self._engines: list[_EnginePair] = []
        self._available = False
        self._unavailable_reason: str | None = None

        if _GPU_LOAD_IMPORT_ERROR is not None:
            self._unavailable_reason = (
                f'gpu_load perf helpers unavailable: {_GPU_LOAD_IMPORT_ERROR}')
            return

        classes = engine_classes if engine_classes is not None else _ENGINE_CLASSES

        pmu = _discover_xe_pmu()
        if pmu is None:
            self._unavailable_reason = (
                'no Xe PMU at /sys/bus/event_source/devices/xe_*/ '
                '(other GPU vendor, or driver not loaded)')
            return
        pmu_type, pmu_cpu = pmu

        for name, klass in classes.items():
            try:
                active_fd = self._open(pmu_type, pmu_cpu,
                                       _EVENT_ACTIVE_TICKS, klass)
                total_fd = self._open(pmu_type, pmu_cpu,
                                      _EVENT_TOTAL_TICKS, klass)
            except PermissionError as e:
                # perf_event_paranoid >= 2 blocks unprivileged uncore
                # reads. Degrade: no GPU engine metrics at all.
                self._unavailable_reason = (
                    f'perf_event_open denied ({e.strerror}); lower '
                    f'kernel.perf_event_paranoid or grant CAP_PERFMON')
                self.close()
                return
            except OSError as e:
                # This engine class probably doesn't exist on this GPU
                # (e.g. no video_enhance). Skip it, keep the rest.
                log.debug('[telemetry] engine %r unavailable: %s', name, e)
                continue
            pair = _EnginePair(name=name, active_fd=active_fd, total_fd=total_fd)
            pair.prev_active = _read_counter(active_fd)
            pair.prev_total = _read_counter(total_fd)
            self._engines.append(pair)

        self._available = bool(self._engines)
        if not self._available and self._unavailable_reason is None:
            self._unavailable_reason = 'no Xe engine counters opened'

    @staticmethod
    def _open(pmu_type: int, pmu_cpu: int, event: int, klass: int) -> int:
        attr = _PerfEventAttr()
        attr.type = pmu_type
        attr.size = ctypes.sizeof(_PerfEventAttr)
        attr.config = _build_config(event, klass)
        # pid=-1, cpu=N -> system-wide on the PMU's CPU (uncore-style).
        return _perf_event_open(attr, pid=-1, cpu=pmu_cpu,
                                group_fd=-1, flags=0)

    @property
    def available(self) -> bool:
        return self._available

    @property
    def unavailable_reason(self) -> str | None:
        return self._unavailable_reason

    def sample(self) -> dict[str, float]:
        """Busy fraction per engine since the previous call."""
        out: dict[str, float] = {}
        if not self._available:
            return out
        for pair in self._engines:
            try:
                active = _read_counter(pair.active_fd)
                total = _read_counter(pair.total_fd)
            except OSError:
                continue
            d_active = active - pair.prev_active
            d_total = total - pair.prev_total
            pair.prev_active = active
            pair.prev_total = total
            out[pair.name] = (d_active / d_total) if d_total else 0.0
        return out

    def close(self) -> None:
        import os
        for pair in self._engines:
            for fd in (pair.active_fd, pair.total_fd):
                try:
                    os.close(fd)
                except OSError:
                    pass
        self._engines.clear()
        self._available = False


# ---------------------------------------------------------------------------
# VRAM query (one-shot) via Vulkan VK_EXT_memory_budget
# ---------------------------------------------------------------------------

def query_vram_once(device_index: int = 0) -> dict[str, Any] | None:
    """Best-effort one-shot VRAM query via a short-lived Vulkan instance.

    Returns a dict with total / used / budget bytes for the device-local
    heap, plus the device name, or None on any failure.

    WHY A NEW INSTANCE: the live wallpaper is OpenGL, so there is no
    Vulkan device to reuse (see module docstring). We create a minimal
    Vulkan instance, read the memory budget, and tear it down.

    GOTCHA (verify on-machine): VkPhysicalDeviceMemoryBudgetPropertiesEXT
    reports heapUsage as an estimate for THIS process only — a fresh
    probe instance that has allocated nothing will report ~0 usage, NOT
    the GL wallpaper's usage. heapBudget, however, is the amount this
    process may use given system-wide pressure, i.e. total minus what
    OTHER processes (the GL wallpaper, the compositor, etc.) already hold.
    So the useful system-wide estimate is:

        system_used ~= heap.size - heapBudget   (+ this probe's heapUsage)

    We return size, budget, and usage separately so the offline analysis
    can decide. Mark this estimate as needing validation against a known
    ground truth (e.g. compare size-budget to what the renderer's buffer
    allocations predict at the live canvas resolution).

    RISK: creating a Vulkan instance inside the GL render process is
    untested. This function is wrapped by the caller so a Python-level
    exception is logged and nulled; a hard GPU/driver crash is NOT
    catchable here. If on-machine testing shows any instability, switch
    to the subprocess-isolated variant described in the hookmap doc.
    """
    try:
        import vulkan as vk
    except Exception as e:
        log.debug('[telemetry] vulkan module unavailable: %s', e)
        return None

    instance = None
    try:
        app_info = vk.VkApplicationInfo(
            pApplicationName='flame-sheep-telemetry',
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName='flame-sheep',
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_API_VERSION_1_1,
        )
        # VK_KHR_get_physical_device_properties2 is core in 1.1 but we
        # request the instance extension too for driver portability.
        want_inst_ext = ['VK_KHR_get_physical_device_properties2']
        try:
            avail = {e.extensionName for e in
                     vk.vkEnumerateInstanceExtensionProperties(None)}
            inst_ext = [e for e in want_inst_ext if e in avail]
        except Exception:
            inst_ext = []
        create_info = vk.VkInstanceCreateInfo(
            pApplicationInfo=app_info,
            enabledExtensionCount=len(inst_ext),
            ppEnabledExtensionNames=inst_ext,
        )
        instance = vk.vkCreateInstance(create_info, None)

        devices = vk.vkEnumeratePhysicalDevices(instance)
        if not devices:
            return None
        phys = devices[min(device_index, len(devices) - 1)]
        props = vk.vkGetPhysicalDeviceProperties(phys)
        device_name = str(props.deviceName)

        # Try the budget path (properties2 + VK_EXT_memory_budget chain).
        # The exact pNext-chaining API of the `vulkan` binding is fiddly;
        # if it is not supported we fall back to total-heap-size only.
        budget = None
        usage = None
        total = None
        device_local_heap = None
        try:
            get_props2 = getattr(vk, 'vkGetPhysicalDeviceMemoryProperties2', None)
            budget_struct = getattr(
                vk, 'VkPhysicalDeviceMemoryBudgetPropertiesEXT', None)
            mem_props2_struct = getattr(
                vk, 'VkPhysicalDeviceMemoryProperties2', None)
            if get_props2 and budget_struct and mem_props2_struct:
                bud = budget_struct()
                mp2 = mem_props2_struct(pNext=bud)
                get_props2(phys, mp2)
                heaps = mp2.memoryProperties.memoryHeaps
                n_heaps = mp2.memoryProperties.memoryHeapCount
                dl_bit = vk.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
                for i in range(n_heaps):
                    if heaps[i].flags & dl_bit:
                        device_local_heap = i
                        total = int(heaps[i].size)
                        budget = int(bud.heapBudget[i])
                        usage = int(bud.heapUsage[i])
                        break
        except Exception as e:
            log.debug('[telemetry] memory_budget path failed: %s', e)

        if total is None:
            # Fallback: total device-local heap size only (Vulkan 1.0).
            mem_props = vk.vkGetPhysicalDeviceMemoryProperties(phys)
            dl_bit = vk.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
            for i in range(mem_props.memoryHeapCount):
                if mem_props.memoryHeaps[i].flags & dl_bit:
                    device_local_heap = i
                    total = int(mem_props.memoryHeaps[i].size)
                    break

        return {
            'device_name': device_name,
            'device_local_heap': device_local_heap,
            'total_bytes': total,
            'budget_bytes': budget,
            'usage_bytes': usage,
            # Convenience estimate; see docstring caveat.
            'system_used_estimate_bytes':
                (total - budget) if (total is not None and budget is not None)
                else None,
        }
    except Exception as e:
        log.debug('[telemetry] VRAM query failed: %s', e)
        return None
    finally:
        if instance is not None:
            try:
                import vulkan as vk
                vk.vkDestroyInstance(instance, None)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Genome cost-feature extraction
# ---------------------------------------------------------------------------

def extract_genome_features(genome: Any,
                            db_id: int | None) -> dict[str, Any]:
    """Per-iter cost features + a stable id for a genome.

    Identity:
      - db_id: stable library id when present (None for the ephemeral
        lerp-intermediate genome shown mid-morph).
      - signature: variation_signature() — a sorted bag of active
        variation ids ("+F" suffixed when a final xform is present). This
        is a stable, human-readable structural fingerprint that is the
        same across runs for structurally-identical genomes, so it serves
        as the id for ephemeral/morphed genomes.

    Cost features (what drives per-iteration GPU work in the chaos game):
      - n_transforms: branch count per walker step.
      - n_active_vars: total active variation evaluations per step.
      - active_var_ids: the distinct variation set. The chaos shader is
        specialized per variation-set (see renderer._get_chaos_shader_for_genome
        and scheduler.precompile_warm._genome_tuple), so different sets are
        literally different (differently-costed) pipelines. Some variations
        are far more expensive than others, so logging the set — not just a
        count — lets the offline analysis learn per-variation cost.
      - has_final_xform: an extra unconditional xform applied every step.

    Fails soft: on any error returns {'id_error': ...} with whatever was
    computable, so a weird genome never kills a sample.
    """
    feat: dict[str, Any] = {'db_id': db_id}
    try:
        from flame_sheep.transitions import variation_signature
        feat['signature'] = variation_signature(genome)
    except Exception as e:
        feat['signature'] = None
        feat['id_error'] = repr(e)
    try:
        transforms = getattr(genome, 'transforms', []) or []
        feat['n_transforms'] = len(transforms)
        feat['has_final_xform'] = getattr(genome, 'final_xform', None) is not None
        var_ids: set[int] = set()
        n_active = 0
        import numpy as np
        for tr in transforms:
            variations = getattr(tr, 'variations', None)
            if variations is not None:
                idx = np.where(np.abs(variations) > 1e-6)[0]
                n_active += int(len(idx))
                var_ids.update(int(i) for i in idx)
            pre = getattr(tr, 'pre_variations', None)
            if pre is not None:
                idx = np.where(np.abs(pre) > 1e-6)[0]
                n_active += int(len(idx))
                var_ids.update(int(i) for i in idx)
        feat['n_active_vars'] = n_active
        feat['active_var_ids'] = sorted(var_ids)
    except Exception as e:
        feat.setdefault('id_error', repr(e))
    return feat


# ---------------------------------------------------------------------------
# Audio-reactivity metric extraction
# ---------------------------------------------------------------------------

def _extract_audio_metrics(snap: AudioSnapshot | None) -> dict[str, Any]:
    """Scalar audio-reactivity metrics from an AudioSnapshot.

    Excludes arrays (spectrum/waveform). These are the features that the
    detail/genome/palette axes actually react to, so they are what we need
    to correlate iter_count and GPU load against. Fails soft field by
    field.
    """
    if snap is None:
        return {}
    out: dict[str, Any] = {}
    for key in ('mode', 'bpm', 'effective_bpm', 'tempo_confidence',
                'percussiveness', 'spectral_novelty', 'section_change',
                'break_intensity', 'centroid', 'centroid_delta',
                'centroid_rms', 'centroid_harmonic_rms',
                'slow_centroid_harmonic_rms'):
        try:
            val = getattr(snap, key)
            out[key] = round(val, 4) if isinstance(val, float) else val
        except Exception:
            out[key] = None
    # Per-band rms / onset density — the detail axis keys off centroid_rms,
    # but per-band is useful for the controller phase.
    try:
        bands: dict[str, dict[str, float]] = {}
        for name, bs in (snap.bands or {}).items():
            bands[name] = {
                'rms': round(bs.rms, 4),
                'onset_density': round(bs.onset_density, 3),
            }
        out['bands'] = bands
    except Exception:
        out['bands'] = None
    return out


# ---------------------------------------------------------------------------
# The sampler
# ---------------------------------------------------------------------------

@dataclass
class _LiveState:
    """Latest per-frame values published by the render loop. Stored by
    reference under the sampler's lock; copied/extracted on the sampler
    thread so publish() stays O(1) and non-blocking."""
    iterations: int | None = None
    genome: Any = None
    genome_db_id: int | None = None
    audio: AudioSnapshot | None = None
    frame_index: int | None = None


class TelemetrySampler:
    """Background GPU + genome telemetry sampler.

    Lifecycle:
        sampler = TelemetrySampler(path=...)
        sampler.start()
        # each frame (cheap, non-blocking):
        sampler.publish(iterations=..., genome=..., genome_db_id=...,
                        audio=orch.audio_state, frame_index=_frame)
        # on shutdown:
        sampler.stop()

    publish() only stores references under a lock. The sampler thread
    wakes every `interval_s`, snapshots the latest published state,
    reads the GPU engines, extracts genome features, and appends one
    JSONL row. The render loop is never blocked by sampling or I/O.
    """

    def __init__(self,
                 path: str | Path = DEFAULT_TELEMETRY_FILE,
                 interval_s: float = DEFAULT_INTERVAL_S,
                 query_vram: bool = True,
                 vram_device_index: int = 0) -> None:
        self._path = Path(path).expanduser()
        self._interval_s = max(0.25, float(interval_s))
        self._query_vram = query_vram
        self._vram_device_index = vram_device_index

        self._lock = threading.Lock()
        self._live = _LiveState()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._file = None
        self._gpu: PerEngineGpuSampler | None = None

    # -- render-loop facing API ------------------------------------------

    def publish(self, *, iterations: int | None = None,
                genome: Any = None,
                genome_db_id: int | None = None,
                audio: AudioSnapshot | None = None,
                frame_index: int | None = None) -> None:
        """Store the latest live values. O(1), non-blocking, never raises
        out (swallows its own errors so a telemetry bug can't take down
        the render loop)."""
        try:
            with self._lock:
                self._live = _LiveState(
                    iterations=iterations,
                    genome=genome,
                    genome_db_id=genome_db_id,
                    audio=audio,
                    frame_index=frame_index,
                )
        except Exception:
            pass

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self._path, 'a')
        except Exception:
            log.exception('[telemetry] could not open %s; disabled', self._path)
            self._file = None
            return
        self._thread = threading.Thread(
            target=self._run, name='telemetry-sampler', daemon=True)
        self._thread.start()
        log.info('[telemetry] sampling to %s every %.1fs',
                 self._path, self._interval_s)

    def stop(self) -> None:
        self._stop_evt.set()
        t = self._thread
        if t is not None:
            t.join(timeout=self._interval_s + 1.0)
        if self._gpu is not None:
            self._gpu.close()
        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                pass
        log.info('[telemetry] stopped')

    # -- sampler thread ---------------------------------------------------

    def _run(self) -> None:
        # Build the GPU sampler on this thread (perf fds are process-wide,
        # but keeping all GPU-sampler state thread-local is tidy).
        try:
            self._gpu = PerEngineGpuSampler()
            if not self._gpu.available:
                log.warning('[telemetry] GPU engine sampling unavailable: %s',
                            self._gpu.unavailable_reason)
        except Exception:
            log.exception('[telemetry] GPU sampler init failed')
            self._gpu = None

        # One-shot header (static per machine / per resolution).
        self._write_header()

        # Prime the per-engine deltas so the first sample's window is real.
        if self._gpu is not None and self._gpu.available:
            self._gpu.sample()

        while not self._stop_evt.wait(self._interval_s):
            try:
                self._write_sample()
            except Exception:
                # Never let a sampling error kill the thread.
                log.exception('[telemetry] sample failed; continuing')

    def _write_header(self) -> None:
        header: dict[str, Any] = {
            'record_type': 'header',
            't': time.time(),
            'interval_s': self._interval_s,
            'schema_version': 1,
        }
        if self._gpu is not None:
            header['gpu_engines_available'] = self._gpu.available
            header['gpu_unavailable_reason'] = self._gpu.unavailable_reason
        if self._query_vram:
            # Done on the sampler thread so it never blocks render startup.
            header['vram'] = query_vram_once(self._vram_device_index)
        else:
            header['vram'] = None
        self._emit(header)

    def _write_sample(self) -> None:
        with self._lock:
            live = self._live
        row: dict[str, Any] = {
            'record_type': 'sample',
            't': time.time(),
            'frame_index': live.frame_index,
            'iter_count': live.iterations,
        }
        # Genome features
        if live.genome is not None:
            row['genome'] = extract_genome_features(live.genome,
                                                    live.genome_db_id)
        else:
            row['genome'] = {'db_id': live.genome_db_id}
        # Audio-reactivity metrics
        row['audio'] = _extract_audio_metrics(live.audio)
        # GPU per-engine busy fractions over the window since last sample.
        # Missing engines -> explicit null so the schema is stable.
        engines = self._gpu.sample() if (self._gpu and self._gpu.available) else {}
        row['gpu_busy'] = {
            'compute': engines.get('compute'),
            'render': engines.get('render'),
            'video': _sum_or_none(engines.get('video_decode'),
                                  engines.get('video_enhance')),
            'video_decode': engines.get('video_decode'),
            'video_enhance': engines.get('video_enhance'),
        }
        self._emit(row)

    def _emit(self, record: dict[str, Any]) -> None:
        if self._file is None:
            return
        try:
            self._file.write(json.dumps(record) + '\n')
            self._file.flush()
        except Exception:
            log.exception('[telemetry] write failed')


def _sum_or_none(*vals: float | None) -> float | None:
    """Sum the non-None values; return None only if all are None."""
    present = [v for v in vals if v is not None]
    return sum(present) if present else None
