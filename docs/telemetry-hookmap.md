# Telemetry hook map (Phase 1 — measurement only)

Seed material for the telemetry system and a future flame-sheep dev skill.
Goal of Phase 1: log, per sample over long real-usage periods, the data that
will (a) answer whether a local Whisper STT job can share the Intel Arc GPU
with the wallpaper, and (b) later seed an adaptive iteration-budget
controller. **No controller in this phase.** Nothing here feeds back into
the render loop.

All file:line references are against the tree as of this writing; names are
the load-bearing part if lines drift.

---

## TL;DR — three findings that change the plan

1. **The live wallpaper is OpenGL, not Vulkan.** `_run_wallpaper` builds a
   `moderngl` context (`runtime/wallpaper.py:125`), and the chaos game runs as
   GL compute shaders (`rendering/chaos.py:134 dispatch`). Vulkan
   (`wallpaper_ml.VkCompute`) is used **only** by the offline ML trainers, in
   separate processes. There is **no live Vulkan device to reuse** for a VRAM
   query. The task's "query VRAM via VK_EXT_memory_budget on the existing
   device" premise does not hold for the live process — see **VRAM query**.

2. **`intel_gpu_top` will not work on this machine.** The project already
   reads the GPU via the **Xe** PMU (`scheduler/gpu_load.py`), and its header
   states plainly: the user is on the **Xe driver** and `intel_gpu_top` is
   **i915-only**. So the task's primary suggestion (shell out to
   `intel_gpu_top`) is a dead end here; the DRM PMU path (its listed
   alternative) is the one that works and is already in the codebase. The
   telemetry module reuses that plumbing.

3. **VRAM is effectively static — CONFIRMED (with caveats).** GPU buffers are
   allocated once at init, sized by canvas resolution, and written into (not
   reallocated) per frame / per genome. See **VRAM-static verdict**.

---

## Architecture overview

```
_run_wallpaper (runtime/wallpaper.py)
  ├─ WallpaperSession → moderngl GL context  (OpenGL, EGL, wlr-layer-shell)
  ├─ FlameRenderer (rendering/renderer.py)   ← owns all GL buffers (SSBOs)
  │    └─ ChaosGame (rendering/chaos.py)      ← per-frame compute dispatch
  ├─ Orchestrator (runtime/orchestrator.py)  ← owns audio engine + ctl pipe
  │    └─ audio_state : AudioSnapshot         ← overwritten every orch.tick()
  └─ FlameSheepCore (runtime/core.py)         ← visual axes state machine
       ├─ GenomeAxis   → current/target Genome, morph
       ├─ DetailAxis   → iterations (audio energy → iter count)   ★
       ├─ PaletteAxis, ZoomAxis, BrightnessAxis
       └─ tick(frame_time) → FrameState(genome, palette, spectrum,
                                        brightness, iterations)

Main loop (runtime/wallpaper.py:397): per frame
   orch.tick()                      → refresh audio_state, deliver events
   frame = core.tick(frame_time)    → advance axes, produce FrameState
   renderer.upload_genome(frame.genome); upload_palette; upload_audio
   renderer.dispatch_chaos_game(iterations=frame.iterations)   ★ iter count
   renderer.render_tonemap(...) per surface; session.swap(...)
```

GA / background workers run in **separate processes/threads** and do not
touch the live render GPU path: `genome.score_worker.BackgroundCpuScorer`,
`transitions.BackgroundTransitionScorer`, `genome.pruner.BackgroundPruner`
(started at `runtime/wallpaper.py:318-324`); the GPU render worker is run
manually as its own process, never inside the wallpaper.

---

## Hook points

### 1. Main frame / render loop
- **File:** `flame_sheep/runtime/wallpaper.py`
- **Function:** `_run_wallpaper(...)`, the `while` loop at **`:397`**.
- Per-frame landmarks:
  - `orch.tick()` — **`:400`** (refreshes `orch.audio_state`).
  - `frame = core.tick(frame_time)` — **`:476`** (produces `FrameState`).
  - `renderer.dispatch_chaos_game(iterations=frame.iterations)` — **`:521`**
    (the GL compute dispatch; `iterations` is the live iter count).
  - Optional existing `AudioFeatureLogger.tick()` — **`:401-402`** (the model
    to follow for a non-blocking in-loop logger call).
- Frame counter `_frame` (**`:355`**), framerate cap via `cfg.max_fps`
  (**`:455`**).

### 2. Audio-reactivity state (energy / beat / onset metrics)
- **Held on:** `Orchestrator.audio_state : AudioSnapshot`
  (`runtime/orchestrator.py:73`, refreshed in `Orchestrator.tick()` at
  **`:110-117`** via `self.audio.drain()`).
- **Schema:** `AudioSnapshot` (`flame_sheep_audio/src/flame_sheep_audio/_types.py:78`).
  Scalar reactivity fields: `centroid`, `centroid_delta`, `centroid_rms`,
  `centroid_harmonic_rms`, `slow_centroid_harmonic_rms`, `percussiveness`,
  `spectral_novelty`, `section_change`, `bpm`, `effective_bpm`,
  `tempo_confidence`, `break_intensity`, `mode`; per-band `bands[name]`
  (`BandState`: `rms`, `harmonic_rms`, `onset_density`, `density_delta`, …
  at `_types.py:26`); discrete `events: list[BeatEvent]`.
- Core re-bundles a subset into `AudioState` each tick
  (`runtime/core.py:123-141`) for the axes.

### 3. Mapping audio → ITERATION COUNT (the per-frame iter count)
- **File:** `flame_sheep/axes/detail_axis.py`
- **Class/method:** `DetailAxis.tick()` — **`:41-48`**. Maps an asymmetric
  envelope of `audio.centroid_rms` into `[min_iters, max_iters]`
  (`LIVE_ITER_MIN=100`, `LIVE_ITER_MAX=500`, `render_params.py:22-23`), stores
  on `self.iterations`.
- **Published to the frame:** `DetailAxis.contribute()` sets
  `frame.iterations` (**`detail_axis.py:50-51`**); `FrameState.iterations`
  field is declared at `runtime/core.py:106` and assigned at
  `runtime/core.py:168`.
- **Consumed:** `renderer.dispatch_chaos_game(iterations=frame.iterations)`
  at `runtime/wallpaper.py:521` → `ChaosGame.dispatch()` sets
  `u_iterations` (`rendering/chaos.py:134-142`).
- So the live iter count for a given frame is exactly `frame.iterations`
  (equivalently `core._detail_axis.iterations`).

### 4. Genome object — identity + cost features
- **Class:** `Genome` dataclass (`flame_sheep/genome/__init__.py:72`).
- **Identity:**
  - `Genome.db_id : int | None` (**`:84`**) — stable library id, set when
    loaded from the DB; **None for ephemeral genomes**, including the
    lerp-intermediate genome that is actually on screen mid-morph.
  - `FlameSheepCore.active_genome_db_id` (`runtime/core.py:248-257`) — db_id
    of the **dominant** endpoint (current if `morph_t < 0.5` else target).
    This is the right "which library genome is this" signal to log.
  - **No hash / `__hash__` exists.** For a stable id of the *actual rendered*
    (possibly ephemeral) genome, use `variation_signature(genome)`
    (`transitions/signature.py:32`) — a sorted bag of active variation ids
    with a `+F` suffix for the final xform. Stable across runs, human
    readable. (A content hash of `to_gpu_arrays()` bytes would be an
    alternative but is overkill and not needed here.)
- **Cost-relevant features** (drive per-iteration GPU work):
  - `len(genome.transforms)` — branch count per walker step (`MAX_TRANSFORMS=6`).
  - active variation **set** per transform (`tr.variations`, `tr.pre_variations`,
    threshold `|w|>1e-6`). The chaos shader is **specialized per
    variation-set** — see `renderer._get_chaos_shader_for_genome`
    (`rendering/renderer.py:470`) and the existing cost key
    `scheduler.precompile_warm._genome_tuple` → `(n_transforms, has_final,
    frozenset(keep_vars))` (`scheduler/precompile_warm.py:28-39`). Different
    sets are different, differently-costed pipelines, and some variations are
    much more expensive, so **log the set, not just a count**.
  - `genome.final_xform is not None` (**`:76`**) — an extra unconditional
    xform every step.
  - No per-genome iteration/quality knob exists; quality knobs that matter are
    global: `N_WALKERS` (fixed 65536, `render_params.py:12`) and the live
    `iterations` (hook 3). Tonemap-only knobs (`flam3_brightness`,
    `flam3_gamma`, `:87-88`) do not affect chaos-game cost.
  - `telemetry.extract_genome_features()` packages all of this.

### 5. Vulkan device handle (for VRAM)
- **There is no live Vulkan device.** The live renderer is `moderngl`
  (`runtime/wallpaper.py:125`, `rendering/gpu_context.py` re-exports
  `viz_authoring.gpu_context.GpuContext`, a GL wrapper).
- The only Vulkan device-handle code is `wallpaper_ml.VkCompute`
  (`wallpaper_ml/src/wallpaper_ml/vk_compute.py:90`): `self.instance`
  (`:136`), `self.physical_device` (`:142`), `self.device` (`:167`),
  `self.mem_props = vkGetPhysicalDeviceMemoryProperties(...)` (`:145`).
  Note it requests **`VK_API_VERSION_1_0`** and enables **no** instance/device
  extensions, so it cannot query `VK_EXT_memory_budget` as-is, and it is only
  ever constructed in trainer **subprocesses**, not the wallpaper.
- Consequence for VRAM: see below.

### 6. VRAM-static verdict — CONFIRMED (with two caveats)
All GL buffers are allocated once in `FlameRenderer._create_resources()`
(`rendering/renderer.py:310-398`), sized by canvas resolution:
- histogram SSBO `n_px*2*4` (`:318-321`); DE in/out histograms (`:324-329`);
  per-transform hits `n_px*MAX_TRANSFORMS*4` (`:332-334`); downsample buffer
  (`:339-341`); `_max_buf` (`:344-345`); palette + audio textures
  (`:348-356`); genome SSBOs sized by `MAX_TRANSFORMS+1` slots — **fixed**,
  not per-genome (`:358-391`); walker SSBO sized by `n_walkers` (fixed 65536,
  `:393-398`).
- Per **genome**: `renderer.upload_genome()` **writes into** the fixed genome
  SSBOs (no allocation). Per **frame**: `ChaosGame.dispatch()` only sets
  uniforms + runs (`rendering/chaos.py:134-142`). So VRAM for data buffers is
  static once the canvas is known — **the user's belief is correct.**
- **Caveat 1 (compare mode):** `ChaosGame.ensure_double_histogram()`
  (`chaos.py:48-72`) reallocates the histogram to 2× once, only when compare
  mode is entered. Not the normal wallpaper path.
- **Caveat 2 (shader cache):** `renderer._chaos_shader_cache`
  (`rendering/renderer.py:160`) compiles a **new GL compute program per
  distinct variation-set** encountered. That is pipeline/program memory, not
  buffer memory, and it grows with the *variety* of genomes shown over a
  session (bounded by distinct variation-sets). Worth noting in the header so
  the offline analysis isn't surprised if VRAM creeps slightly over a long
  run. A single header capture is still the right call for Phase 1; the
  controller phase may want a periodic re-check.

---

## VRAM query — recommended approach + the open decision

Because the live process is GL, the telemetry module's `query_vram_once()`
creates its **own** short-lived Vulkan instance (API 1.1 +
`VK_KHR_get_physical_device_properties2`), reads
`VkPhysicalDeviceMemoryBudgetPropertiesEXT` via
`vkGetPhysicalDeviceMemoryProperties2`, and tears it down. Done **once** on
the sampler thread (so it never blocks render startup), fully fail-soft.

Two things to **verify on-machine**:

1. **Safety.** Creating a second graphics-API (Vulkan) context inside the
   live GL render process is **untested**. The `wallpaper-ml` skill warns the
   Mesa Xe KMD has crashed sway under GPU-sharing (fork-based) scenarios. A
   Python exception is caught and nulled, but a hard driver crash is not
   catchable. **If this perturbs the live GL context at all, switch to the
   subprocess-isolated variant:** a tiny `python -m` helper that creates the
   Vulkan instance, prints the budget JSON, and exits — parent reads it with a
   timeout. That matches the project's "GPU work in a separate process"
   discipline and cannot crash sway. (The task said "don't shell out for
   VRAM," meaning don't use a vendor CLI; running our own Vulkan code in a
   child process is a different thing and is the safe fallback.)

2. **Semantics.** `heapUsage` in the budget extension is a **per-process**
   estimate — a fresh probe instance reports ~0 usage, **not** the GL
   wallpaper's usage. `heapBudget` is what *this* process may use given
   system-wide pressure, i.e. `total − (other processes' usage)`. So the
   useful system-wide number is `system_used ≈ heap.size − heapBudget`. The
   module returns `total_bytes`, `budget_bytes`, `usage_bytes`, and the
   `system_used_estimate_bytes = total − budget` so the offline analysis can
   decide. Validate `size − budget` against the renderer's predicted buffer
   footprint at the live canvas resolution before trusting it.

   Sysfs is a possible no-Vulkan fallback for VRAM on Xe
   (`/sys/class/drm/card*/device/...` / fdinfo), but the exact attribute names
   differ by kernel/driver version and some need debugfs/root — left
   unimplemented pending on-machine inspection.

---

## GPU engine utilization — what the module does

- **Source:** the **Xe PMU** via `perf_event_open`, reusing the plumbing in
  `scheduler/gpu_load.py` (`_discover_xe_pmu`, `_build_config`,
  `_PerfEventAttr`, `_perf_event_open`, `_read_counter`, the
  `_EVENT_ACTIVE_TICKS`/`_EVENT_TOTAL_TICKS` ids). `GpuLoadSampler` there sums
  render+compute into one fraction; telemetry needs them split, so
  `PerEngineGpuSampler` opens per-engine fds and returns a per-engine
  busy-fraction dict.
- **Engine classes:** RENDER=0 and COMPUTE=4 are **confirmed** by
  `gpu_load.py` (kernel `xe_hw_engine_types.h`). VIDEO_DECODE=2 and
  VIDEO_ENHANCE=3 are **INFERRED** from the same header — **verify on-machine**
  (dump `/sys/bus/event_source/devices/xe_*/events/`). A wrong video class
  just reports null for that engine; it doesn't break the others. Multi-
  instance media engines are sampled at instance 0 only — verify whether a
  second video instance exists and needs summing.
- **Driver note:** this is **Xe-specific**. `intel_gpu_top` is i915-only and
  will not provide this on the user's machine (per `gpu_load.py` header).
- **Permissions:** `perf_event_open` for uncore needs
  `kernel.perf_event_paranoid <= 1` or `CAP_PERFMON`. On denial the sampler
  degrades to null GPU metrics (it does not crash). **Verify** the wallpaper's
  runtime has the needed paranoid level / capability.
- **BUG-CLASS follow-up:** two callers now want per-engine Xe data. Ideally
  `gpu_load.py` grows a first-class per-engine API and `GpuLoadSampler`
  becomes a thin `sum([render, compute])` wrapper over it. Deferred to keep
  this phase read-only on existing code; the telemetry module imports the
  private helpers to avoid a parallel perf_event_open implementation.

---

## Output schema (JSONL, append-mode, configurable path)

Default path `~/.local/share/flame-sheep/telemetry.jsonl` (mirrors the audio
feature logger). Two record types:

**Header** (once at startup):
```json
{"record_type":"header","t":<unix>,"interval_s":2.0,"schema_version":1,
 "gpu_engines_available":true,"gpu_unavailable_reason":null,
 "vram":{"device_name":"...","total_bytes":...,"budget_bytes":...,
         "usage_bytes":...,"system_used_estimate_bytes":...}}
```

**Sample** (every `interval_s`):
```json
{"record_type":"sample","t":<unix>,"frame_index":12345,"iter_count":317,
 "genome":{"db_id":1234,"signature":"18,23,26+F","n_transforms":4,
           "n_active_vars":6,"active_var_ids":[18,23,26],
           "has_final_xform":true},
 "audio":{"mode":"beat","bpm":128.0,"centroid_rms":0.21,"percussiveness":0.6,
          "break_intensity":0.0,"bands":{"low":{"rms":..,"onset_density":..}}},
 "gpu_busy":{"compute":0.07,"render":0.44,"video":0.0,
             "video_decode":0.0,"video_enhance":0.0}}
```
Any unavailable field is `null`; the key set is stable across rows.

---

## Integration plan (edits are the user's to apply — NOT applied here)

All edits are in `flame_sheep/runtime/wallpaper.py`, alongside the existing
`AudioFeatureLogger` wiring. ~4 touch points, ~8 lines total. `publish()` is
O(1) and lock-guarded; it does no GPU work and no I/O, so it will not pace the
render loop.

1. **Construct + start** — near the feature-logger block (`:349-352`), after
   `orch.start()` (`:346`):
   ```python
   telemetry = None
   if cfg_telemetry_enabled:                      # or an argparse flag
       from ..telemetry import TelemetrySampler
       telemetry = TelemetrySampler(
           path=cfg.telemetry.path,               # see config stub below
           interval_s=cfg.telemetry.interval_s)
       telemetry.start()
   ```
   (Simplest MVP: unconditionally construct with defaults; gate later.)

2. **Publish each frame** — immediately after `frame = core.tick(frame_time)`
   (`:476`), before the GPU-paused check:
   ```python
   if telemetry is not None:
       telemetry.publish(
           iterations=frame.iterations,
           genome=frame.genome,
           genome_db_id=core.active_genome_db_id,
           audio=orch.audio_state,
           frame_index=_frame)
   ```

3. **Stop on shutdown** — in the `finally:` block, beside
   `if feature_logger: feature_logger.close()` (`:629-630`):
   ```python
   if telemetry is not None:
       telemetry.stop()
   ```

4. **(Optional) config** — add a `telemetry` block to
   `config.py:DEFAULTS` and `default_config.toml`:
   ```toml
   [telemetry]
   enabled = false
   path = "~/.local/share/flame-sheep/telemetry.jsonl"
   interval_s = 2.0
   ```
   Access via `cfg.telemetry.enabled` etc. (A `--log-telemetry` argparse flag
   in `main.py` next to `--log-features` is the alternative; pick one — don't
   ship both as competing toggles.)

No other files need edits. The sampler reads `orch.audio_state`,
`frame.genome`, `frame.iterations`, and `core.active_genome_db_id` — all
already live at the publish point.

---

## Open questions / verify on-machine (user is away from the box)

1. **Xe video engine class ids** (2 / 3) — confirm against
   `/sys/bus/event_source/devices/xe_*/events/`; check for multiple video
   engine instances.
2. **`perf_event_paranoid` / `CAP_PERFMON`** in the wallpaper's runtime —
   does `perf_event_open` for uncore succeed unprivileged? (The systemd user
   service may need adjustment.)
3. **Vulkan-in-GL-process safety** — does `query_vram_once()` run cleanly
   alongside the live moderngl context, or does it perturb/crash it? If any
   instability, move it to the subprocess-isolated variant.
4. **`vulkan` binding specifics** — confirm `vkGetPhysicalDeviceMemoryProperties2`
   and `VkPhysicalDeviceMemoryBudgetPropertiesEXT` pNext-chaining work in the
   installed `vulkan` package; otherwise it falls back to total-heap-size only
   (used/budget come back null).
5. **`heapBudget` semantics on Arc** — validate `total − budget` against the
   renderer's predicted buffer footprint at the live canvas resolution.
6. **Shader-cache VRAM creep** — over a long session, does
   `_chaos_shader_cache` growth move VRAM enough to matter? If so the
   controller phase wants periodic VRAM re-capture, not just the header.
7. **Sampling cadence** — 2s default; confirm it's frequent enough to catch
   STT-contention transients but not so frequent the log bloats over
   multi-hour runs.
