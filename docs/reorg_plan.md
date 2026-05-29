# Codebase Reorganization Plan

The `flame_sheep/` package has grown organically from a wayland-wallpaper-running-electric-sheep into a multi-system research codebase. The flat namespace and several junk-drawer files no longer reflect the actual structure. This plan reshapes the layout so the directory structure communicates the design.

This is for the project author's own sensibilities — internal-satisfaction is the driver, not external review. So choices favor "where would I look for X" over "what would impress a reviewer."

## Status (as of 2026-05-28)

| Stage | Description | Status |
|-------|-------------|--------|
| 0 | FlameRenderer decomposition | **Done** — GpuContext extracted earlier; this session split FlameRenderer into 3 sub-components: `ChaosGame` (chaos.py, dispatch + histogram mgmt), `TonemapPipeline` (tonemap.py, render + blur + temporal blend), `SnapshotPipeline` (snapshot.py, PNG snapshot variants + DE). renderer.py 808 → 419 lines; each unit passes the "describe in one sentence" test. Old method names kept as delegators — no caller changes. |
| 1 | Extract `wallpaper_ml/` to standalone package | **Done** — package at `wallpaper_ml/src/wallpaper_ml/`; thin re-export shim at `flame_sheep/wallpaper_ml.py` for back-compat. |
| 2 | Split `storage.py` + audio cross-package cleanup | **Done** — `storage/{library,schema,serialization,catalog,esheep_parser}.py`; `palette/scoring.py`; `audio/tempo.py`. |
| 3 | Group siblings into packages (runtime/, rendering/, audio/, ui/, etc.) | **Done** — 3a/b/d/e committed during May 2026 reorg work. |
| 4 | Object-organize genome / palette / transitions / loops | **Done** — 4a transitions/, 4b genome/ + workers + motion_field (1-4), 4c loops/ split + score_loop move. |
| 5 | Tests mirror + docs sweep | **Done** — `tests/` mirrors source package layout (commits `6b48bbb`, `8120492`, `0217d42`); this status section is the docs sweep. |
| 6 | Extract shared `load_aware.py` from workers | **Done** (commit `ede7ffd`) — 5 workers migrated; `LOAD_THRESHOLD`/`LOAD_CHECK_INTERVAL` centralized; `IDLE_CHECK_INTERVAL` kept per-worker (principled differences). |
| 7 | Comm-pipe API polish (`debug/` uses only public names) | **Done** — all production code (`flame_sheep/`) and `tools/` import only from `flame_sheep_audio` public surface. `a_weight_curve` promoted to public. Tests retain underscore reaches by design (they test internals). |
| 8a | Worker subprocess-bootstrap dedup (logging/nice/DB) | **Done** — new `flame_sheep/worker_bootstrap.py` `init_worker_subprocess()` helper; pruner / score / transitions migrated. Render worker keeps bespoke bootstrap because of its 10-second GPU-yield sleep interleaving with bootstrap steps. Dead `BackgroundScorer` deleted. |
| 8b | Full worker patterns (`GpuWorker` / `CpuWorkerPool` / `ScheduledTask`) + `LoadMonitor` pub/sub | **Pending** — bundled with Vulkan transition since GPU-aware scheduling needs Vulkan primitives. Feeds the GPU scheduler design (see `~/.claude/projects/-home-chaos/memory/project_gpu_scheduler_design.md`). |
| 9 | `viz_authoring/` extraction | **Pending** — gated on 6+7+8. |

The remainder of this document is the original planning text — kept as historical reference for *why* each stage was structured the way it was. Body uses future-tense ("will move X → Y"); read past tense for stages 0-5.

## Goals

- **Organize by object, not by operation**: each domain object (genome, palette, loop, transition) owns its scoring, generation, evolution, etc. No top-level `scoring/` directory crossing object boundaries — `score_loop` lives in `loops/`, `score_palette` in `palette/`, etc.
- **Layout communicates structure**: directory should tell you what each thing is
- **Files do one job**: split the junk drawers (storage.py 76KB, wallpaper_ml.py 40KB)
- **Sibling pattern → package**: when N files share a suffix, that's a missed package (workers, etc.)
- **Naming clarifies purpose**: rename log.py / logger.py since they're not what their names suggest
- **Staged migration**: each stage independently shippable, doesn't require breaking the codebase mid-refactor

## Non-goals

- Renaming internal symbols (functions, classes) unless they're moving anyway
- Behavior changes — pure reorganization
- Performance changes — pure reorganization
- Refactoring tools/ aggressively (lower-stakes than the package proper)
- Refactoring tests/ until the package layout is stable (then mirror it)

## Repository structure

Three sibling installable packages in the monorepo plus a future fourth, plus tools/docs:

```
flame-sheep/                  # repo root
  pyproject.toml              # top-level workspace
  flame_sheep/                # main package — the wallpaper application
  flame_sheep_audio/          # audio analysis package (already separate)
    pyproject.toml
    src/flame_sheep_audio/
      ...
      tempo.py                # ABSORB flame_sheep/tempo.py here if non-duplicate
  wallpaper_ml/               # NEW: Vulkan ML framework, extracted from flame_sheep
    pyproject.toml
    src/wallpaper_ml/
      __init__.py
      vk_compute.py
      layers.py
      losses.py
      models.py
  viz_authoring/              # FUTURE (not in this refactor): viz author helper library
    # See "Future extraction" section
  tools/
  docs/
  tests/
```

`wallpaper_ml` becomes a sibling to `flame_sheep_audio`. Both are generic infrastructure that any project could depend on. `flame_sheep` depends on both. `viz_authoring` is a planned future extraction (see below) — not part of this refactor but the rendering layout should anticipate it.

## Target `flame_sheep/` structure (object-organized)

```
flame_sheep/
  __init__.py
  __main__.py
  config.py                  # unchanged
  default_config.toml        # unchanged
  main.py                    # unchanged (entry point)
  logging.py                 # RENAMED from log.py

  runtime/                   # live-wallpaper coordination layer
    __init__.py
    core.py                  # FlameSheepCore
    orchestrator.py
    control.py
    session.py
    mpris.py
    command_handlers.py
    role_mapper.py
    wallpaper.py             # the wallpaper command's main loop

  axes/                      # unchanged (already organized)

  genome/                    # everything genome — model, breeding, scoring, workers
    __init__.py              # re-exports Transform, Genome
    transform.py             # Transform dataclass
    genome.py                # Genome dataclass + breed/mutate methods
    symmetry.py              # moved from top-level
    motion_field.py          # MOVED from storage — geometric, not persistence
    render_worker.py         # MOVED from gpu_render_worker.py
    score_worker.py          # MOVED from cpu_score_worker.py
    pruner.py                # MOVED from pruner_worker.py
    scoring/                 # genome-specific scoring (multiple files justify subdir)
      __init__.py
      cnn_scorer.py
      scoring_channels.py
      cluster_scorer.py
      image_scorer.py
      scorer.py              # subprocess entrypoint — verify python -m path after move
      gpu.py                 # MOVED from gpu_scorer.py (was missed in earlier plan)
      experimental_metrics.py
      histogram.py           # _score_from_histogram (from genome.py)
      symmetry_score.py      # _score_symmetry (from genome.py)

  palette/                   # everything palette — generation, scoring
    __init__.py
    generation.py            # _random_palette + future spline generators
    scoring.py               # score_palette (MOVED from storage.py)
    # future: cnn_scorer.py once distilled

  transitions/               # genome-pair distance (shared by genome + loops)
    __init__.py
    distance.py              # compute_transition_distance + helpers (from transition.py)
    signature.py             # variation_signature etc.
    worker.py                # MOVED from flame_sheep/transition_worker.py

  loops/                     # everything loops — composition, evolution, scoring
    __init__.py
    candidate.py             # LoopCandidate dataclass
    compose.py               # compose_loops, compose_loops_graph
    evolve.py                # crossover, mutate_*, evolve_loops
    prune.py                 # prune_duplicate, prune_low_fitness
    sequence.py              # loop_sequence, cycle_length
    scoring.py               # score_loop (MOVED from storage.py)

  storage/                   # persistence layer (no scoring code; pure storage)
    __init__.py              # re-exports Library
    library.py               # Library class
    schema.py                # _ensure_schema + all migrations
    serialization.py         # _transform_to_dict, _genome_to_json
    catalog.py               # absorbed from top-level flame_sheep/catalog.py
    esheep_parser.py         # absorbed from top-level
    # NOTE: compute_motion_field + motion_field_coherence move to genome/motion_field.py
    # (they're geometric operations on genome pairs, not persistence — see "Cycle avoidance" below)

  rendering/                 # GPU rendering — internal structure depends on Stage 0
    __init__.py
    # If Stage 0 (FlameRenderer decomposition) has run:
    #   gpu_context.py, window.py, surface.py, shader_loader.py, pipeline.py — generic plumbing
    #   flame_renderer.py — flame-specific dispatch
    # If Stage 0 has NOT run yet:
    #   renderer.py — the existing monolithic FlameRenderer (1174 lines)
    #   window.py — RENAMED from wayland_window.py
    #   surface.py — RENAMED from display.py
    shaders/                 # flame fractal shaders (unchanged location)

  # NOTE: no workers/ package — all workers are object-specific and live with their object:
  #   genome/render_worker.py     (was gpu_render_worker.py)
  #   genome/score_worker.py      (was cpu_score_worker.py)
  #   genome/pruner.py            (was pruner_worker.py)
  #   transitions/worker.py       (was transition_worker.py)

  audio/                     # consumer of flame_sheep_audio (the package)
    __init__.py
    client.py                # RENAMED from audio_client.py
    feature_logger.py        # RENAMED from logger.py

  ui/                        # interactive UI surfaces
    __init__.py
    compare.py
    library_cli.py
    benchmark.py

  variations/                # unchanged
  eval/                      # unchanged (osu eval lands here)
  protocol/                  # unchanged
  data/                      # unchanged (data files only)
  debug/                     # unchanged
```

Note the dependency flow this enables (downstream to upstream):
- `genome/` depends on `palette/` (Genome dataclass has palette field) — and own motion_field, workers, etc.
- `palette/` depends on nothing else
- `transitions/` depends on `genome/`
- `loops/` depends on `genome/`, `transitions/` (NOT palette — loops hold genome IDs, palette is genome-internal)
- `storage/` depends on `genome/`, `palette/`, `loops/` (Library calls `score_loop` from loops/scoring.py)
- `rendering/` depends on `genome/`, `palette/` (renders genomes with palettes)
- `audio/` depends on `flame_sheep_audio` (external package)
- `runtime/`, `ui/` depend on everything

### Cycle avoidance (important — caught in review)

A naive arrangement would create `storage → loops → storage`:
- `score_loop` moves to `loops/scoring.py`
- `compute_loop_scores` in storage calls `score_loop` (storage → loops)
- `loops/compose.py` imports `compute_motion_field` (loops → storage)
- Cycle.

The fix: **`compute_motion_field` and `motion_field_coherence` move to `genome/motion_field.py`, not into `storage/`**. They're geometric operations on genome pairs, not persistence. With that move:
- `storage → loops` for loop scoring
- `loops → genome` for motion field
- `genome` doesn't import either — DAG is clean

Worker dependencies are runtime-only (lazy imports inside methods), not import-time. So `genome/render_worker.py` importing `FlameRenderer` from `rendering/` at call time doesn't create an import-time cycle even though it'd look like one in the static dep graph.

## Decisions

1. **Object-organized, not operation-organized.** No top-level `scoring/` package. Each object owns its own scoring code: `loops/scoring.py`, `palette/scoring.py`, `genome/scoring/*.py`.

2. **`tempo.py` folds into `flame_sheep_audio`.** Already has tempo modules there (`tempo.py`, `tempo_acf.py`, `tempo_btrack.py`, `tempo_scaler.py`). The `flame_sheep/tempo.py` is either a duplicate of one of those or a thin consumer wrapper — diff before deleting to confirm.

3. **`transition.py` becomes its own `transitions/` sub-package.** Not because transitions are first-class objects but because they're a shared concept between `genome/` and `loops/`. Putting them in either would force an asymmetric dependency.

4. **`wallpaper_ml` extracts to top-level standalone package.** Same shape as `flame_sheep_audio`. Generic ML infrastructure usable beyond flame-sheep. Has its own `pyproject.toml`, installs separately, gets a separate test suite.

5. **`storage/` splits.** 76KB junk drawer → 5-6 focused files. No code lives in storage that doesn't actually do persistence (score_palette and score_loop move to their respective object packages).

6. **`__init__.py` re-exports at package level.** External code writes `from flame_sheep.genome import Genome`, not `from flame_sheep.genome.genome import Genome`. Internal code inside `genome/` can use the full path freely.

## Deferred to a future pass

- **`tests/` mirror**: reorganize `tests/visualizer/` etc. to match the new package layout. Done in Stage 5 once package structure is stable.
- **`tools/` reorganization**: lower stakes, can be done independently later.
- **`viz_authoring/` extraction**: see below — a future package extraction once the current refactor is settled.

## Future extraction: `viz_authoring/` package

After this refactor lands, the platform vision is to extract a `viz_authoring` package as a sibling to `flame_sheep_audio` and `wallpaper_ml`. This isn't part of the current refactor — it's a separate future undertaking — but the current refactor should anticipate it so the extraction is straightforward.

### Purpose

A helper library for third-party visualization authors. Provides the generic infrastructure for building audio-reactive Wayland visualizations without each author needing to figure out windowing, GPU context, shader loading, etc. from scratch.

### What goes in `viz_authoring`

From `flame_sheep/rendering/`:
- `gpu_context.py` — GPU context setup
- `window.py` — Wayland window management
- `surface.py` — surface management
- `shader_loader.py` — shader compilation utilities
- `pipeline.py` — generic pipeline scaffolding

Audio helpers (heavier — `flame_sheep_audio`'s lib is daemon-client-only):
- Helpers for working with raw spectrum/waveform from the daemon's shmem
- Envelope followers, easing functions, smoothers
- Audio-to-shader-uniform mapping conveniences (the "high-level tier" from the viz interface design)

Other generic scaffolding:
- Configuration / settings infrastructure
- Hot-reload infrastructure (for shader iteration during dev)

### What stays in `flame_sheep`

- `flame_renderer.py` — the FlameRenderer class with flame-specific dispatch logic
- `shaders/` — the chaos-game shader, histogram, tonemap shaders, all flame-specific GLSL
- Everything genome-specific in `genome/`
- Loops, transitions, palette code
- Runtime coordination
- Compare-mode + library CLI + benchmark

### Dependency relationships after extraction

- `flame_sheep_audio` — daemon + thin client lib. Standalone.
- `wallpaper_ml` — standalone Vulkan ML framework, no internal deps
- `viz_authoring` — depends on `flame_sheep_audio` (client lib for reading from daemon); possibly on `wallpaper_ml` (if ML helpers join later)
- `flame_sheep` — depends on `viz_authoring`, `flame_sheep_audio`, `wallpaper_ml`

This shape lets a third-party author write a viz that depends on `viz_authoring` (which transitively pulls in the audio client lib), without needing to know about flame-fractal-specific code.

### The architectural split: daemon vs library

`flame_sheep_audio` is fundamentally a **daemon-first** package:
- The daemon runs continuously, captures audio, runs all analysis algorithms (beats, tempo, onsets, etc.), publishes results to shmem
- The library portion is just "how to read me" — client API + data types for connecting to the daemon's shmem
- Analysis algorithms live inside the daemon, not exposed as embedded library calls

`viz_authoring` provides the **convenience layer** on top of that:
- Helpers for processing the raw spectrum/waveform from shmem (band-specific analysis, etc.)
- Envelope followers, easing, smoothing
- The "high-level tier" mapping from analysis data → shader uniforms (per the viz interface design)

The benefit of daemon-only analysis: every visualization gets the same beat detection, tempo, onsets. No drift between concurrent visualizations. Single source of truth for "what the music is doing right now."

The cost: third-party authors who want to compute custom features have to do so from the raw data the daemon publishes, not from a Python analysis library. That's what `viz_authoring`'s audio helpers are for — making the raw-data path ergonomic without duplicating the daemon's algorithms.

### Why anticipate now

The renderer.py file currently mixes generic plumbing with flame-specific dispatch logic. The current refactor (Stage 4 above) splits it into separate modules along that boundary. That means when extraction happens later, the move is mechanical (`mv flame_sheep/rendering/gpu_context.py viz_authoring/src/viz_authoring/`), not a re-architecture. Pre-drawing the boundary now is much cheaper than redrawing it twice.

## Staged migration

Each stage is independently shippable and can land as its own PR/commit. Tests pass at the end of each stage.

**Sequence**: Stage 0 → Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5 → (future: Stage 6 viz_authoring extraction).

The viz_authoring near-term horizon (~3 months, debug/ already exists as a toy consumer) means Stage 0 (FlameRenderer decomposition) is justified before the reorg — otherwise Stage 3b's rendering/ split is aspirational, and the viz_authoring extraction later has to do refactor+move simultaneously.

### Stage 0: FlameRenderer decomposition

**Standalone refactor — no file moves, no package boundary changes.** Goal: separate the existing monolithic 1174-line `FlameRenderer` class into units with clear internal boundaries, so the future "what's generic vs flame-specific" question has answers.

Targets:
- **GPU context / windowing** — could be reused by any viz
- **Shader compilation + resource allocation** — generic infrastructure
- **Flame-specific dispatch logic** — chaos game, histogram, tonemap, snapshot — stays flame-specific

Acceptance test: can you describe each unit in one sentence? If not, it's not separable.

Output: still one renderer.py file (or 2-3 sibling files inside flame_sheep/), with the FlameRenderer class either split into multiple classes or with a clear internal layering. No external moves yet. ~1-2 days.

This stage is **only worth doing if viz_authoring is real-near-term**. Skip if the platform vision shifts to "someday" — the reorg works without it (Stage 3b just moves renderer.py as a single file).

### Stage 1: Extract `wallpaper_ml` to standalone package

The biggest structural change — moving code out of `flame_sheep/` entirely.

- Create `wallpaper_ml/` directory at repo root with `pyproject.toml`, `src/wallpaper_ml/`, `tests/`
- Split `flame_sheep/wallpaper_ml.py` (40KB) into:
  - `wallpaper_ml/src/wallpaper_ml/layers.py` (VkLayer + Vk{Conv2d,Linear,GAPLinear,GAPMLP,GRU})
  - `wallpaper_ml/src/wallpaper_ml/losses.py` (bce_loss_dispatch, cross_entropy_dispatch)
  - `wallpaper_ml/src/wallpaper_ml/models.py` (VkModel, build_cnn_scorer, build_beat_crnn)
- Move `flame_sheep/vk_compute.py` → `wallpaper_ml/src/wallpaper_ml/vk_compute.py`
- Move `flame_sheep/shaders/` ML-related shaders to `wallpaper_ml/src/wallpaper_ml/shaders/` (the rendering shaders stay)
- Set up `wallpaper_ml/pyproject.toml` mirroring `flame_sheep_audio`'s structure
- Add `wallpaper_ml` as a dependency in `flame_sheep`'s `pyproject.toml`
- `pip install -e ./wallpaper_ml` then re-install flame_sheep
- Update imports in `flame_sheep/cnn_scorer.py`, `tools/train_beat_rnn.py`, `tools/train_beat_rnn_continuous.py`, `tools/finetune_cnn_vk.py`, etc.

Done when: `python -m flame_sheep --help` runs, `pytest` doesn't grow new failures, training scripts can still launch.

### Stage 2: Split `storage.py` and consolidate `tempo.py`

The other big junk drawer + the audio cross-package cleanup.

**2a. Split `storage.py` (76KB → 5 files + score_palette extract)**
- Create `flame_sheep/storage/` directory
- Move `Library` class → `storage/library.py` (keep `score_loop` here temporarily — moves in Stage 4 cleanly)
- Move `_ensure_schema` + migrations → `storage/schema.py`
- Move `_transform_to_dict`, `_genome_to_json` → `storage/serialization.py`
- Move `catalog.py` and `esheep_parser.py` from `flame_sheep/` → `storage/`
- **Motion field**: do NOT move into storage/. Compute_motion_field + coherence are geometric operations on genomes — defer to Stage 4 when they land in `genome/motion_field.py`. For Stage 2, leave them as module-level functions in storage.py if that's needed for in-flight imports, OR create `flame_sheep/_motion_field_tmp.py` as a temporary home.
- Move `score_palette` → `flame_sheep/palette/scoring.py` (create palette/ here in Stage 2 — small change, decouples from later stages)
- **Leave `score_loop` in storage/library.py temporarily**. No `_loop_scoring_tmp.py` placeholder. It moves cleanly in Stage 4 when loops/ is created.
- Create `storage/__init__.py` re-exporting `Library`
- Update imports throughout codebase
- Delete `storage.py`

**2b. Resolve `tempo.py` placement** (alter ego confirmed not duplicate)

`flame_sheep/tempo.py` is NOT a duplicate of `flame_sheep_audio/src/flame_sheep_audio/tempo.py`. It's an onset-IOI-histogram BPM estimator on the consumer side. Decision:
- Move to `flame_sheep/audio/tempo.py` (keep as consumer-side logic for now), OR
- Upstream the algorithm into the daemon and consume the published BPM from client side

Per the earlier discussion: tempo is "commonly wanted + expensive to compute multiple times," which favors upstream-to-daemon. But that's a bigger move (daemon code changes) than a pure file relocation. Recommendation: in this stage, move it to `flame_sheep/audio/tempo.py` as consumer logic. The upstream-to-daemon decision can land as a separate piece of work later, after the reorg settles.

After Stage 2, two of the biggest files are gone, scoring code is moving to its proper objects, and the audio cross-package boundary is clean.

### Stage 3: Group siblings into packages (mechanical)

Pure moves, no code changes inside files.

**3a. Create `runtime/` package**
- Move `core.py`, `orchestrator.py`, `control.py`, `session.py`, `mpris.py`, `command_handlers.py`, `role_mapper.py`, `wallpaper.py` → `runtime/`

**3b. Create `rendering/` package**
- Move `renderer.py`, `wayland_window.py`, `display.py` → `rendering/`
- Move `flame_sheep/shaders/` (rendering portion) → `rendering/shaders/`

**3c. No workers/ package — workers go with their object**
- `gpu_render_worker.py` → will become `genome/render_worker.py` in Stage 4
- `cpu_score_worker.py` → will become `genome/score_worker.py` in Stage 4
- `pruner_worker.py` → will become `genome/pruner.py` in Stage 4
- `transition_worker.py` → will become `transitions/worker.py` in Stage 4
- For Stage 3, leave them in place at top level (they'll move when their object package is created)

**3d. Create `audio/` package and `ui/` package**

Note: `flame_sheep/audio/` already exists as an empty dir — Stage 3d is "move files into existing audio/," not "create."

- `audio/`: move + rename `audio_client.py` → `audio/client.py`, `logger.py` → `audio/feature_logger.py`, also `tempo.py` from Stage 2b
- `ui/`: move `compare.py`, `library_cli.py`, `benchmark.py` → `ui/`

**3e. Rename `log.py` → `logging.py`** (or `logging_config.py` if stdlib collision)

### Stage 4: Object-organize genome/palette/transitions/loops

The conceptual reorganization — putting scoring with its objects.

**4a. Create `flame_sheep/transitions/` package**
- Move `flame_sheep/transition.py` → `transitions/distance.py` (compute_transition_distance + transform distance helpers)
- Split out `variation_signature`, `signature_distance` → `transitions/signature.py`
- Move `flame_sheep/transition_worker.py` → `transitions/worker.py`
- Create `transitions/__init__.py` with re-exports
- Update imports

**4b. Split `genome.py` (44KB → ~3 files + scoring/ subdir) and pull workers + motion_field in**
- Create `flame_sheep/genome/` directory
- Move `Transform` class → `genome/transform.py`
- Move `Genome` class + breeding methods → `genome/genome.py`
- Move `_score_from_histogram` → `genome/scoring/histogram.py`
- Move `_score_symmetry` → `genome/scoring/symmetry_score.py`
- Move existing `flame_sheep/symmetry.py` → `genome/symmetry.py`
- Move `cnn_scorer.py`, `scoring_channels.py`, `cluster_scorer.py`, `image_scorer.py`, `scorer.py`, `experimental_metrics.py` → `genome/scoring/`
- **Move `gpu_scorer.py` → `genome/scoring/gpu.py`** — caught by alter ego review, missed in initial plan
  - Note: gpu_scorer.py has a `python -m flame_sheep.gpu_scorer` entrypoint; verify it still works as `python -m flame_sheep.genome.scoring.gpu` or add a thin shim
  - Also consider: refactor gpu_scorer to use the same WorkerProcess pattern as the other genome workers (it's structurally a worker that wasn't built that way)
- **Move workers into genome/:**
  - `gpu_render_worker.py` → `genome/render_worker.py`
  - `cpu_score_worker.py` → `genome/score_worker.py`
  - `pruner_worker.py` → `genome/pruner.py`
  - **Auto-start CPU workers (`score_worker`, `pruner`, `transitions/worker`) with the wallpaper.** These don't use the GPU, so they have no isolation requirement — they could be auto-started now or as part of Stage 4. Currently separate manual-launch processes, which means "user forgets to start them → gen advance stalls indefinitely → silent failure." Anything the wallpaper requires to function should start with the wallpaper; making it a separate manual step is an accident of how they grew, not a design.
  - **`genome/render_worker.py` is the exception**: it uses the GPU and currently requires separate-process isolation for sway safety (per `project_gpu_contention.md`: Arc+Xe handles separate processes OK, but fork crashes sway). Auto-starting render_worker is gated on the full Vulkan conversion completing — once Vulkan is everywhere, the worker can share GPU context with the wallpaper and auto-start becomes safe. Until then, render_worker stays manual-launch as the safety mechanism. Sequence for it specifically: Vulkan conversion → reorg → auto-start wiring.
- **Move `compute_motion_field`, `motion_field_coherence` from storage.py → `genome/motion_field.py`** — cycle avoidance (see Cycle avoidance section above). This is the move that breaks the storage→loops→storage cycle.
- Move `_random_palette` and friends → `flame_sheep/palette/generation.py`
- Move `_lerp_arr` → `flame_sheep/palette/generation.py` (or a small utility module if used elsewhere)
- Create `genome/__init__.py`, `genome/scoring/__init__.py`, `palette/__init__.py` with re-exports
- Update imports

**4c. Move `score_loop` from storage to loops** — was deferred from Stage 2, lands here naturally with the loops/ split below.

**4c. Split `loops.py` (47KB → ~5 files + scoring)**
- Create `flame_sheep/loops/` directory
- `LoopCandidate` + dedup helpers → `loops/candidate.py`
- `prune_*` functions → `loops/prune.py`
- `loop_sequence`, `cycle_length` → `loops/sequence.py`
- `compose_loops`, `compose_loops_graph`, `_build_*`, `_too_similar`, `save_best_loops` → `loops/compose.py`
- `crossover`, `mutate_*`, `refine_loop`, `insert_genome`, `delete_genome`, `jitter_loop`, `mutate_structure`, `explore_breed`, `evolve_loops` → `loops/evolve.py`
- Move `score_loop` (parked in `_loop_scoring_tmp.py` during Stage 2) → `loops/scoring.py`
- Create `loops/__init__.py` with re-exports
- Update imports, delete `loops.py` and `_loop_scoring_tmp.py`

After Stage 4, the entire object-organized structure is in place. The big files (storage 76K, loops 47K, genome 44K, wallpaper_ml 40K) are all gone, replaced by focused modules grouped by what they belong to.

### Stage 5: Tests mirror, docs sweep

- Reorganize `tests/` to match `flame_sheep/` layout
- Update CLAUDE.md (if it references paths)
- Update docs that reference paths (this plan itself, `principles.md`, `generational_architecture.md` if it mentions storage.py specifically, etc.)
- Verify `python -m flame_sheep --help` + `pytest` + actual wallpaper run still work
- Sweep imports with mypy/pyright if available, or by running test scripts

## Risk + mitigation

**Risk**: import errors during migration. **Mitigation**: each stage moves a self-contained subset; tests run between stages. Stage 1's storage split is the highest-risk because it touches `Library` which is imported widely — do it first when context is fresh.

**Risk**: merge conflicts with other in-flight work. **Mitigation**: do this on a dedicated branch, sequence stages tightly (don't leave the codebase half-reorganized for weeks).

**Risk**: breaking external scripts in `tools/` that import from specific paths. **Mitigation**: re-exports in `__init__.py` files preserve the public API. Internal imports in tools/ may still need updates but won't silently break.

**Risk**: scope creep into "while we're here, let's rename functions / refactor logic too". **Mitigation**: discipline. Stage 1-3 are pure moves. Once the structure is stable, individual files become safe to refactor in isolation.

## Estimated effort

- Stage 0 (FlameRenderer decomposition): ~1-2 days, *if viz_authoring is real-near-term*
- Stage 1 (wallpaper_ml extraction): ~half a day — leaf module, only 9 import sites
- Stage 2 (storage split + tempo placement): ~1 day
- Stage 3 (group siblings into packages): few hours of mechanical moves
- Stage 4 (genome + loops object-organize): **~2 days** — splitting class files needs cohesion judgment, not just cut-paste
- Stage 5 (tests mirror + docs sweep): half day

Total ~5-6 days of focused work if Stage 0 happens. ~4 days without. Splits cleanly across evenings — each stage commits cleanly with passing tests.

## Future stages (post-refactor)

- **Stage 6 (small, can be done in parallel with anything)**: Extract `os.getloadavg() + pause` from the workers into shared `load_aware.py` utility. Removes per-worker duplication. ~half day. Pure refactor, no behavior change. Prerequisite for Stage 7.
- **Stage 7**: Comm-pipe API polish. Migrate `debug/` to import only from public `flame_sheep_audio` names (no underscore-prefix reaches). The exports needed to make that work are the public API. ~half day.
- **Stage 8**: Worker consolidation. The current per-task worker breakdown (`genome/render_worker`, `genome/score_worker`, `genome/pruner`, `transitions/worker`) grew organically. Consolidate by resource contention into three patterns:
  - **GpuWorker** — owns GPU context, GPU-load-aware (combines render + score: they pipeline naturally, score reads render's histogram)
  - **CpuWorkerPool** — N workers + job queue, system-load-aware (handles transition computation, thumbs-up breeding, future CPU-bound tasks)
  - **ScheduledTask** — episodic subprocess, skips when loaded (pruning, retrain checks, library audits)
  
  Within-package refactor. Doesn't expose anything yet. Establishes the shape that viz_authoring will publicize. Sets up `LoadMonitor` as an event bus with three publishers (system / GPU / disk) — not a single composite score.
  
  Why subprocess rather than thread for ScheduledTask: the wallpaper main process IS the render loop, and Python's GIL means even an unrelated thread doing CPU work during a render dispatch can cause frame jitter. Subprocess pays ~50ms startup but eliminates GIL contention entirely.
  
- **Stage 9**: viz_authoring extraction. With Stage 0 (FlameRenderer split), Stage 7 (comm pipe public API), and Stage 8 (worker consolidation) done, this is mostly moves + thin convenience layer:
  - Generic plumbing modules → `viz_authoring/src/viz_authoring/`
  - `GpuWorker`, `CpuWorkerPool`, `ScheduledTask` published as public API
  - Audio convenience helpers (envelope followers, easing, shader-uniform mapping)
  - `debug/` becomes the reference implementation
  
  Days to weeks depending on scope. Once landed, third-party viz authors can build against `viz_authoring` + `flame_sheep_audio` without touching flame-specific code.

### Worker patterns: docs notes

Each of the three patterns has roughly three hard questions that aren't visible from the API surface. A third-party author choosing any of them should find the trade-offs spelled out, not have to discover them. To-document list:

**Worker (single long-running subprocess)**
- IPC contract: what protocol does parent ↔ child speak (pipe, shmem, sqlite)?
- Lifecycle: clean shutdown vs orphan-cleanup-on-crash; how does parent know child died?
- Resource cleanup on crash: GPU contexts, file handles, sqlite connections — what releases when?

**WorkerPool (N workers + job queue)**
- Queue ordering: FIFO vs priority vs LIFO; partial-failure behavior (does one bad job kill the queue?)
- N-tuning: when to add/remove workers based on load; thrashing prevention (don't add/remove more than every N seconds)
- Partial-failure: if one worker dies, pool keeps running with remaining workers; how is the dead worker replaced?

**ScheduledTask (periodic subprocess)**
- Re-entrancy: interval fires while previous run is still going — default skip, configurable to queue or run concurrently
- Crash recovery: subprocess crashes (segfault, OOM, signal) → log + skip next interval as soft backoff; >3 consecutive crashes disables + alerts
- Shutdown signaling: how does the app exit signal a running scheduled task to stop (vs leak the subprocess)?

### LoadMonitor: signal source notes

Within "GPU contention" the signal source is plural:
- **Wallpaper FPS dropping** — consequence signal (something is making the GPU slow)
- **sysfs GPU utilization** — cause signal (something is using the GPU)

These can disagree: FPS can drop from a CPU stall (no GPU contention but FPS still falls), or GPU utilization can spike from a brief render burst (no sustained contention). A good GpuWorker subscribes to both and pauses on either.

This is the pub/sub shape doing its job. The antipattern to avoid is composite stress scoring — Brendan Gregg writes about this for system performance: "single number for system health" gets reified, drives wrong decisions. Keep the signals separate and let each worker subscribe to what matters for its work.

### Stage 6 (load_aware.py) gives free diagnostic data

When you extract the duplicated `os.getloadavg() + pause` logic, you'll find that workers have hand-rolled different thresholds, sleep intervals, polling frequencies. Either the differences are **principled** (this worker really needs different thresholds) or **accidental** (someone copy-pasted and changed a constant).

Either answer is useful. Principled differences tell you what the LoadMonitor abstraction needs to parameterize. Accidental differences tell you which workers had loose tuning that can be standardized. Stage 6 is small but it's where the design conversation becomes concrete instead of theoretical.

## Verification at each stage

- `python -m flame_sheep --help` runs (entry point unbroken)
- `python -m pytest` passes (or at least doesn't grow new failures)
- `grep -rn "from flame_sheep" flame_sheep tools tests` finds no obviously-broken paths
- Spot-check that wallpaper still runs end-to-end before declaring a stage done

Once all stages land, the codebase layout will communicate what it actually is: a research codebase with clearly-labeled components, rather than a flat namespace where you have to read filenames to figure out what each thing does.
