# Vulkan chaos-shader performance reproducer

This directory contains a focused benchmark harness that reproduces a
significant performance gap between Mesa's GL and Vulkan compilers for
flame-sheep's chaos-game compute shader on Intel Arc (Xe driver).

## What it measures

The chaos shader compiles the same GLSL source on both backends.
Measured on Intel Arc A770, Mesa 26.0.5:

| Backend (per-genome trimmed shader) | gpu_chaos / dispatch |
|---|---|
| GL (Mesa-Iris-equivalent compiler) | ~4-8ms |
| Vk (Mesa-Xe Vulkan compiler) | ~14-28ms |

The 2.5-4× gap persists even after per-genome variation-switch
trimming (which was the workaround that got Vk out of an even worse
spillage regime — see commit `36` and the
`[Vk chaos perf]` memory note).

The shader source is structurally identical on both backends. The
gap is in the ISA Mesa emits.

## Why this directory exists

This was the production wallpaper rendering path until 2026-05-31,
at which point we reverted the wallpaper to GL because the gap was
the dominant perf cost on this hardware. The Vk chaos-compute path
stays around as:

1. **An active reproducer for the Mesa-Xe Vulkan compiler bug** that
   should be filed upstream (task #37 in the project tracker).
2. **A canary for future Mesa-Xe improvements** — re-run the bench
   periodically; once the gap closes, the wallpaper can move back to
   Vulkan (queue priorities + explicit barriers + spec consts are
   real advantages once the compiler isn't holding things back).
3. **Validation surface for cross-backend correctness** — the Vk
   golden-master tests (`tests/variations/test_vk_*.py`) keep the
   chaos game's numerical output verified against the GL path.

## What got pruned with the wallpaper revert

The production wallpaper integration was deleted:
- `flame_sheep/runtime/wallpaper_vk.py` — wallpaper entry point
- `flame_sheep/ui/compare_vk.py` — Vk compare-mode renderer
- `viz_authoring/vk/{layer_shell,multi_wallpaper,wallpaper_demo,
  swapchain,surface,triangle_demo,flame_demo}.py` — wallpaper-shell
  WSI and demo apps

## What stays in viz_authoring/vk for the bench

- `context.py` — VkContext (instance, device, queues, command pool)
- `pipeline.py` — ComputePipeline, GraphicsPipeline
- `buffer.py`, `image.py` — resource wrappers
- `chaos_game.py` — the actual chaos-game compute path; per-genome
  spec-const trim variant lives here
- `headless.py` — HeadlessVkRenderer; still used by
  `flame_sheep.genome.render_worker` for catalog scoring renders
- `pipeline_warm.py` — cross-process warm-marker convention
- `shaders/*` — the GLSL source itself

## Running the bench

```bash
python -m tools.vk_perf_diag.bench_chaos_frame
```

When the gap closes upstream, also re-run on a fresh wallpaper revert
attempt to confirm the production path can move back to Vk.
