"""Wallpaper mode entry point — Vulkan backend.

Parallel to wallpaper.py (the moderngl/OpenGL runtime). Reuses the
Orchestrator + FlameSheepCore + Library + background workers + command
machinery unchanged; swaps the GL renderer + WallpaperSession for the
Vulkan stack (viz_authoring.vk.ChaosGame + tonemap pipeline + per-
output LayerShellSurface).

Out of scope for the first cut (each with a known follow-up):
  - Compare mode (eval-time only; commands.comparing always False here)
  - Test pattern
  - Audio FFT uniform texture (deferred per the user, not used by
    flame-sheep currently)
  - GpuRingTimer (CPU-side timing only for now)
  - VT switch + suspend recovery (the Vulkan stack hasn't been tested
    against compositor suspend; for now we exit on Wayland disconnect
    and rely on systemd to restart)
  - Hot-plug / output reconfiguration (compute layout once at start)

What's preserved verbatim:
  - Library auto-seed on empty DB
  - Background CPU scorer + transition scorer + pruner
  - Watchdog
  - Frame-rate cap from cfg.max_fps
  - Per-frame genome / palette / brightness from FlameSheepCore.tick
  - Walker re-randomize on genome change
"""

from __future__ import annotations

import logging
import os
import struct
import threading
import time

import cffi
import numpy as np
import vulkan as vk

from ..config import cfg
from ..genome import Genome
from .core import FlameSheepCore
from .orchestrator import Orchestrator

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Genome → ChaosGame.set_genome kwarg adapter
# ---------------------------------------------------------------------------

def _genome_to_chaos_kwargs(genome: Genome) -> dict:
    """Convert an in-memory Genome to the kwarg dict ChaosGame.set_genome
    expects. Mirror of viz_authoring.vk.wallpaper_demo._genome_from_catalog
    minus the DB load step — used here for the per-frame genome refresh
    coming out of FlameSheepCore.tick()."""
    arrays = genome.to_gpu_arrays()
    av = arrays['active_vars'].reshape(7, 8, 10)
    pv = arrays['pre_active_vars'].reshape(7, 8, 10)
    return dict(
        affines=arrays['affines'],
        post_affines=arrays['post_affines'],
        active_vars=av, pre_vars=pv,
        colors=arrays['colors'],
        color_speeds=arrays['color_speeds'],
        weights=arrays['weights'],
        n_transforms=len(genome.transforms),
        has_final_xform=arrays['has_final_xform'],
    )


def _palette_to_rgba8(palette_f32: np.ndarray) -> np.ndarray:
    """(256, 3) float32 [0,1] → (1, 256, 4) uint8 with alpha=255."""
    rgb = (np.clip(palette_f32, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgba = np.zeros((1, 256, 4), dtype=np.uint8)
    rgba[0, :, :3] = rgb
    rgba[0, :, 3] = 255
    return rgba


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _run_wallpaper_vk(audio_device: str | int | None, test_audio: bool,
                     blur_radius: float = 1.0,
                     log_features: bool = False, log_file: str | None = None,
                     test_pattern: bool = False) -> None:
    """Vulkan-backed wallpaper runtime — same external contract as
    _run_wallpaper(). blur_radius and test_pattern are accepted for CLI
    parity but currently ignored (TODOs in the unsupported-features
    list above)."""
    if test_pattern:
        log.warning('--test-pattern not supported by the Vulkan backend yet '
                    '— ignored')

    from ..rendering import _ensure_singleton
    _ensure_singleton()

    # Lazy import so flame_sheep can be imported on systems without Vulkan
    # (the GL backend remains the default until --backend vk is passed).
    from viz_authoring.vk.context import VkContext
    from viz_authoring.vk.chaos_game import ChaosGame
    from viz_authoring.vk.layer_shell import LayerShellSurface
    from viz_authoring.vk.swapchain import Swapchain
    from viz_authoring.vk.pipeline import (
        GraphicsPipeline, make_color_attachment_render_pass,
        make_intermediate_render_pass)
    from viz_authoring.vk.image import (
        create_image_rgba8, create_color_attachment_image,
        upload_image_rgba8, create_linear_sampler)
    from ..rendering.surface import _get_output_layout

    # --- 1. Discover outputs + compute physical-mm layout (same approach as
    # multi_wallpaper.py) ----------------------------------------------------

    raw_layout = _get_output_layout()
    if not raw_layout:
        raise RuntimeError('No active sway outputs found — is sway running?')
    # Auto-order left-to-right by physical x (matches multi_wallpaper).
    output_names = sorted(raw_layout.keys(), key=lambda n: raw_layout[n]['x'])
    layout = {n: raw_layout[n] for n in output_names}
    for name, g in layout.items():
        log.info(f'{name}: {g["w"]}x{g["h"]}px @ {g["ppi"]:.1f}ppi, '
                 f'physical {g["phys_w_mm"]:.0f}x{g["phys_h_mm"]:.0f}mm')

    # Physical-mm viewport math (matches MultiMonitorWallpaper).
    max_ppi = max(g['ppi'] for g in layout.values())
    RENDER_SCALE = 0.5
    canvas_ppmm = max_ppi / 25.4 * RENDER_SCALE
    phys_x_mm = {}
    cur = 0.0
    for name in output_names:
        phys_x_mm[name] = cur
        cur += layout[name]['phys_w_mm']
    max_phys_h_mm = max(g['phys_h_mm'] for g in layout.values())
    canvas_w = int(cur * canvas_ppmm)
    canvas_h = int(max_phys_h_mm * canvas_ppmm)
    log.info(f'canvas: {canvas_w}x{canvas_h}px @ {canvas_ppmm:.2f} px/mm')

    # --- 2. Per-output LayerShellSurface ------------------------------------

    surfaces: dict[str, LayerShellSurface] = {}
    for name in output_names:
        surfaces[name] = LayerShellSurface(output_name=name,
                                            install_sigint=False)

    # --- 3. Vulkan context + per-output swapchains ---------------------------

    ctx = VkContext(
        instance_extensions=LayerShellSurface.required_instance_extensions(),
        app_name='flame-sheep wallpaper (vk)',
    )
    for s in surfaces.values():
        s.create_surface(ctx)
    ctx.select_device(surface=surfaces[output_names[0]].surface)
    log.info(f'device: {ctx.device_name}')

    first_sc = Swapchain(ctx, surfaces[output_names[0]].surface,
                          requested_extent=surfaces[output_names[0]].framebuffer_size())
    # Two render passes: the swapchain target (PRESENT_SRC_KHR final
    # layout) and an intermediate target (SHADER_READ_ONLY_OPTIMAL final
    # layout, RGBA8_UNORM format) for the blur passes. The intermediate
    # render pass is fixed to RGBA8_UNORM since that's what our color-
    # attachment images use.
    swapchain_pass = make_color_attachment_render_pass(ctx, first_sc.format)
    intermediate_pass = make_intermediate_render_pass(
        ctx, vk.VK_FORMAT_R8G8B8A8_UNORM)
    first_sc.build_framebuffers(swapchain_pass)
    swapchains: dict[str, Swapchain] = {output_names[0]: first_sc}
    for name in output_names[1:]:
        sc = Swapchain(ctx, surfaces[name].surface,
                        requested_extent=surfaces[name].framebuffer_size())
        if sc.format != first_sc.format:
            raise NotImplementedError(
                f'output {name} surface format != first output\'s; '
                f'per-output render passes needed')
        sc.build_framebuffers(swapchain_pass)
        swapchains[name] = sc

    # Per-output viewport rectangles (canvas pixels)
    viewports: dict[str, tuple[int, int, int, int]] = {}
    for name in output_names:
        g = layout[name]
        y_off_mm = (max_phys_h_mm - g['phys_h_mm']) / 2.0
        viewports[name] = (
            int(phys_x_mm[name] * canvas_ppmm),
            int(y_off_mm * canvas_ppmm),
            int(g['phys_w_mm'] * canvas_ppmm),
            int(g['phys_h_mm'] * canvas_ppmm),
        )
        log.info(f'{name}: viewport {viewports[name]}')

    # --- 4. Chaos game + palette image + tonemap pipeline -------------------

    chaos = ChaosGame(ctx, canvas_w, canvas_h, n_walkers=65536)

    # Precompile driver is constructed below after `lib` exists (the
    # background enumeration thread closes over `lib`).
    precompile_driver = None

    palette_img = create_image_rgba8(ctx, 256, 1)
    # Initial palette is zeros — first frame upload from core.tick() fills it.
    upload_image_rgba8(ctx, palette_img,
                        np.zeros((1, 256, 4), dtype=np.uint8))
    palette_sampler = create_linear_sampler(ctx)

    SHADER_DIR = (os.path.dirname(__file__) + '/../../viz_authoring/src/'
                  'viz_authoring/vk/shaders')
    from pathlib import Path
    shader_dir = Path(SHADER_DIR).resolve()

    # Per-output intermediate images for the blur path:
    #   imgA = tonemap output (read by blurH)
    #   imgB = blurH output (read by blurV, which writes to swapchain)
    # Plus per-output framebuffers wrapping them at intermediate_pass.
    intermediate_a: dict[str, object] = {}
    intermediate_b: dict[str, object] = {}
    framebuffer_a: dict[str, int] = {}
    framebuffer_b: dict[str, int] = {}
    intermediate_sampler = create_linear_sampler(ctx)
    for name in output_names:
        sc = swapchains[name]
        w, h = sc.extent.width, sc.extent.height
        intermediate_a[name] = create_color_attachment_image(ctx, w, h)
        intermediate_b[name] = create_color_attachment_image(ctx, w, h)
        fb_a = vk.vkCreateFramebuffer(ctx.device, vk.VkFramebufferCreateInfo(
            renderPass=intermediate_pass,
            attachmentCount=1, pAttachments=[intermediate_a[name].view],
            width=w, height=h, layers=1), None)
        fb_b = vk.vkCreateFramebuffer(ctx.device, vk.VkFramebufferCreateInfo(
            renderPass=intermediate_pass,
            attachmentCount=1, pAttachments=[intermediate_b[name].view],
            width=w, height=h, layers=1), None)
        framebuffer_a[name] = fb_a
        framebuffer_b[name] = fb_b

    # Tonemap now renders into the intermediate render pass (its color
    # attachment is the per-output imgA, set per command-buffer record).
    tonemap = GraphicsPipeline(
        ctx, intermediate_pass,
        shader_dir / 'tonemap.vert',
        shader_dir / 'tonemap.frag',
        extent=first_sc.extent,  # ignored — dynamic viewport
        storage_buffers=[chaos.histogram, chaos.max_buf],
        sampled_images=[(palette_img, palette_sampler)],
        push_constant_size=28,
        dynamic_viewport=True,
    )

    # Per-output blurH pipeline (reads its imgA, writes its imgB via the
    # intermediate render pass). Pipeline descriptor sets are baked at
    # creation time so we need ONE pipeline per (output, A-image) pair.
    # Same for blurV.
    BLUR_PUSH_SIZE = 12   # vec2 (8 + alignment) + float = 12, but vec2 is
                          # 8-byte aligned so we pack vec2 first then float
                          # → 4 + 4 + 4 = 12. Actually vec2 needs 8-byte
                          # alignment so layout: vec2 (0..7), float (8..11).
    # Use 16 to give std140 alignment headroom — push constant min size
    # is 128 bytes anyway, so over-allocating costs nothing.
    BLUR_PUSH_SIZE = 16
    blur_h: dict[str, GraphicsPipeline] = {}
    blur_v: dict[str, GraphicsPipeline] = {}
    for name in output_names:
        blur_h[name] = GraphicsPipeline(
            ctx, intermediate_pass,
            shader_dir / 'tonemap.vert',  # same fullscreen-quad vert
            shader_dir / 'blur.frag',
            extent=first_sc.extent,
            sampled_images=[(intermediate_a[name], intermediate_sampler)],
            push_constant_size=BLUR_PUSH_SIZE,
            dynamic_viewport=True,
        )
        blur_v[name] = GraphicsPipeline(
            ctx, swapchain_pass,
            shader_dir / 'tonemap.vert',
            shader_dir / 'blur.frag',
            extent=first_sc.extent,
            sampled_images=[(intermediate_b[name], intermediate_sampler)],
            push_constant_size=BLUR_PUSH_SIZE,
            dynamic_viewport=True,
        )

    # --- 5. Pre-record per-(output, image-index) command buffers ------------

    ffi = cffi.FFI()

    BLUR_RADIUS = float(blur_radius)

    def _record_for_output(name: str):
        """3-pass per-frame command buffer per swapchain image:
          1. tonemap → imgA  (intermediate render pass)
          2. blur H samples imgA → imgB  (intermediate render pass)
          3. blur V samples imgB → swapchain image  (swapchain render pass)
        All three passes use the same fullscreen-quad vertex shader; only
        the fragment shader + descriptor set + render pass differ.
        """
        sc = swapchains[name]
        cmd_bufs = ctx.allocate_command_buffers(len(sc.framebuffers))
        vx, vy, vw, vh = viewports[name]
        # Tonemap viewport spans the per-output surface; blur passes
        # render fullscreen at the same surface dimensions.
        viewport_full = vk.VkViewport(
            x=0.0, y=0.0,
            width=float(sc.extent.width), height=float(sc.extent.height),
            minDepth=0.0, maxDepth=1.0,
        )
        scissor_full = vk.VkRect2D(offset=vk.VkOffset2D(x=0, y=0),
                                     extent=sc.extent)
        clear = vk.VkClearValue(
            color=vk.VkClearColorValue(float32=[0, 0, 0, 1]))

        tonemap_push = struct.pack(
            '6if', canvas_w, canvas_h, vx, vy, vw, vh, 1.0)
        tonemap_pc = ffi.new('char[]', tonemap_push)
        # Blur push: vec2 direction + float radius (alignment forces 16).
        # Horizontal: direction = (1/w, 0); vertical: (0, 1/h).
        blur_h_push = struct.pack(
            '2f f 4x', 1.0 / sc.extent.width, 0.0, BLUR_RADIUS)
        blur_h_pc = ffi.new('char[]', blur_h_push)
        blur_v_push = struct.pack(
            '2f f 4x', 0.0, 1.0 / sc.extent.height, BLUR_RADIUS)
        blur_v_pc = ffi.new('char[]', blur_v_push)

        def _bind_and_draw(cb, pipe, push, pc_ptr):
            vk.vkCmdBindPipeline(cb, vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                                   pipe.pipeline)
            vk.vkCmdBindDescriptorSets(
                cb, vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipe.layout, 0, 1, [pipe.descriptor_set], 0, None)
            vk.vkCmdPushConstants(
                cb, pipe.layout,
                vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                0, len(push), ffi.cast('void*', pc_ptr))
            vk.vkCmdDraw(cb, 6, 1, 0, 0)

        for i, cb in enumerate(cmd_bufs):
            vk.vkBeginCommandBuffer(cb, vk.VkCommandBufferBeginInfo())
            vk.vkCmdSetViewport(cb, 0, 1, [viewport_full])
            vk.vkCmdSetScissor(cb, 0, 1, [scissor_full])

            # Pass 1: tonemap → imgA
            vk.vkCmdBeginRenderPass(cb, vk.VkRenderPassBeginInfo(
                renderPass=intermediate_pass,
                framebuffer=framebuffer_a[name],
                renderArea=vk.VkRect2D(offset=vk.VkOffset2D(x=0, y=0),
                                         extent=sc.extent),
                clearValueCount=1, pClearValues=[clear]),
                vk.VK_SUBPASS_CONTENTS_INLINE)
            _bind_and_draw(cb, tonemap, tonemap_push, tonemap_pc)
            vk.vkCmdEndRenderPass(cb)

            # Pass 2: blur H samples imgA → imgB
            vk.vkCmdBeginRenderPass(cb, vk.VkRenderPassBeginInfo(
                renderPass=intermediate_pass,
                framebuffer=framebuffer_b[name],
                renderArea=vk.VkRect2D(offset=vk.VkOffset2D(x=0, y=0),
                                         extent=sc.extent),
                clearValueCount=1, pClearValues=[clear]),
                vk.VK_SUBPASS_CONTENTS_INLINE)
            _bind_and_draw(cb, blur_h[name], blur_h_push, blur_h_pc)
            vk.vkCmdEndRenderPass(cb)

            # Pass 3: blur V samples imgB → swapchain image
            vk.vkCmdBeginRenderPass(cb, vk.VkRenderPassBeginInfo(
                renderPass=swapchain_pass,
                framebuffer=sc.framebuffers[i],
                renderArea=vk.VkRect2D(offset=vk.VkOffset2D(x=0, y=0),
                                         extent=sc.extent),
                clearValueCount=1, pClearValues=[clear]),
                vk.VK_SUBPASS_CONTENTS_INLINE)
            _bind_and_draw(cb, blur_v[name], blur_v_push, blur_v_pc)
            vk.vkCmdEndRenderPass(cb)

            vk.vkEndCommandBuffer(cb)
        return cmd_bufs

    command_buffers: dict[str, list] = {
        name: _record_for_output(name) for name in output_names
    }

    # --- 6. Per-output sync primitives --------------------------------------

    image_available = {}
    render_finished = {}
    in_flight = {}
    for name in output_names:
        image_available[name] = vk.vkCreateSemaphore(
            ctx.device, vk.VkSemaphoreCreateInfo(), None)
        render_finished[name] = vk.vkCreateSemaphore(
            ctx.device, vk.VkSemaphoreCreateInfo(), None)
        in_flight[name] = vk.vkCreateFence(
            ctx.device, vk.VkFenceCreateInfo(
                flags=vk.VK_FENCE_CREATE_SIGNALED_BIT), None)

    # --- 7. Library auto-seed + workers + orchestrator/core (unchanged) ---

    from ..storage import Library
    from ..loops import compose_loops, save_best_loops
    lib = Library()

    if lib.genome_count() == 0:
        log.info('Empty library — generating initial genomes...')
        rng = np.random.default_rng()
        for i in range(200):
            g = Genome.random(rng)
            lib.save_genome(g, g.aesthetic_score())
            if (i + 1) % 50 == 0:
                log.info(f'  {i+1}/200 genomes')
        log.info(f'Generated {lib.genome_count()} genomes')

    if lib.loop_count() == 0 and lib.genome_count() >= 20:
        log.info('No loops — composing initial loops...')
        candidates = compose_loops(lib, n_attempts=200, loop_length=6)
        if candidates:
            save_best_loops(lib, candidates, n_keep=20)
        log.info(f'Composed {lib.loop_count()} loops')

    # --- Shader precompile worker (best-effort) ---------------------
    # Spawns a subprocess that warms Mesa's shader cache for every
    # genome's (n_transforms, has_final_xform, variation_set) tuple,
    # so the wallpaper's per-genome compile drops from ~300-500ms
    # cold to ~3ms cache-hit. Catalog enumeration runs in a thread
    # so it doesn't delay startup.
    #
    # Best-effort: if anything fails (can't spawn subprocess etc.)
    # the wallpaper continues with the synchronous compile path —
    # visible as occasional 500ms stutters on first-use of new tuples.
    wallpaper_signal_writer = None
    try:
        from ..scheduler.policy import Policy
        from ..scheduler.precompile_driver import PrecompileDriver
        from ..scheduler.wallpaper_signal import (
            WallpaperSignalReader, WallpaperSignalWriter)

        # The wallpaper publishes its frame-budget usage; the scheduler
        # reads the same shmem region. Both share a default path under
        # $XDG_RUNTIME_DIR.
        target_ms = 1000.0 / cfg.max_fps
        wallpaper_signal_writer = WallpaperSignalWriter(target_ms=target_ms)
        signal_reader = WallpaperSignalReader()

        # Thresholds calibrated against chaos_ms / target_ms semantics:
        # at iters=100 baseline chaos ≈ 14ms / 16.7ms ≈ 0.84 → SLOW.
        # If chaos exceeds the frame budget itself → PAUSE.
        _policy = Policy(signal_reader,
                          sample_interval_s=0.25,
                          slow_threshold=0.85,
                          pause_threshold=1.00)
        precompile_driver = PrecompileDriver(
            canvas_w=canvas_w, canvas_h=canvas_h, policy=_policy)
        precompile_driver.start()

        # Background enumeration — scans catalog (~3s) and enqueues
        # at LOW priority. Done in a daemon thread so it doesn't
        # block wallpaper startup.
        def _enumerate_and_enqueue():
            try:
                from ..scheduler.precompile_warm import (
                    enumerate_catalog_tuples)
                # Don't pass main-thread's `lib` — sqlite3 connections
                # are not cross-thread safe. The helper opens its own.
                tuples = enumerate_catalog_tuples()
                added = precompile_driver.enqueue(
                    tuples, priority=10)
                log.info(f'[precompile] enqueued {added} catalog '
                         f'tuples at priority=10')
            except Exception:
                log.exception(
                    '[precompile] catalog enumeration failed')
        threading.Thread(target=_enumerate_and_enqueue,
                          name='precompile-enum',
                          daemon=True).start()
    except Exception:
        log.exception('[precompile] init failed; continuing without')
        precompile_driver = None

    orch = Orchestrator(audio_device=audio_device, test_audio=test_audio)
    core = FlameSheepCore(orchestrator=orch, lib=lib)

    from ..genome.score_worker import BackgroundCpuScorer
    from ..transitions import BackgroundTransitionScorer
    from ..genome.pruner import BackgroundPruner
    from ..storage import _db_path
    db = str(_db_path())
    cpu_scorer = BackgroundCpuScorer(db_path=db)
    transition_scorer = BackgroundTransitionScorer(db_path=db)
    pruner = BackgroundPruner(db_path=db)
    cpu_scorer.start()
    transition_scorer.start()
    pruner.start()

    orch.start()

    # Feature logger (optional)
    feature_logger = None
    if log_features:
        from ..audio.feature_logger import AudioFeatureLogger
        feature_logger = AudioFeatureLogger(orch, path=log_file or None)

    # --- 8. Watchdog (same pattern as wallpaper.py) -------------------------

    _watchdog_last = [time.perf_counter()]
    _quit = [False]

    def _pet_watchdog() -> None:
        _watchdog_last[0] = time.perf_counter()

    import signal as _signal
    _signal.signal(_signal.SIGINT, lambda *_: _quit.__setitem__(0, True))

    def _watchdog():
        start = time.perf_counter()
        while not _quit[0]:
            time.sleep(2.0)
            elapsed = time.perf_counter() - start
            timeout = 15.0 if elapsed < 20.0 else 3.0
            if time.perf_counter() - _watchdog_last[0] > timeout:
                log.error('[watchdog] render loop stalled, forcing exit')
                os._exit(1)

    threading.Thread(target=_watchdog, daemon=True).start()

    # --- 9. Render loop ----------------------------------------------------

    last_time = time.perf_counter()
    last_genome_id: int | None = None
    last_palette_id: int | None = None  # palette is a numpy array; id() ok for change detection
    last_loop_id: int | None = None  # track loop changes for precompile MED hints
    perf_frames = 0
    perf_accum: dict[str, float] = {}
    perf_iters: list[int] = []
    PERF_INTERVAL = 60

    def _accum(stage, dt):
        perf_accum[stage] = perf_accum.get(stage, 0.0) + dt

    _pet_watchdog()
    log.info('vulkan wallpaper render loop starting')

    try:
        while not _quit[0] and not any(s.should_close() for s in surfaces.values()):
            _pet_watchdog()
            orch.tick()
            if feature_logger:
                feature_logger.tick()

            for s in surfaces.values():
                s.dispatch_events(blocking=False)

            now = time.perf_counter()
            frame_time = now - last_time
            MIN_FRAME_TIME = 1.0 / cfg.max_fps
            if frame_time < MIN_FRAME_TIME:
                time.sleep(MIN_FRAME_TIME - frame_time)
                now = time.perf_counter()
                frame_time = now - last_time
            last_time = now
            _accum('frame_total', frame_time)

            frame = core.tick(frame_time)
            _pet_watchdog()

            # --- Genome upload (only on change — saves per-frame buffer churn)
            gid = id(frame.genome)
            _ts = time.perf_counter()
            if gid != last_genome_id:
                chaos.set_genome(**_genome_to_chaos_kwargs(frame.genome))
                last_genome_id = gid
                # New genome → re-randomize walkers (matches GL upload_genome
                # path's reset on change behavior).
                chaos.reset_walkers()

                # Precompile MED hint: when loop changes, push all the
                # new loop's members so they're cached before the next
                # set_genome lands. Cheap (small loops, dedupe in driver).
                if precompile_driver is not None:
                    try:
                        loop_id = core._genome_axis.active_loop_id
                        if loop_id != last_loop_id:
                            last_loop_id = loop_id
                            from ..scheduler.precompile_warm import (
                                _genome_tuple)
                            loop_tuples = [
                                _genome_tuple(g) for g in
                                core._genome_axis._loop.loop_genomes]
                            n = precompile_driver.enqueue(
                                loop_tuples, priority=0)
                            if n:
                                log.debug(
                                    f'[precompile] enqueued {n} loop '
                                    f'#{loop_id} members at MED')
                    except Exception:
                        log.exception('[precompile] loop hint failed')
            _accum('upload_genome', time.perf_counter() - _ts)

            # --- Palette upload (small, do every frame; cheap memcpy)
            _ts = time.perf_counter()
            upload_image_rgba8(ctx, palette_img,
                                _palette_to_rgba8(frame.palette))
            _accum('upload_palette', time.perf_counter() - _ts)

            # --- Chaos game pass (decay 0.3 matches GL wallpaper).
            # frame() batches clear + chaos + reduce_max into ONE
            # command buffer with memory barriers between dispatches —
            # one fence-wait per frame instead of three.
            _ts = time.perf_counter()
            chaos.frame(
                iterations=frame.iterations,
                zoom=(frame.genome.zoom, frame.genome.zoom),
                rotation=frame.genome.rotation,
                center=tuple(frame.genome.center),
                decay=0.3,
            )
            chaos_dt = time.perf_counter() - _ts
            _accum('chaos_game', chaos_dt)
            perf_iters.append(frame.iterations)

            # Publish frame budget usage to the scheduler. busy_fraction
            # downstream = chaos_dt / target_ms — drives precompile
            # throttling.
            if wallpaper_signal_writer is not None:
                wallpaper_signal_writer.update(
                    frame_ms=frame_time * 1000.0,
                    chaos_ms=chaos_dt * 1000.0)

            # --- Tonemap + present per output ---
            # NOTE: brightness is not yet wired into the pre-recorded command
            # buffers' push constants — gamma stays at 1.0. Hooking it up
            # cleanly means either re-recording per-frame or using a
            # uniform buffer; deferred until the brightness pulse feels off
            # in practice.
            _ts = time.perf_counter()
            for name in output_names:
                sc = swapchains[name]
                vk.vkWaitForFences(ctx.device, 1, [in_flight[name]],
                                    vk.VK_TRUE, 0xFFFFFFFFFFFFFFFF)
                vk.vkResetFences(ctx.device, 1, [in_flight[name]])
                img_idx = sc.acquire_next_image(image_available[name])
                submit = vk.VkSubmitInfo(
                    waitSemaphoreCount=1,
                    pWaitSemaphores=[image_available[name]],
                    pWaitDstStageMask=[
                        vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT],
                    commandBufferCount=1,
                    pCommandBuffers=[command_buffers[name][img_idx]],
                    signalSemaphoreCount=1,
                    pSignalSemaphores=[render_finished[name]],
                )
                vk.vkQueueSubmit(ctx.graphics_queue, 1, [submit],
                                  in_flight[name])
                sc.present(img_idx, render_finished[name])
            _accum('tonemap+present', time.perf_counter() - _ts)
            _pet_watchdog()

            perf_frames += 1
            if perf_frames >= PERF_INTERVAL:
                ft = perf_accum.pop('frame_total', 0.0) / perf_frames * 1000
                parts = sorted(perf_accum.items(), key=lambda kv: -kv[1])
                s = '  '.join(f'{k}={v/perf_frames*1000:.2f}ms'
                              for k, v in parts)
                iters_avg = sum(perf_iters) / len(perf_iters)
                log.info(f'[vk perf {perf_frames} frames]  '
                         f'frame={ft:.2f}ms  {s}  '
                         f'iters={min(perf_iters)}/{iters_avg:.0f}/{max(perf_iters)}')
                perf_frames = 0
                perf_accum.clear()
                perf_iters.clear()

    except Exception:
        log.exception('[vk] exception in render loop')
        raise
    finally:
        log.info('vulkan wallpaper render loop exiting')
        # Stop precompile first so its worker isn't competing during
        # the GPU teardown.
        if precompile_driver is not None:
            try:
                precompile_driver.stop()
            except Exception:
                log.exception('[precompile] stop failed')
        if wallpaper_signal_writer is not None:
            try:
                wallpaper_signal_writer.close()
            except Exception:
                log.exception('[scheduler] signal writer close failed')
        vk.vkDeviceWaitIdle(ctx.device)
        # Cleanup — best-effort; OS reclaims on exit anyway.
        for name in output_names:
            vk.vkDestroySemaphore(ctx.device, image_available[name], None)
            vk.vkDestroySemaphore(ctx.device, render_finished[name], None)
            vk.vkDestroyFence(ctx.device, in_flight[name], None)
            blur_h[name].cleanup()
            blur_v[name].cleanup()
            vk.vkDestroyFramebuffer(ctx.device, framebuffer_a[name], None)
            vk.vkDestroyFramebuffer(ctx.device, framebuffer_b[name], None)
            intermediate_a[name].destroy()
            intermediate_b[name].destroy()
            swapchains[name].cleanup()
        intermediate_sampler.destroy()
        tonemap.cleanup()
        palette_img.destroy()
        palette_sampler.destroy()
        chaos.cleanup()
        vk.vkDestroyRenderPass(ctx.device, intermediate_pass, None)
        vk.vkDestroyRenderPass(ctx.device, swapchain_pass, None)
        for s in surfaces.values():
            s.destroy_surface()
        ctx.cleanup()
        for s in surfaces.values():
            s.cleanup()
