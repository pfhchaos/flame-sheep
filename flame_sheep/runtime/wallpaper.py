"""Wallpaper mode entry point — _run_wallpaper.

Sets up the Wayland session, multi-monitor canvas geometry, renderer,
orchestrator, command handlers, and runs the main render loop with
VT-switch / suspend recovery and a watchdog.
"""

from __future__ import annotations

import logging
import os
import threading
import time

import numpy as np
import moderngl

from ..config import cfg
from ..genome import Genome
from .orchestrator import Orchestrator
from .core import FlameSheepCore
from ..rendering import (
    _monitor_cfg, _get_output_layout, _ensure_singleton,
    FlameRenderer, GpuContext, GpuRingTimer, Viewport,
)
from ..screen_layout import PhysicalViewport
from .command_handlers import WallpaperCommands

log = logging.getLogger(__name__)


def _run_wallpaper(audio_device: str | int | None, test_audio: bool,
                   blur_radius: float = 1.0,
                   log_features: bool = False, log_file: str | None = None,
                   test_pattern: bool = False) -> None:
    """
    Wallpaper mode — one continuous flame fractal image across all monitors.
    """
    _ensure_singleton()
    from ..rendering import WallpaperSession

    # --- discover outputs and sway layout ---
    output_names = WallpaperSession.list_outputs()
    log.info(f'outputs: {output_names}')

    layout = _get_output_layout()
    active = {n: layout[n] for n in output_names if n in layout}
    if not active:
        raise RuntimeError('No active sway outputs found — is sway running?')

    # --- compute physical positions (mm) for each output ---
    # We need to convert pixel positions to physical positions.
    # Use the leftmost monitor's left edge as physical origin.
    # For vertical alignment, align monitors by their physical center.
    
    # First, find physical dimensions and pixel positions
    for name, g in active.items():
        log.info(f'{name}: {g["w"]}x{g["h"]}px @ {g["ppi"]:.1f}ppi, '
              f'physical {g["phys_w_mm"]:.0f}x{g["phys_h_mm"]:.0f}mm')
    
    # Sort outputs by x position (left to right)
    sorted_outputs = sorted(active.items(), key=lambda x: x[1]['x'])
    
    # Compute physical x positions by accumulating physical widths
    phys_x = {}  # mm from left edge
    current_x_mm = 0.0
    for name, g in sorted_outputs:
        phys_x[name] = current_x_mm
        current_x_mm += g['phys_w_mm']
    total_phys_w_mm = current_x_mm
    
    # For vertical: center all monitors on the tallest one's center
    max_phys_h_mm = max(g['phys_h_mm'] for g in active.values())
    phys_y = {}  # mm from top (centered)
    for name, g in active.items():
        # Center this monitor vertically
        phys_y[name] = (max_phys_h_mm - g['phys_h_mm']) / 2.0
    total_phys_h_mm = max_phys_h_mm
    
    # --- canvas resolution: render at a fixed density ---
    # Use the highest-PPI monitor as reference to avoid quality loss
    max_ppi = max(g['ppi'] for g in active.values())
    RENDER_SCALE = 0.5  # render at half the max PPI for performance
    canvas_ppmm = max_ppi / 25.4 * RENDER_SCALE  # pixels per mm in canvas
    
    canvas_w = int(total_phys_w_mm * canvas_ppmm)
    canvas_h = int(total_phys_h_mm * canvas_ppmm)
    log.info(f'physical canvas: {total_phys_w_mm:.0f}x{total_phys_h_mm:.0f}mm '
          f'-> render {canvas_w}x{canvas_h}px @ {canvas_ppmm:.2f} px/mm')

    # --- compute viewport for each output ---
    # PhysicalViewport describes screen geometry in mm (the natural unit for
    # multi-monitor composition); to_viewport(ppmm) projects into pixel-space
    # Viewports for the renderer. Keeps mm logic out of the renderer's API.
    phys_viewports: dict[str, PhysicalViewport] = {
        name: PhysicalViewport(
            x_mm=phys_x[name],
            y_mm=phys_y[name],
            w_mm=g['phys_w_mm'],
            h_mm=g['phys_h_mm'],
        )
        for name, g in active.items()
    }
    viewports: dict[str, Viewport] = {
        name: pv.to_viewport(canvas_ppmm) for name, pv in phys_viewports.items()
    }
    for name, vp in viewports.items():
        log.info(f'{name}: viewport {vp.w}x{vp.h}+{vp.x},{vp.y} (canvas px)')

    # --- one session: one wl_display, one EGL display, one GL context ---
    session = WallpaperSession()
    surfaces: dict[str, 'OutputSurface'] = {}
    first_surf = None
    for name in output_names:
        if name not in active:
            continue
        surf = session.add_output(name)
        surfaces[name] = surf
        if first_surf is None:
            first_surf = surf

    # Bind context to first surface and create the moderngl wrapper
    session.make_current(first_surf)
    ctx = session.create_moderngl_context()
    renderer = FlameRenderer(GpuContext(ctx, canvas_w, canvas_h, ppmm=canvas_ppmm))
    renderer.blur_radius = blur_radius

    renderer.temporal_decay = 0.0  # image-space temporal off (using histogram decay instead)

    _test_pattern = test_pattern

    # --- library + evolution state ---
    from ..storage import Library
    from ..loops import evolve_loops, compose_loops, compose_loops_graph, save_best_loops
    lib = Library()

    # Auto-seed library if empty
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

    # --- Orchestrator: owns audio, control pipe, MPRIS, session ---
    orch = Orchestrator(audio_device=audio_device, test_audio=test_audio)
    core = FlameSheepCore(orchestrator=orch, lib=lib)

    # --- Wallpaper-signal scheduler + GL precompile worker ---
    # Same architecture as wallpaper_vk: a shared mmap signal file
    # publishes per-frame budget usage; Policy throttles batch
    # workers (precompile + render) based on it; PrecompileDriver
    # spawns a subprocess that pre-warms Mesa's GL shader cache for
    # upcoming variation sets so the render thread never compiles
    # inline. Compile-starved system → GenomeAxis defers genome
    # swaps until the next-shader is ready (upstream gate).
    from ..scheduler.policy import Policy
    from ..scheduler.precompile_driver import PrecompileDriver
    # GL doesn't have a useful per-frame busy signal: the chaos
    # dispatch returns to the CPU immediately (driver queues async)
    # and only `eglSwapBuffers` actually waits, which is vsync-locked
    # to ~16.7ms even when nothing's happening on the GPU. Using
    # frame_time as the policy signal would keep busy_fraction ≈ 1.0
    # forever and the worker would never run. Skip the signal entirely
    # on GL — _child_preexec already applies nice +19 + ionice idle
    # to the worker subprocess, so OS scheduling handles contention.
    # If GL later grows GPU timing telemetry we can revisit and feed
    # the wallpaper signal again.
    wallpaper_signal_writer = None
    precompile_driver = None
    class _NoSignal:
        available = False
        def sample(self): return None
    try:
        # Policy with unavailable sampler stays in RUN forever — see
        # Policy.__init__: state = RUN if not sampler.available.
        precompile_policy = Policy(
            _NoSignal(), sample_interval_s=0.25,
            slow_threshold=0.55, pause_threshold=0.80,
            pause_hold_s=2.0)
        precompile_driver = PrecompileDriver(
            canvas_w=0, canvas_h=0,  # GL worker ignores canvas dims
            policy=precompile_policy,
            worker_module='flame_sheep.scheduler.precompile_worker_gl')
        precompile_driver.start()
        log.info('[precompile-gl] driver started (no signal — runs at '
                 'nice +19, OS handles contention)')
    except Exception:
        log.exception('[precompile-gl] init failed; continuing without')
        precompile_driver = None

    # Upstream gate: GenomeAxis filters target candidates through
    # the predicate. Cold candidate → defer swap + request compile +
    # stay on current genome. Same shape as the Vk side.
    def _genome_is_ready(g) -> bool:
        try:
            gpu = g.to_gpu_arrays()
            keep_vars = renderer._extract_keep_vars(gpu)
            return renderer.is_shader_warm(keep_vars)
        except Exception:
            log.exception('[axis-gate] readiness probe failed')
            return True  # fail open — don't deadlock the axis

    def _genome_request_precompile(g) -> None:
        if precompile_driver is None:
            return
        try:
            from ..scheduler.precompile_warm import _genome_tuple
            n_added = precompile_driver.enqueue(
                [_genome_tuple(g)], priority=-100)
            if n_added:
                log.debug(
                    f'[axis-gate] requested compile for genome '
                    f'#{getattr(g, "db_id", "?")} (axis deferred swap)')
        except Exception:
            log.exception('[axis-gate] precompile enqueue failed')

    # renderer was constructed earlier (above); predicate closures
    # bind it via late lookup. Wire into the axis now so the very
    # first genome-swap decision the axis makes goes through us.
    core._genome_axis.set_pipeline_readiness(
        is_ready=_genome_is_ready,
        on_request=_genome_request_precompile)

    # --- background workers ---
    # GPU render worker is NOT started here — it competes for the GPU and
    # kills desktop performance. Run separately:
    #   python -m flame_sheep.genome.render_worker
    from ..genome.score_worker import BackgroundCpuScorer
    from ..transitions import BackgroundTransitionScorer
    from ..genome.pruner import BackgroundPruner
    from ..storage import _db_path
    db = str(_db_path())
    cpu_scorer = BackgroundCpuScorer(db_path=db)
    transition_scorer = BackgroundTransitionScorer(db_path=db)
    pruner = BackgroundPruner(db_path=db)
    if lib is not None:
        cpu_scorer.start()
        transition_scorer.start()
        pruner.start()

    # Watchdog timing state (mutable container so command handlers can
    # pet the watchdog during slow operations like compare-mode init,
    # which would otherwise block the render loop long enough to trip
    # the 3s timeout).
    _watchdog_last = [time.perf_counter()]

    def _pet_watchdog() -> None:
        _watchdog_last[0] = time.perf_counter()

    # --- Register command handlers on orchestrator ---
    from .command_handlers import WallpaperCommands
    commands = WallpaperCommands(
        core=core, lib=lib, orch=orch, renderer=renderer, ctx=ctx,
        viewports=viewports, surfaces=surfaces, first_surf=first_surf,
        canvas_ppmm=canvas_ppmm,
        pet_watchdog=_pet_watchdog,
    )
    commands.register_all()

    orch.start()

    # Feature logger (optional)
    feature_logger = None
    if log_features:
        from ..audio.feature_logger import AudioFeatureLogger
        feature_logger = AudioFeatureLogger(orch, path=log_file or None)

    last_time = time.perf_counter()
    _frame = 0
    # _watchdog_last container already initialized before commands so the
    # pet callback could be passed in. Bump it now so the watchdog grace
    # period starts from "render loop about to begin," not "wallpaper
    # started initializing."
    _pet_watchdog()

    # Main-mode per-stage timing — counterpart to CompareRenderer's
    # internal perf accumulator. Lets us compare per-stage cost between
    # the two modes directly.
    MAIN_PERF_INTERVAL = 60
    _main_perf_frames = 0
    _main_perf_accum: dict[str, float] = {}
    # Loop-change detection drives precompile_driver.enqueue.
    last_loop_id: int | None = None

    def _main_accum(stage: str, dt: float) -> None:
        _main_perf_accum[stage] = _main_perf_accum.get(stage, 0.0) + dt

    _main_gpu_chaos_timer: GpuRingTimer | None = None

    def _watchdog():
        """Background thread: force exit if render loop stops making progress.

        Grace period: first 15s after start allows slow shader compilation
        on large canvases (3+ monitors, Arc GPU). After that, 3s stall = exit.
        Command handlers (e.g. _handle_compare) may pet the watchdog from
        their own threads during long-running synchronous setup.
        """
        start = time.perf_counter()
        while not commands.quit_requested:
            time.sleep(2.0)
            elapsed = time.perf_counter() - start
            timeout = 15.0 if elapsed < 20.0 else 3.0
            if time.perf_counter() - _watchdog_last[0] > timeout:
                log.error('[watchdog] render loop stalled, forcing exit')
                os._exit(1)

    _wd_thread = threading.Thread(target=_watchdog, daemon=True)
    _wd_thread.start()

    try:
        while not commands.quit_requested and not all(s.should_close for s in surfaces.values()):
            _frame += 1
            _pet_watchdog()
            orch.tick()
            if feature_logger:
                feature_logger.tick()

            # Dispatch pending Wayland events (delivers frame callbacks)
            if not session.dispatch():
                if orch.session.gpu_paused:
                    log.info('[render] wayland connection lost during VT switch, '
                             'waiting for session to resume...')
                    while not commands.quit_requested and not orch.session.is_active():
                        time.sleep(0.5)
                        _pet_watchdog()
                        orch.tick()
                    if not commands.quit_requested:
                        log.info('[render] session resumed, restarting wallpaper')
                        # Re-exec ourselves to get fresh Wayland connection
                        import sys
                        os.execv(sys.executable, [sys.executable] + sys.argv)
                log.error('wayland connection lost, exiting')
                break

            # Block until the compositor signals it wants a new frame.
            # If no frame callback arrives for 2s, assume VT switch or
            # compositor suspended — pause rendering and keep watchdog alive.
            _wait_start = time.perf_counter()
            while not commands.quit_requested:
                ready = {n: s for n, s in surfaces.items()
                         if s._frame_pending and not s.should_close}
                if ready or all(s.should_close for s in surfaces.values()):
                    break
                session.wait_for_events(timeout=0.016)
                _pet_watchdog()
                # VT switch detection: no frame callbacks for 2s
                if time.perf_counter() - _wait_start > 2.0:
                    log.info('[render] no frame callbacks — compositor suspended? '
                             'Pausing render loop.')
                    while not commands.quit_requested:
                        session.wait_for_events(timeout=0.5)
                        _pet_watchdog()
                        orch.tick()
                        ready = {n: s for n, s in surfaces.items()
                                 if s._frame_pending and not s.should_close}
                        if ready:
                            log.info('[render] frame callbacks resumed')
                            break
                        if all(s.should_close for s in surfaces.values()):
                            break

            if not ready:
                continue

            now        = time.perf_counter()
            frame_time = now - last_time

            # Cap framerate — no point rendering faster than the display
            MIN_FRAME_TIME = 1.0 / cfg.max_fps
            if frame_time < MIN_FRAME_TIME:
                time.sleep(MIN_FRAME_TIME - frame_time)
                now = time.perf_counter()
                frame_time = now - last_time

            last_time  = now

            if _frame % 300 == 0:
                fps = 1.0 / frame_time if frame_time > 0 else 0
                log.debug(f'[perf] frame={_frame} fps={fps:.1f} dt={frame_time*1000:.1f}ms iters={frame.iterations}')

            # Attribute total per-frame wall clock to whichever mode is active
            # so we can see (frame_total − measured CPU stages) = GPU sync +
            # other unmeasured time. GPU-bound mode shows large `swap` and
            # large `frame_total` with small `chaos_game`.
            if commands.comparing and commands.compare_mode:
                commands.compare._accum('frame_total', frame_time)
            else:
                _main_accum('frame_total', frame_time)

            frame = core.tick(frame_time)
            _pet_watchdog()

            # Skip ALL GL calls if GPU is paused (VT switch)
            if orch.session.gpu_paused:
                _pet_watchdog()
                time.sleep(0.1)
                continue

            # Compute pass — surface doesn't matter for compute, keep first
            if not session.make_current(first_surf):
                break  # surfaces died (sway reload?)
            # Double-check after make_current — VT switch can happen between
            # the check above and here
            if orch.session.gpu_paused:
                session.release_current()
                continue

            if commands.comparing and commands.compare_mode:
                commands.compare.dispatch(
                    commands.compare_mode.pair, frame,
                    rotation_phase=core._genome_axis._rotation.phase,
                    override_genome=core._pinned_genome)

            elif not _test_pattern:
                # --- Normal mode: single chaos game ---
                _ts = time.perf_counter()
                renderer.upload_audio(frame.spectrum)
                _main_accum('upload_audio', time.perf_counter() - _ts)
                _ts = time.perf_counter()
                renderer.upload_genome(frame.genome)
                _main_accum('upload_genome', time.perf_counter() - _ts)
                _ts = time.perf_counter()
                renderer.upload_palette(frame.palette)
                _main_accum('upload_palette', time.perf_counter() - _ts)
                if core.needs_walker_reset:
                    renderer.reset_walkers()
                    core.needs_walker_reset = False
                _ts = time.perf_counter()
                renderer.clear_histogram(decay=0.3)
                _main_accum('clear_hist', time.perf_counter() - _ts)
                _ts = time.perf_counter()
                if _main_gpu_chaos_timer is None:
                    _main_gpu_chaos_timer = GpuRingTimer(ctx, logger=log)
                with _main_gpu_chaos_timer:
                    renderer.dispatch_chaos_game(iterations=frame.iterations)
                _main_accum('chaos_game', time.perf_counter() - _ts)
                _main_accum('gpu_chaos', _main_gpu_chaos_timer.last_ns / 1e9)
                _ts = time.perf_counter()
                ctx.memory_barrier()
                _main_accum('barrier', time.perf_counter() - _ts)
                _ts = time.perf_counter()
                renderer.reduce_histogram_max()
                _main_accum('reduce_max', time.perf_counter() - _ts)

                # Loop-change precompile hint: when GenomeAxis picks
                # a new loop, enqueue the loop's members + 2-hop
                # graph-flood neighborhood so the precompile worker
                # warms shaders for likely-soon-played genomes. Same
                # bounded-flood pattern as the Vk wallpaper.
                if precompile_driver is not None:
                    try:
                        loop_id = core._genome_axis.active_loop_id
                        if loop_id != last_loop_id:
                            last_loop_id = loop_id
                            from ..scheduler.precompile_warm import (
                                _genome_tuple, near_tuples)
                            loop_genomes = (
                                core._genome_axis._loop.loop_genomes)
                            loop_tuples = [_genome_tuple(g)
                                           for g in loop_genomes]
                            n_med = precompile_driver.enqueue(
                                loop_tuples, priority=0)
                            seed_ids = [g.db_id for g in loop_genomes
                                         if g.db_id is not None]
                            n_low = 0
                            if seed_ids and lib is not None:
                                flood = near_tuples(lib, seed_ids,
                                                      hops=2,
                                                      fan_per_node=5)
                                n_low = precompile_driver.enqueue(
                                    flood, priority=10)
                            if n_med or n_low:
                                log.info(
                                    f'[precompile-gl] loop #{loop_id}: '
                                    f'MED={n_med} (loop members) '
                                    f'LOW={n_low} (2-hop flood)')
                    except Exception:
                        log.exception(
                            '[precompile-gl] loop hint failed')

                _main_perf_frames += 1
                if _main_perf_frames >= MAIN_PERF_INTERVAL:
                    n = _main_perf_frames
                    # Headline metrics: frame budget, GPU-sync wait,
                    # measured CPU. The "other" gap is unmeasured time
                    # (compositor / make_current / ready-wait / etc.)
                    frame_total = _main_perf_accum.pop('frame_total', 0.0) / n * 1000
                    swap = _main_perf_accum.pop('swap', 0.0) / n * 1000
                    cpu_stages = sum(_main_perf_accum.values()) / n * 1000
                    other = max(0, frame_total - swap - cpu_stages)
                    parts = sorted(_main_perf_accum.items(),
                                   key=lambda kv: -kv[1])
                    s = '  '.join(f'{k}={v / n * 1000:.2f}ms'
                                  for k, v in parts)
                    log.info(
                        f'[main perf {n} frames]  '
                        f'frame_total={frame_total:.2f}ms  '
                        f'swap={swap:.2f}ms  '
                        f'cpu={cpu_stages:.2f}ms  '
                        f'other={other:.2f}ms  ({s})'
                    )
                    _main_perf_frames = 0
                    _main_perf_accum.clear()
            _pet_watchdog()

            # Tonemap pass — only swap surfaces the compositor is ready for
            for name, surf in ready.items():
                if not session.make_current(surf):
                    continue  # this surface is dead, skip it
                if _test_pattern:
                    renderer.gpu.render_test_pattern(viewports[name], surf.width, surf.height)
                elif commands.comparing and name == commands.compare.surf_name:
                    commands.compare.tonemap_surface(surf, frame)
                elif commands.comparing:
                    # Side monitors: black during compare mode
                    ctx.clear(0.0, 0.0, 0.0, 1.0)
                else:
                    renderer.render_tonemap(viewports[name], surf.width, surf.height,
                                           brightness=frame.brightness)
                _swap_ts = time.perf_counter()
                if not session.swap(surf):
                    break  # wayland connection lost
                _swap_dt = time.perf_counter() - _swap_ts
                if commands.comparing and commands.compare_mode:
                    commands.compare._accum('swap', _swap_dt)
                else:
                    _main_accum('swap', _swap_dt)
                _pet_watchdog()

    except Exception as e:
        log.error(f'[render] exception in render loop: {e}', exc_info=True)
    finally:
        # If surfaces closed during VT switch, wait and restart
        if not commands.quit_requested and orch.session.gpu_paused:
            log.info('[render] surfaces closed during VT switch, waiting for resume...')
            while not orch.session.is_active():
                time.sleep(0.5)
            log.info('[render] session resumed, restarting')
            import sys
            os.execv(sys.executable, [sys.executable, '-m', 'flame_sheep'] + sys.argv[1:])

        log.debug(f'[render] exiting render loop (commands.quit_requested={commands.quit_requested})')
        if feature_logger:
            feature_logger.close()
        # Stop precompile first so its worker isn't competing during
        # the GPU teardown.
        if precompile_driver is not None:
            try:
                precompile_driver.stop()
            except Exception:
                log.exception('[precompile-gl] stop failed')
        if wallpaper_signal_writer is not None:
            try:
                wallpaper_signal_writer.close()
            except Exception:
                log.exception('[scheduler] signal writer close failed')
        cpu_scorer.stop()
        transition_scorer.stop()
        pruner.stop()
        orch.stop()
        session.destroy()
