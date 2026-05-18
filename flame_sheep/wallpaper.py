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

from .config import cfg
from .genome import Genome
from .orchestrator import Orchestrator
from .core import FlameSheepCore
from .display import _monitor_cfg, _get_output_layout, _ensure_singleton
from .renderer import FlameRenderer, Viewport
from .command_handlers import WallpaperCommands

log = logging.getLogger(__name__)


def _run_wallpaper(audio_device: str | int | None, test_audio: bool,
                   blur_radius: float = 1.0,
                   log_features: bool = False, log_file: str | None = None,
                   spectrum_engine: str = 'octave_bank',
                   test_pattern: bool = False) -> None:
    """
    Wallpaper mode — one continuous flame fractal image across all monitors.
    """
    _ensure_singleton()
    from .wayland_window import WallpaperSession

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
    
    # Overscan: expand canvas vertically so the trapezoid from perspective
    # skew is fully covered with fractal data (no black corners).
    overscan = _monitor_cfg('__default__', 'overscan', 0.02)
    from types import SimpleNamespace as _NS
    if isinstance(overscan, _NS):
        overscan = 0.02
    has_skew = any(abs(_monitor_cfg(n, 'skew', 0.0)) > 0.1 for n in active)
    if has_skew:
        total_phys_h_mm *= (1.0 + 2.0 * overscan)

    canvas_w = int(total_phys_w_mm * canvas_ppmm)
    canvas_h = int(total_phys_h_mm * canvas_ppmm)
    log.info(f'physical canvas: {total_phys_w_mm:.0f}x{total_phys_h_mm:.0f}mm '
          f'-> render {canvas_w}x{canvas_h}px @ {canvas_ppmm:.2f} px/mm')

    # --- compute viewport for each output (in canvas pixels) ---
    viewports: dict[str, Viewport] = {}
    for name, g in active.items():
        vx = int(phys_x[name] * canvas_ppmm)
        vy = int(phys_y[name] * canvas_ppmm)
        vw = int(g['phys_w_mm'] * canvas_ppmm)
        vh = int(g['phys_h_mm'] * canvas_ppmm)
        # Shift viewports down by overscan amount to center in expanded canvas
        if has_skew:
            vy += int(canvas_h * overscan / (1.0 + 2.0 * overscan))
        viewports[name] = Viewport(vx, vy, vw, vh)
        log.info(f'{name}: viewport {vw}x{vh}+{vx},{vy} (canvas px)')

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
    renderer = FlameRenderer(ctx, canvas_w, canvas_h)
    renderer.blur_radius = blur_radius
    renderer.set_ppmm(canvas_ppmm)

    # Per-monitor perspective skew from config
    _monitor_skew = {name: _monitor_cfg(name, 'skew', 0.0) for name in viewports}
    renderer.temporal_decay = 0.0  # image-space temporal off (using histogram decay instead)

    _blur_comparison = False
    _test_pattern = test_pattern

    # --- library + evolution state ---
    from .storage import Library
    from .loops import evolve_loops, compose_loops, compose_loops_graph, save_best_loops
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
    orch = Orchestrator(audio_device=audio_device, test_audio=test_audio,
                        spectrum_engine=spectrum_engine)
    core = FlameSheepCore(orchestrator=orch, lib=lib)

    # --- background workers ---
    # GPU render worker is NOT started here — it competes for the GPU and
    # kills desktop performance. Run separately: python -m flame_sheep.gpu_render_worker
    from .cpu_score_worker import BackgroundCpuScorer
    from .transition_worker import BackgroundTransitionScorer
    from .pruner_worker import BackgroundPruner
    from .storage import _db_path
    db = str(_db_path())
    cpu_scorer = BackgroundCpuScorer(db_path=db)
    transition_scorer = BackgroundTransitionScorer(db_path=db)
    pruner = BackgroundPruner(db_path=db)
    if lib is not None:
        cpu_scorer.start()
        transition_scorer.start()
        pruner.start()

    # --- Register command handlers on orchestrator ---
    from .command_handlers import WallpaperCommands
    commands = WallpaperCommands(
        core=core, lib=lib, orch=orch, renderer=renderer, ctx=ctx,
        viewports=viewports, surfaces=surfaces, first_surf=first_surf,
        canvas_ppmm=canvas_ppmm,
    )
    commands.register_all()

    orch.start()

    # Feature logger (optional)
    feature_logger = None
    if log_features:
        from .logger import AudioFeatureLogger
        feature_logger = AudioFeatureLogger(orch, path=log_file or None)

    last_time = time.perf_counter()
    _frame = 0
    _watchdog_last = time.perf_counter()

    def _watchdog():
        """Background thread: force exit if render loop stops making progress.

        Grace period: first 15s after start allows slow shader compilation
        on large canvases (3+ monitors, Arc GPU). After that, 3s stall = exit.
        """
        start = time.perf_counter()
        while not commands.quit_requested:
            time.sleep(2.0)
            elapsed = time.perf_counter() - start
            timeout = 15.0 if elapsed < 20.0 else 3.0
            if time.perf_counter() - _watchdog_last > timeout:
                log.error('[watchdog] render loop stalled, forcing exit')
                os._exit(1)

    _wd_thread = threading.Thread(target=_watchdog, daemon=True)
    _wd_thread.start()

    try:
        while not commands.quit_requested and not all(s.should_close for s in surfaces.values()):
            _frame += 1
            _watchdog_last = time.perf_counter()
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
                        _watchdog_last = time.perf_counter()
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
                _watchdog_last = time.perf_counter()
                # VT switch detection: no frame callbacks for 2s
                if time.perf_counter() - _wait_start > 2.0:
                    log.info('[render] no frame callbacks — compositor suspended? '
                             'Pausing render loop.')
                    while not commands.quit_requested:
                        session.wait_for_events(timeout=0.5)
                        _watchdog_last = time.perf_counter()
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

            frame = core.tick(frame_time)
            _watchdog_last = time.perf_counter()

            # Skip ALL GL calls if GPU is paused (VT switch)
            if orch.session.gpu_paused:
                _watchdog_last = time.perf_counter()
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
                    rotation_phase=core._genome_axis._rotation.phase)

            elif not _test_pattern:
                # --- Normal mode: single chaos game ---
                renderer.upload_audio(frame.spectrum)
                renderer.upload_genome(frame.genome)
                renderer.upload_palette(frame.palette)
                if core.needs_walker_reset:
                    renderer.reset_walkers()
                    core.needs_walker_reset = False
                renderer.clear_histogram(decay=0.3)
                renderer.dispatch_chaos_game(iterations=frame.iterations)
                ctx.memory_barrier()
                renderer.reduce_histogram_max()
            _watchdog_last = time.perf_counter()

            # Tonemap pass — only swap surfaces the compositor is ready for
            for name, surf in ready.items():
                if not session.make_current(surf):
                    continue  # this surface is dead, skip it
                renderer.set_skew(_monitor_skew.get(name, 0.0))
                if _test_pattern:
                    renderer.render_test_pattern(viewports[name], surf.width, surf.height)
                elif commands.comparing and name == commands.compare.surf_name:
                    commands.compare.tonemap_surface(surf, frame)
                elif commands.comparing:
                    # Side monitors: black during compare mode
                    ctx.clear(0.0, 0.0, 0.0, 1.0)
                elif _blur_comparison and surf.width >= 3000:
                    renderer.render_blur_comparison(viewports[name], surf.width, surf.height,
                                                    brightness=frame.brightness,
                                                    radii=(0.6, 1.0))
                else:
                    renderer.render_tonemap(viewports[name], surf.width, surf.height,
                                           brightness=frame.brightness)
                if not session.swap(surf):
                    break  # wayland connection lost
                _watchdog_last = time.perf_counter()

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
        cpu_scorer.stop()
        transition_scorer.stop()
        pruner.stop()
        orch.stop()
        session.destroy()
