"""
flame-sheep — entry point and main loop.

Two modes:
  default    — GLFW window via moderngl-window (for development / testing)
  --wallpaper — wlr-layer-shell BACKGROUND surface (true wallpaper on sway)

Core rendering/audio/genome logic lives in FlameSheepCore so it can be
shared between both modes.
"""

from __future__ import annotations

import faulthandler
faulthandler.enable()  # print traceback on SIGSEGV instead of silent death

import signal
import os

def _sigsegv_restart(signum: int, frame: object) -> None:
    """On SIGSEGV (GPU crash on VT switch), restart ourselves."""
    import sys
    os.execv(sys.executable, [sys.executable] + sys.argv)

signal.signal(signal.SIGSEGV, _sigsegv_restart)

import logging

log = logging.getLogger(__name__)


import argparse
import sys
import time
import os
import threading
import numpy as np
import moderngl_window as mglw
from moderngl_window import settings

from .config import cfg
from .genome import Genome
from flame_sheep_audio import DEFAULT_DEVICE
from .orchestrator import Orchestrator
from .core import FlameSheepCore
from .display import _monitor_cfg, _get_output_layout, _ensure_singleton
from .benchmark import _run_variation_benchmark
from .library_cli import _run_library_commands

import moderngl
from .renderer import FlameRenderer, Viewport
from .control import ControlPipe, ControlEvent


# Set by main() before run_window_config — workaround for moderngl-window
# not passing CLI args through to WindowConfig.__init__
_AUDIO_DEVICE: str | int = DEFAULT_DEVICE
_TEST_AUDIO:   bool = False
_SPECTRUM_ENGINE: str = 'octave_bank'



class FlameSheepApp(mglw.WindowConfig):
    """moderngl-window wrapper — used for windowed/dev mode."""
    title       = 'flame-sheep'
    gl_version  = (4, 3)
    resizable   = True
    vsync       = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._orch = Orchestrator(audio_device=_AUDIO_DEVICE, test_audio=_TEST_AUDIO,
                                  spectrum_engine=_SPECTRUM_ENGINE)
        self._core = FlameSheepCore(orchestrator=self._orch)
        self._orch.start()
        w, h = self.window_size
        self._renderer = FlameRenderer(self.ctx, w, h)
        self._viewport = Viewport(0, 0, w, h)
        log.info('flame-sheep started. Press Q to quit, F to force genome swap.')

    def on_render(self, time_val: float, frame_time: float):
        self.ctx.clear(0.0, 0.0, 0.0)
        self._orch.tick()
        frame = self._core.tick(frame_time)
        self._renderer.upload_audio(frame.spectrum)
        self._renderer.upload_genome(frame.genome)
        self._renderer.upload_palette(frame.palette)
        if self._core.needs_walker_reset:
            self._renderer.reset_walkers()
            self._core.needs_walker_reset = False
        self._renderer.clear_histogram(decay=0.3)
        self._renderer.dispatch_chaos_game(iterations=frame.iterations)
        self.ctx.memory_barrier()
        self._renderer.reduce_histogram_max()
        w, h = self.window_size
        self._renderer.render_tonemap(self._viewport, w, h, brightness=frame.brightness)

    def key_event(self, key, action, modifiers):
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.Q:
                self._orch.stop()
                self.wnd.close()
            elif key == self.wnd.keys.F:
                self._core.force_genome_swap()

    def close(self):
        self._orch.stop()



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



def main() -> None:
    parser = argparse.ArgumentParser(
        description='flame-sheep: audio-reactive flame fractal wallpaper',
        # Don't error on moderngl-window's own flags — we strip ours then
        # leave the rest for mglw to handle.
        add_help=False,
    )
    parser.add_argument('-h', '--help', action='store_true')
    parser.add_argument('--wallpaper', action='store_true',
                        help='run as wlr-layer-shell wallpaper (no window chrome)')
    parser.add_argument('--fullscreen', action='store_true', help='run fullscreen (windowed mode only)')
    parser.add_argument('--width',  type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--list-audio', action='store_true', help='list audio devices and exit')
    parser.add_argument('--audio-device', default=DEFAULT_DEVICE,
                        help=f'audio input device name or index (default: {DEFAULT_DEVICE!r})')
    parser.add_argument('--test-audio', action='store_true',
                        help='use synthetic metronome instead of real audio (120bpm, predictable beats)')
    parser.add_argument('--debug', action='store_true',
                        help='open debug overlay window (audio analysis visualization)')
    parser.add_argument('--test-pattern', action='store_true',
                        help='show grid test pattern instead of fractal (alignment debugging)')
    parser.add_argument('--generate-genomes', type=int, metavar='N',
                        help='generate N random genomes into the library and exit')
    parser.add_argument('--compose-loops', type=int, metavar='N', default=None,
                        help='compose N loops from the genome library (use with --generate-genomes or standalone)')
    parser.add_argument('--evolve', action='store_true',
                        help='run one evolution cycle on stored loops and exit')
    parser.add_argument('--loop-length', type=int, default=6,
                        help='genomes per loop (default: 6)')
    parser.add_argument('--generate-palettes', type=int, metavar='N', default=None,
                        help='generate N random palettes into the library')
    parser.add_argument('--prune-loops', action='store_true',
                        help='remove duplicate and low-fitness loops')
    parser.add_argument('--stats', action='store_true',
                        help='print library statistics and exit')
    parser.add_argument('--blur-radius', type=float, default=0.6,
                        help='wallpaper blur strength (0=off, 1=light, 2+=heavy; default: 1.0)')
    parser.add_argument('--log-features', action='store_true',
                        help='log audio features as JSON lines for offline analysis')
    parser.add_argument('--log-file', type=str, default=None,
                        help='path for feature log (default: ~/.local/share/flame-sheep/features.jsonl)')
    parser.add_argument('--benchmark-variations', action='store_true',
                        help='benchmark each variation solo (GPU timing) and exit')
    parser.add_argument('--render-catalog', type=str, metavar='DIR', default=None,
                        help='generate genome catalog for evaluation (renders PNGs)')
    parser.add_argument('--catalog-count', type=int, default=50,
                        help='number of genomes to generate (default: 50)')
    parser.add_argument('--catalog-evolve', type=int, default=3,
                        help='evolution generations before rendering (default: 3)')
    parser.add_argument('--render-unrated', type=str, metavar='DIR', default=None,
                        help='render existing unrated genomes for evaluation')
    parser.add_argument('--import-catalog', type=str, metavar='DIR', default=None,
                        help='import votes from sorted catalog (good/bad folders)')
    parser.add_argument('--spectrum-engine', choices=['octave_bank', 'cqt'],
                        default='cqt',
                        help='spectrum analysis engine (default: cqt)')
    parser.add_argument('--log-level', action='append', default=[],
                        help='logging verbosity: global (INFO) or per-component '
                             '(flame_sheep_audio.tempo_acf=DEBUG). Repeatable.')

    # parse_known_args so moderngl-window's own flags don't cause errors here
    args, remaining = parser.parse_known_args()

    if args.help:
        parser.print_help()
        return

    from .log import setup_logging
    # Parse --log-level args: bare value = global, name=LEVEL = per-component
    global_level = 'INFO'
    component_levels: dict[str, str] = {}
    # Config TOML levels (baseline)
    from .config import cfg
    if hasattr(cfg, 'logging') and hasattr(cfg.logging, 'levels'):
        config_levels = cfg.logging.levels
        if isinstance(config_levels, dict):
            component_levels.update(config_levels)
    # CLI overrides
    for spec in args.log_level:
        if '=' in spec:
            name, level = spec.split('=', 1)
            component_levels[name] = level.upper()
        else:
            global_level = spec.upper()
    setup_logging(level=global_level, component_levels=component_levels)

    global _AUDIO_DEVICE, _TEST_AUDIO, _SPECTRUM_ENGINE
    _AUDIO_DEVICE = args.audio_device
    _TEST_AUDIO   = args.test_audio
    _SPECTRUM_ENGINE = args.spectrum_engine

    if args.list_audio:
        from flame_sheep_audio import list_monitor_devices
        for d in list_monitor_devices():
            print(f"  [{d['index']:2d}] {d['name']}")
        return

    if args.debug:
        from .debug import run_debug_overlay
        audio_device = args.audio_device
        if audio_device is None:
            from .config import cfg
            audio_device = cfg.audio_device
        run_debug_overlay(audio_device=audio_device, test_audio=args.test_audio)
        return

    if args.benchmark_variations:
        _run_variation_benchmark()
        return

    if args.render_catalog:
        from .catalog import generate_catalog
        generate_catalog(args.render_catalog, n_genomes=args.catalog_count,
                         n_evolve=args.catalog_evolve)
        return

    if args.render_unrated:
        from .catalog import render_unrated
        render_unrated(args.render_unrated)
        return

    if args.import_catalog:
        from .catalog import import_catalog
        import_catalog(args.import_catalog)
        return

    if args.generate_genomes or args.generate_palettes is not None or args.compose_loops is not None or args.evolve or args.prune_loops or args.stats:
        _run_library_commands(args)
        return

    if args.wallpaper:
        # CLI flag overrides config; config overrides auto-detect
        audio_device = args.audio_device
        if audio_device is None:
            from .config import cfg
            audio_device = cfg.audio_device
        _run_wallpaper(audio_device, args.test_audio, blur_radius=args.blur_radius,
                       log_features=args.log_features, log_file=args.log_file,
                       spectrum_engine=args.spectrum_engine,
                       test_pattern=args.test_pattern)
        return

    # Strip our flags from sys.argv so moderngl-window's arg parser
    # doesn't choke on arguments it doesn't know about.
    sys.argv = [sys.argv[0]] + remaining

    settings.WINDOW['class']      = 'moderngl_window.context.glfw.Window'
    settings.WINDOW['size']       = (args.width, args.height)
    settings.WINDOW['fullscreen'] = args.fullscreen
    settings.WINDOW['title']      = 'flame-sheep'

    mglw.run_window_config(FlameSheepApp)


if __name__ == '__main__':
    main()
