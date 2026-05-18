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
import moderngl_window as mglw
from moderngl_window import settings

from .config import cfg
from flame_sheep_audio import DEFAULT_DEVICE
from .orchestrator import Orchestrator
from .core import FlameSheepCore
from .renderer import FlameRenderer, Viewport
from .benchmark import _run_variation_benchmark
from .library_cli import _run_library_commands
from .wallpaper import _run_wallpaper


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
