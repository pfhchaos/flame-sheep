"""
flame-sheep — entry point and main loop.

This is a wallpaper system. The primary entry is --wallpaper (wlr-layer-shell
BACKGROUND surface on sway). Other modes (--debug overlay, library commands,
benchmarks) layer on top of that core. A future fullscreen mode would just be
the same wallpaper code with a different wl_layer.
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

from .config import cfg
from flame_sheep_audio import DEFAULT_DEVICE
from .ui.benchmark import _run_variation_benchmark
from .ui.library_cli import _run_library_commands


def main() -> None:
    parser = argparse.ArgumentParser(
        description='flame-sheep: audio-reactive flame fractal wallpaper',
    )
    parser.add_argument('--wallpaper', action='store_true',
                        help='run as wlr-layer-shell wallpaper (no window chrome)')
    parser.add_argument('--list-audio', action='store_true', help='list audio devices and exit')
    parser.add_argument('--audio-device', default=DEFAULT_DEVICE,
                        help=f'audio input device name or index (default: {DEFAULT_DEVICE!r})')
    parser.add_argument('--test-audio', action='store_true',
                        help='use synthetic metronome instead of real audio (120bpm, predictable beats)')
    # --debug overlay was a GL-mode-only audio-analysis window built
    # on top of FlameRenderer. Removed with the GL stack; will be
    # rebuilt against viz_authoring.vk if needed.
    parser.add_argument('--debug', action='store_true',
                        help='(removed) was GL-only debug overlay')
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
    parser.add_argument('--sync-chaos', action='store_true',
                        help='force synchronous chaos.frame() (debug only; '
                             'default is async — one frame CPU-ahead of GPU)')
    # --backend flag retained as a no-op for back-compat scripts that
    # still pass it; only `vk` is supported now. The GL wallpaper was
    # removed in favor of the single Vulkan stack.
    parser.add_argument('--backend', choices=['vk'], default='vk',
                        help='rendering backend (vk only; GL removed)')
    parser.add_argument('--log-features', action='store_true',
                        help='log audio features as JSON lines for offline analysis')
    parser.add_argument('--log-file', type=str, default=None,
                        help='path for feature log (default: ~/.local/share/flame-sheep/features.jsonl)')
    parser.add_argument('--benchmark-variations', action='store_true',
                        help='benchmark each variation solo (GPU timing) and exit')
    parser.add_argument('--bench-sample', choices=['random', 'top', 'stratified'],
                        default='random',
                        help='library-genome sampling mode for benchmark (default: random)')
    parser.add_argument('--bench-n', type=int, default=50,
                        help='number of library genomes to benchmark (default: 50)')
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
    parser.add_argument('--log-level', action='append', default=[],
                        help='logging verbosity: global (INFO) or per-component '
                             '(flame_sheep_audio.tempo_acf=DEBUG). Repeatable.')

    args = parser.parse_args()

    from .logging_config import setup_logging
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

    if args.list_audio:
        from flame_sheep_audio import list_monitor_devices
        for d in list_monitor_devices():
            print(f"  [{d['index']:2d}] {d['name']}")
        return

    if args.debug:
        log.error('--debug overlay was removed with the GL stack. '
                  'Rebuild against viz_authoring.vk if you need it.')
        sys.exit(2)

    if args.benchmark_variations:
        _run_variation_benchmark(sample_mode=args.bench_sample,
                                 n_genomes=args.bench_n)
        return

    if args.render_catalog:
        from .storage.catalog import generate_catalog
        generate_catalog(args.render_catalog, n_genomes=args.catalog_count,
                         n_evolve=args.catalog_evolve)
        return

    if args.render_unrated:
        from .storage.catalog import render_unrated
        render_unrated(args.render_unrated)
        return

    if args.import_catalog:
        from .storage.catalog import import_catalog
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
        from .runtime.wallpaper_vk import _run_wallpaper_vk
        _run_wallpaper_vk(audio_device, args.test_audio,
                          blur_radius=args.blur_radius,
                          log_features=args.log_features,
                          log_file=args.log_file,
                          test_pattern=args.test_pattern,
                          sync_chaos=args.sync_chaos)
        return

    # No mode given — print help. flame-sheep is a wallpaper system; running
    # without a mode is almost always a user error.
    parser.print_help()
    sys.exit(2)


if __name__ == '__main__':
    main()
