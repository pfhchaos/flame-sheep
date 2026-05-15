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
from collections.abc import Callable
from dataclasses import dataclass
import os
import threading
from typing import TYPE_CHECKING
import numpy as np
import moderngl_window as mglw
from moderngl_window import settings

if TYPE_CHECKING:
    from .storage import Library

from .config import cfg
from .genome import Genome, _lerp_arr
from flame_sheep_audio import BeatEvent, AudioState, DEFAULT_DEVICE
from flame_sheep_audio.mode import Mode
from .orchestrator import Orchestrator

import moderngl
from .renderer import FlameRenderer, Viewport
from .control import ControlPipe, ControlEvent
from .axes.zoom_axis import ZoomAxis
from .axes.brightness_axis import BrightnessAxis
from .axes.detail_axis import DetailAxis
from .axes.genome_axis import GenomeAxis
from .axes.palette_axis import PaletteAxis
from .role_mapper import RoleMapper


# Set by main() before run_window_config — workaround for moderngl-window
# not passing CLI args through to WindowConfig.__init__
_AUDIO_DEVICE: str | int = DEFAULT_DEVICE
_TEST_AUDIO:   bool = False
_SPECTRUM_ENGINE: str = 'octave_bank'


class FlameSheepCore:
    """
    Visualization state — axes, drift mode, genome morphing.
    Consumes an Orchestrator for audio state and events.
    Does NOT hold a renderer or GL context — that lives in each window.

    Each frame, call:
      genome, spectrum = core.tick(frame_time)   # advance state, get current genome
    Then render with those values in your per-window renderer.

    Call .force_genome_swap() to kick a degenerate image.
    """

    def __init__(self, orchestrator: Orchestrator,
                 lib: Library | None = None,
                 clock: Callable[[], float] | None = None,
                 genome_factory: Callable[[], Genome] | None = None) -> None:
        self._orch = orchestrator
        self._consumer_id: str = orchestrator.register('wallpaper')
        self._clock: Callable[[], float] = clock or time.perf_counter
        self.rng: np.random.Generator = np.random.default_rng()
        self._lib = lib

        # --- Visual axes ---
        if genome_factory:
            factory = genome_factory
        elif lib is not None and lib.genome_count() > 0:
            # Pre-load top genomes — no DB access from prefetch thread
            _pool = [lib.load_genome(gid) for gid, _ in lib.top_genomes(n=50)]
            factory = lambda: _pool[int(self.rng.integers(0, len(_pool)))]
        else:
            factory = lambda: Genome.random(self.rng)
        # Role mapping (band names → visual roles)
        role_cfg = cfg.roles
        role_map = {k: getattr(role_cfg, k) for k in ('downbeat', 'backbeat', 'subdivision', 'energy')}
        self._role = RoleMapper(role_map)

        bin_freqs = orchestrator.audio.bin_freqs
        self._genome_axis = GenomeAxis(genome_factory=factory, role=self._role, lib=lib, rng=self.rng)
        self._palette_axis = PaletteAxis(
            initial_palette=self._genome_axis.current_genome.palette,
            role=self._role, lib=lib, rng=self.rng, freqs=bin_freqs)
        self._zoom_axis = ZoomAxis(role=self._role)
        self._brightness_axis = BrightnessAxis(role=self._role)
        self._detail_axis = DetailAxis(role=self._role)
        self._pending_song_start = False

    @dataclass
    class FrameState:
        """Per-frame output from tick() — three orthogonal axes + audio."""
        genome: 'Genome'          # low axis: transforms, zoom, rotation, center (includes zoom pulse)
        palette: np.ndarray       # mid axis: 256x3 float32 color palette
        spectrum: np.ndarray      # raw FFT spectrum for audio-reactive tonemap
        brightness: float         # RMS-driven brightness for tonemap
        iterations: int           # chaos game iterations — scales with bass energy

    def tick(self, frame_time: float) -> 'FlameSheepCore.FrameState':
        """Advance visualization one frame.
        Returns FrameState with three independent axes.
        """
        snap = self._orch.audio_state
        timestamped = self._orch.drain_events(self._consumer_id)
        now  = self._clock()

        # Collect beat events from timestamped wrappers
        events = [te.event for te in timestamped]
        if self._pending_song_start:
            events.insert(0, BeatEvent(kind='song_start', energy=0.0))
            self._pending_song_start = False

        # Build AudioState for axes
        audio = AudioState(
            events=events,
            spectrum=snap.spectrum,
            bands=snap.bands,
            centroid=snap.centroid,
            centroid_delta=snap.centroid_delta,
            centroid_rms=snap.centroid_rms,
            centroid_harmonic_rms=snap.centroid_harmonic_rms,
            slow_centroid_harmonic_rms=snap.slow_centroid_harmonic_rms,
            percussiveness=snap.percussiveness,
            spectral_novelty=snap.spectral_novelty,
            section_change=snap.section_change,
            bpm=snap.bpm,
            effective_bpm=snap.effective_bpm,
            tempo_confidence=snap.tempo_confidence,
            tempo_saturated=snap.tempo_saturated,
            break_intensity=snap.break_intensity,
            mode=snap.mode,
        )

        # Cross-band inhibition: when multiple bands fire simultaneously,
        # suppress weaker events that are likely bleed from the dominant hit.
        # Events within 60% of the strongest are kept (independent hits).
        if len(audio.events) > 1:
            best_energy = max(e.energy for e in audio.events)
            audio.events = [e for e in audio.events
                            if e.energy > best_energy * 0.6]

        current_mode = Mode(snap.mode)

        # Tick all axes — genome_axis handles mode internally
        self._brightness_axis.tick(audio, frame_time, now)
        self._detail_axis.tick(audio, frame_time, now)
        self._genome_axis.tick(audio, frame_time, now)

        if current_mode == Mode.BEAT:
            self._palette_axis.tick(audio, frame_time, now)
            self._zoom_axis.tick(audio, frame_time, now)

        # Assemble frame state
        frame = self.FrameState(
            genome=self.current_genome,  # overwritten by contribute
            palette=self._palette_axis.palette_current,  # overwritten by contribute
            spectrum=snap.spectrum,
            brightness=self._brightness_axis.brightness,
            iterations=self._detail_axis.iterations,
        )

        self._genome_axis.contribute(frame)
        self._palette_axis.contribute(frame)
        self._zoom_axis.contribute(frame)

        return frame

    # --- Backward-compat properties delegating to axes ---

    @property
    def palette_t(self):
        return self._palette_axis.palette_t

    @property
    def palette_target(self):
        return self._palette_axis.palette_target

    @property
    def zoom_boost(self):
        return self._zoom_axis.zoom_boost

    @zoom_boost.setter
    def zoom_boost(self, value):
        self._zoom_axis.zoom_boost = value

    @property
    def current_genome(self):
        return self._genome_axis.current_genome

    @current_genome.setter
    def current_genome(self, value):
        self._genome_axis.current_genome = value

    @property
    def target_genome(self):
        return self._genome_axis.target_genome

    @target_genome.setter
    def target_genome(self, value):
        self._genome_axis.target_genome = value

    @property
    def morph_t(self):
        return self._genome_axis.morph_t

    @morph_t.setter
    def morph_t(self, value):
        self._genome_axis.morph_t = value

    @property
    def morph_speed(self):
        return self._genome_axis.morph_speed

    @morph_speed.setter
    def morph_speed(self, value):
        self._genome_axis.morph_speed = value

    @property
    def needs_walker_reset(self):
        return self._genome_axis.needs_walker_reset

    @needs_walker_reset.setter
    def needs_walker_reset(self, value):
        self._genome_axis.needs_walker_reset = value

    @property
    def active_loop_id(self):
        return self._genome_axis.active_loop_id

    @property
    def active_genome_db_id(self) -> int | None:
        """DB ID of the genome currently dominant on screen."""
        ga = self._genome_axis
        if ga.morph_t < 0.5:
            g = ga.current_genome
        else:
            g = ga.target_genome
        return g.db_id if g is not None else None

    def force_genome_swap(self) -> None:
        """Immediately swap to a new genome — call when image looks degenerate."""
        self._genome_axis.force_swap()

    def load_loop(self, loop_id: int) -> None:
        self._genome_axis.load_loop(loop_id)

    def next_loop(self) -> None:
        self._genome_axis.next_loop()

    def user_next(self) -> None:
        self._genome_axis.user_next()

    def song_started(self) -> None:
        """Signal new song started — resets tempo, bands, drop detectors, mode.
        Injects a song_start event on the next tick via _pending_song_start.
        """
        self._orch.audio.song_started()
        self._orch.audio.reset_bands()
        self._pending_song_start = True
        log.info('[song] reset tempo, bands, drop detectors, mode')

    def hint_tempo(self, bpm: float) -> None:
        """Provide tempo hint from external source."""
        self._orch.audio.hint_tempo(bpm)
        log.info(f'[tempo] hint: {bpm:.1f} BPM')



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


# ---------------------------------------------------------------------------
# Sway output layout — read once at startup
# ---------------------------------------------------------------------------

def _monitor_cfg(name: str, key: str, default=None):
    """Read per-monitor config: [monitors.NAME].key or [monitors].default_*."""
    mon = cfg.monitors if hasattr(cfg, 'monitors') else {}
    # Check per-monitor override first
    per = mon.get(name, {}) if isinstance(mon, dict) else getattr(mon, name, None)
    if per is not None:
        if isinstance(per, dict):
            val = per.get(key)
        else:
            val = getattr(per, key, None)
        if val is not None:
            return val
    # Fall back to default
    if isinstance(mon, dict):
        return mon.get(f'default_{key}', mon.get(key, default))
    return getattr(mon, f'default_{key}', getattr(mon, key, default))


def _get_output_layout() -> dict[str, dict]:
    """
    Query Wayland outputs for geometry via wl_output + xdg-output-manager.

    Uses xdg-output-manager-v1 for logical positions (compositor-agnostic),
    falling back to wl_output.geometry and then swaymsg.

    Returns {name: {x, y, w, h, ppi, phys_w_mm, phys_h_mm}}.
    """
    import math
    from pywayland.client import Display
    from pywayland.protocol.wayland import WlOutput

    display = Display()
    display.connect()
    registry = display.get_registry()
    outputs = []
    xdg_output_manager = None

    # Try to bind xdg-output-manager for logical positions
    try:
        from .protocol.xdg_output_unstable_v1.zxdg_output_manager_v1 import ZxdgOutputManagerV1
        _has_xdg_output = True
    except ImportError:
        _has_xdg_output = False

    def _on_global(reg, name, interface, version):
        nonlocal xdg_output_manager
        if interface == WlOutput.name:
            out = reg.bind(name, WlOutput, min(version, 4))
            outputs.append(out)
        elif _has_xdg_output and interface == ZxdgOutputManagerV1.name:
            xdg_output_manager = reg.bind(name, ZxdgOutputManagerV1, min(version, 3))

    registry.dispatcher['global'] = _on_global
    display.roundtrip()

    # Collect output info via events
    output_info = {}

    def _make_handlers(out):
        info = {'name': None, 'x': 0, 'y': 0, 'phys_w_mm': 0, 'phys_h_mm': 0,
                'mode_w': 0, 'mode_h': 0, 'xdg_x': None, 'xdg_y': None}
        output_info[id(out)] = info

        def _on_geometry(output, x, y, phys_w, phys_h, subpixel, make, model, transform):
            info['x'] = x
            info['y'] = y
            info['phys_w_mm'] = phys_w
            info['phys_h_mm'] = phys_h
            info['transform'] = transform

        def _on_mode(output, flags, width, height, refresh):
            if flags & 0x1:  # WL_OUTPUT_MODE_CURRENT
                info['mode_w'] = width
                info['mode_h'] = height

        def _on_name(output, name):
            info['name'] = name

        out.dispatcher['geometry'] = _on_geometry
        out.dispatcher['mode'] = _on_mode
        out.dispatcher['name'] = _on_name

        # If xdg-output-manager is available, get logical position
        if xdg_output_manager is not None:
            xdg_out = xdg_output_manager.get_xdg_output(out)

            def _on_logical_position(xdg_output, x, y):
                info['xdg_x'] = x
                info['xdg_y'] = y

            def _on_logical_size(xdg_output, w, h):
                pass  # we use mode size instead

            def _on_done(xdg_output):
                pass

            xdg_out.dispatcher['logical_position'] = _on_logical_position
            xdg_out.dispatcher['logical_size'] = _on_logical_size
            xdg_out.dispatcher['done'] = _on_done

    for out in outputs:
        _make_handlers(out)

    display.roundtrip()
    display.disconnect()

    result = {}
    for info in output_info.values():
        name = info['name']
        if not name:
            continue
        w = info['mode_w']
        h = info['mode_h']
        if w == 0 or h == 0:
            continue

        # Use xdg-output position if available, else wl_output geometry
        x = info['xdg_x'] if info['xdg_x'] is not None else info['x']
        y = info['xdg_y'] if info['xdg_y'] is not None else info['y']

        # Compute PPI from physical size or fallback
        if info['phys_w_mm'] > 0 and info['phys_h_mm'] > 0:
            phys_w_mm = info['phys_w_mm']
            phys_h_mm = info['phys_h_mm']
            diag_mm = math.sqrt(phys_w_mm**2 + phys_h_mm**2)
            diag_px = math.sqrt(w**2 + h**2)
            ppi = diag_px / (diag_mm / 25.4) if diag_mm > 0 else 96.0
        else:
            diag_px = math.sqrt(w**2 + h**2)
            diag_inches = _monitor_cfg(name, 'diagonal', 27.0)
            ppi = diag_px / diag_inches
            phys_w_mm = w / ppi * 25.4
            phys_h_mm = h / ppi * 25.4

        # Account for rotation: swap physical w/h if transform is 90 or 270
        transform = info.get('transform', 0)
        if transform in (1, 3, 6, 7):  # 90°, 270°, flipped variants
            phys_w_mm, phys_h_mm = phys_h_mm, phys_w_mm
            w, h = h, w  # swap pixel dimensions to match physical

        result[name] = {
            'x': x, 'y': y,
            'w': w, 'h': h,
            'ppi': ppi,
            'phys_w_mm': phys_w_mm,
            'phys_h_mm': phys_h_mm,
        }

    # If xdg-output didn't work and positions are all zero, try swaymsg
    if result and all(g['x'] == 0 and g['y'] == 0 for g in result.values()):
        sway_result = _swaymsg_fallback(result)
        if sway_result:
            return sway_result

    return result


def _swaymsg_fallback(wl_result: dict[str, dict]) -> dict[str, dict]:
    """swaymsg fallback for output positions — sway-specific."""
    import json, subprocess, math
    try:
        raw = subprocess.check_output(['swaymsg', '-t', 'get_outputs'], timeout=3)
        outputs = json.loads(raw)
        result = {}
        for o in outputs:
            if not o.get('active'):
                continue
            r = o['rect']
            name = o['name']
            if wl_result and name in wl_result:
                phys_w_mm = wl_result[name]['phys_w_mm']
                phys_h_mm = wl_result[name]['phys_h_mm']
                ppi = wl_result[name]['ppi']
            else:
                mode = o.get('current_mode', {})
                native_w = mode.get('width', r['width'])
                native_h = mode.get('height', r['height'])
                diag_px = math.sqrt(native_w**2 + native_h**2)
                diag_inches = _monitor_cfg(name, 'diagonal', 27.0)
                ppi = diag_px / diag_inches
                phys_w_mm = r['width'] / ppi * 25.4
                phys_h_mm = r['height'] / ppi * 25.4
            result[name] = {
                'x': r['x'], 'y': r['y'],
                'w': r['width'], 'h': r['height'],
                'ppi': ppi,
                'phys_w_mm': phys_w_mm,
                'phys_h_mm': phys_h_mm,
            }
        return result
    except Exception as e:
        log.error(f'swaymsg fallback failed: {e}')
        return {}


def _ensure_singleton() -> None:
    """Kill any existing flame-sheep wallpaper instance.

    Uses a pidfile at ~/.local/share/flame-sheep/pid. If a previous
    instance is still running, SIGTERM it and wait briefly for cleanup.
    """
    import signal
    pid_path = os.path.expanduser('~/.local/share/flame-sheep/pid')
    os.makedirs(os.path.dirname(pid_path), exist_ok=True)

    # Check for existing instance
    try:
        with open(pid_path, 'r') as f:
            old_pid = int(f.read().strip())
        # Check if it's actually running
        os.kill(old_pid, 0)
        if old_pid == os.getpid():
            pass  # that's us (re-exec after VT switch)
        else:
            log.info(f'killing previous instance (pid {old_pid})')
            os.kill(old_pid, signal.SIGTERM)
        # Wait briefly for it to die
        for _ in range(20):
            time.sleep(0.1)
            try:
                os.kill(old_pid, 0)
            except ProcessLookupError:
                break
    except (FileNotFoundError, ValueError, ProcessLookupError):
        pass  # no previous instance

    # Write our PID
    with open(pid_path, 'w') as f:
        f.write(str(os.getpid()))


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

    _vote_count = 0
    _votes_per_evolve = 5       # first few cycles need more votes
    _evolve_count = 0
    _evolving = False

    def _maybe_evolve():
        """Trigger evolution if enough votes have accumulated."""
        nonlocal _vote_count, _votes_per_evolve, _evolve_count, _evolving
        _vote_count += 1
        if _evolving:
            log.debug(f'[evolve] already running, vote queued ({_vote_count})')
            return
        if _vote_count < _votes_per_evolve:
            log.debug(f'[evolve] {_vote_count}/{_votes_per_evolve} votes until next evolution')
            return
        if lib.loop_count() < 2:
            log.warning('not enough loops to evolve')
            return
        _vote_count = 0
        _evolve_count += 1
        if _evolve_count >= 3:
            _votes_per_evolve = 3
        _evolving = True
        log.info(f'[evolve] starting evolution cycle {_evolve_count} in subprocess...')
        import subprocess, sys as _sys
        proc = subprocess.Popen(
            [_sys.executable, '-m', 'flame_sheep', '--evolve', '--loop-length', str(6)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        def _wait_evolve():
            nonlocal _evolving
            out, _ = proc.communicate()
            if out:
                for line in out.decode().strip().split('\n'):
                    log.info(f'[evolve] {line}')
            _evolving = False
        threading.Thread(target=_wait_evolve, daemon=True).start()

    # --- background workers ---
    # GPU render worker is NOT started here — it competes for the GPU and
    # kills desktop performance. Run separately: python -m flame_sheep.gpu_render_worker
    from .cpu_score_worker import BackgroundCpuScorer
    from .transition_worker import BackgroundTransitionScorer
    from .storage import _db_path
    db = str(_db_path())
    cpu_scorer = BackgroundCpuScorer(db_path=db)
    transition_scorer = BackgroundTransitionScorer(db_path=db)
    if lib is not None:
        cpu_scorer.start()
        transition_scorer.start()

    # --- Register command handlers on orchestrator ---
    quit_requested = False

    def _handle_quit(event):
        nonlocal quit_requested
        quit_requested = True

    def _handle_swap(event):
        core.force_genome_swap()

    def _handle_like(event):
        gid = core.active_genome_db_id
        if gid is not None:
            lib.rate('genome', gid, +1)
            log.debug(f'[ctl] liked genome #{gid}')
            _maybe_evolve()
        else:
            log.warning('no active genome to rate')

    def _handle_dislike(event):
        gid = core.active_genome_db_id
        if gid is not None:
            lib.rate('genome', gid, -1)
            log.debug(f'[ctl] disliked genome #{gid}')
            _maybe_evolve()
        core.user_next()

    def _handle_next(event):
        core.user_next()
        log.debug(f'[ctl] next loop #{core.active_loop_id}')

    def _handle_song(event):
        core.song_started()
        core.force_genome_swap()

    def _handle_tempo(event):
        if event.args:
            try:
                bpm = float(event.args[0])
                core.hint_tempo(bpm)
            except ValueError:
                log.info(f'[ctl] invalid tempo: {event.args[0]}')

    def _handle_pause(event):
        core._genome_axis.on_playback_paused()

    def _handle_resume(event):
        core._genome_axis.on_playback_resumed()

    def _handle_seek(event):
        orch.audio.reset_tempo()
        log.debug('[ctl] seek — reset tempo + drop state')

    def _handle_config(event):
        if event.args and event.args[0] == 'reload':
            from .config import cfg
            from flame_sheep_audio.config import cfg as audio_cfg
            cfg.reload()
            audio_cfg.reload()
            log.debug('[ctl] config reloaded (viz + audio)')

    orch.on_command('quit', _handle_quit)
    orch.on_command('swap', _handle_swap)
    orch.on_command('like', _handle_like)
    orch.on_command('dislike', _handle_dislike)
    orch.on_command('next', _handle_next)
    orch.on_command('song', _handle_song)
    orch.on_command('tempo', _handle_tempo)
    orch.on_command('pause', _handle_pause)
    orch.on_command('resume', _handle_resume)
    orch.on_command('seek', _handle_seek)
    orch.on_command('config', _handle_config)

    def _handle_evolve(event):
        """Force an evolution cycle regardless of vote count."""
        nonlocal _vote_count
        _vote_count = _votes_per_evolve  # pretend we have enough votes
        _maybe_evolve()
    orch.on_command('evolve', _handle_evolve)

    # --- Compare mode ---
    from .compare import CompareMode
    _compare_mode: CompareMode | None = None
    _comparing = False
    _compare_needs_reset = False

    def _handle_compare(event):
        nonlocal _compare_mode, _comparing, _compare_needs_reset
        if _comparing:
            return
        _compare_mode = CompareMode(lib)
        _compare_mode.pick_pair()
        _comparing = True
        _compare_needs_reset = True
        log.info('[ctl] entered compare mode')

    def _handle_left(event):
        nonlocal _compare_needs_reset
        if _comparing and _compare_mode:
            _compare_mode.on_left_wins()
            _compare_needs_reset = True

    def _handle_right(event):
        nonlocal _compare_needs_reset
        if _comparing and _compare_mode:
            _compare_mode.on_right_wins()
            _compare_needs_reset = True

    def _handle_skip(event):
        nonlocal _compare_needs_reset
        if _comparing and _compare_mode:
            _compare_mode.on_skip()
            _compare_needs_reset = True

    def _handle_wallpaper(event):
        nonlocal _comparing
        if _comparing:
            _comparing = False
            # Restore normal buffers
            renderer.histogram_buf.bind_to_storage_buffer(0)
            renderer.walker_buf.bind_to_storage_buffer(1)
            renderer._max_buf.bind_to_storage_buffer(8)
            renderer.reset_walkers()
            log.info('[ctl] exited compare mode')

    orch.on_command('compare', _handle_compare)
    orch.on_command('left', _handle_left)
    orch.on_command('right', _handle_right)
    orch.on_command('skip', _handle_skip)
    orch.on_command('wallpaper', _handle_wallpaper)

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
        while not quit_requested:
            time.sleep(2.0)
            elapsed = time.perf_counter() - start
            timeout = 15.0 if elapsed < 20.0 else 3.0
            if time.perf_counter() - _watchdog_last > timeout:
                log.error('[watchdog] render loop stalled, forcing exit')
                os._exit(1)

    _wd_thread = threading.Thread(target=_watchdog, daemon=True)
    _wd_thread.start()

    try:
        while not quit_requested and not all(s.should_close for s in surfaces.values()):
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
                    while not quit_requested and not orch.session.is_active():
                        time.sleep(0.5)
                        _watchdog_last = time.perf_counter()
                        orch.tick()
                    if not quit_requested:
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
            while not quit_requested:
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
                    while not quit_requested:
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

            if _comparing and _compare_mode:
                # --- Compare mode: single renderer, two sequential dispatches ---
                # Swap histogram + walker buffers between left/right dispatches.
                # Same decay, iterations, brightness as normal mode.
                pair = _compare_mode.pair
                if pair.left is not None and pair.right is not None:
                    rot = core._genome_axis._rotation_phase
                    left_g = pair.left.rotated(rot) if rot != 0.0 else pair.left
                    right_g = pair.right.rotated(rot) if rot != 0.0 else pair.right

                    # Find the widest monitor for compare split
                    _compare_surf_name = max(viewports, key=lambda n: viewports[n].w)
                    _compare_vp = viewports[_compare_surf_name]
                    _compare_surf = surfaces.get(_compare_surf_name, first_surf)
                    _cw, _ch = _compare_surf.width, _compare_surf.height

                    # Ensure FBO for left side snapshot
                    if not hasattr(renderer, '_compare_left_fbo') or \
                       renderer._compare_left_fbo.size != (_cw // 2, _ch):
                        renderer._compare_left_tex = ctx.texture((_cw // 2, _ch), 4)
                        renderer._compare_left_fbo = ctx.framebuffer(
                            color_attachments=[renderer._compare_left_tex])

                    # Ensure separate histogram + walker buffers for right side
                    if not hasattr(renderer, '_compare_right_hist'):
                        import numpy as _np
                        from .renderer import N_WALKERS
                        n_pixels = renderer.canvas_w * renderer.canvas_h
                        renderer._compare_right_hist = ctx.buffer(
                            _np.zeros(n_pixels * 2, dtype=_np.uint32).tobytes())
                        renderer._compare_right_walkers = ctx.buffer(
                            _np.random.uniform(-1, 1, (N_WALKERS, 3)).astype(_np.float32).tobytes())
                        renderer._compare_right_max = ctx.buffer(
                            _np.zeros(1, dtype=_np.uint32).tobytes())

                    if _compare_needs_reset:
                        import numpy as _np
                        from .renderer import N_WALKERS
                        renderer.reset_walkers()
                        renderer._compare_right_walkers.write(
                            _np.random.uniform(-1, 1, (N_WALKERS, 3)).astype(_np.float32).tobytes())
                        _compare_needs_reset = False

                    # --- Left genome (uses renderer's own histogram/walkers) ---
                    renderer.histogram_buf.bind_to_storage_buffer(0)
                    renderer.walker_buf.bind_to_storage_buffer(1)
                    renderer._max_buf.bind_to_storage_buffer(8)
                    renderer.upload_audio(frame.spectrum)
                    renderer.upload_genome(left_g)
                    renderer.upload_palette(frame.palette)
                    renderer.clear_histogram(decay=0.3)
                    renderer.dispatch_chaos_game(iterations=frame.iterations)
                    ctx.memory_barrier()
                    renderer.reduce_histogram_max()

                    # Tonemap left genome into FBO
                    renderer.render_tonemap(_compare_vp, _cw // 2, _ch,
                                           brightness=frame.brightness,
                                           target_fbo=renderer._compare_left_fbo)

                    # --- Right genome (swap in right-side buffers) ---
                    renderer._compare_right_hist.bind_to_storage_buffer(0)
                    renderer._compare_right_walkers.bind_to_storage_buffer(1)
                    renderer._compare_right_max.bind_to_storage_buffer(8)
                    renderer.upload_genome(right_g)
                    renderer.clear_histogram(decay=0.3)
                    renderer.dispatch_chaos_game(iterations=frame.iterations)
                    ctx.memory_barrier()
                    renderer.reduce_histogram_max()

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
                elif _comparing and name == _compare_surf_name:
                    # Split the center monitor: left FBO + right histogram
                    from .renderer import _bind_default_framebuffer
                    _bind_default_framebuffer()
                    vp = viewports[name]
                    half_w = surf.width // 2

                    # Left half: blit pre-rendered left FBO
                    ctx.viewport = (0, 0, half_w, surf.height)
                    renderer._compare_left_tex.use(location=0)
                    bp = renderer.blur_program
                    bp['u_texture'] = 0
                    bp['u_direction'] = (0.0, 0.0)
                    bp['u_radius'] = 0.0
                    renderer.blur_vao.render(moderngl.TRIANGLES)

                    # Right half: tonemap right histogram (still bound from compute)
                    renderer._compare_right_hist.bind_to_storage_buffer(0)
                    renderer._compare_right_max.bind_to_storage_buffer(8)
                    renderer.render_tonemap(vp, surf.width, surf.height,
                                           brightness=frame.brightness,
                                           screen_rect=(half_w, 0, surf.width - half_w, surf.height))

                    # Restore normal histogram for other surfaces
                    renderer.histogram_buf.bind_to_storage_buffer(0)
                    renderer._max_buf.bind_to_storage_buffer(8)
                elif _comparing:
                    # Non-center monitors: render left genome normally
                    renderer.histogram_buf.bind_to_storage_buffer(0)
                    renderer._max_buf.bind_to_storage_buffer(8)
                    renderer.render_tonemap(viewports[name], surf.width, surf.height,
                                           brightness=frame.brightness)
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
        if not quit_requested and orch.session.gpu_paused:
            log.info('[render] surfaces closed during VT switch, waiting for resume...')
            while not orch.session.is_active():
                time.sleep(0.5)
            log.info('[render] session resumed, restarting')
            import sys
            os.execv(sys.executable, [sys.executable, '-m', 'flame_sheep'] + sys.argv[1:])

        log.debug(f'[render] exiting render loop (quit_requested={quit_requested})')
        if feature_logger:
            feature_logger.close()
        cpu_scorer.stop()
        transition_scorer.stop()
        orch.stop()
        session.destroy()


def _run_variation_benchmark() -> None:
    """Benchmark variations and library genomes on the GPU.

    Two phases:
      1. Per-variation: each variation solo with 3 transforms (isolates cost)
      2. Library genomes: actual genomes from the library (real-world cost)

    Outputs a cost table with budget classification:
      OK     — under 2x linear baseline
      HEAVY  — 2-4x baseline
      COSTLY — over 4x baseline (consider iteration reduction)
    """
    import time
    import moderngl
    from .genome import (Genome, Transform, Variation, NUM_VARIATIONS,
                         MAX_TRANSFORMS, MAX_VAR_PARAMS)
    from .renderer import FlameRenderer

    # Variation names for display
    var_names = {}
    for name in dir(Variation):
        if name.startswith('_'):
            continue
        val = getattr(Variation, name)
        if isinstance(val, int) and 0 <= val < NUM_VARIATIONS:
            var_names[val] = name

    ctx = moderngl.create_context(standalone=True)
    renderer = FlameRenderer(ctx, 1920, 1080)

    n_warmup = 5
    n_frames = 30
    rng = np.random.default_rng(42)

    def _bench_genome(g: Genome) -> float:
        """Returns ms/frame for a genome at 500 iterations."""
        renderer.upload_genome(g)
        for _ in range(n_warmup):
            renderer.clear_histogram(decay=0.3)
            renderer.dispatch_chaos_game(iterations=500)
            ctx.finish()
        t0 = time.perf_counter()
        for _ in range(n_frames):
            renderer.clear_histogram(decay=0.3)
            renderer.dispatch_chaos_game(iterations=500)
            ctx.finish()
        return (time.perf_counter() - t0) / n_frames * 1000

    # --- Phase 1: Per-variation ---
    print('\n=== Per-Variation Benchmark (3 transforms, solo) ===\n')
    results = []
    for var_idx in range(NUM_VARIATIONS):
        g = Genome.random(rng, n_transforms=3)
        for tr in g.transforms:
            tr.variations[:] = 0.0
            tr.variations[var_idx] = 1.0
            if var_idx in (Variation.JULIAN, Variation.JULIASCOPE):
                tr.var_params = {'julian_power': 3.0, 'julian_dist': 1.0}
            elif var_idx == Variation.SPLITS:
                tr.var_params = {'splits_x': 0.5, 'splits_y': 0.5}
            elif var_idx == Variation.CURL:
                tr.var_params = {'curl_c1': 0.5, 'curl_c2': 0.0}

        ms = _bench_genome(g)
        name = var_names.get(var_idx, f'var_{var_idx}')
        results.append((var_idx, name, ms))

    baseline = next(r[2] for r in results if r[0] == 0)  # LINEAR
    print(f'{"idx":>3}  {"variation":<20}  {"ms/frame":>9}  {"rel":>6}  {"budget"}')
    print('-' * 58)

    for idx, name, ms in sorted(results, key=lambda r: -r[2]):
        rel = ms / baseline if baseline > 0 else 0
        if rel > 4.0:
            budget = 'COSTLY'
        elif rel > 2.0:
            budget = 'HEAVY'
        else:
            budget = 'ok'
        print(f'{idx:3d}  {name:<20}  {ms:9.3f}  {rel:5.2f}x  {budget}')

    print(f'\nBaseline (linear): {baseline:.3f} ms/frame')
    print(f'Budget at 60fps: {16.7:.1f} ms/frame')

    # --- Phase 2: Library genomes ---
    from .storage import Library
    lib = Library()
    n_genomes = lib.genome_count()
    if n_genomes > 0:
        print(f'\n=== Library Genome Benchmark ({min(n_genomes, 50)} genomes) ===\n')
        top = lib.top_genomes(n=50)
        genome_results = []
        for gid, _score in top:
            g = lib.load_genome(gid)
            ms = _bench_genome(g)
            # Identify dominant variations
            dom_vars = []
            for tr in g.transforms:
                if tr.weight < 0.05:
                    continue
                for j, w in enumerate(tr.variations):
                    if w > 0.1:
                        dom_vars.append(var_names.get(j, f'v{j}'))
            genome_results.append((gid, ms, dom_vars))

        print(f'{"id":>5}  {"ms/frame":>9}  {"fps":>5}  {"budget":<8}  variations')
        print('-' * 70)

        for gid, ms, dom_vars in sorted(genome_results, key=lambda r: -r[1]):
            fps = 1000 / ms if ms > 0 else 999
            if ms > 16.7:
                budget = 'COSTLY'
            elif ms > 10.0:
                budget = 'HEAVY'
            else:
                budget = 'ok'
            vars_str = ', '.join(sorted(set(dom_vars)))[:40]
            print(f'{gid:5d}  {ms:9.3f}  {fps:5.1f}  {budget:<8}  {vars_str}')

        avg = np.mean([r[1] for r in genome_results])
        worst = max(r[1] for r in genome_results)
        best = min(r[1] for r in genome_results)
        costly = sum(1 for r in genome_results if r[1] > 16.7)
        print(f'\nAvg: {avg:.1f}ms  Best: {best:.1f}ms  Worst: {worst:.1f}ms')
        print(f'{costly}/{len(genome_results)} genomes over 60fps budget (16.7ms)')

    lib.close()
    ctx.release()


def _run_library_commands(args: argparse.Namespace) -> None:
    """Handle --generate-genomes, --compose-loops, --evolve, --stats."""
    from .genome import Genome
    from .storage import Library
    from .loops import save_best_loops, evolve_loops

    lib = Library()

    if args.generate_genomes:
        n = args.generate_genomes
        print(f'Generating {n} genomes...')
        rng = np.random.default_rng()
        for i in range(n):
            g = Genome.random(rng)
            lib.save_genome(g)
            if (i + 1) % 10 == 0 or i == n - 1:
                print(f'  {i + 1}/{n}')
        print(f'Library now has {lib.genome_count()} genomes')

    if args.generate_palettes is not None:
        from .genome import _random_palette
        n = args.generate_palettes
        print(f'Generating {n} palettes...')
        rng = np.random.default_rng()
        for i in range(n):
            lib.save_palette(_random_palette(rng))
            if (i + 1) % 50 == 0 or i == n - 1:
                print(f'  {i + 1}/{n}')
        print(f'Library now has {lib.palette_count()} palettes')

    if args.compose_loops is not None:
        from .loops import compose_loops_graph
        n = args.compose_loops
        print(f'Composing loops (target: {n}, length: {args.loop_length})...')
        candidates = compose_loops_graph(
            lib, loop_length=args.loop_length, n_attempts=n * 10,
        )
        if candidates:
            ids = save_best_loops(lib, candidates, n_keep=n)
            print(f'Saved {len(ids)} loops')
            for lid in ids:
                scores = lib.loop_fitness(lid)
                print(f'  #{lid}: fitness={scores["fitness"]:.3f}, '
                      f'coherence={scores["mean_coherence"]:.3f}')
        else:
            print('No viable loops found — try generating more genomes')

    if args.evolve:
        n_loops = lib.loop_count()
        if n_loops < 2:
            print('Need at least 2 loops to evolve. Run --compose-loops first.')
        else:
            print(f'Evolving from {n_loops} loops...')
            top = evolve_loops(lib, n_generations=3, n_offspring=8,
                               pool_size=min(30, lib.genome_count()),
                               loop_length=args.loop_length)
            print(f'Top {len(top)} loops after evolution ({lib.loop_count()} total):')
            for lid in top:
                scores = lib.loop_fitness(lid)
                print(f'  #{lid}: fitness={scores["fitness"]:.3f}')

    if getattr(args, 'prune_loops', False):
        from .loops import prune_loops
        before = lib.loop_count()
        n = prune_loops(lib)
        print(f'Pruned {n} loops ({before} -> {lib.loop_count()})')

    if args.stats:
        n_genomes = lib.genome_count()
        n_loops = lib.loop_count()
        n_palettes = lib.palette_count()
        print(f'\nLibrary: {n_genomes} genomes, {n_loops} loops, {n_palettes} palettes')
        if n_loops > 0:
            top = lib.top_loops(n=5)
            print(f'Top loops:')
            for lid, scores in top:
                gids = lib.loop_genome_ids(lid)
                rating = lib.net_rating('loop', lid)
                print(f'  #{lid}: {len(gids)} genomes, '
                      f'fitness={scores["fitness"]:.3f}, '
                      f'smoothness={scores.get("smoothness", 0):.3f}, '
                      f'coherence={scores["mean_coherence"]:.3f}, '
                      f'diversity={scores["diversity"]:.3f}, '
                      f'votes={rating:+d}')

    lib.close()


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
