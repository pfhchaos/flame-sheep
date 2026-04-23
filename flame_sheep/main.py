"""
flame-sheep — entry point and main loop.

Two modes:
  default    — GLFW window via moderngl-window (for development / testing)
  --wallpaper — wlr-layer-shell BACKGROUND surface (true wallpaper on sway)

Core rendering/audio/genome logic lives in FlameSheepCore so it can be
shared between both modes.
"""

import logging

log = logging.getLogger(__name__)


import argparse
import sys
import time
from dataclasses import dataclass
import os
import threading
import numpy as np
import moderngl_window as mglw
from moderngl_window import settings

from .genome import Genome, _lerp_arr
from .audio import AudioProcessor, SyntheticAudioProcessor, BeatEvent, AudioState, DEFAULT_DEVICE
from .audio.drop_detector import DropDetector
from .renderer import FlameRenderer, Viewport
from .control import ControlPipe, ControlEvent
from .tempo import TempoTracker
from .axes.zoom_axis import ZoomAxis
from .axes.brightness_axis import BrightnessAxis
from .axes.detail_axis import DetailAxis
from .axes.genome_axis import GenomeAxis
from .axes.palette_axis import PaletteAxis
from .axes.drift_axis import DriftAxis


# Set by main() before run_window_config — workaround for moderngl-window
# not passing CLI args through to WindowConfig.__init__
_AUDIO_DEVICE: str | int = DEFAULT_DEVICE
_TEST_AUDIO:   bool = False


class FlameSheepCore:
    """
    Audio + genome state, shared across all outputs.
    Does NOT hold a renderer or GL context — that lives in each window.

    Each frame, call:
      genome, spectrum = core.tick(frame_time)   # advance state, get current genome
    Then render with those values in your per-window renderer.

    Call .force_genome_swap() to kick a degenerate image.
    Call .stop() on shutdown.
    """

    def __init__(self, audio_device=DEFAULT_DEVICE, test_audio: bool = False,
                 lib=None, clock=None, genome_factory=None,
                 ):
        self._clock = clock or time.perf_counter
        self.rng = np.random.default_rng()
        self._lib = lib

        # --- Visual axes ---
        factory = genome_factory or (lambda: Genome.random(self.rng))
        self._genome_axis = GenomeAxis(genome_factory=factory, lib=lib, rng=self.rng)
        self._palette_axis = PaletteAxis(
            initial_palette=self._genome_axis.current_genome.palette,
            lib=lib, rng=self.rng)
        self._zoom_axis = ZoomAxis()
        self._brightness_axis = BrightnessAxis()
        self._detail_axis = DetailAxis()
        self._drift_axis = DriftAxis(self._genome_axis)
        # Tempo tracking + drop detection
        self.tempo = TempoTracker()
        self._drop_detector = DropDetector()
        self._pending_song_start = False

        if test_audio:
            self.audio = SyntheticAudioProcessor(
                kick_interval=0.5,
                snare_interval=1.0,
                hihat_interval=0.25,
                clock=clock,
            )
        else:
            self.audio = AudioProcessor(device=audio_device)
            log.info(f'audio device: {audio_device!r}')
        self.audio.start()

    @dataclass
    class FrameState:
        """Per-frame output from tick() — three orthogonal axes + audio."""
        genome: 'Genome'          # kick axis: transforms, zoom, rotation, center (includes zoom pulse)
        palette: np.ndarray       # snare axis: 256x3 float32 color palette
        spectrum: np.ndarray      # raw FFT spectrum for audio-reactive tonemap
        brightness: float         # RMS-driven brightness for tonemap
        iterations: int           # chaos game iterations — scales with bass energy

    def tick(self, frame_time: float) -> 'FlameSheepCore.FrameState':
        """Advance audio/genome state one frame.
        Returns FrameState with three independent axes.
        """
        snap = self.audio.drain()
        now  = self._clock()

        # Filter events through tempo gating
        filtered_events = self._handle_beats(snap.events)

        # Inject song_start event if pending
        if self._pending_song_start:
            filtered_events.insert(0, BeatEvent(kind='song_start', energy=0.0))
            self._pending_song_start = False

        # Run drop detector — may inject a 'drop' event
        drop = self._drop_detector.detect(
            filtered_events, snap.centroid_rms,
            self.tempo.bpm, self._drift_axis.drifting, frame_time)
        if drop is not None:
            filtered_events.append(drop)

        # Build AudioState for axes
        audio = AudioState(
            events=filtered_events,
            rms=snap.rms,
            percussiveness=snap.percussiveness,
            centroid=snap.centroid,
            centroid_delta=snap.centroid_delta,
            centroid_rms=snap.centroid_rms,
            bpm=self.tempo.bpm,
            drifting=self._drift_axis.drifting,
        )

        # Tick visual axes (drift must tick before genome to trigger swaps)
        self._drift_axis.tick(audio, frame_time, now)
        self._genome_axis.tick(audio, frame_time, now)
        self._palette_axis.tick(audio, frame_time, now)
        self._zoom_axis.tick(audio, frame_time, now)
        self._brightness_axis.tick(audio, frame_time, now)
        self._detail_axis.tick(audio, frame_time, now)

        # Assemble frame state
        frame = self.FrameState(
            genome=self.current_genome,  # overwritten by contribute
            palette=self._palette_axis.palette_current,  # overwritten by contribute
            spectrum=snap.spectrum,
            brightness=self._brightness_axis.brightness,
            iterations=self._detail_axis.iterations,
        )

        # Axes contribute (order: genome first, then zoom modifies genome.zoom)
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

    def force_genome_swap(self):
        """Immediately swap to a new genome — call when image looks degenerate."""
        self._genome_axis.force_swap()

    def stop(self):
        self.audio.stop()

    def load_loop(self, loop_id: int):
        self._genome_axis.load_loop(loop_id)

    def next_loop(self):
        self._genome_axis.next_loop()

    def song_started(self):
        """Signal new song started — resets tempo, bands, drop detector.
        Injects a song_start event on the next tick via _pending_song_start.
        """
        self.tempo.song_started()
        self.audio.reset_bands()
        self._drop_detector.reset()
        self._pending_song_start = True
        log.info('[song] reset tempo, bands, drop detector')
        
    def hint_tempo(self, bpm: float):
        """Provide tempo hint from external source."""
        self.tempo.hint_tempo(bpm)
        log.info(f'[tempo] hint: {bpm:.1f} BPM')

    def _handle_beats(self, events: list[BeatEvent]) -> list[BeatEvent]:
        """Feed events to tempo tracker and return them for axes.
        All events pass through — no gating."""
        now = self._clock()
        for event in events:
            if event.kind in ('kick', 'snare', 'clap', 'hihat'):
                self.tempo.process_onset(event.kind, now)
        return events



class FlameSheepApp(mglw.WindowConfig):
    """moderngl-window wrapper — used for windowed/dev mode."""
    title       = 'flame-sheep'
    gl_version  = (4, 3)
    resizable   = True
    vsync       = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._core     = FlameSheepCore(audio_device=_AUDIO_DEVICE, test_audio=_TEST_AUDIO)
        w, h = self.window_size
        self._renderer = FlameRenderer(self.ctx, w, h)
        self._viewport = Viewport(0, 0, w, h)
        log.info('flame-sheep started. Press Q to quit, F to force genome swap.')

    def on_render(self, time_val: float, frame_time: float):
        self.ctx.clear(0.0, 0.0, 0.0)
        frame = self._core.tick(frame_time)
        self._renderer.upload_audio(frame.spectrum)
        self._renderer.upload_genome(frame.genome)
        self._renderer.upload_palette(frame.palette)
        if self._core.needs_walker_reset:
            self._renderer.reset_walkers()
            self._core.needs_walker_reset = False
        self._renderer.clear_histogram()
        self._renderer.dispatch_chaos_game(iterations=frame.iterations)
        self.ctx.memory_barrier()
        w, h = self.window_size
        self._renderer.render_tonemap(self._viewport, w, h, brightness=frame.brightness)

    def key_event(self, key, action, modifiers):
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.Q:
                self._core.stop()
                self.wnd.close()
            elif key == self.wnd.keys.F:
                self._core.force_genome_swap()

    def close(self):
        self._core.stop()


# ---------------------------------------------------------------------------
# Sway output layout — read once at startup
# ---------------------------------------------------------------------------

# Monitor physical specs: diagonal size in inches
# Used to compute PPI for physical alignment across mixed-DPI displays
MONITOR_SIZES = {
    'LS24A600U': 24.0,   # Samsung 24" 2560x1440
    'ASUS XG438': 43.0,  # ASUS 43" 3840x2160
}
DEFAULT_MONITOR_SIZE = 27.0  # fallback assumption


def _get_sway_layout() -> dict[str, dict]:
    """
    Query swaymsg for active output geometry and compute PPI.
    Returns {name: {x, y, w, h, ppi, phys_w_mm, phys_h_mm}}.
    """
    import json, subprocess, math
    try:
        raw = subprocess.check_output(['swaymsg', '-t', 'get_outputs'], timeout=3)
        outputs = json.loads(raw)
        result = {}
        for o in outputs:
            if not o.get('active'):
                continue
            r = o['rect']
            mode = o.get('current_mode', {})
            model = o.get('model', '')
            
            # Native resolution (before rotation)
            native_w = mode.get('width', r['width'])
            native_h = mode.get('height', r['height'])
            
            # Get diagonal size for this model
            diag_inches = MONITOR_SIZES.get(model, DEFAULT_MONITOR_SIZE)
            
            # Calculate PPI from native resolution and diagonal
            diag_px = math.sqrt(native_w**2 + native_h**2)
            ppi = diag_px / diag_inches
            
            # Physical size in mm (display coords, after rotation)
            phys_w_mm = r['width'] / ppi * 25.4
            phys_h_mm = r['height'] / ppi * 25.4
            
            result[o['name']] = {
                'x': r['x'], 'y': r['y'],
                'w': r['width'], 'h': r['height'],
                'ppi': ppi,
                'phys_w_mm': phys_w_mm,
                'phys_h_mm': phys_h_mm,
            }
        return result
    except Exception as e:
        log.error(f'swaymsg failed: {e}')
        return {}


def _ensure_singleton():
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


def _run_wallpaper(audio_device, test_audio: bool, blur_radius: float = 1.0):
    """
    Wallpaper mode — one continuous flame fractal image across all monitors.
    """
    _ensure_singleton()
    from .wayland_window import WallpaperSession

    # --- discover outputs and sway layout ---
    output_names = WallpaperSession.list_outputs()
    log.info(f'outputs: {output_names}')

    layout = _get_sway_layout()
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

    # --- compute viewport for each output (in canvas pixels) ---
    viewports: dict[str, Viewport] = {}
    for name, g in active.items():
        vx = int(phys_x[name] * canvas_ppmm)
        vy = int(phys_y[name] * canvas_ppmm)
        vw = int(g['phys_w_mm'] * canvas_ppmm)
        vh = int(g['phys_h_mm'] * canvas_ppmm)
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

    # --- library + evolution state ---
    from .storage import Library
    from .loops import evolve_loops
    lib = Library()

    core = FlameSheepCore(audio_device=audio_device, test_audio=test_audio, lib=lib,
                          )

    _vote_count = 0
    _votes_per_evolve = 5       # first few cycles need more votes
    _evolve_count = 0
    _evolving = False

    def _maybe_evolve():
        """Trigger evolution if enough votes have accumulated."""
        nonlocal _vote_count, _votes_per_evolve, _evolve_count, _evolving
        _vote_count += 1
        if _evolving:
            log.info(f'[evolve] already running, vote queued ({_vote_count})')
            return
        if _vote_count < _votes_per_evolve:
            log.info(f'[evolve] {_vote_count}/{_votes_per_evolve} votes until next evolution')
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
        import subprocess
        proc = subprocess.Popen(
            [sys.executable, '-m', 'flame_sheep', '--evolve', '--loop-length', str(6)],
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

    # --- opportunistic symmetry scoring ---
    def _try_score_symmetry(core, renderer, lib):
        """Score the currently displayed genome's symmetry if not yet computed."""
        ga = core._genome_axis
        if not ga._loop_genomes or ga._loop_pos >= len(ga._loop_genomes):
            return
        # Find genome ID for current loop position
        gids = lib.loop_genome_ids(ga.active_loop_id) if ga.active_loop_id else []
        if not gids:
            return
        gid = gids[ga._loop_pos % len(gids)]
        # Check if already scored
        row = lib.conn.execute(
            'SELECT symmetry_max FROM genomes WHERE id = ?', (gid,)
        ).fetchone()
        if row and row[0] is not None:
            return  # already scored
        # Read histogram and compute symmetry
        from .genome import _score_symmetry
        hit_counts, _ = renderer.histogram_data()
        sym = _score_symmetry(hit_counts.astype(np.float64))
        lib.conn.execute(
            '''UPDATE genomes SET symmetry_max=?, rotational=?, reflective=?,
               radial=?, periodic=?, fractal_dim=? WHERE id=?''',
            (sym['symmetry_max'], sym['rotational'], sym['reflective'],
             sym['radial'], sym['periodic'], sym['fractal_dim'], gid),
        )
        lib.conn.commit()
        log.debug(f'[symmetry] genome #{gid}: sym={sym["symmetry_max"]:.3f} '
                  f'fd={sym["fractal_dim"]:.2f}')

    # --- control pipe for external commands ---
    control = ControlPipe()
    control.start()
    log.info(f'wallpaper running. Control pipe: {control.pipe_path}')

    # --- MPRIS listener for song change detection ---
    from .mpris import MprisListener
    mpris = MprisListener(ctl_path=control.pipe_path)
    mpris.start()

    def handle_control_events() -> bool:
        """Process control events. Returns True if quit requested."""
        for event in control.poll_all():
            log.info(f'[ctl] {event.command} {" ".join(event.args)}')

            if event.command == 'quit':
                return True
            elif event.command == 'swap':
                core.force_genome_swap()
            elif event.command == 'like':
                if core.active_loop_id is not None:
                    lib.rate('loop', core.active_loop_id, +1)
                    # Propagate to recent palettes
                    for pid in core._palette_axis.palette_history:
                        lib.rate('palette', pid, +1)
                    lib.update_loop_fitness(core.active_loop_id)
                    log.info(f'[ctl] liked loop #{core.active_loop_id} '
                          f'(+{len(core._palette_axis.palette_history)} palettes)')
                    _maybe_evolve()
                else:
                    log.warning('no active loop to rate')
            elif event.command == 'dislike':
                if core.active_loop_id is not None:
                    lib.rate('loop', core.active_loop_id, -1)
                    for pid in core._palette_axis.palette_history:
                        lib.rate('palette', pid, -1)
                    lib.update_loop_fitness(core.active_loop_id)
                    log.info(f'[ctl] disliked loop #{core.active_loop_id} '
                          f'(+{len(core._palette_axis.palette_history)} palettes)')
                    _maybe_evolve()
                core.next_loop()  # switch to a different loop
            elif event.command == 'next':
                core.next_loop()
                log.info(f'[ctl] next loop #{core.active_loop_id}')
            elif event.command == 'song':
                core.song_started()
                core.force_genome_swap()  # fresh visual for new song
            elif event.command == 'tempo':
                if event.args:
                    try:
                        bpm = float(event.args[0])
                        core.hint_tempo(bpm)
                    except ValueError:
                        log.info(f'[ctl] invalid tempo: {event.args[0]}')
            elif event.command == 'seek':
                core.tempo.reset()
                core._drop_detector.reset()
                log.info('[ctl] seek — reset tempo + drop state')
            elif event.command == 'pause':
                # TODO: enter ambient/slow mode
                pass
            elif event.command == 'resume':
                # TODO: exit ambient mode
                pass
        return False

    last_time = time.perf_counter()
    quit_requested = False
    _frame = 0
    _watchdog_last = time.perf_counter()

    def _watchdog():
        """Background thread: force exit if render loop stops making progress."""
        while not quit_requested:
            time.sleep(2.0)
            if time.perf_counter() - _watchdog_last > 3.0:
                log.error('[watchdog] render loop stalled, forcing exit')
                os._exit(1)

    _wd_thread = threading.Thread(target=_watchdog, daemon=True)
    _wd_thread.start()

    try:
        while not quit_requested and not all(s.should_close for s in surfaces.values()):
            _frame += 1
            _watchdog_last = time.perf_counter()
            quit_requested = handle_control_events()

            # Dispatch pending Wayland events (delivers frame callbacks)
            if not session.dispatch():
                log.error('wayland connection lost, exiting')
                break

            # Block until the compositor signals it wants a new frame.
            # This replaces eglSwapInterval(1) as our frame pacer.
            # Wait for compositor to signal a frame callback.
            # Audio thread runs independently — no need to tick audio here.
            while not quit_requested:
                ready = {n: s for n, s in surfaces.items()
                         if s._frame_pending and not s.should_close}
                if ready or all(s.should_close for s in surfaces.values()):
                    break
                session.wait_for_events(timeout=0.016)
                _watchdog_last = time.perf_counter()

            if not ready:
                continue

            now        = time.perf_counter()
            frame_time = now - last_time
            last_time  = now

            frame = core.tick(frame_time)

            # Compute pass — surface doesn't matter for compute, keep first
            if not session.make_current(first_surf):
                break  # surfaces died (sway reload?)
            renderer.upload_audio(frame.spectrum)
            renderer.upload_genome(frame.genome)
            renderer.upload_palette(frame.palette)
            if core.needs_walker_reset:
                renderer.reset_walkers()
                core.needs_walker_reset = False
            renderer.clear_histogram()
            renderer.dispatch_chaos_game(iterations=frame.iterations)
            ctx.memory_barrier()

            # Opportunistic symmetry scoring — score the current genome if unscored
            # Runs at most once every 120 frames (~2s) to avoid GPU readback spam
            if _frame % 120 == 60 and core.active_loop_id is not None:
                _try_score_symmetry(core, renderer, lib)

            # Tonemap pass — only swap surfaces the compositor is ready for
            for name, surf in ready.items():
                if not session.make_current(surf):
                    continue  # this surface is dead, skip it
                renderer.render_tonemap(viewports[name], surf.width, surf.height,
                                       brightness=frame.brightness)
                if not session.swap(surf):
                    break  # wayland connection lost

    finally:
        mpris.stop()
        control.stop()
        core.stop()
        session.destroy()


def _run_variation_benchmark():
    """Benchmark each variation solo on the GPU."""
    import time
    import moderngl
    from .genome import (Genome, Transform, Variation, NUM_VARIATIONS,
                         MAX_TRANSFORMS, MAX_VAR_PARAMS)
    from .renderer import FlameRenderer

    # Variation names for display (skip dunder attrs like __firstlineno__)
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

    # Build a base genome with 3 transforms
    base = Genome.random(rng, n_transforms=3)

    results = []
    for var_idx in range(NUM_VARIATIONS):
        # Set all transforms to use only this variation
        g = Genome.random(rng, n_transforms=3)
        for tr in g.transforms:
            tr.variations[:] = 0.0
            tr.variations[var_idx] = 1.0
            # Set params for parametric variations
            if var_idx in (Variation.JULIAN, Variation.JULIASCOPE):
                tr.var_params = {'julian_power': 3.0, 'julian_dist': 1.0}
            elif var_idx == Variation.SPLITS:
                tr.var_params = {'splits_x': 0.5, 'splits_y': 0.5}
            elif var_idx == Variation.CURL:
                tr.var_params = {'curl_c1': 0.5, 'curl_c2': 0.0}

        renderer.upload_genome(g)

        # Warmup
        for _ in range(n_warmup):
            renderer.clear_histogram()
            renderer.dispatch_chaos_game()
            ctx.finish()

        # Timed
        t0 = time.perf_counter()
        for _ in range(n_frames):
            renderer.clear_histogram()
            renderer.dispatch_chaos_game()
            ctx.finish()
        elapsed = time.perf_counter() - t0

        ms_per_frame = (elapsed / n_frames) * 1000
        name = var_names.get(var_idx, f'var_{var_idx}')
        results.append((var_idx, name, ms_per_frame))

    # Also time a baseline with linear only
    print(f'\n{"idx":>3}  {"variation":<15}  {"ms/frame":>9}  {"rel":>6}')
    print('-' * 42)

    baseline = next(r[2] for r in results if r[0] == 0)  # LINEAR
    for idx, name, ms in results:
        rel = ms / baseline if baseline > 0 else 0
        marker = ' **' if rel > 2.0 else ''
        print(f'{idx:3d}  {name:<15}  {ms:9.3f}  {rel:5.2f}x{marker}')

    ctx.release()


def _run_library_commands(args):
    """Handle --generate-genomes, --compose-loops, --evolve, --stats."""
    from .genome import Genome
    from .storage import Library
    from .loops import compose_loops, save_best_loops, evolve_loops

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
        n = args.compose_loops
        print(f'Composing loops (target: {n}, length: {args.loop_length})...')
        candidates = compose_loops(
            lib, pool_size=min(40, lib.genome_count()),
            loop_length=args.loop_length,
            n_attempts=n * 10,
            min_coherence=-1.0,
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
            print(f'Top {len(top)} loops after evolution:')
            for lid in top:
                scores = lib.loop_fitness(lid)
                print(f'  #{lid}: fitness={scores["fitness"]:.3f}')

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


def main():
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
    parser.add_argument('--stats', action='store_true',
                        help='print library statistics and exit')
    parser.add_argument('--blur-radius', type=float, default=1.0,
                        help='wallpaper blur strength (0=off, 1=light, 2+=heavy; default: 1.0)')
    parser.add_argument('--benchmark-variations', action='store_true',
                        help='benchmark each variation solo (GPU timing) and exit')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='logging verbosity (default: INFO)')

    # parse_known_args so moderngl-window's own flags don't cause errors here
    args, remaining = parser.parse_known_args()

    if args.help:
        parser.print_help()
        return

    from .log import setup_logging
    setup_logging(level=args.log_level)

    global _AUDIO_DEVICE, _TEST_AUDIO
    _AUDIO_DEVICE = args.audio_device
    _TEST_AUDIO   = args.test_audio

    if args.list_audio:
        from .audio import list_monitor_devices
        for d in list_monitor_devices():
            print(f"  [{d['index']:2d}] {d['name']}")
        return

    if args.benchmark_variations:
        _run_variation_benchmark()
        return

    if args.generate_genomes or args.generate_palettes is not None or args.compose_loops is not None or args.evolve or args.stats:
        _run_library_commands(args)
        return

    if args.wallpaper:
        _run_wallpaper(args.audio_device, args.test_audio, blur_radius=args.blur_radius)
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
