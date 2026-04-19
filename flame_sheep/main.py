"""
flame-sheep — entry point and main loop.

Two modes:
  default    — GLFW window via moderngl-window (for development / testing)
  --wallpaper — wlr-layer-shell BACKGROUND surface (true wallpaper on sway)

Core rendering/audio/genome logic lives in FlameSheepCore so it can be
shared between both modes.
"""

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
from .audio import AudioProcessor, SyntheticAudioProcessor, BeatEvent, DEFAULT_DEVICE
from .renderer import FlameRenderer, Viewport
from .control import ControlPipe, ControlEvent
from .tempo import TempoTracker


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

    MIN_GENOME_DISTANCE = 0.15
    DRIFT_RMS_THRESHOLD = 0.002
    DRIFT_SWAP_FRAMES   = 60 * 8
    DRIFT_MORPH_SPEED   = 0.003
    KICK_SWAP_EVERY     = 4     # swap genome every Nth kick (1 bar in 4/4)
    KICK_MORPH_PULSE    = 0.03  # morph speed boost on non-swap kicks

    # Zoom axis (hihat) — zoom pulse on hits, decays back
    ZOOM_BOOST_MAX  = 0.3    # maximum zoom boost (30%)
    ZOOM_DECAY      = 0.95   # per-frame decay back to baseline

    def __init__(self, audio_device=DEFAULT_DEVICE, test_audio: bool = False,
                 lib=None):
        self.rng = np.random.default_rng()

        # --- Kick axis: genome (transforms, zoom, rotation, center) ---
        self.current_genome = Genome.random(self.rng)
        self.target_genome  = Genome.random(self.rng)
        self.morph_t        = 0.0
        self.morph_speed    = self.DRIFT_MORPH_SPEED

        self._next_genome: Genome | None = None
        self._genome_lock = threading.Lock()

        # --- Snare axis: palette graph traversal ---
        self.palette_current = self.current_genome.palette.copy()
        self.palette_target  = self.target_genome.palette.copy()
        self.palette_t       = 0.0
        self.palette_speed   = self.DRIFT_MORPH_SPEED
        self._current_palette_id: int | None = None

        # --- Hihat axis: zoom pulse ---
        self.zoom_boost = 0.0  # additive zoom, decays back to 0

        # Kick counting for swap-every-N
        self._kick_count = 0
        self._recent_kick_energy = 0.5  # running average for downbeat detection

        # Loop playback state
        self._lib = lib
        self._loop_genomes: list[Genome] = []   # genomes in the active loop
        self._loop_pos: int = 0                  # current position in loop
        self.active_loop_id: int | None = None   # for vote tracking

        if lib is not None and lib.loop_count() > 0:
            self._load_top_loop()
        else:
            self._prefetch_genome()

        self.needs_walker_reset = False
        self._quiet_frames: int = 0
        self._last_beat_time: dict[str, float] = {'kick': 0.0, 'snare': 0.0, 'hihat': 0.0}

        # Tempo tracking for rhythm coherence gating
        self.tempo = TempoTracker()

        if test_audio:
            self.audio = SyntheticAudioProcessor(
                kick_interval=0.5,
                snare_interval=1.0,
                hihat_interval=0.25,
            )
        else:
            self.audio = AudioProcessor(device=audio_device)
            print(f'audio device: {audio_device!r}')
        self.audio.start()

    @dataclass
    class FrameState:
        """Per-frame output from tick() — three orthogonal axes + audio."""
        genome: 'Genome'          # kick axis: transforms, zoom, rotation, center (includes zoom pulse)
        palette: np.ndarray       # snare axis: 256x3 float32 color palette
        spectrum: np.ndarray      # raw FFT spectrum for audio-reactive tonemap
        brightness: float         # RMS-driven brightness for tonemap

    def tick(self, frame_time: float) -> 'FlameSheepCore.FrameState':
        """Advance audio/genome state one frame.
        Returns FrameState with three independent axes.
        """
        beat_events = self.audio.process()
        spectrum    = self.audio.spectrum
        self._handle_beats(beat_events)

        # --- Kick axis: genome morph ---
        self.morph_t = min(1.0, self.morph_t + self.morph_speed)
        display_genome = self.current_genome.lerp(self.target_genome, self.morph_t)

        if self.morph_t >= 1.0:
            self.current_genome = self.target_genome
            self.morph_t        = 0.0
            self._swap_next_genome()
            self.morph_speed    = self.DRIFT_MORPH_SPEED

        # --- Snare axis: palette morph ---
        self.palette_t = min(1.0, self.palette_t + self.palette_speed)
        display_palette = _lerp_arr(self.palette_current, self.palette_target, self.palette_t)

        if self.palette_t >= 1.0:
            self.palette_current = self.palette_target.copy()
            self.palette_t       = 0.0
            self.palette_speed   = self.DRIFT_MORPH_SPEED

        # --- Hihat axis: zoom pulse decay ---
        self.zoom_boost *= self.ZOOM_DECAY
        if self.zoom_boost < 0.001:
            self.zoom_boost = 0.0
        display_genome.zoom *= (1.0 + self.zoom_boost)

        # --- Drift (quiet) ---
        rms = self.audio.rms
        if rms < self.DRIFT_RMS_THRESHOLD:
            self._quiet_frames += 1
            if self._quiet_frames >= self.DRIFT_SWAP_FRAMES:
                self._quiet_frames = 0
                self._swap_next_genome()
                self.morph_t     = 0.0
                self.morph_speed = self.DRIFT_MORPH_SPEED
                dist = self.current_genome.distance(self.target_genome)
                print(f'[drift] rms={rms:.5f}  dist={dist:.3f}')
        else:
            self._quiet_frames = 0

        self.morph_speed = max(self.DRIFT_MORPH_SPEED, self.morph_speed * 0.98)
        self.palette_speed = max(self.DRIFT_MORPH_SPEED, self.palette_speed * 0.98)

        # Map RMS to brightness — quiet=3.0 (dim), loud=9.0 (vivid)
        brightness = 3.0 + min(rms / 0.05, 1.0) * 6.0

        return self.FrameState(
            genome=display_genome,
            palette=display_palette,
            spectrum=spectrum,
            brightness=brightness,
        )

    def force_genome_swap(self):
        """Immediately swap to a new genome — call when image looks degenerate."""
        self.current_genome = self.current_genome.lerp(self.target_genome, self.morph_t)
        self._swap_next_genome()
        self.morph_t     = 0.0
        self.morph_speed = 0.15
        self.needs_walker_reset = True
        print('[force swap]')

    def stop(self):
        self.audio.stop()

    def _load_top_loop(self):
        """Load the highest-fitness loop from the library."""
        top = self._lib.top_loops(n=1)
        if not top:
            self._loop_genomes = []
            self.active_loop_id = None
            self._prefetch_genome()
            return
        loop_id = top[0][0]
        self._start_loop(loop_id)

    def load_loop(self, loop_id: int):
        """Switch to a specific loop by ID."""
        self._start_loop(loop_id)

    def _start_loop(self, loop_id: int):
        """Load a loop and start at a random position."""
        items = self._lib.load_loop(loop_id)
        self._loop_genomes = [genome for _, genome, _ in items]
        n = len(self._loop_genomes)
        start = int(self.rng.integers(0, n))
        self._loop_pos = (start + 1) % n
        self.active_loop_id = loop_id
        self.current_genome = self._loop_genomes[start]
        self.target_genome = self._loop_genomes[self._loop_pos]
        self.morph_t = 0.0
        self.morph_speed = 0.05
        self.needs_walker_reset = True
        print(f'[loop] loaded #{loop_id} ({n} genomes, start={start})')

    def next_loop(self):
        """Switch to the next best loop from the library."""
        if self._lib is None or self._lib.loop_count() < 1:
            return
        top = self._lib.top_loops(n=10)
        # Pick one we're not already playing
        for lid, _ in top:
            if lid != self.active_loop_id:
                self.load_loop(lid)
                return
        # All the same — just reload current
        if top:
            self.load_loop(top[0][0])

    def _prefetch_genome(self):
        current_snapshot = self.current_genome
        def _gen():
            rng = np.random.default_rng()
            for _ in range(20):
                g = Genome.random(rng)
                if g.distance(current_snapshot) >= FlameSheepCore.MIN_GENOME_DISTANCE:
                    break
            with self._genome_lock:
                self._next_genome = g
        threading.Thread(target=_gen, daemon=True).start()

    def _swap_next_genome(self):
        if self._loop_genomes:
            # Advance through the loop
            self._loop_pos = (self._loop_pos + 1) % len(self._loop_genomes)
            self.target_genome = self._loop_genomes[self._loop_pos]
        else:
            # Fallback: random genomes
            with self._genome_lock:
                if self._next_genome is not None:
                    self.target_genome = self._next_genome
                    self._next_genome  = None
                else:
                    self.target_genome = Genome.random(self.rng)
            self._prefetch_genome()

    def song_started(self):
        """Signal new song started — resets tempo and enables trusted mode."""
        self.tempo.song_started()
        print('[tempo] song started (trusted mode)')
        
    def hint_tempo(self, bpm: float):
        """Provide tempo hint from external source."""
        self.tempo.hint_tempo(bpm)
        print(f'[tempo] hint: {bpm:.1f} BPM')

    def _handle_beats(self, events: list[BeatEvent]):
        now = time.perf_counter()
        for event in events:
            # Feed all onsets to tempo tracker, gate based on rhythm coherence
            if not self.tempo.process_onset(event.kind, now):
                # Onset rejected as non-rhythmic (e.g., speech)
                continue

            since = now - self._last_beat_time[event.kind]
            self._last_beat_time[event.kind] = now

            if event.kind == 'kick':
                # Track average kick energy for downbeat detection
                self._recent_kick_energy = (
                    self._recent_kick_energy * 0.8 + event.energy * 0.2)

                # If this kick is significantly stronger than average,
                # it's likely a downbeat — reset the counter
                if (event.energy > self._recent_kick_energy * 1.5
                        and self._kick_count > 1):
                    self._kick_count = 0

                self._kick_count += 1
                if self._kick_count >= self.KICK_SWAP_EVERY:
                    # Swap kick: advance to next genome in loop
                    self._kick_count = 0
                    self.current_genome = self.current_genome.lerp(
                        self.target_genome, self.morph_t)
                    self._swap_next_genome()
                    self.morph_t = 0.0
                    # Scale morph speed to fill the gap until next swap
                    # Aim to complete ~80% of the morph before the next swap kick
                    kick_period = since if since < 2.0 else 0.5
                    frames_until_swap = (kick_period * self.KICK_SWAP_EVERY * 60) * 0.8
                    self.morph_speed = max(0.01, 1.0 / frames_until_swap)
                    dist = self.current_genome.distance(self.target_genome)
                    print(f'[SWAP]  +{since:.3f}s  energy={event.energy:.2f}  '
                          f'dist={dist:.3f}  spd={self.morph_speed:.3f}')
                else:
                    # Non-swap kick: pulse the morph speed forward
                    self.morph_speed = min(0.15,
                        self.morph_speed + event.energy * self.KICK_MORPH_PULSE)
                    print(f'[kick]  +{since:.3f}s  energy={event.energy:.2f}  '
                          f'beat={self._kick_count}/{self.KICK_SWAP_EVERY}')

            elif event.kind == 'snare':
                # Snare axis: walk the palette graph
                # Energy controls jump distance — louder snares = bigger color shift
                self.palette_current = _lerp_arr(
                    self.palette_current, self.palette_target, self.palette_t)
                next_palette = self._pick_next_palette(event.energy)
                if next_palette is not None:
                    self.palette_target = next_palette
                self.palette_t     = 0.0
                self.palette_speed = 0.03 + event.energy * 0.1
                print(f'[snare] +{since:.3f}s  energy={event.energy:.2f}')

            elif event.kind == 'hihat':
                # Hihat axis: zoom pulse
                self.zoom_boost = min(
                    self.ZOOM_BOOST_MAX,
                    self.zoom_boost + event.energy * 0.15)
                print(f'[hihat] +{since:.3f}s  energy={event.energy:.2f}  '
                      f'zoom_boost={self.zoom_boost:.3f}')

    def _pick_next_palette(self, energy: float) -> np.ndarray | None:
        """
        Walk the palette graph. Energy controls jump distance:
          low energy  → small step (nearby palette, subtle shift)
          high energy → big step (distant palette, dramatic shift)

        Falls back to random palette if the library has no palettes.
        """
        if self._lib is None or self._lib.palette_count() < 2:
            from .genome import _random_palette
            return _random_palette(self.rng)

        # Map energy [0,1] to distance range
        # Low energy: stay close (0.02-0.15), high energy: jump far (0.15-0.6)
        min_dist = 0.02 + energy * 0.13
        max_dist = 0.15 + energy * 0.45

        if self._current_palette_id is not None:
            neighbors = self._lib.palette_neighbors(
                self._current_palette_id, min_dist=min_dist, max_dist=max_dist,
                min_fitness=0.5)
            if neighbors:
                # Pick randomly from neighbors, weighted toward closer ones
                ids = [n[0] for n in neighbors]
                dists = np.array([n[1] for n in neighbors])
                weights = 1.0 / (dists + 0.01)
                weights /= weights.sum()
                choice = int(self.rng.choice(ids, p=weights))
                self._current_palette_id = choice
                return self._lib.load_palette(choice)

        # No current position or no neighbors — pick a random palette
        all_ids = self._lib.all_palette_ids()
        if all_ids:
            choice = int(self.rng.choice(all_ids))
            self._current_palette_id = choice
            return self._lib.load_palette(choice)

        from .genome import _random_palette
        return _random_palette(self.rng)


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
        print('flame-sheep started. Press Q to quit, F to force genome swap.')

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
        self._renderer.dispatch_chaos_game()
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


def _shift_palette(palette: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """Rotate the palette hue slightly."""
    shifted = np.roll(palette, int(amount * 64), axis=0)
    return shifted


def _perturb_genome(genome: Genome, scale: float, rng: np.random.Generator):
    """Add small random noise to affine coefficients."""
    for tr in genome.transforms:
        tr.affine += rng.uniform(-scale, scale, tr.affine.shape).astype(np.float32)


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
        print(f'[wallpaper] swaymsg failed: {e}')
        return {}


def _run_wallpaper(audio_device, test_audio: bool, blur_radius: float = 1.0):
    """
    Wallpaper mode — one continuous flame fractal image across all monitors.

    Architecture:
      - Query sway for the logical layout of all outputs.
      - Compute a virtual canvas in physical (mm) coordinates for correct
        alignment across mixed-DPI displays.
      - Render the flame at a fixed pixels-per-mm density.
      - One WallpaperWindow per output, all sharing ONE EGL context.
      - Each frame: compute once, then for each window switch EGL surface
        and tonemap its viewport slice.
    """
    from .wayland_window import WallpaperSession

    # --- discover outputs and sway layout ---
    output_names = WallpaperSession.list_outputs()
    print(f'[wallpaper] outputs: {output_names}')

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
        print(f'[wallpaper] {name}: {g["w"]}x{g["h"]}px @ {g["ppi"]:.1f}ppi, '
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
    print(f'[wallpaper] physical canvas: {total_phys_w_mm:.0f}x{total_phys_h_mm:.0f}mm '
          f'-> render {canvas_w}x{canvas_h}px @ {canvas_ppmm:.2f} px/mm')

    # --- compute viewport for each output (in canvas pixels) ---
    viewports: dict[str, Viewport] = {}
    for name, g in active.items():
        vx = int(phys_x[name] * canvas_ppmm)
        vy = int(phys_y[name] * canvas_ppmm)
        vw = int(g['phys_w_mm'] * canvas_ppmm)
        vh = int(g['phys_h_mm'] * canvas_ppmm)
        viewports[name] = Viewport(vx, vy, vw, vh)
        print(f'[wallpaper] {name}: viewport {vw}x{vh}+{vx},{vy} (canvas px)')

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

    core = FlameSheepCore(audio_device=audio_device, test_audio=test_audio, lib=lib)

    _vote_count = 0
    _votes_per_evolve = 5       # first few cycles need more votes
    _evolve_count = 0
    _evolving = False

    def _maybe_evolve():
        """Trigger evolution if enough votes have accumulated."""
        nonlocal _vote_count, _votes_per_evolve, _evolve_count, _evolving
        _vote_count += 1
        if _evolving:
            print(f'[evolve] already running, vote queued ({_vote_count})')
            return
        if _vote_count < _votes_per_evolve:
            print(f'[evolve] {_vote_count}/{_votes_per_evolve} votes until next evolution')
            return
        if lib.loop_count() < 2:
            print('[evolve] not enough loops to evolve')
            return
        _vote_count = 0
        _evolve_count += 1
        if _evolve_count >= 3:
            _votes_per_evolve = 3
        _evolving = True
        print(f'[evolve] starting evolution cycle {_evolve_count} in subprocess...')
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
                    print(f'[evolve] {line}')
            _evolving = False
        threading.Thread(target=_wait_evolve, daemon=True).start()

    # --- control pipe for external commands ---
    control = ControlPipe()
    control.start()
    print(f'flame-sheep wallpaper running. Control pipe: {control.pipe_path}')

    def handle_control_events() -> bool:
        """Process control events. Returns True if quit requested."""
        for event in control.poll_all():
            print(f'[ctl] {event.command} {" ".join(event.args)}')

            if event.command == 'quit':
                return True
            elif event.command == 'swap':
                core.force_genome_swap()
            elif event.command == 'like':
                if core.active_loop_id is not None:
                    lib.rate('loop', core.active_loop_id, +1)
                    print(f'[ctl] liked loop #{core.active_loop_id} '
                          f'(net: {lib.net_rating("loop", core.active_loop_id):+d})')
                    _maybe_evolve()
                else:
                    print('[ctl] no active loop to rate')
            elif event.command == 'dislike':
                if core.active_loop_id is not None:
                    lib.rate('loop', core.active_loop_id, -1)
                    print(f'[ctl] disliked loop #{core.active_loop_id} '
                          f'(net: {lib.net_rating("loop", core.active_loop_id):+d})')
                    _maybe_evolve()
                core.next_loop()  # switch to a different loop
            elif event.command == 'song':
                core.song_started()
                core.force_genome_swap()  # fresh visual for new song
            elif event.command == 'tempo':
                if event.args:
                    try:
                        bpm = float(event.args[0])
                        core.hint_tempo(bpm)
                    except ValueError:
                        print(f'[ctl] invalid tempo: {event.args[0]}')
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
                print('[wallpaper] watchdog: render loop stalled, forcing exit')
                os._exit(1)

    _wd_thread = threading.Thread(target=_watchdog, daemon=True)
    _wd_thread.start()

    try:
        while not quit_requested and not all(s.should_close for s in surfaces.values()):
            _frame += 1
            _watchdog_last = time.perf_counter()
            quit_requested = handle_control_events()
            if not session.connection_alive():
                print('[wallpaper] wayland connection lost, exiting')
                break
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
            renderer.dispatch_chaos_game()
            ctx.memory_barrier()

            # Tonemap pass — switch surface per output, same GL context
            for name, surf in surfaces.items():
                if not session.make_current(surf):
                    continue  # this surface is dead, skip it
                renderer.render_tonemap(viewports[name], surf.width, surf.height,
                                       brightness=frame.brightness)
                if not session.swap(surf):
                    break  # wayland connection lost

    finally:
        control.stop()
        core.stop()
        session.destroy()


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

    # parse_known_args so moderngl-window's own flags don't cause errors here
    args, remaining = parser.parse_known_args()

    if args.help:
        parser.print_help()
        return

    global _AUDIO_DEVICE, _TEST_AUDIO
    _AUDIO_DEVICE = args.audio_device
    _TEST_AUDIO   = args.test_audio

    if args.list_audio:
        from .audio import list_monitor_devices
        for d in list_monitor_devices():
            print(f"  [{d['index']:2d}] {d['name']}")
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
