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
import threading
import numpy as np
import moderngl_window as mglw
from moderngl_window import settings

from .genome import Genome
from .audio import AudioProcessor, SyntheticAudioProcessor, BeatEvent, DEFAULT_DEVICE
from .renderer import FlameRenderer


# Set by main() before run_window_config — workaround for moderngl-window
# not passing CLI args through to WindowConfig.__init__
_AUDIO_DEVICE: str | int = DEFAULT_DEVICE
_TEST_AUDIO:   bool = False


class FlameSheepCore:
    """
    All audio/genome/rendering logic, independent of window system.
    Requires a moderngl.Context and (width, height) at construction.
    Call .render_frame(frame_time) each frame.
    Call .force_genome_swap() to kick a degenerate image.
    Call .stop() on shutdown.
    """

    # Minimum genome distance before accepting a prefetched candidate.
    MIN_GENOME_DISTANCE = 0.15

    # Autonomous drift mode — active when RMS is below threshold
    DRIFT_RMS_THRESHOLD = 0.002
    DRIFT_SWAP_FRAMES   = 60 * 8
    DRIFT_MORPH_SPEED   = 0.003

    def __init__(self, ctx: 'moderngl.Context', width: int, height: int,
                 audio_device=DEFAULT_DEVICE, test_audio: bool = False):
        self.rng      = np.random.default_rng()
        self.renderer = FlameRenderer(ctx, width, height)

        self.current_genome = Genome.random(self.rng)
        self.target_genome  = Genome.random(self.rng)
        self.morph_t        = 0.0
        self.morph_speed    = self.DRIFT_MORPH_SPEED

        self._next_genome: Genome | None = None
        self._genome_lock = threading.Lock()
        self._prefetch_genome()

        self._quiet_frames: int = 0
        self._last_beat_time: dict[str, float] = {'kick': 0.0, 'snare': 0.0, 'hihat': 0.0}

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

        self.renderer.upload_genome(self.current_genome)

    def render_frame(self, frame_time: float):
        """Run one frame. frame_time is seconds since last frame."""
        beat_events = self.audio.process()
        spectrum    = self.audio.spectrum
        self.renderer.upload_audio(spectrum)
        self._handle_beats(beat_events)

        self.morph_t = min(1.0, self.morph_t + self.morph_speed)
        display_genome = self.current_genome.lerp(self.target_genome, self.morph_t)

        if self.morph_t >= 1.0:
            self.current_genome = self.target_genome
            self.morph_t        = 0.0
            self._swap_next_genome()
            self.morph_speed    = self.DRIFT_MORPH_SPEED

        self.renderer.upload_genome(display_genome)
        self.renderer.clear_histogram()
        self.renderer.dispatch_chaos_game(n_iterations=1000)
        self.renderer.ctx.memory_barrier()
        self.renderer.render_tonemap()

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

    def resize(self, width: int, height: int):
        self.renderer.resize(width, height)

    def force_genome_swap(self):
        """Immediately swap to a new genome — call when image looks degenerate."""
        self.current_genome = self.current_genome.lerp(self.target_genome, self.morph_t)
        self._swap_next_genome()
        self.morph_t     = 0.0
        self.morph_speed = 0.15
        print('[force swap]')

    def stop(self):
        self.audio.stop()

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
        with self._genome_lock:
            if self._next_genome is not None:
                self.target_genome = self._next_genome
                self._next_genome  = None
            else:
                self.target_genome = Genome.random(self.rng)
        self._prefetch_genome()

    def _handle_beats(self, events: list[BeatEvent]):
        now = time.perf_counter()
        for event in events:
            since = now - self._last_beat_time[event.kind]
            self._last_beat_time[event.kind] = now

            if event.kind == 'kick':
                self.current_genome = self.current_genome.lerp(
                    self.target_genome, self.morph_t)
                self._swap_next_genome()
                self.morph_t     = 0.0
                self.morph_speed = 0.05 + event.energy * 0.15
                dist = self.current_genome.distance(self.target_genome)
                print(f'[kick]  +{since:.3f}s  energy={event.energy:.2f}  '
                      f'dist={dist:.3f}  spd={self.morph_speed:.3f}')

            elif event.kind == 'snare':
                self.target_genome.palette = _shift_palette(
                    self.target_genome.palette, event.energy, self.rng)
                print(f'[snare] +{since:.3f}s  energy={event.energy:.2f}')

            elif event.kind == 'hihat':
                _perturb_genome(self.target_genome, event.energy * 0.05, self.rng)
                print(f'[hihat] +{since:.3f}s  energy={event.energy:.2f}')


class FlameSheepApp(mglw.WindowConfig):
    """moderngl-window wrapper — used for windowed/dev mode."""
    title       = 'flame-sheep'
    gl_version  = (4, 3)
    resizable   = True
    vsync       = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._core = FlameSheepCore(
            self.ctx, *self.window_size,
            audio_device=_AUDIO_DEVICE,
            test_audio=_TEST_AUDIO,
        )
        print('flame-sheep started. Press Q to quit, F to force genome swap.')

    def on_render(self, time_val: float, frame_time: float):
        self.ctx.clear(0.0, 0.0, 0.0)
        self._core.render_frame(frame_time)

    def resize(self, width: int, height: int):
        self._core.resize(width, height)

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


def _run_wallpaper(width: int, height: int, audio_device, test_audio: bool):
    """Wallpaper mode: wlr-layer-shell BACKGROUND surface."""
    from .wayland_window import WallpaperWindow

    win  = WallpaperWindow(width, height)
    core = FlameSheepCore(
        win.ctx, win.width, win.height,
        audio_device=audio_device,
        test_audio=test_audio,
    )
    print('flame-sheep wallpaper running. Ctrl-C or SIGTERM to quit.')
    print('  swaymsg \'[app_id=flame-sheep] focus\' then press F to force genome swap')

    last_time = time.perf_counter()
    try:
        while not win.should_close:
            now        = time.perf_counter()
            frame_time = now - last_time
            last_time  = now

            win.ctx.clear(0.0, 0.0, 0.0)
            core.render_frame(frame_time)
            win.swap()
    finally:
        core.stop()
        win.destroy()


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

    if args.wallpaper:
        _run_wallpaper(args.width, args.height, args.audio_device, args.test_audio)
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
