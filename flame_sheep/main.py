"""
flame-sheep — entry point and main loop.

Uses moderngl-window for window management (handles Wayland/X11 transparently).
Eventually: replace window with wlr-layer-shell for true wallpaper mode.
"""

import argparse
import threading
import numpy as np
import moderngl_window as mglw
from moderngl_window import settings

from .genome import Genome
from .audio import AudioProcessor, BeatEvent
from .renderer import FlameRenderer


# Set by main() before run_window_config — workaround for moderngl-window
# not passing CLI args through to WindowConfig.__init__
_AUDIO_DEVICE: int = 6


class FlameSheepApp(mglw.WindowConfig):
    title       = 'flame-sheep'
    gl_version  = (4, 3)   # compute shaders need 4.3+
    resizable   = True
    vsync       = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.rng      = np.random.default_rng()
        self.renderer = FlameRenderer(self.ctx, *self.window_size)

        # Genome state
        self.current_genome = Genome.random(self.rng)
        self.target_genome  = Genome.random(self.rng)
        self.morph_t        = 0.0    # 0 = current, 1 = target
        self.morph_speed    = 0.003  # per frame, slow baseline drift

        # Pre-generate next genome in background so beat response is instant
        self._next_genome: Genome | None = None
        self._genome_lock = threading.Lock()
        self._prefetch_genome()

        # Audio
        self.audio = AudioProcessor(device=_AUDIO_DEVICE)
        self.audio.start()

        # Upload initial state
        self.renderer.upload_genome(self.current_genome)

        print('flame-sheep started. Press Q to quit.')

    def on_render(self, time_val: float, frame_time: float):
        self.ctx.clear(0.0, 0.0, 0.0)

        # --- Audio ---
        beat_events = self.audio.process()
        spectrum    = self.audio.spectrum
        self.renderer.upload_audio(spectrum)
        self._handle_beats(beat_events)

        # --- Morph genome ---
        self.morph_t = min(1.0, self.morph_t + self.morph_speed)
        display_genome = self.current_genome.lerp(self.target_genome, self.morph_t)

        # When morph completes, current becomes target, fetch next
        if self.morph_t >= 1.0:
            self.current_genome = self.target_genome
            self.morph_t        = 0.0
            self._swap_next_genome()

        self.renderer.upload_genome(display_genome)

        # --- GPU render ---
        self.renderer.clear_histogram()
        self.renderer.dispatch_chaos_game(n_iterations=1000)
        self.ctx.memory_barrier()   # ensure compute writes are visible
        self.renderer.render_tonemap()

        # Decay morph speed back toward baseline after a beat spike
        self.morph_speed = max(0.005, self.morph_speed * 0.98)

    def _prefetch_genome(self):
        """Generate next genome in background thread with its own RNG."""
        def _gen():
            # Use a fresh RNG per thread — numpy generators aren't thread-safe
            rng = np.random.default_rng()
            g = Genome.random(rng)
            with self._genome_lock:
                self._next_genome = g
        threading.Thread(target=_gen, daemon=True).start()

    def _swap_next_genome(self):
        """Use prefetched genome as new target, start prefetching another."""
        with self._genome_lock:
            if self._next_genome is not None:
                self.target_genome = self._next_genome
                self._next_genome  = None
            else:
                # Prefetch wasn't ready — generate synchronously as fallback
                self.target_genome = Genome.random(self.rng)
        self._prefetch_genome()

    def _handle_beats(self, events: list[BeatEvent]):
        for event in events:
            if event.kind == 'kick':
                # Major mutation: swap in prefetched genome instantly, speed up morph
                self._swap_next_genome()
                self.morph_t       = 0.0
                self.morph_speed   = 0.05 + event.energy * 0.15
                print(f'[kick]  energy={event.energy:.2f}  new target genome')

            elif event.kind == 'snare':
                # Shift palette of target genome
                self.target_genome.palette = _shift_palette(
                    self.target_genome.palette, event.energy, self.rng
                )
                print(f'[snare] energy={event.energy:.2f}  palette shift')

            elif event.kind == 'hihat':
                # Small perturbation to affine coefficients
                _perturb_genome(self.target_genome, event.energy * 0.05, self.rng)

    def resize(self, width: int, height: int):
        self.renderer.resize(width, height)

    def key_event(self, key, action, modifiers):
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.Q:
                self.audio.stop()
                self.wnd.close()

    def close(self):
        self.audio.stop()


def _shift_palette(palette: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """Rotate the palette hue slightly."""
    shifted = np.roll(palette, int(amount * 64), axis=0)
    return shifted


def _perturb_genome(genome: Genome, scale: float, rng: np.random.Generator):
    """Add small random noise to affine coefficients."""
    for tr in genome.transforms:
        tr.affine += rng.uniform(-scale, scale, tr.affine.shape).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='flame-sheep: audio-reactive flame fractal wallpaper')
    parser.add_argument('--fullscreen', action='store_true', help='run fullscreen')
    parser.add_argument('--width',  type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--list-audio', action='store_true', help='list audio devices and exit')
    parser.add_argument('--audio-device', type=int, default=6, help='audio input device index (default: 6 pipewire)')
    args = parser.parse_args()

    global _AUDIO_DEVICE
    _AUDIO_DEVICE = args.audio_device

    if args.list_audio:
        from .audio import list_monitor_devices
        for d in list_monitor_devices():
            print(f"  [{d['index']:2d}] {d['name']}")
        return

    settings.WINDOW['class']      = 'moderngl_window.context.glfw.Window'
    settings.WINDOW['size']       = (args.width, args.height)
    settings.WINDOW['fullscreen'] = args.fullscreen
    settings.WINDOW['title']      = 'flame-sheep'

    mglw.run_window_config(FlameSheepApp)


if __name__ == '__main__':
    main()
