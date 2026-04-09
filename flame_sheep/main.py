"""
flame-sheep — entry point and main loop.

Uses moderngl-window for window management (handles Wayland/X11 transparently).
Eventually: replace window with wlr-layer-shell for true wallpaper mode.
"""

import sys
import time
import argparse
import numpy as np
import moderngl_window as mglw
from moderngl_window import settings

from .genome import Genome
from .audio import AudioProcessor, BeatEvent
from .renderer import FlameRenderer


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
        self.morph_speed    = 0.02   # per frame, slow baseline drift

        # Audio
        self.audio = AudioProcessor()  # TODO: allow device selection via CLI
        self.audio.start()

        # Upload initial state
        self.renderer.upload_genome(self.current_genome)

        print('flame-sheep started. Press Q to quit.')

    def render(self, time_val: float, frame_time: float):
        self.ctx.clear(0.0, 0.0, 0.0)

        # --- Audio ---
        beat_events = self.audio.process()
        spectrum    = self.audio.spectrum
        self.renderer.upload_audio(spectrum)
        self._handle_beats(beat_events)

        # --- Morph genome ---
        self.morph_t = min(1.0, self.morph_t + self.morph_speed)
        display_genome = self.current_genome.lerp(self.target_genome, self.morph_t)

        # When morph completes, current becomes target, pick new target
        if self.morph_t >= 1.0:
            self.current_genome = self.target_genome
            self.target_genome  = Genome.random(self.rng)
            self.morph_t        = 0.0

        self.renderer.upload_genome(display_genome)

        # --- GPU render ---
        self.renderer.clear_histogram()
        self.renderer.dispatch_chaos_game(n_iterations=200)
        self.ctx.memory_barrier()   # ensure compute writes are visible
        self.renderer.render_tonemap()

        # Decay morph speed back toward baseline after a beat spike
        self.morph_speed = max(0.005, self.morph_speed * 0.98)

    def _handle_beats(self, events: list[BeatEvent]):
        for event in events:
            if event.kind == 'kick':
                # Major mutation: jump toward a new genome, speed up morph
                self.target_genome = Genome.random(self.rng)
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
    args = parser.parse_args()

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
