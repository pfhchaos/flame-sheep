"""
ModernGL renderer — manages GPU resources and shader execution.

Render pipeline each frame:
  1. Upload genome uniforms (affines, variations, palette, global params)
  2. Upload audio texture (FFT spectrum)
  3. Dispatch compute shader (chaos game — accumulates histogram)
  4. Full-screen quad pass (log-density tonemap + color)
  5. Clear histogram buffer for next frame (or accumulate, TBD)

The histogram is a 2-channel float32 texture:
  channel 0: hit count (accumulated)
  channel 1: color accumulator (weighted average of transform colors)
"""

import moderngl
import numpy as np
from pathlib import Path

from .genome import Genome, MAX_TRANSFORMS, NUM_VARIATIONS

SHADER_DIR = Path(__file__).parent / 'shaders'

# Number of parallel chaos game walkers in compute shader
N_WALKERS = 1024 * 64   # 65536 — adjust for perf


class FlameRenderer:
    """Owns all ModernGL state. Call update() + render() each frame."""

    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx    = ctx
        self.width  = width
        self.height = height

        self._load_shaders()
        self._create_resources()

    def _load_shaders(self):
        self.compute_shader = self.ctx.compute_shader(
            (SHADER_DIR / 'flame.comp').read_text()
        )
        self.tonemap_program = self.ctx.program(
            vertex_shader   = (SHADER_DIR / 'tonemap.vert').read_text(),
            fragment_shader = (SHADER_DIR / 'tonemap.frag').read_text(),
        )

    def _create_resources(self):
        w, h = self.width, self.height

        # Histogram texture: RG32F — R = hit count, G = color accumulator
        self.histogram = self.ctx.texture((w, h), components=2, dtype='f4')
        self.histogram.bind_to_image(0, read=True, write=True)

        # Palette texture: 256 x 1 RGB32F
        self.palette_tex = self.ctx.texture((256, 1), components=3, dtype='f4')
        self.palette_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Audio spectrum texture: N_BINS x 1 R32F
        from .audio import N_BINS
        self.audio_tex = self.ctx.texture((N_BINS, 1), components=1, dtype='f4')
        self.audio_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Walker state buffer: each walker needs (x, y, color) — 3 floats
        walker_data = np.random.uniform(-1, 1, (N_WALKERS, 3)).astype(np.float32)
        self.walker_buf = self.ctx.buffer(walker_data.tobytes())
        self.walker_buf.bind_to_storage_buffer(1)

        # Fullscreen quad
        quad_verts = np.array([
            -1, -1,   1, -1,   -1,  1,
             1, -1,   1,  1,   -1,  1,
        ], dtype=np.float32)
        self.quad_vbo = self.ctx.buffer(quad_verts.tobytes())
        self.quad_vao = self.ctx.vertex_array(
            self.tonemap_program,
            [(self.quad_vbo, '2f', 'in_pos')],
        )

    def upload_genome(self, genome: Genome):
        """Pack genome into uniform arrays and upload to both shaders."""
        affines, variations, colors, weights = genome.to_gpu_arrays()

        cs = self.compute_shader
        # Flat uploads — GLSL std140 packing means we upload as flat float arrays
        cs['u_affines'].write(affines.tobytes())
        cs['u_variations'].write(variations.tobytes())
        cs['u_colors'].write(colors.tobytes())
        cs['u_weights'].write(weights.tobytes())
        cs['u_n_transforms'] = len(genome.transforms)
        cs['u_zoom']         = genome.zoom
        cs['u_rotation']     = genome.rotation
        cs['u_center']       = tuple(genome.center)
        cs['u_width']        = self.width
        cs['u_height']       = self.height

        # Palette to texture
        self.palette_tex.write(genome.palette.tobytes())

    def upload_audio(self, spectrum: np.ndarray):
        """Upload FFT spectrum as 1D texture."""
        self.audio_tex.write(spectrum.astype(np.float32).tobytes())

    def clear_histogram(self):
        """Zero the histogram texture between frames."""
        zeros = np.zeros((self.width * self.height * 2,), dtype=np.float32)
        self.histogram.write(zeros.tobytes())

    def dispatch_chaos_game(self, n_iterations: int = 100):
        """
        Dispatch compute shader.
        Each invocation runs n_iterations of the chaos game for one walker.
        """
        self.compute_shader['u_iterations'] = n_iterations
        # Dispatch enough groups to cover all walkers
        # local_size_x = 64 in shader
        groups = N_WALKERS // 64
        self.compute_shader.run(group_x=groups)

    def render_tonemap(self):
        """Full-screen tonemap pass — reads histogram, writes to screen."""
        self.histogram.use(location=0)
        self.palette_tex.use(location=1)
        self.audio_tex.use(location=2)

        self.tonemap_program['u_histogram'] = 0
        self.tonemap_program['u_palette']   = 1
        self.tonemap_program['u_audio']     = 2
        self.tonemap_program['u_width']     = self.width
        self.tonemap_program['u_height']    = self.height

        self.quad_vao.render(moderngl.TRIANGLES)

    def resize(self, width: int, height: int):
        """Handle window resize — recreate resolution-dependent resources."""
        self.width  = width
        self.height = height
        # Recreate histogram and walker buffer at new size
        self._create_resources()
