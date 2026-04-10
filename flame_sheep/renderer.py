"""
ModernGL renderer — manages GPU resources and shader execution.

Render pipeline each frame:
  1. Upload genome uniforms (affines, variations, palette, global params)
  2. Upload audio texture (FFT spectrum)
  3. Dispatch compute shader (chaos game — accumulates histogram SSBO)
  4. Memory barrier — ensure compute writes visible to fragment shader
  5. Full-screen quad pass (log-density tonemap + color)
  6. Clear histogram SSBO for next frame

Histogram layout (SSBO, binding=0):
  uint hit_count[width * height]   — how many times each pixel was hit
  uint color_acc[width * height]   — color accumulator (scaled to uint)
  Both arrays packed sequentially in one buffer.

Walker state (SSBO, binding=1):
  float[N_WALKERS * 3]  — [x, y, color] per walker, persists between frames
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
        self.clear_shader = self.ctx.compute_shader(
            (SHADER_DIR / 'clear.comp').read_text()
        )
        self.tonemap_program = self.ctx.program(
            vertex_shader   = (SHADER_DIR / 'tonemap.vert').read_text(),
            fragment_shader = (SHADER_DIR / 'tonemap.frag').read_text(),
        )

    def _create_resources(self):
        w, h = self.width, self.height
        n_pixels = w * h

        # Histogram SSBO (binding=0): two packed uint arrays
        #   [0          .. n_pixels-1] = hit_count
        #   [n_pixels   .. 2*n_pixels-1] = color_acc
        # Initialized to zeros
        histogram_data = np.zeros(n_pixels * 2, dtype=np.uint32)
        self.histogram_buf = self.ctx.buffer(histogram_data.tobytes())
        self.histogram_buf.bind_to_storage_buffer(0)

        # Palette texture: 256 x 1 RGB32F
        self.palette_tex = self.ctx.texture((256, 1), components=3, dtype='f4')
        self.palette_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Audio spectrum texture: N_BINS x 1 R32F
        from .audio import N_BINS
        self.audio_tex = self.ctx.texture((N_BINS, 1), components=1, dtype='f4')
        self.audio_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Genome array SSBOs (binding=2,3,4,5): affines, variations, colors, weights
        # Allocated at max size, written each frame with current genome
        affines_data    = np.zeros((MAX_TRANSFORMS, 6),              dtype=np.float32)
        variations_data = np.zeros((MAX_TRANSFORMS, NUM_VARIATIONS), dtype=np.float32)
        colors_data     = np.zeros(MAX_TRANSFORMS,                   dtype=np.float32)
        weights_data    = np.zeros(MAX_TRANSFORMS,                   dtype=np.float32)
        self.affines_buf    = self.ctx.buffer(affines_data.tobytes())
        self.variations_buf = self.ctx.buffer(variations_data.tobytes())
        self.colors_buf     = self.ctx.buffer(colors_data.tobytes())
        self.weights_buf    = self.ctx.buffer(weights_data.tobytes())
        self.affines_buf.bind_to_storage_buffer(2)
        self.variations_buf.bind_to_storage_buffer(3)
        self.colors_buf.bind_to_storage_buffer(4)
        self.weights_buf.bind_to_storage_buffer(5)

        # Walker state SSBO (binding=1): [x, y, color] per walker
        # Randomize initial positions so walkers spread across attractor quickly
        walker_data = np.random.uniform(-1, 1, (N_WALKERS, 3)).astype(np.float32)
        self.walker_buf = self.ctx.buffer(walker_data.tobytes())
        self.walker_buf.bind_to_storage_buffer(1)

        # Fullscreen quad — two triangles covering clip space
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
        """Pack genome into arrays and upload to GPU.

        Large arrays (affines, variations) go via SSBOs (binding 2,3) to avoid
        moderngl uniform array limitations. Scalar uniforms upload directly.
        """
        affines, variations, colors, weights = genome.to_gpu_arrays()

        # All arrays via SSBOs — moderngl doesn't support uniform array element access
        self.affines_buf.write(affines.tobytes())
        self.variations_buf.write(variations.tobytes())
        self.colors_buf.write(colors.tobytes())
        self.weights_buf.write(weights.tobytes())

        cs = self.compute_shader
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
        """Zero the histogram SSBO on the GPU — avoids slow CPU→GPU transfer."""
        size = self.width * self.height * 2
        self.clear_shader['u_size'] = size
        # ceil(size / 64) workgroups
        groups = (size + 63) // 64
        self.clear_shader.run(group_x=groups)
        self.ctx.memory_barrier()

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
        """Full-screen tonemap pass — reads histogram SSBO, writes to screen."""
        # Histogram is already bound as SSBO at binding=0 — fragment shader
        # reads it directly via the buffer binding, no texture needed.
        # Palette and audio are still textures.
        self.palette_tex.use(location=0)
        self.audio_tex.use(location=1)

        self.tonemap_program['u_palette'] = 0
        self.tonemap_program['u_audio']   = 1
        self.tonemap_program['u_width']   = self.width
        self.tonemap_program['u_height']  = self.height

        self.quad_vao.render(moderngl.TRIANGLES)

    def resize(self, width: int, height: int):
        """Handle window resize — recreate resolution-dependent resources."""
        self.width  = width
        self.height = height
        # Recreate histogram and walker buffer at new size
        self._create_resources()
