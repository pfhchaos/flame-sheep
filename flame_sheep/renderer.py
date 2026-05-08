"""
ModernGL renderer — manages GPU resources and shader execution.

Virtual canvas architecture (wallpaper mode):
  One histogram covers the full multi-monitor virtual canvas (at half-res).
  Each window/output samples its own rectangular slice via viewport uniforms.
  This produces a single continuous image across all monitors.

Render pipeline each frame:
  1. Upload genome uniforms (affines, variations, palette, global params)
  2. Upload audio texture (FFT spectrum)
  3. clear_histogram()      — zero the SSBO on GPU
  4. dispatch_chaos_game()  — accumulate walkers into full canvas histogram
  5. ctx.memory_barrier()
  6. For each window:
       render_tonemap(viewport)  — tonemap this window's slice to its FBO/screen

Histogram layout (SSBO, binding=0):
  uint hit_count[canvas_w * canvas_h]
  uint color_acc[canvas_w * canvas_h]
  Both packed sequentially.

Walker state (SSBO, binding=1):
  float[N_WALKERS * 3]  — [x, y, color] per walker, persists between frames
"""
from __future__ import annotations

import ctypes
import moderngl
import numpy as np
from pathlib import Path

# glBindFramebuffer(GL_FRAMEBUFFER, 0) to return to the EGL surface
try:
    from OpenGL import GL as _GL
    def _bind_default_framebuffer() -> None:
        _GL.glBindFramebuffer(_GL.GL_FRAMEBUFFER, 0)
except ImportError:
    # Fallback: call glBindFramebuffer via ctypes
    _libGL = ctypes.CDLL('libGL.so.1')
    def _bind_default_framebuffer() -> None:
        _libGL.glBindFramebuffer(0x8D40, 0)  # GL_FRAMEBUFFER = 0x8D40

from .genome import Genome, MAX_TRANSFORMS, NUM_VARIATIONS, MAX_ACTIVE_VARS, SLOT_SIZE

SHADER_DIR = Path(__file__).parent / 'shaders'


def _resolve_includes(source: str, shader_dir: Path) -> str:
    """Resolve #include "file.glsl" directives by inlining file contents."""
    import re
    def _replace(m: re.Match[str]) -> str:
        path = shader_dir / m.group(1)
        return path.read_text()
    return re.sub(r'#include\s+"(.+?)"', _replace, source)

# ---------------------------------------------------------------------------
# Walker / iteration tuning
#
# We render the full virtual canvas at half resolution for performance.
# Walker count and iterations are chosen to give reasonable density at
# the half-res canvas size without hammering the GPU.
# ---------------------------------------------------------------------------
N_WALKERS  = 1024 * 64   # 65536 walkers — more parallelism for modern GPUs
N_ITERS    = 150          # iterations per walker per frame (same total points)


class Viewport:
    """Describes one window's rectangle within the virtual canvas (render coords)."""
    __slots__ = ('x', 'y', 'w', 'h')

    def __init__(self, x: int, y: int, w: int, h: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __repr__(self) -> str:
        return f'Viewport(x={self.x}, y={self.y}, w={self.w}, h={self.h})'


class FlameRenderer:
    """
    Owns all GL resources for the full virtual canvas.

    canvas_w, canvas_h  — full virtual canvas size in render pixels (may be
                          half the physical size for performance).
    """

    def __init__(self, ctx: moderngl.Context, canvas_w: int, canvas_h: int,
                 scoring: bool = False):
        self.ctx      = ctx
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h

        self._load_shaders(scoring=scoring)
        self._create_resources()

    def _load_shaders(self, scoring: bool = False) -> None:
        from .variations._symmetry_groups import generate_glsl
        flame_src = (SHADER_DIR / 'flame.comp').read_text()
        flame_src = _resolve_includes(flame_src, SHADER_DIR)
        flame_src = flame_src.replace('// {{SYMMETRY_GROUPS}}', generate_glsl())
        if scoring:
            flame_src = '#define SCORING_MODE\n' + flame_src
        self.compute_shader = self.ctx.compute_shader(flame_src)
        self.clear_shader = self.ctx.compute_shader(
            (SHADER_DIR / 'clear.comp').read_text()
        )
        self.tonemap_program = self.ctx.program(
            vertex_shader   = (SHADER_DIR / 'tonemap.vert').read_text(),
            fragment_shader = (SHADER_DIR / 'tonemap.frag').read_text(),
        )
        self.blur_program = self.ctx.program(
            vertex_shader   = (SHADER_DIR / 'tonemap.vert').read_text(),
            fragment_shader = (SHADER_DIR / 'blur.frag').read_text(),
        )
        self.downsample_hist_shader = self.ctx.compute_shader(
            (SHADER_DIR / 'downsample_hist.comp').read_text()
        )

    def _create_resources(self) -> None:
        w, h = self.canvas_w, self.canvas_h
        n_pixels = w * h

        # Histogram SSBO (binding=0): two packed uint arrays
        #   [0        .. n_pixels-1] = hit_count
        #   [n_pixels .. 2*n_pixels-1] = color_acc
        histogram_data = np.zeros(n_pixels * 2, dtype=np.uint32)
        self.histogram_buf = self.ctx.buffer(histogram_data.tobytes())
        self.histogram_buf.bind_to_storage_buffer(0)

        # Per-transform hit counts SSBO (binding=7)
        xform_hits_data = np.zeros(n_pixels * MAX_TRANSFORMS, dtype=np.uint32)
        self.transform_hits_buf = self.ctx.buffer(xform_hits_data.tobytes())
        self.transform_hits_buf.bind_to_storage_buffer(7)

        # Downsampled histogram for symmetry scoring (binding=6)
        self._ds_w = min(256, w)
        self._ds_h = min(256, h)
        ds_data = np.zeros(self._ds_w * self._ds_h, dtype=np.uint32)
        self._ds_buf = self.ctx.buffer(ds_data.tobytes())
        self._ds_buf.bind_to_storage_buffer(6)

        # Palette texture: 256 x 1 RGB32F
        self.palette_tex = self.ctx.texture((256, 1), components=3, dtype='f4')
        self.palette_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Audio spectrum texture: N_BINS x 1 R32F
        from flame_sheep_audio import N_BINS
        self.audio_tex = self.ctx.texture((N_BINS, 1), components=1, dtype='f4')
        self.audio_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Genome SSBOs (binding=2,3,4,5)
        affines_data     = np.zeros((MAX_TRANSFORMS, 6), dtype=np.float32)
        # Packed active variations: (index, weight, p0..p5) per slot
        active_vars_data = np.zeros((MAX_TRANSFORMS, MAX_ACTIVE_VARS * SLOT_SIZE),
                                     dtype=np.float32)
        colors_data      = np.zeros(MAX_TRANSFORMS, dtype=np.float32)
        weights_data     = np.zeros(MAX_TRANSFORMS, dtype=np.float32)
        self.affines_buf     = self.ctx.buffer(affines_data.tobytes())
        self.active_vars_buf = self.ctx.buffer(active_vars_data.tobytes())
        self.colors_buf      = self.ctx.buffer(colors_data.tobytes())
        self.weights_buf     = self.ctx.buffer(weights_data.tobytes())
        self.affines_buf.bind_to_storage_buffer(2)
        self.active_vars_buf.bind_to_storage_buffer(3)
        self.colors_buf.bind_to_storage_buffer(4)
        self.weights_buf.bind_to_storage_buffer(5)

        # Walker state SSBO (binding=1)
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
        self.blur_vao = self.ctx.vertex_array(
            self.blur_program,
            [(self.quad_vbo, '2f', 'in_pos')],
        )

        # Blur FBOs — created lazily per surface size
        self._blur_fbos = {}
        self.blur_radius = 1.0  # adjustable blur strength

        # Temporal decay tracking for tonemap normalization
        self._decay = 0.0  # set by clear_histogram()

        # Debug overlay (lazy init)
        self._debug_circle_prog = None
        self._debug_circle_vao = None

    def reset_walkers(self) -> None:
        """Re-randomize walker positions. Call after a genome swap to avoid
        stuck walkers that escaped to infinity under a degenerate genome."""
        walker_data = np.random.uniform(-1, 1, (N_WALKERS, 3)).astype(np.float32)
        self.walker_buf.write(walker_data.tobytes())

    # ------------------------------------------------------------------
    # Per-frame API
    # ------------------------------------------------------------------

    def upload_genome(self, genome: Genome) -> None:
        affines, active_vars, colors, weights = genome.to_gpu_arrays()
        self.affines_buf.write(affines.tobytes())
        self.active_vars_buf.write(active_vars.tobytes())
        self.colors_buf.write(colors.tobytes())
        self.weights_buf.write(weights.tobytes())

        cs = self.compute_shader
        import math
        cs['u_n_transforms'] = len(genome.transforms)
        cs['u_zoom']         = genome.zoom
        cs['u_cos_rot']      = math.cos(genome.rotation)
        cs['u_sin_rot']      = math.sin(genome.rotation)
        cs['u_center']       = tuple(genome.center)
        cs['u_width']        = self.canvas_w
        cs['u_height']       = self.canvas_h

        self.palette_tex.write(genome.palette.tobytes())

    def upload_palette(self, palette: np.ndarray) -> None:
        """Upload a palette independently of the genome.
        palette: shape (256, 3) float32."""
        self.palette_tex.write(palette.tobytes())

    def upload_audio(self, spectrum: np.ndarray) -> None:
        # Resize spectrum to match GPU texture width
        # Octave bank produces 108 bins; interpolate to fill the texture
        tex_width = self.audio_tex.width
        if len(spectrum) != tex_width:
            x_old = np.linspace(0, 1, len(spectrum))
            x_new = np.linspace(0, 1, tex_width)
            spectrum = np.interp(x_new, x_old, spectrum).astype(np.float32)
        self.audio_tex.write(spectrum.astype(np.float32).tobytes())

    def clear_histogram(self, decay: float = 0.0) -> None:
        """Clear or decay the histogram.

        decay: 0.0 = full clear, 0.9 = keep 90% of previous frame's hits.
        """
        self._decay = decay
        size = self.canvas_w * self.canvas_h * 2
        self.clear_shader['u_size'] = size
        self.clear_shader['u_decay_num'] = int(decay * 256)
        groups = (size + 63) // 64
        self.clear_shader.run(group_x=groups)
        self.ctx.memory_barrier()

    def clear_transform_hits(self) -> None:
        """Zero the per-transform hit counts buffer."""
        n = self.canvas_w * self.canvas_h * MAX_TRANSFORMS
        self.transform_hits_buf.write(np.zeros(n, dtype=np.uint32).tobytes())
        self.ctx.memory_barrier()

    def dispatch_chaos_game(self, iterations: int = N_ITERS) -> None:
        self.compute_shader['u_iterations'] = iterations
        groups = N_WALKERS // 64
        self.compute_shader.run(group_x=groups)

    def _get_blur_fbos(self, w: int, h: int) -> tuple[moderngl.Framebuffer, moderngl.Texture, moderngl.Framebuffer, moderngl.Texture]:
        """Get or create a pair of FBOs for two-pass blur at the given size."""
        key = (w, h)
        if key not in self._blur_fbos:
            tex_a = self.ctx.texture((w, h), components=4, dtype='f2')
            tex_a.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex_a.repeat_x = False
            tex_a.repeat_y = False
            fbo_a = self.ctx.framebuffer(color_attachments=[tex_a])

            tex_b = self.ctx.texture((w, h), components=4, dtype='f2')
            tex_b.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex_b.repeat_x = False
            tex_b.repeat_y = False
            fbo_b = self.ctx.framebuffer(color_attachments=[tex_b])

            self._blur_fbos[key] = (fbo_a, tex_a, fbo_b, tex_b)
        return self._blur_fbos[key]

    def render_tonemap(self, viewport: Viewport, surface_w: int, surface_h: int,
                       brightness: float = 6.0) -> None:
        """
        Render the tonemap pass for one window's viewport slice,
        then apply a two-pass gaussian blur.

        viewport    — the slice of the canvas this surface displays (in canvas coords)
        surface_w/h — the actual EGL surface size (physical pixels)
        brightness  — tone mapping brightness (driven by audio RMS)

        The caller must have already made the target window's EGL surface
        current before calling this. After this returns, call win.swap().
        """
        # Effective accumulated frames from temporal decay (geometric series)
        accum_frames = 1.0 / (1.0 - self._decay) if self._decay > 0 else 1.0

        # Fast path: no blur — render tonemap directly to screen
        if self.blur_radius <= 0:
            self.ctx.viewport = (0, 0, surface_w, surface_h)

            self.palette_tex.use(location=0)

            p = self.tonemap_program
            p['u_palette']       = 0
            p['u_width']         = self.canvas_w
            p['u_height']        = self.canvas_h
            p['u_viewport_x']    = viewport.x
            p['u_viewport_y']    = viewport.y
            p['u_viewport_w']    = viewport.w
            p['u_viewport_h']    = viewport.h
            p['u_surface_w']     = surface_w
            p['u_surface_h']     = surface_h
            p['u_brightness']    = brightness
            p['u_accum_frames']  = accum_frames

            self.quad_vao.render(moderngl.TRIANGLES)
            return

        fbo_a, tex_a, fbo_b, tex_b = self._get_blur_fbos(surface_w, surface_h)

        # Pass 1: tonemap into FBO A
        fbo_a.use()
        self.ctx.viewport = (0, 0, surface_w, surface_h)

        self.palette_tex.use(location=0)

        p = self.tonemap_program
        p['u_palette']       = 0
        p['u_width']         = self.canvas_w
        p['u_height']        = self.canvas_h
        p['u_viewport_x']    = viewport.x
        p['u_viewport_y']    = viewport.y
        p['u_viewport_w']    = viewport.w
        p['u_viewport_h']    = viewport.h
        p['u_surface_w']     = surface_w
        p['u_surface_h']     = surface_h
        p['u_brightness']    = brightness
        p['u_accum_frames']  = accum_frames

        self.quad_vao.render(moderngl.TRIANGLES)

        # Pass 2: horizontal blur from FBO A into FBO B
        fbo_b.use()
        tex_a.use(location=0)
        bp = self.blur_program
        bp['u_texture']   = 0
        bp['u_direction'] = (1.0 / surface_w, 0.0)
        bp['u_radius']    = self.blur_radius
        self.blur_vao.render(moderngl.TRIANGLES)

        # Pass 3: vertical blur from FBO B to screen (EGL surface = framebuffer 0)
        _bind_default_framebuffer()
        self.ctx.viewport = (0, 0, surface_w, surface_h)
        tex_b.use(location=0)
        bp['u_texture']   = 0
        bp['u_direction'] = (0.0, 1.0 / surface_h)
        bp['u_radius']    = self.blur_radius
        self.blur_vao.render(moderngl.TRIANGLES)

    def draw_debug_circle(self, ndc_x: float, ndc_y: float,
                          surface_w: int, surface_h: int,
                          radius_px: float = 30.0,
                          color: tuple[float, float, float] = (1.0, 0.0, 0.0)) -> None:
        """Draw a circle overlay at NDC coords (-1..1) on current framebuffer.

        ndc_x, ndc_y: position in normalized device coords (-1..1)
        radius_px: circle radius in pixels
        color: RGB float tuple
        """
        if self._debug_circle_prog is None:
            self._debug_circle_prog = self.ctx.program(
                vertex_shader="""
                #version 430
                in vec2 in_pos;
                void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
                """,
                fragment_shader="""
                #version 430
                uniform vec2 u_center;
                uniform float u_radius;
                uniform vec2 u_resolution;
                uniform vec3 u_color;
                out vec4 fragColor;
                void main() {
                    vec2 pixel = gl_FragCoord.xy;
                    vec2 center_px = (u_center * 0.5 + 0.5) * u_resolution;
                    float dist = length(pixel - center_px);
                    float ring = smoothstep(u_radius - 2.0, u_radius - 1.0, dist)
                               * (1.0 - smoothstep(u_radius + 1.0, u_radius + 2.0, dist));
                    if (ring < 0.01) discard;
                    fragColor = vec4(u_color * ring, ring);
                }
                """,
            )
            self._debug_circle_vao = self.ctx.vertex_array(
                self._debug_circle_prog,
                [(self.quad_vbo, '2f', 'in_pos')],
            )

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        p = self._debug_circle_prog
        p['u_center'] = (ndc_x, ndc_y)
        p['u_radius'] = radius_px
        p['u_resolution'] = (float(surface_w), float(surface_h))
        p['u_color'] = color
        self._debug_circle_vao.render(moderngl.TRIANGLES)

        self.ctx.disable(moderngl.BLEND)

    def histogram_centroid_ndc(self) -> tuple[float, float] | None:
        """Compute the hit-weighted centroid from the coarse histogram.

        Returns (ndc_x, ndc_y) in (-1..1) or None if histogram is empty.
        Uses the GPU-downsampled histogram (~256x256), cheap enough for debug.
        """
        hist = self.histogram_data_coarse()
        total = hist.sum()
        if total == 0:
            return None
        h, w = hist.shape
        ys, xs = np.mgrid[0:h, 0:w]
        cx = float(np.sum(xs * hist) / total)
        cy = float(np.sum(ys * hist) / total)
        # Convert from pixel coords to NDC (-1..1)
        ndc_x = (cx / w) * 2.0 - 1.0
        ndc_y = (cy / h) * 2.0 - 1.0
        return (ndc_x, ndc_y)

    def histogram_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Read back the histogram SSBO after a render pass.

        Returns (hit_counts, color_accs), each shape (canvas_h, canvas_w) uint32.

        Triggers a GPU->CPU readback — slow, don't call every frame.
        Use only when scoring a candidate genome.
        """
        n_pixels = self.canvas_w * self.canvas_h
        raw = np.frombuffer(self.histogram_buf.read(), dtype=np.uint32)
        hit_counts = raw[:n_pixels].reshape(self.canvas_h, self.canvas_w)
        color_accs = raw[n_pixels:].reshape(self.canvas_h, self.canvas_w)
        return hit_counts, color_accs

    def transform_hits_data(self) -> np.ndarray:
        """Read back per-transform hit counts.

        Returns shape (canvas_h, canvas_w, MAX_TRANSFORMS) uint32.
        """
        raw = np.frombuffer(self.transform_hits_buf.read(), dtype=np.uint32)
        return raw.reshape(self.canvas_h, self.canvas_w, MAX_TRANSFORMS)

    def histogram_data_coarse(self) -> np.ndarray:
        """Downsample hit_counts on the GPU, read back ~256KB instead of ~38MB.

        Returns hit_counts shape (ds_h, ds_w) uint32.
        """
        ds = self.downsample_hist_shader
        ds['u_canvas_w'] = self.canvas_w
        ds['u_canvas_h'] = self.canvas_h
        ds['u_out_w'] = self._ds_w
        ds['u_out_h'] = self._ds_h
        gx = (self._ds_w + 15) // 16
        gy = (self._ds_h + 15) // 16
        ds.run(group_x=gx, group_y=gy)
        self.ctx.memory_barrier()
        raw = np.frombuffer(self._ds_buf.read(), dtype=np.uint32)
        return raw.reshape(self._ds_h, self._ds_w)
