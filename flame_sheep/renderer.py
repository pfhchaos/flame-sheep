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
            # Insert after #version line — GLSL requires #version first
            flame_src = flame_src.replace('\n', '\n#define SCORING_MODE\n', 1)
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
        self.reduce_max_shader = self.ctx.compute_shader(
            (SHADER_DIR / 'reduce_max.comp').read_text()
        )
        self.test_pattern_program = self.ctx.program(
            vertex_shader   = (SHADER_DIR / 'tonemap.vert').read_text(),
            fragment_shader = (SHADER_DIR / 'test_pattern.frag').read_text(),
        )
        # Deferred — VAO created after quad_vbo exists
        self._test_pattern_vao = None
        self.de_shader = self.ctx.compute_shader(
            (SHADER_DIR / 'density_estimation.comp').read_text()
        )
        self.tonemap_flam3_program = self.ctx.program(
            vertex_shader   = (SHADER_DIR / 'tonemap.vert').read_text(),
            fragment_shader = (SHADER_DIR / 'tonemap_flam3.frag').read_text(),
        )

    def _create_resources(self) -> None:
        w, h = self.canvas_w, self.canvas_h
        n_pixels = w * h
        self._rng_frame_counter = 0

        # Histogram SSBO (binding=0): two packed uint arrays
        #   [0        .. n_pixels-1] = hit_count
        #   [n_pixels .. 2*n_pixels-1] = color_acc
        histogram_data = np.zeros(n_pixels * 2, dtype=np.uint32)
        self.histogram_buf = self.ctx.buffer(histogram_data.tobytes())
        self.histogram_buf.bind_to_storage_buffer(0)

        # DE input/output histogram SSBOs (binding=11, 12)
        self._de_in_buf = self.ctx.buffer(histogram_data.tobytes())
        self._de_out_buf = self.ctx.buffer(histogram_data.tobytes())
        self._de_in_buf.bind_to_storage_buffer(11)
        self._de_out_buf.bind_to_storage_buffer(12)

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

        # Max hit count buffer (binding=8): single uint for tonemap normalization
        self._max_buf = self.ctx.buffer(np.zeros(1, dtype=np.uint32).tobytes())
        self._max_buf.bind_to_storage_buffer(8)

        # Palette texture: 256 x 1 RGB32F
        self.palette_tex = self.ctx.texture((256, 1), components=3, dtype='f4')
        self.palette_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Audio spectrum texture: N_BINS x 1 R32F
        from flame_sheep_audio import N_BINS
        self.audio_tex = self.ctx.texture((N_BINS, 1), components=1, dtype='f4')
        self.audio_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Genome SSBOs — all have MAX_TRANSFORMS+1 slots (+1 for final xform)
        n_slots = MAX_TRANSFORMS + 1
        IDENTITY = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)

        affines_data     = np.zeros((n_slots, 6), dtype=np.float32)
        active_vars_data = np.full((n_slots, MAX_ACTIVE_VARS * SLOT_SIZE),
                                    -1.0, dtype=np.float32)
        colors_data      = np.zeros(n_slots, dtype=np.float32)
        weights_data     = np.zeros(MAX_TRANSFORMS, dtype=np.float32)
        post_affines_data = np.tile(IDENTITY, (n_slots, 1))
        pre_vars_data     = np.full((n_slots, MAX_ACTIVE_VARS * SLOT_SIZE),
                                     -1.0, dtype=np.float32)

        self.affines_buf      = self.ctx.buffer(affines_data.tobytes())
        self.active_vars_buf  = self.ctx.buffer(active_vars_data.tobytes())
        self.colors_buf       = self.ctx.buffer(colors_data.tobytes())
        self.weights_buf      = self.ctx.buffer(weights_data.tobytes())
        self.post_affines_buf = self.ctx.buffer(post_affines_data.tobytes())
        self.pre_vars_buf     = self.ctx.buffer(pre_vars_data.tobytes())

        self.affines_buf.bind_to_storage_buffer(2)
        self.active_vars_buf.bind_to_storage_buffer(3)
        self.colors_buf.bind_to_storage_buffer(4)
        self.weights_buf.bind_to_storage_buffer(5)
        self.post_affines_buf.bind_to_storage_buffer(9)
        self.pre_vars_buf.bind_to_storage_buffer(10)

        # Walker state SSBO (binding=1)
        walker_data = np.random.uniform(-1, 1, (N_WALKERS, 3)).astype(np.float32)
        self.walker_buf = self.ctx.buffer(walker_data.tobytes())
        self.walker_buf.bind_to_storage_buffer(1)

        # Fullscreen quad — 4 triangles fanning from center to avoid
        # diagonal seam artifact when W-based perspective skew is active.
        # Center vertex at (0,0) splits the quad into 4 triangles:
        #   bottom: (-1,-1) (1,-1) (0,0)
        #   right:  (1,-1) (1,1) (0,0)
        #   top:    (1,1) (-1,1) (0,0)
        #   left:   (-1,1) (-1,-1) (0,0)
        quad_verts = np.array([
            -1, -1,   1, -1,   0,  0,   # bottom
             1, -1,   1,  1,   0,  0,   # right
             1,  1,  -1,  1,   0,  0,   # top
            -1,  1,  -1, -1,   0,  0,   # left
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

        # Per-surface skew amount (0 = head-on)
        self._skew = 0.0

        # Temporal blend — per-surface previous frame FBOs
        self._temporal_fbos: dict[tuple[int, int], tuple] = {}
        self.temporal_decay = 0.0  # per-second decay rate (0=off, 0.5=gentle, 0.9=heavy trails)
        self._temporal_mix_prog = self.ctx.program(
            vertex_shader   = (SHADER_DIR / 'tonemap.vert').read_text(),
            fragment_shader = (SHADER_DIR / 'temporal_mix.frag').read_text(),
        )
        self._temporal_mix_vao = self.ctx.vertex_array(
            self._temporal_mix_prog,
            [(self.quad_vbo, '2f', 'in_pos')],
        )

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
        (affines, active_vars, colors, weights,
         post_affines, pre_active_vars, has_final) = genome.to_gpu_arrays()
        self.affines_buf.write(affines.tobytes())
        self.active_vars_buf.write(active_vars.tobytes())
        self.colors_buf.write(colors.tobytes())
        self.weights_buf.write(weights.tobytes())
        self.post_affines_buf.write(post_affines.tobytes())
        self.pre_vars_buf.write(pre_active_vars.tobytes())

        cs = self.compute_shader
        import math
        cs['u_n_transforms'] = len(genome.transforms)
        cs['u_zoom']         = genome.zoom
        cs['u_cos_rot']      = math.cos(genome.rotation)
        cs['u_sin_rot']      = math.sin(genome.rotation)
        cs['u_center']       = tuple(genome.center)
        cs['u_width']        = self.canvas_w
        cs['u_height']       = self.canvas_h
        cs['u_has_final_xform'] = 1 if has_final else 0

        self._last_zoom = genome.zoom
        self.palette_tex.write(genome.palette.tobytes())

    def set_skew(self, angle_deg: float = 0.0) -> None:
        """Set perspective skew for angled monitors.

        angle_deg: physical angle of the monitor relative to viewer.
        Positive = angled away on right, negative = angled away on left.
        0 = facing viewer directly.
        """
        import math
        self._skew = math.tan(math.radians(angle_deg)) * 0.5 if abs(angle_deg) > 0.1 else 0.0

    def set_rotation(self, angle: float) -> None:
        """Update only the rotation uniforms (no genome re-upload)."""
        import math
        self.compute_shader['u_cos_rot'] = math.cos(angle)
        self.compute_shader['u_sin_rot'] = math.sin(angle)

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
        self.compute_shader['u_rng_seed'] = self._rng_frame_counter
        self._rng_frame_counter += 1
        groups = N_WALKERS // 64
        self.compute_shader.run(group_x=groups)

    def reduce_histogram_max(self) -> None:
        """Compute max hit count from histogram via GPU reduction.
        Result is stored in _max_buf SSBO (binding=8) for tonemap to read."""
        n_pixels = self.canvas_w * self.canvas_h
        # Reset max to 0 before reduction
        self._max_buf.write(np.zeros(1, dtype=np.uint32).tobytes())
        self.reduce_max_shader['u_n_pixels'] = n_pixels
        groups = (n_pixels + 255) // 256
        self.reduce_max_shader.run(group_x=groups)
        self.ctx.memory_barrier()

    def apply_density_estimation(self, max_radius: int = 9,
                                  curve: float = 0.4,
                                  min_density: float = 0.0) -> None:
        """Apply flam3-style density estimation to the histogram.

        Spatially-varying gaussian blur: sparse pixels get wide kernels,
        dense pixels stay sharp. Reveals filament structure that would
        otherwise be invisible single-pixel dots.

        Modifies the histogram in-place (copies to DE buffers, runs shader,
        copies result back).

        Args:
            max_radius: maximum blur kernel radius (flam3 estimator_radius)
            curve: density falloff curve (flam3 estimator_curve, typically 0.4-0.6)
            min_density: minimum hits to consider (below = noise)
        """
        w, h = self.canvas_w, self.canvas_h

        # Need max_hits for the shader
        self.reduce_histogram_max()

        # Copy current histogram → DE input buffer
        data = self.histogram_buf.read()
        self._de_in_buf.write(data)

        # Run DE shader
        de = self.de_shader
        de['u_width'] = w
        de['u_height'] = h
        de['u_max_radius'] = max_radius
        de['u_curve'] = float(curve)
        de['u_min_density'] = float(min_density)

        groups_x = (w + 15) // 16
        groups_y = (h + 15) // 16
        de.run(group_x=groups_x, group_y=groups_y)
        self.ctx.memory_barrier()

        # Copy DE output → main histogram buffer
        result = self._de_out_buf.read()
        self.histogram_buf.write(result)

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

    def render_test_pattern(self, viewport: Viewport, surface_w: int, surface_h: int) -> None:
        """Render a grid test pattern for multi-monitor alignment.

        Uses the same vertex shader (with skew) as the tonemap, so
        perspective correction and viewport slicing are identical.
        """
        if self._test_pattern_vao is None:
            self._test_pattern_vao = self.ctx.vertex_array(
                self.test_pattern_program,
                [(self.quad_vbo, '2f', 'in_pos')],
            )

        _bind_default_framebuffer()
        self.ctx.viewport = (0, 0, surface_w, surface_h)

        try:
            self.test_pattern_program['u_skew'] = self._skew
        except KeyError:
            pass

        p = self.test_pattern_program
        p['u_width']      = self.canvas_w
        p['u_height']     = self.canvas_h
        p['u_viewport_x'] = viewport.x
        p['u_viewport_y'] = viewport.y
        p['u_viewport_w'] = viewport.w
        p['u_viewport_h'] = viewport.h
        p['u_surface_w']  = surface_w
        p['u_surface_h']  = surface_h
        if 'u_ppmm' in p:
            p['u_ppmm']   = getattr(self, '_ppmm', 1.0)

        self._test_pattern_vao.render(moderngl.TRIANGLES)

    def set_ppmm(self, ppmm: float) -> None:
        """Set canvas pixels per millimeter (for test pattern physical grid)."""
        self._ppmm = ppmm

    def render_tonemap(self, viewport: Viewport, surface_w: int, surface_h: int,
                       brightness: float = 6.0, dt: float = 1/60) -> None:
        """
        Render the tonemap pass for one window's viewport slice,
        optionally with gaussian blur and temporal blend.

        viewport    — the slice of the canvas this surface displays (in canvas coords)
        surface_w/h — the actual EGL surface size (physical pixels)
        brightness  — tone mapping brightness (driven by audio RMS)
        dt          — frame time in seconds (for frame-rate-independent temporal blend)

        The caller must have already made the target window's EGL surface
        current before calling this. After this returns, call win.swap().
        """
        # Effective accumulated frames from temporal decay (geometric series)

        # Frame-rate-independent temporal blend.
        # temporal_decay: blend strength in seconds (half-life of previous frame).
        # 0 = off, 0.05 = subtle speckle smoothing, 0.5 = visible trails.
        # mix_factor is how much NEW frame to show (1 = all new, 0 = frozen).
        use_temporal = self.temporal_decay > 0.0
        if use_temporal:
            # Exponential decay: after temporal_decay seconds, previous
            # frame has faded to 50%. Per-frame retention = 0.5^(dt/half_life).
            temporal_mix = 1.0 - 0.5 ** (dt / self.temporal_decay)
        else:
            temporal_mix = 1.0
        if use_temporal:
            # 3 FBOs: accum (previous blended result), current (this frame),
            # blend (output of mix → becomes new accum next frame)
            (accum_fbo, accum_tex), (current_fbo, current_tex), (blend_fbo, blend_tex) = \
                self._get_temporal_fbos(surface_w, surface_h)

        # Set skew for angled monitors
        try:
            self.tonemap_program['u_skew'] = self._skew
        except KeyError:
            pass  # uniform optimized out when always 0

        if self.blur_radius <= 0:
            # No spatial blur — tonemap directly
            target = current_fbo if use_temporal else None
            if target:
                target.use()
            else:
                _bind_default_framebuffer()
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
            p['u_gamma']         = brightness

            self.quad_vao.render(moderngl.TRIANGLES)
        else:
            # Spatial blur: tonemap → FBO A → blur H → FBO B → blur V → temp/screen
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
            p['u_gamma']         = brightness

            self.quad_vao.render(moderngl.TRIANGLES)

            # Pass 2: horizontal blur A → B
            fbo_b.use()
            tex_a.use(location=0)
            bp = self.blur_program
            bp['u_texture']   = 0
            bp['u_direction'] = (1.0 / surface_w, 0.0)
            bp['u_radius']    = self.blur_radius
            self.blur_vao.render(moderngl.TRIANGLES)

            # Pass 3: vertical blur B → temp/screen
            if use_temporal:
                current_fbo.use()
            else:
                _bind_default_framebuffer()
            self.ctx.viewport = (0, 0, surface_w, surface_h)
            tex_b.use(location=0)
            bp['u_texture']   = 0
            bp['u_direction'] = (0.0, 1.0 / surface_h)
            bp['u_radius']    = self.blur_radius
            self.blur_vao.render(moderngl.TRIANGLES)

        # Temporal blend: mix current + accumulated → blend_fbo, display,
        # then rotate FBOs so blend becomes new accumulator.
        if use_temporal:
            # Mix current + accum → blend_fbo
            blend_fbo.use()
            self.ctx.viewport = (0, 0, surface_w, surface_h)
            current_tex.use(location=0)
            accum_tex.use(location=1)
            tp = self._temporal_mix_prog
            tp['u_current']    = 0
            tp['u_previous']   = 1
            tp['u_mix_factor'] = temporal_mix
            tp['u_comparison'] = 0
            self._temporal_mix_vao.render(moderngl.TRIANGLES)

            # Blit blend result to screen
            _bind_default_framebuffer()
            self.ctx.viewport = (0, 0, surface_w, surface_h)
            blend_tex.use(location=0)
            bp = self.blur_program
            bp['u_texture']   = 0
            bp['u_direction'] = (0.0, 0.0)
            bp['u_radius']    = 0.0
            self.blur_vao.render(moderngl.TRIANGLES)

            # Rotate: blend → accum for next frame
            key = (surface_w, surface_h)
            self._temporal_fbos[key] = [
                (blend_fbo, blend_tex),     # new accum
                (accum_fbo, accum_tex),      # new current (scratch)
                (current_fbo, current_tex),  # new blend (scratch)
            ]

    def _render_comparison_old(self, viewport: Viewport, surface_w: int, surface_h: int,
                          brightness: float = 6.0, dt: float = 1/60) -> None:
        """Render 4 quadrants: raw / gaussian / temporal / both.

        Temporarily overrides blur_radius and temporal_decay to render
        each combination into a quadrant of the screen.
        """
        hw = surface_w // 2
        hh = surface_h // 2

        saved_blur = self.blur_radius
        saved_decay = self.temporal_decay

        # Scale viewport to fit each quadrant
        # Each quadrant shows the full fractal at half resolution
        quadrants = [
            (0,  hh, hw, hh, 0.0, 0.0),    # top-left: raw
            (hw, hh, hw, hh, saved_blur, 0.0),  # top-right: gaussian only
            (0,  0,  hw, hh, 0.0, saved_decay),  # bottom-left: temporal only
            (hw, 0,  hw, hh, saved_blur, saved_decay),  # bottom-right: both
        ]

        for qx, qy, qw, qh, blur, decay in quadrants:
            self.blur_radius = blur
            self.temporal_decay = decay

            # Render tonemap to a temp FBO at quadrant size, then blit
            # to the correct quadrant on screen.
            # Simpler approach: just set the GL viewport and render directly.

            # For temporal blend, each quadrant needs its own accum FBOs
            # keyed by quadrant position. Hack: offset the size key.
            if decay > 0:
                # Use unique key per quadrant for temporal FBOs
                orig_get = self._get_temporal_fbos
                _qkey = (qw + qx, qh + qy)  # unique per quadrant
                if _qkey not in self._temporal_fbos:
                    fbos = []
                    for _ in range(3):
                        tex = self.ctx.texture((qw, qh), components=4, dtype='f2')
                        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                        fbo = self.ctx.framebuffer(color_attachments=[tex])
                        fbo.use()
                        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
                        fbos.append((fbo, tex))
                    self._temporal_fbos[_qkey] = fbos

            self.render_tonemap(viewport, qw, qh, brightness=brightness, dt=dt)

            # If no temporal, the result is on the default framebuffer at
            # viewport (0,0,qw,qh). Need to blit to the correct quadrant.
            # For temporal, the blit already happened.
            # This is messy — let me use a simpler approach.

        self.blur_radius = saved_blur
        self.temporal_decay = saved_decay

    def render_comparison_v2(self, viewport: Viewport, surface_w: int, surface_h: int,
                             brightness: float = 6.0, dt: float = 1/60) -> None:
        """Render 4 vertical strips comparing blur modes on one fractal.

        Left to right: raw | gaussian | temporal | both.
        Uses the temporal_mix shader in comparison mode.
        """
        temporal_mix = 1.0 - 0.5 ** (dt / self.temporal_decay) if self.temporal_decay > 0 else 1.0

        # Get/create comparison FBOs (keyed specially to not conflict)
        cmp_key = ('cmp', surface_w, surface_h)
        if cmp_key not in self._temporal_fbos:
            fbos = []
            for _ in range(3):
                tex = self.ctx.texture((surface_w, surface_h), components=4, dtype='f2')
                tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                fbo = self.ctx.framebuffer(color_attachments=[tex])
                fbo.use()
                self.ctx.clear(0.0, 0.0, 0.0, 1.0)
                fbos.append((fbo, tex))
            self._temporal_fbos[cmp_key] = fbos

        (accum_fbo, accum_tex), (current_fbo, current_tex), (gauss_fbo, gauss_tex) = \
            self._temporal_fbos[cmp_key]

        # Step 1: tonemap → current_fbo (raw, unblurred)
        current_fbo.use()
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
        p['u_gamma']         = brightness
        self.quad_vao.render(moderngl.TRIANGLES)

        # Step 2: gaussian blur current → gauss_fbo (reuse blur FBOs for temp)
        blur_a, blur_a_tex, blur_b, blur_b_tex = self._get_blur_fbos(surface_w, surface_h)
        # Horizontal
        blur_a.use()
        self.ctx.viewport = (0, 0, surface_w, surface_h)
        current_tex.use(location=0)
        bp = self.blur_program
        bp['u_texture']   = 0
        bp['u_direction'] = (1.0 / surface_w, 0.0)
        bp['u_radius']    = self.blur_radius
        self.blur_vao.render(moderngl.TRIANGLES)
        # Vertical → gauss_fbo
        gauss_fbo.use()
        blur_a_tex.use(location=0)
        bp['u_texture']   = 0
        bp['u_direction'] = (0.0, 1.0 / surface_h)
        bp['u_radius']    = self.blur_radius
        self.blur_vao.render(moderngl.TRIANGLES)

        # Step 3: comparison blend → screen
        # The shader splits into 4 strips using u_comparison=1
        _bind_default_framebuffer()
        self.ctx.viewport = (0, 0, surface_w, surface_h)
        current_tex.use(location=0)   # u_current = raw frame
        accum_tex.use(location=1)     # u_previous = accumulated
        gauss_tex.use(location=2)     # u_gauss = blurred frame
        tp = self._temporal_mix_prog
        tp['u_current']    = 0
        tp['u_previous']   = 1
        tp['u_gauss']      = 2
        tp['u_mix_factor'] = temporal_mix
        tp['u_comparison'] = 1
        self._temporal_mix_vao.render(moderngl.TRIANGLES)

        # Step 4: update accumulator — copy current frame into accum
        # for next frame's temporal blend. Simple overwrite, not blend,
        # because the comparison shader handles the blend per-strip.
        self.ctx.copy_framebuffer(accum_fbo, current_fbo)

    def render_blur_comparison(self, viewport: Viewport, surface_w: int, surface_h: int,
                               brightness: float = 6.0,
                               radii: tuple[float, ...] = (0.0, 0.3, 0.6, 1.0)) -> None:
        """Render vertical strips with different gaussian blur radii."""
        # Tonemap to a shared FBO first
        raw_tex = self.ctx.texture((surface_w, surface_h), components=4, dtype='f2')
        raw_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        raw_fbo = self.ctx.framebuffer(color_attachments=[raw_tex])
        raw_fbo.use()
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
        p['u_gamma']         = brightness
        self.quad_vao.render(moderngl.TRIANGLES)

        n_strips = len(radii)
        strip_w = surface_w // n_strips

        _bind_default_framebuffer()

        for i, radius in enumerate(radii):
            if radius <= 0:
                # No blur — blit raw directly to strip
                self.ctx.viewport = (i * strip_w, 0, strip_w, surface_h)
                raw_tex.use(location=0)
                bp = self.blur_program
                bp['u_texture']   = 0
                bp['u_direction'] = (0.0, 0.0)
                bp['u_radius']    = 0.0
                self.blur_vao.render(moderngl.TRIANGLES)
            else:
                # Blur into temp FBOs then blit to strip
                fbo_a, tex_a, fbo_b, tex_b = self._get_blur_fbos(surface_w, surface_h)
                bp = self.blur_program

                # Horizontal blur
                fbo_a.use()
                self.ctx.viewport = (0, 0, surface_w, surface_h)
                raw_tex.use(location=0)
                bp['u_texture']   = 0
                bp['u_direction'] = (1.0 / surface_w, 0.0)
                bp['u_radius']    = radius
                self.blur_vao.render(moderngl.TRIANGLES)

                # Vertical blur → strip on screen
                _bind_default_framebuffer()
                self.ctx.viewport = (i * strip_w, 0, strip_w, surface_h)
                tex_a.use(location=0)
                bp['u_texture']   = 0
                bp['u_direction'] = (0.0, 1.0 / surface_h)
                bp['u_radius']    = radius
                self.blur_vao.render(moderngl.TRIANGLES)

        raw_tex.release()
        raw_fbo.release()

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

    def _get_temporal_fbos(self, w: int, h: int):
        """Get or create FBOs for temporal blending at given size.
        Returns (accum_fbo, accum_tex, current_fbo, current_tex, blend_fbo, blend_tex).
        Three FBOs: current frame, accumulated result, blend output."""
        key = (w, h)
        if key not in self._temporal_fbos:
            fbos = []
            for _ in range(3):
                tex = self.ctx.texture((w, h), components=4, dtype='f2')
                tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                fbo = self.ctx.framebuffer(color_attachments=[tex])
                fbo.use()
                self.ctx.clear(0.0, 0.0, 0.0, 1.0)
                fbos.append((fbo, tex))
            self._temporal_fbos[key] = fbos
        return self._temporal_fbos[key]

    def snapshot_png(self, brightness: float = 6.0) -> bytes:
        """Tonemap the current histogram to an RGBA PNG and return as bytes.

        Uses a temporary FBO — does not affect screen output.
        """
        import io
        from PIL import Image

        w, h = self.canvas_w, self.canvas_h
        fbo_tex = self.ctx.texture((w, h), components=4, dtype='f1')
        fbo = self.ctx.framebuffer(color_attachments=[fbo_tex])
        fbo.use()
        self.ctx.viewport = (0, 0, w, h)

        # Compute max hit count for tonemap normalization
        self.reduce_histogram_max()

        viewport = Viewport(0, 0, w, h)
        self.palette_tex.use(location=0)

        p = self.tonemap_program
        p['u_palette']       = 0
        p['u_width']         = w
        p['u_height']        = h
        p['u_viewport_x']    = 0
        p['u_viewport_y']    = 0
        p['u_viewport_w']    = w
        p['u_viewport_h']    = h
        p['u_surface_w']     = w
        p['u_surface_h']     = h
        p['u_gamma']         = brightness

        self.quad_vao.render(moderngl.TRIANGLES)

        data = fbo.read(components=4)
        img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 4)
        img = img[::-1].copy()  # flip Y (OpenGL origin is bottom-left)

        fbo.release()
        fbo_tex.release()

        buf = io.BytesIO()
        Image.fromarray(img, 'RGBA').save(buf, format='PNG', optimize=True)
        return buf.getvalue()

    def snapshot_de_png(self, brightness: float = 6.0,
                        max_radius: int = 9, curve: float = 0.4) -> bytes:
        """Render with density estimation, then tonemap to PNG.

        Applies spatially-varying DE blur to the histogram before
        tonemapping. Sparse regions get wide blurs revealing filament
        structure; dense regions stay sharp.
        """
        self.apply_density_estimation(max_radius=max_radius, curve=curve)
        return self.snapshot_png(brightness=brightness)

    def snapshot_flam3_png(self, brightness: float = 4.0, gamma: float = 4.0,
                           vibrancy: float = 1.0, contrast: float = 1.0,
                           sample_density: float = 1.0,
                           highlight_power: float = -1.0) -> bytes:
        """Tonemap using flam3-accurate pipeline and return as PNG bytes.

        Uses absolute brightness scaling (k1/k2) and pow(density, 1/gamma)
        gamma correction, matching flam3's rendering math.

        Args:
            brightness: flam3 brightness parameter (typically 2-100)
            gamma: flam3 gamma parameter (typically 1-5)
            vibrancy: color saturation blend (0=grayscale, 1=full color)
            contrast: flam3 contrast parameter (typically 1.0)
            sample_density: total samples per pixel (for k2 normalization)
            highlight_power: hue preservation power (-1 = disabled)
        """
        import io
        from PIL import Image

        w, h = self.canvas_w, self.canvas_h

        # Compute k1 and k2 matching flam3's rect.c
        # k1 = contrast * brightness * PREFILTER_WHITE * 268 / 256
        # k2 = oversample² * nbatches / (contrast * area * WHITE_LEVEL * sample_density * sumfilt)
        #
        # area = image_w * image_h / (ppux * ppuy) — world-space area of viewport
        # For us: viewport spans (-1/zoom, 1/zoom) in each axis, so:
        #   world_area = (2/zoom)² = 4/zoom²
        # ppux = ppuy = zoom * w/2, so area = w*h / (zoom*w/2)² = 4/zoom²
        # oversample=1, nbatches=1, sumfilt=1 for our simple case
        k1 = contrast * brightness * 255.0 * 268.0 / 256.0
        area = 4.0 / max(self._last_zoom ** 2, 1e-10) if hasattr(self, '_last_zoom') else 4.0
        k2 = 1.0 / max(contrast * area * 255.0 * max(sample_density, 0.01), 1e-10)

        fbo_tex = self.ctx.texture((w, h), components=4, dtype='f1')
        fbo = self.ctx.framebuffer(color_attachments=[fbo_tex])
        fbo.use()
        self.ctx.viewport = (0, 0, w, h)

        self.palette_tex.use(location=0)

        p = self.tonemap_flam3_program
        p['u_palette']       = 0
        p['u_width']         = w
        p['u_height']        = h
        p['u_viewport_x']    = 0
        p['u_viewport_y']    = 0
        p['u_viewport_w']    = w
        p['u_viewport_h']    = h
        p['u_surface_w']     = w
        p['u_surface_h']     = h
        p['u_k1']            = float(k1)
        p['u_k2']            = float(k2)
        p['u_gamma']         = 1.0 / max(gamma, 0.01)  # pre-invert for shader
        p['u_vibrancy']      = float(vibrancy)
        # These may be optimized out by the compiler
        if 'u_lin_thresh' in p:
            p['u_lin_thresh']    = 0.01
        if 'u_highlight_power' in p:
            p['u_highlight_power'] = float(highlight_power)

        self.quad_vao.render(moderngl.TRIANGLES)

        data = fbo.read(components=4)
        img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 4)
        img = img[::-1].copy()

        fbo.release()
        fbo_tex.release()

        buf = io.BytesIO()
        Image.fromarray(img, 'RGBA').save(buf, format='PNG', optimize=True)
        return buf.getvalue()

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
