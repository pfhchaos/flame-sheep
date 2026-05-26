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

import moderngl
import numpy as np
from pathlib import Path

# Framework-side GPU plumbing lives in gpu_context.py. We re-export here so
# existing `from .renderer import Viewport, GpuContext, SHADER_DIR,
# _resolve_includes, _bind_default_framebuffer` calls keep working unchanged
# while new code can also import directly from .gpu_context.
from .gpu_context import (
    SHADER_DIR,
    Viewport,
    GpuContext,
    _resolve_includes,
    _bind_default_framebuffer,
)

from ..genome import Genome, MAX_TRANSFORMS, NUM_VARIATIONS, MAX_ACTIVE_VARS, SLOT_SIZE


# ---------------------------------------------------------------------------
# Walker / iteration tuning
#
# We render the full virtual canvas at half resolution for performance.
# Walker count and iterations are chosen to give reasonable density at
# the half-res canvas size without hammering the GPU.
# ---------------------------------------------------------------------------
N_WALKERS  = 1024 * 64   # 65536 walkers — more parallelism for modern GPUs
N_ITERS    = 150          # default iterations per walker per frame

# Live render iteration range. At runtime, DetailAxis maps audio energy to
# an iteration count in [LIVE_ITER_MIN, LIVE_ITER_MAX] per dispatch.
# Anything that wants to mirror "what the user will actually see" — batch
# rendering, first-hit snapshots, benchmarks — should use these constants
# so changes propagate.
LIVE_ITER_MIN = 100
LIVE_ITER_MAX = 500



class FlameRenderer:
    """
    Owns flame-specific GPU resources for the full virtual canvas.
    Composes with a GpuContext for shared infrastructure.

    canvas_w / canvas_h are read from self.gpu — full virtual canvas
    size in render pixels (may be half the physical size for performance).
    """

    def __init__(self, gpu: GpuContext, scoring: bool = False,
                 n_walkers: int = N_WALKERS):
        self.gpu        = gpu
        self.ctx        = gpu.ctx  # local alias — same moderngl context
        self.n_walkers  = n_walkers

        self._load_shaders(scoring=scoring)
        self._create_resources()

    @property
    def canvas_w(self) -> int:
        return self.gpu.canvas_w

    @property
    def canvas_h(self) -> int:
        return self.gpu.canvas_h

    def _load_shaders(self, scoring: bool = False) -> None:
        from .variations._symmetry_groups import generate_glsl

        def _inject_symmetry(src: str) -> str:
            return src.replace('// {{SYMMETRY_GROUPS}}', generate_glsl())

        flame_defines = {'SCORING_MODE': ''} if scoring else None
        self.compute_shader = self.gpu.compile_compute_shader(
            SHADER_DIR / 'flame.comp',
            defines=flame_defines,
            source_transform=_inject_symmetry,
        )
        self.clear_shader = self.gpu.compile_compute_shader(
            SHADER_DIR / 'clear.comp')
        self.tonemap_program = self.gpu.compile_program(
            SHADER_DIR / 'tonemap.vert', SHADER_DIR / 'tonemap.frag')
        self.blur_program = self.gpu.compile_program(
            SHADER_DIR / 'tonemap.vert', SHADER_DIR / 'blur.frag')
        self.downsample_hist_shader = self.gpu.compile_compute_shader(
            SHADER_DIR / 'downsample_hist.comp')
        self.reduce_max_shader = self.gpu.compile_compute_shader(
            SHADER_DIR / 'reduce_max.comp')
        self.de_shader = self.gpu.compile_compute_shader(
            SHADER_DIR / 'density_estimation.comp')
        self.tonemap_flam3_program = self.gpu.compile_program(
            SHADER_DIR / 'tonemap.vert', SHADER_DIR / 'tonemap_flam3.frag'
        )

    def _create_resources(self) -> None:
        w, h = self.canvas_w, self.canvas_h
        n_pixels = w * h
        self._rng_frame_counter = 0

        # Histogram SSBO (binding=0): two packed uint arrays
        #   [0        .. n_pixels-1] = hit_count
        #   [n_pixels .. 2*n_pixels-1] = color_acc
        histogram_bytes = bytes(n_pixels * 2 * 4)  # uint32
        self.histogram_buf = self.gpu.allocate_storage_buffer(
            len(histogram_bytes), initial=histogram_bytes)
        self.histogram_buf.bind_to_storage_buffer(0)

        # DE input/output histogram SSBOs (binding=11, 12)
        self._de_in_buf = self.gpu.allocate_storage_buffer(
            len(histogram_bytes), initial=histogram_bytes)
        self._de_out_buf = self.gpu.allocate_storage_buffer(
            len(histogram_bytes), initial=histogram_bytes)
        self._de_in_buf.bind_to_storage_buffer(11)
        self._de_out_buf.bind_to_storage_buffer(12)

        # Per-transform hit counts SSBO (binding=7)
        self.transform_hits_buf = self.gpu.allocate_storage_buffer(
            n_pixels * MAX_TRANSFORMS * 4)  # uint32 zero-init
        self.transform_hits_buf.bind_to_storage_buffer(7)

        # Downsampled histogram for symmetry scoring (binding=6)
        self._ds_w = min(256, w)
        self._ds_h = min(256, h)
        self._ds_buf = self.gpu.allocate_storage_buffer(
            self._ds_w * self._ds_h * 4)  # uint32 zero-init
        self._ds_buf.bind_to_storage_buffer(6)

        # Max hit count buffer (binding=8): single uint for tonemap normalization
        self._max_buf = self.gpu.allocate_storage_buffer(4)  # one uint32
        self._max_buf.bind_to_storage_buffer(8)

        # Palette texture: 256 x 1 RGB32F
        self.palette_tex = self.gpu.allocate_texture(
            (256, 1), components=3, dtype='f4',
            filter=(moderngl.LINEAR, moderngl.LINEAR))

        # Audio spectrum texture: N_BINS x 1 R32F
        from flame_sheep_audio import N_BINS
        self.audio_tex = self.gpu.allocate_texture(
            (N_BINS, 1), components=1, dtype='f4',
            filter=(moderngl.LINEAR, moderngl.LINEAR))

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

        color_speeds_data = np.full(n_slots, 0.5, dtype=np.float32)

        def _ssbo(data: np.ndarray):
            b = data.tobytes()
            return self.gpu.allocate_storage_buffer(len(b), initial=b)

        self.affines_buf      = _ssbo(affines_data)
        self.active_vars_buf  = _ssbo(active_vars_data)
        self.colors_buf       = _ssbo(colors_data)
        self.color_speeds_buf = _ssbo(color_speeds_data)
        self.weights_buf      = _ssbo(weights_data)
        self.post_affines_buf = _ssbo(post_affines_data)
        self.pre_vars_buf     = _ssbo(pre_vars_data)

        self.affines_buf.bind_to_storage_buffer(2)
        self.active_vars_buf.bind_to_storage_buffer(3)
        self.colors_buf.bind_to_storage_buffer(4)
        self.weights_buf.bind_to_storage_buffer(5)
        self.color_speeds_buf.bind_to_storage_buffer(6)
        self.post_affines_buf.bind_to_storage_buffer(9)
        self.pre_vars_buf.bind_to_storage_buffer(10)

        # Walker state SSBO (binding=1)
        walker_data = np.random.uniform(-1, 1, (self.n_walkers, 3)).astype(np.float32)
        walker_bytes = walker_data.tobytes()
        self.walker_buf = self.gpu.allocate_storage_buffer(
            len(walker_bytes), initial=walker_bytes)
        self.walker_buf.bind_to_storage_buffer(1)

        # Fullscreen-quad VAOs — bind the shared VBO (owned by GpuContext)
        # to each program that draws a fullscreen pass.
        self.quad_vao = self.gpu.make_quad_vao(self.tonemap_program)
        self.blur_vao = self.gpu.make_quad_vao(self.blur_program)

        # Blur FBOs — created lazily per surface size
        self._blur_fbos = {}
        self.blur_radius = 1.0  # adjustable blur strength

        # Temporal blend — per-surface previous frame FBOs
        self._temporal_fbos: dict[tuple[int, int], tuple] = {}
        self.temporal_decay = 0.0  # per-second decay rate (0=off, 0.5=gentle, 0.9=heavy trails)
        self._temporal_mix_prog = self.gpu.compile_program(
            SHADER_DIR / 'tonemap.vert', SHADER_DIR / 'temporal_mix.frag')
        self._temporal_mix_vao = self.gpu.make_quad_vao(self._temporal_mix_prog)

        # Temporal decay tracking for tonemap normalization
        self._decay = 0.0  # set by clear_histogram()

        # Histogram offset/stride for compare mode (default: normal single-genome)
        n_pixels = self.canvas_w * self.canvas_h
        self._hist_offset = 0
        self.compute_shader['u_hist_offset'] = 0
        self.compute_shader['u_hist_stride'] = n_pixels
        self.clear_shader['u_hist_offset'] = 0
        self.reduce_max_shader['u_hist_offset'] = 0
        self.tonemap_program['u_hist_offset'] = 0
        self.tonemap_program['u_hist_stride'] = n_pixels

    def reset_walkers(self) -> None:
        """Re-randomize walker positions. Call after a genome swap to avoid
        stuck walkers that escaped to infinity under a degenerate genome."""
        walker_data = np.random.uniform(-1, 1, (self.n_walkers, 3)).astype(np.float32)
        self.walker_buf.write(walker_data.tobytes())

    # ------------------------------------------------------------------
    # Per-frame API
    # ------------------------------------------------------------------

    def upload_genome(self, genome: Genome) -> None:
        gpu = genome.to_gpu_arrays()
        self.affines_buf.write(gpu['affines'].tobytes())
        self.active_vars_buf.write(gpu['active_vars'].tobytes())
        self.colors_buf.write(gpu['colors'].tobytes())
        self.color_speeds_buf.write(gpu['color_speeds'].tobytes())
        self.weights_buf.write(gpu['weights'].tobytes())
        self.post_affines_buf.write(gpu['post_affines'].tobytes())
        self.pre_vars_buf.write(gpu['pre_active_vars'].tobytes())

        cs = self.compute_shader
        import math
        cs['u_n_transforms'] = len(genome.transforms)
        # Aspect-correct zoom so circles stay circular.
        # Shorter axis gets the base zoom, longer axis is scaled down.
        if self.canvas_w >= self.canvas_h:
            # Landscape/square: y is the constraining axis
            aspect = self.canvas_h / max(self.canvas_w, 1)
            cs['u_zoom'] = (genome.zoom * aspect, genome.zoom)
        else:
            # Portrait: x is the constraining axis
            aspect = self.canvas_w / max(self.canvas_h, 1)
            cs['u_zoom'] = (genome.zoom, genome.zoom * aspect)
        cs['u_cos_rot']      = math.cos(genome.rotation)
        cs['u_sin_rot']      = math.sin(genome.rotation)
        cs['u_center']       = tuple(genome.center)
        cs['u_width']        = self.canvas_w
        cs['u_height']       = self.canvas_h
        cs['u_has_final_xform'] = 1 if gpu['has_final_xform'] else 0

        self._last_zoom = genome.zoom
        self.palette_tex.write(genome.palette.tobytes())

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

    def ensure_double_histogram(self) -> None:
        """Double the histogram buffer for compare mode.

        Layout: [hits_L(n_px), hits_R(n_px), colors_L(n_px), colors_R(n_px)]
        Stride (hits → colors) = 2*n_px instead of n_px.
        Left offset = 0, right offset = n_px.
        """
        n_pixels = self.canvas_w * self.canvas_h
        if self.histogram_buf.size >= n_pixels * 4 * 4:
            return  # already doubled
        new_buf = self.gpu.allocate_storage_buffer(n_pixels * 4 * 4)  # uint32 zero-init
        new_buf.bind_to_storage_buffer(0)
        self.histogram_buf = new_buf
        # Update stride to account for doubled hits region
        stride = n_pixels * 2
        self.compute_shader['u_hist_stride'] = stride
        self.clear_shader['u_hist_offset'] = 0
        self.reduce_max_shader['u_hist_offset'] = 0
        self.tonemap_program['u_hist_stride'] = stride

    def set_histogram_offset(self, offset: int) -> None:
        """Set histogram offset for compare mode (0 = left/normal, n_pixels = right)."""
        self._hist_offset = offset
        self.compute_shader['u_hist_offset'] = offset
        self.clear_shader['u_hist_offset'] = offset
        self.reduce_max_shader['u_hist_offset'] = offset
        self.tonemap_program['u_hist_offset'] = offset

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

    def bind_buffers(self) -> None:
        """Rebind all SSBOs to their binding points.

        Required when multiple FlameRenderer instances share a GL context
        (e.g. compare mode), since bind_to_storage_buffer is global state.
        """
        self.histogram_buf.bind_to_storage_buffer(0)
        self.walker_buf.bind_to_storage_buffer(1)
        self.affines_buf.bind_to_storage_buffer(2)
        self.active_vars_buf.bind_to_storage_buffer(3)
        self.colors_buf.bind_to_storage_buffer(4)
        self.weights_buf.bind_to_storage_buffer(5)
        self.color_speeds_buf.bind_to_storage_buffer(6)
        self.transform_hits_buf.bind_to_storage_buffer(7)
        self._max_buf.bind_to_storage_buffer(8)
        self.post_affines_buf.bind_to_storage_buffer(9)
        self.pre_vars_buf.bind_to_storage_buffer(10)

    def dispatch_chaos_game(self, iterations: int = N_ITERS) -> None:
        self.compute_shader['u_iterations'] = iterations
        self.compute_shader['u_rng_seed'] = self._rng_frame_counter
        self._rng_frame_counter += 1
        groups = self.n_walkers // 64
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

        # Copy current histogram → DE input buffer
        data = self.histogram_buf.read()
        self._de_in_buf.write(data)

        # Zero the output buffer (scatter accumulates from zero)
        self._de_out_buf.write(b'\x00' * len(data))

        # Run DE scatter shader
        de = self.de_shader
        de['u_width'] = w
        de['u_height'] = h
        de['u_max_radius'] = max_radius
        de['u_curve'] = float(curve)

        groups_x = (w + 15) // 16
        groups_y = (h + 15) // 16
        de.run(group_x=groups_x, group_y=groups_y)
        self.ctx.memory_barrier()

        # Copy DE output → main histogram buffer
        # FP_SCALE is baked into the values but the tonemap normalizes by
        # max_hits anyway, so the scale cancels out — just copy directly.
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

    def render_tonemap(self, viewport: Viewport, surface_w: int, surface_h: int,
                       brightness: float = 6.0, dt: float = 1/60,
                       screen_rect: tuple[int, int, int, int] | None = None,
                       target_fbo: 'moderngl.Framebuffer | None' = None) -> None:
        """
        Render the tonemap pass for one window's viewport slice,
        optionally with gaussian blur and temporal blend.

        viewport    — the slice of the canvas this surface displays (in canvas coords)
        surface_w/h — the actual EGL surface size (physical pixels)
        brightness  — tone mapping brightness (driven by audio RMS)
        dt          — frame time in seconds (for frame-rate-independent temporal blend)

        The caller must have already made the target window's EGL surface
        current before calling this. After this returns, call win.swap().

        screen_rect — optional (x, y, w, h) GL viewport override for split-screen.
                      When set, renders to a sub-region of the surface instead of
                      the full surface. Used by compare mode for side-by-side display.
        """
        # Override surface dimensions for GL viewport if screen_rect provided
        if screen_rect is not None:
            _gl_viewport = screen_rect
            # The tonemap shader maps UVs across the GL viewport, so surface_w/h
            # must match the viewport size for correct aspect ratio
            surface_w = screen_rect[2]
            surface_h = screen_rect[3]
        else:
            _gl_viewport = (0, 0, surface_w, surface_h)

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

        if self.blur_radius <= 0:
            # No spatial blur — tonemap directly
            target = current_fbo if use_temporal else target_fbo
            if target:
                target.use()
            else:
                _bind_default_framebuffer()
            self.ctx.viewport = _gl_viewport

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
                self.ctx.viewport = (0, 0, surface_w, surface_h)
            elif target_fbo:
                target_fbo.use()
                self.ctx.viewport = _gl_viewport
            else:
                _bind_default_framebuffer()
                self.ctx.viewport = _gl_viewport
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

            # Blit blend result to screen/target
            if target_fbo:
                target_fbo.use()
            else:
                _bind_default_framebuffer()
            self.ctx.viewport = _gl_viewport
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

    def snapshot_png(self, brightness: float = 6.0,
                     linear_mode: bool = False) -> bytes:
        """Tonemap the current histogram to an RGBA PNG and return as bytes.

        Uses a temporary FBO — does not affect screen output.

        Args:
            brightness: gamma parameter for tone mapping
            linear_mode: if True, skip log-density (for post-DE histograms)
        """
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
        if 'u_linear_mode' in p:
            p['u_linear_mode'] = 1 if linear_mode else 0

        self.quad_vao.render(moderngl.TRIANGLES)

        png = self.gpu.framebuffer_to_png_bytes(fbo, w, h)
        fbo.release()
        fbo_tex.release()
        return png

    def snapshot_de_png(self, brightness: float = 6.0,
                        max_radius: int = 9, curve: float = 0.4) -> bytes:
        """Render with density estimation, then tonemap to PNG.

        Applies spatially-varying DE blur to the histogram before
        tonemapping. Uses linear mode in tonemap since DE scatter
        already applies log-density scaling.
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

        png = self.gpu.framebuffer_to_png_bytes(fbo, w, h)
        fbo.release()
        fbo_tex.release()
        return png

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
