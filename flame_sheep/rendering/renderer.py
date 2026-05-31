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

import logging
import re
import time as _time

import moderngl
import numpy as np
from pathlib import Path

log = logging.getLogger(__name__)


# Matches one `case N: return var_<name>(...);` line in variations.glsl's
# apply_single_variation switch. Mirror of viz_authoring.vk.chaos_game's
# _VAR_CASE_RE — keep them in sync if either side changes the switch
# format.
_VAR_CASE_RE = re.compile(
    r'^[ \t]+case\s+(\d+):\s*return var_\w+\([^)]*\);\s*\n',
    re.MULTILINE,
)


def _trim_variations_switch(src: str, keep_vars: frozenset[int]) -> str:
    """Strip `case N:` lines from apply_single_variation's switch for
    variations not in keep_vars. The `default:` branch stays.

    Cuts a 127-case dispatcher down to typically 5-15 cases per genome.
    Mesa-Iris register-allocates better on the shorter switch (~1.5-1.8x
    gpu_chaos reduction measured). Same lever as Mesa-Xe Vk's spec-const
    trim, less drastic effect (GL compiler wasn't spilling like Vk's
    was, but still benefits from tighter code)."""
    def _drop(m: re.Match[str]) -> str:
        return m.group(0) if int(m.group(1)) in keep_vars else ''
    return _VAR_CASE_RE.sub(_drop, src)

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

        # Sub-components — see docs/reorg_plan.md Stage 0. Each unit
        # owns a coherent slice of FlameRenderer's behavior; they hold
        # back-refs to this renderer for shared resource access.
        from .chaos import ChaosGame
        from .tonemap import TonemapPipeline
        from .snapshot import SnapshotPipeline
        self.chaos = ChaosGame(self)
        self.tonemap = TonemapPipeline(self)
        self.snapshot = SnapshotPipeline(self)

    @property
    def canvas_w(self) -> int:
        return self.gpu.canvas_w

    @property
    def canvas_h(self) -> int:
        return self.gpu.canvas_h

    def _load_shaders(self, scoring: bool = False) -> None:
        from ..variations._symmetry_groups import generate_glsl

        def _inject_symmetry(src: str) -> str:
            return src.replace('// {{SYMMETRY_GROUPS}}', generate_glsl())

        flame_defines = {'SCORING_MODE': ''} if scoring else None
        # Stash for the per-genome trim path — every cached shader
        # reuses the same defines + symmetry injection, only the
        # case-trimming changes per genome.
        self._flame_defines = flame_defines
        self._inject_symmetry = _inject_symmetry
        self.compute_shader = self.gpu.compile_compute_shader(
            SHADER_DIR / 'flame.comp',
            defines=flame_defines,
            source_transform=_inject_symmetry,
        )
        # Per-genome trimmed-shader cache. Keys = frozenset of
        # variation indices the genome actually uses. The 127-case
        # switch in variations.glsl gets stripped down to just the
        # cases the genome reaches — Mesa-Iris register-allocates
        # better on the shorter switch, giving ~1.5-1.8x gpu_chaos
        # reduction in measurements (see tag experiment/gl-chaos-trim
        # and task #53 for methodology). The universal compile above
        # serves as a fallback for the very first upload before any
        # cache entries exist + for codepaths that don't call
        # upload_genome (test_pattern, smoke tests).
        self._chaos_shader_cache: dict[frozenset, "moderngl.ComputeShader"] = {}
        self._chaos_universal_shader = self.compute_shader
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

    @staticmethod
    def _extract_keep_vars(gpu: dict) -> frozenset[int]:
        """Genome → set of variation indices the chaos shader's switch
        needs to keep. Shared between render-time upload_genome (which
        has gpu arrays in hand) and the precompile worker (which also
        derives keep_vars from a genome's gpu arrays for tuple match)."""
        av = gpu['active_vars'].reshape(7, 8, 10)
        pv = gpu['pre_active_vars'].reshape(7, 8, 10)
        var_ids = np.concatenate([av[..., 0].ravel(), pv[..., 0].ravel()])
        return frozenset(int(v) for v in var_ids if v >= 0)

    def is_shader_warm(self, keep_vars: frozenset[int]) -> bool:
        """Predicate for the GenomeAxis upstream gate. True iff the
        chaos shader for this variation set will swap in fast — either
        already in our in-memory cache, or disk-warm via Mesa's
        implicit cache (a marker file means we OR the precompile
        worker compiled it at least once this filesystem-lifetime).
        """
        if keep_vars in self._chaos_shader_cache:
            return True
        # Reuse the cross-process warm-marker convention from the Vk
        # side. The marker is just "this key has been compiled to
        # Mesa's cache by SOMEONE" — backend-agnostic.
        try:
            from viz_authoring.vk import pipeline_warm
            # n_transforms + has_final_xform don't change the GL shader
            # (only keep_vars does), so synthesize neutral values for
            # the marker key. The marker is informational; misses just
            # cause an inline compile, which produces a hit anyway.
            return pipeline_warm.is_warm(0, False, keep_vars)
        except Exception:
            return False

    def _get_chaos_shader_for_genome(self, gpu: dict
                                      ) -> "moderngl.ComputeShader":
        """Render-thread entry: extract keep_vars from gpu arrays,
        then delegate to _get_chaos_shader_for_keep_vars."""
        return self._get_chaos_shader_for_keep_vars(
            self._extract_keep_vars(gpu))

    def _get_chaos_shader_for_keep_vars(self, keep_vars: frozenset[int]
                                          ) -> "moderngl.ComputeShader":
        """Compile-or-cache lookup keyed on the variation set.

        Trimming the universal 127-case switch in variations.glsl down
        to just the cases the genome uses tightens the compiled
        program. Measured ~1.5-1.8× gpu_chaos reduction on Mesa-Iris
        Arc A770 (see experiment/gl-chaos-trim tag).

        Cold compile is blocking and ~600ms. Steady state is dict
        lookup. The precompile worker (flame_sheep.scheduler.
        precompile_worker_gl) runs this method in a separate process
        to pre-warm Mesa's GL shader cache for upcoming genomes — so
        when the wallpaper hits the same key, the compile completes
        in ~5ms (cache hit) instead of 600ms (cold).
        """
        cached = self._chaos_shader_cache.get(keep_vars)
        if cached is not None:
            return cached
        _t0 = _time.perf_counter()
        def _transform(src: str) -> str:
            src = self._inject_symmetry(src)
            return _trim_variations_switch(src, keep_vars)
        prog = self.gpu.compile_compute_shader(
            SHADER_DIR / 'flame.comp',
            defines=self._flame_defines,
            source_transform=_transform,
        )
        self._chaos_shader_cache[keep_vars] = prog
        dt_ms = (_time.perf_counter() - _t0) * 1000
        log.info(
            f'[gl-chaos-trim] compiled shader for '
            f'|vars|={len(keep_vars)} in {dt_ms:.0f}ms '
            f'(cache size={len(self._chaos_shader_cache)})')
        # Mark warm so peer processes (precompile worker, future
        # FlameRenderer instances) know this compile populated Mesa's
        # implicit cache.
        try:
            from viz_authoring.vk import pipeline_warm
            pipeline_warm.mark_warm(0, False, keep_vars)
        except Exception:
            pass  # marker write is informational; failure non-fatal
        return prog

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

        # Blur strength (adjusted at runtime). Per-size blur FBO cache
        # lives on self.tonemap (created in __init__ after this method).
        self.blur_radius = 1.0
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
        return self.chaos.reset_walkers()

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

        # Select the per-genome trimmed compute shader (compiling on
        # first sight of a new keep_vars). The uniform writes below
        # all target self.compute_shader, so the swap MUST happen
        # before the `cs = self.compute_shader` line. Render thread
        # blocks here on cold compile (~600ms on Arc per measurement);
        # commits 3 and 4 of the GL revert plan move this off the
        # render thread via KHR_parallel_shader_compile +
        # ARB_get_program_binary cache.
        self.compute_shader = self._get_chaos_shader_for_genome(gpu)

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
        return self.chaos.ensure_double_histogram()

    def set_histogram_offset(self, offset: int) -> None:
        return self.chaos.set_histogram_offset(offset)

    def clear_histogram(self, decay: float = 0.0) -> None:
        return self.chaos.clear_histogram(decay)

    def clear_transform_hits(self) -> None:
        return self.chaos.clear_transform_hits()

    def bind_buffers(self) -> None:
        return self.chaos.bind_buffers()

    def dispatch_chaos_game(self, iterations: int = N_ITERS) -> None:
        return self.chaos.dispatch(iterations)

    def reduce_histogram_max(self) -> None:
        return self.chaos.reduce_histogram_max()

    def apply_density_estimation(self, max_radius: int = 9,
                                  curve: float = 0.4,
                                  min_density: float = 0.0) -> None:
        return self.snapshot.apply_density_estimation(
            max_radius=max_radius, curve=curve, min_density=min_density)

    def render_tonemap(self, viewport: Viewport, surface_w: int, surface_h: int,
                       brightness: float = 6.0, dt: float = 1/60,
                       screen_rect: tuple[int, int, int, int] | None = None,
                       target_fbo: 'moderngl.Framebuffer | None' = None) -> None:
        return self.tonemap.render(viewport, surface_w, surface_h,
                                   brightness=brightness, dt=dt,
                                   screen_rect=screen_rect, target_fbo=target_fbo)

    def snapshot_png(self, brightness: float = 6.0,
                     linear_mode: bool = False) -> bytes:
        return self.snapshot.png(brightness=brightness, linear_mode=linear_mode)

    def snapshot_de_png(self, brightness: float = 6.0,
                        max_radius: int = 9, curve: float = 0.4) -> bytes:
        return self.snapshot.de_png(brightness=brightness,
                                    max_radius=max_radius, curve=curve)

    def snapshot_flam3_png(self, brightness: float = 4.0, gamma: float = 4.0,
                           vibrancy: float = 1.0, contrast: float = 1.0,
                           sample_density: float = 1.0,
                           highlight_power: float = -1.0) -> bytes:
        return self.snapshot.flam3_png(brightness=brightness, gamma=gamma,
                                       vibrancy=vibrancy, contrast=contrast,
                                       sample_density=sample_density,
                                       highlight_power=highlight_power)


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
