"""ChaosGame — Vulkan port of flame_sheep/rendering/chaos.py.

Owns all the buffers + pipelines needed to run the chaos game on Vulkan.
Standalone (unlike the GL version which holds a back-reference to
FlameRenderer's shared buffers) — the renderer port plugs this in
directly. One ChaosGame instance per canvas/genome configuration;
swap genomes via set_genome() without rebuilding pipelines.

Buffer layout matches flame_chaos.comp's binding order:
    0: histogram (uint, 2*n_pixels — [hits | colors])
    1: walkers (float, n_walkers * 3 — [x,y,c per walker])
    2: affines, 3: active_vars, 4: colors, 5: weights, 6: color_speeds
    7: transform_hits (per-transform per-pixel counts; for cluster scoring)
    8: post_affines, 9: pre_variations

The pipeline holds 4 ComputePipelines: clear, chaos, density_estimation,
reduce_max. downsample_hist isn't included here — it's part of the
tonemap path (different bindings).
"""
from __future__ import annotations

import re
import struct
from pathlib import Path

import logging

import numpy as np
import vulkan as vk

from viz_authoring.vk.pipeline import ComputePipeline
from . import pipeline_warm

log = logging.getLogger(__name__)

SHADER_DIR = Path(__file__).parent / 'shaders'

MAX_TRANSFORMS = 6      # matches flame_chaos.comp + flame_sheep/genome
MAX_ACTIVE_VARS = 8
SLOT_SIZE = 10          # bytes per variation slot (var_idx, weight, 8 params)
WORKGROUP_WALKERS = 64  # flame_chaos.comp local_size_x


def _symmetry_inject(src: str) -> str:
    """flame.comp's variations.glsl has a {{SYMMETRY_GROUPS}} placeholder
    that has to be expanded with the WALLPAPER_* / FRIEZE_* constant
    tables before the shader will compile."""
    from flame_sheep.variations._symmetry_groups import generate_glsl
    return src.replace('// {{SYMMETRY_GROUPS}}', generate_glsl())


_VAR_CASE_RE = re.compile(
    r'^[ \t]+case\s+(\d+):\s*return var_\w+\([^)]*\);\s*\n',
    re.MULTILINE,
)


def _trim_variations_switch(src: str, keep_vars: frozenset) -> str:
    """Strip `case N:` lines from apply_single_variation's switch for
    variations not in keep_vars. The `default: return p;` branch stays
    as a safety fallback. With the active-vars buffer's invariant that
    var_idx is always one of the genome's used variations when the
    switch is reached, trimming is safe — every reachable case is kept.

    Cuts the universal 127-case dispatcher down to typically 5-15 cases
    per genome. That's the lever Mesa's register allocator needed:
    pre-trim, the spill count on the full shader was 1929/3326; post-
    trim it should drop dramatically (most genomes have ≤15 unique vars
    which the spill-threshold table puts comfortably in SIMD8 no-spill
    territory)."""
    def maybe_drop(m):
        idx = int(m.group(1))
        return m.group(0) if idx in keep_vars else ''
    return _VAR_CASE_RE.sub(maybe_drop, src)


def _make_chaos_source_transform(keep_vars: frozenset,
                                    scoring_mode: bool = False):
    """Compose the symmetry-group expansion with the genome-specific
    variation-switch trim. Returned as a single source-transform
    function (the shape ComputePipeline expects).

    scoring_mode: when True, prepends `#define SCORING_MODE` so the
    chaos shader emits per-pixel per-transform hit counts into the
    transform_hits buffer. Wallpaper render path doesn't need this
    (extra atomic per inner-loop iter); scoring renders do.
    """
    def _xform(src: str) -> str:
        src = _symmetry_inject(src)
        src = _trim_variations_switch(src, keep_vars)
        if scoring_mode:
            # Prepend AFTER the #version line — GLSL requires #version
            # to come first.
            lines = src.split('\n', 1)
            src = lines[0] + '\n#define SCORING_MODE\n' + lines[1]
        return src
    return _xform


class ChaosGame:
    """Vulkan chaos-game runner. Allocate-once, reuse across frames /
    genome swaps."""

    def __init__(self, ctx, canvas_w: int, canvas_h: int,
                 n_walkers: int = 65536,
                 scoring_mode: bool = False):
        """scoring_mode: when True, compiled shaders emit per-pixel
        per-transform hit counts into the transform_hits buffer.
        Off by default — wallpaper render path doesn't need it and
        avoids the extra atomic per inner-loop iter. Headless scoring
        renders set this True so downstream metrics (cluster, symmetry,
        balance) can read transform_hits."""
        if n_walkers % WORKGROUP_WALKERS != 0:
            raise ValueError(
                f'n_walkers ({n_walkers}) must be divisible by '
                f'WORKGROUP_WALKERS ({WORKGROUP_WALKERS})')
        self.ctx = ctx
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.n_walkers = n_walkers
        self._scoring_mode = scoring_mode
        n_pixels = canvas_w * canvas_h

        # --- buffer allocation in flame_chaos.comp binding order ---
        T = MAX_TRANSFORMS + 1  # +1 slot for final xform
        SL = MAX_ACTIVE_VARS * SLOT_SIZE

        self.histogram = ctx.create_buffer(n_pixels * 2 * 4, 'storage')   # 0
        self.walkers = ctx.create_buffer(n_walkers * 3 * 4, 'storage')    # 1
        self.affines = ctx.create_buffer(T * 6 * 4, 'storage')             # 2
        self.active_vars = ctx.create_buffer(T * SL * 4, 'storage')       # 3
        self.colors = ctx.create_buffer(T * 4, 'storage')                  # 4
        self.weights = ctx.create_buffer(MAX_TRANSFORMS * 4, 'storage')   # 5
        self.color_speeds = ctx.create_buffer(T * 4, 'storage')           # 6
        self.transform_hits = ctx.create_buffer(
            n_pixels * MAX_TRANSFORMS * 4, 'storage')                      # 7
        self.post_affines = ctx.create_buffer(T * 6 * 4, 'storage')       # 8
        self.pre_vars = ctx.create_buffer(T * SL * 4, 'storage')          # 9

        # Initialize empty-genome state so a render without set_genome()
        # doesn't crash (e.g. pre-set_genome smoke tests).
        ctx.upload(self.weights, np.array([1.0] + [0.0]*(MAX_TRANSFORMS-1),
                                           dtype=np.float32))
        # Identity affines for all transforms
        identity = np.tile(np.array([1, 0, 0, 0, 1, 0], dtype=np.float32),
                            T).reshape(T, 6).flatten()
        ctx.upload(self.affines, identity)
        ctx.upload(self.post_affines, identity)
        # Active vars: transform 0 = linear (idx 0), rest empty
        av = np.full(T * SL, -1.0, dtype=np.float32)
        av[0] = 0.0  # var_idx
        av[1] = 1.0  # weight
        ctx.upload(self.active_vars, av)
        ctx.upload(self.pre_vars, np.full(T * SL, -1.0, dtype=np.float32))
        ctx.upload(self.colors, np.full(T, 0.5, dtype=np.float32))
        ctx.upload(self.color_speeds, np.full(T, 0.5, dtype=np.float32))
        ctx.zero_buffer(self.histogram)
        ctx.zero_buffer(self.transform_hits)
        self.reset_walkers()

        # --- pipelines ---
        # clear: 1 buffer (histogram), 12 bytes push (size, decay, offset)
        self._clear_pipe = ComputePipeline(
            ctx, SHADER_DIR / 'clear_histogram.comp',
            buffers=[self.histogram], push_constant_size=12)

        # chaos: spec-const variant — n_transforms, has_final_xform,
        # width, height are bound at pipeline-creation time so the driver
        # can constant-fold pick_transform's loop bound, dead-strip the
        # final-xform block when unused, and fold width/height literals
        # into world_to_pixel. Measured 2.3× cycle reduction vs the
        # unspec-const'd flame_chaos.comp on production genomes; see
        # tools/vk_perf_diag/measure_spec_const_threshold.py.
        #
        # width / height are fixed for this ChaosGame's lifetime;
        # n_transforms / has_final_xform vary per genome — so we cache
        # one pipeline per (n_transforms, has_final_xform) tuple and
        # swap on set_genome(). At most 6×2 = 12 distinct pipelines.
        self._chaos_pipes: dict = {}
        # Pre-build the default (1, 0, {0}) entry so callers can render
        # before calling set_genome() (smoke tests do this). var_idx=0
        # (linear) matches the empty-genome state set up above.
        self._chaos_pipe = self._get_chaos_pipe(1, 0, frozenset({0}))

        # density_estimation: separate input/output histograms — we use
        # a scratch buffer the same size as histogram and ping-pong.
        self.histogram_scratch = ctx.create_buffer(
            n_pixels * 2 * 4, 'storage')
        self._de_pipe = ComputePipeline(
            ctx, SHADER_DIR / 'density_estimation.comp',
            buffers=[self.histogram, self.histogram_scratch],
            push_constant_size=16)

        # reduce_max: histogram + max_buf
        self.max_buf = ctx.create_buffer(4, 'storage')
        self._reduce_pipe = ComputePipeline(
            ctx, SHADER_DIR / 'reduce_max.comp',
            buffers=[self.histogram, self.max_buf],
            push_constant_size=8)

        self._frame_seed = 0
        # Async-mode resources — allocated lazily on first
        # frame(synchronous=False) call. None for sync-only callers
        # (bench, scoring renders).
        self._async_cb = None
        self._async_fence = None

    def _get_chaos_pipe(self, n_transforms: int, has_final_xform: int,
                          keep_vars: frozenset):
        """Look up or build the chaos pipeline specialized for this
        genome's (n_transforms, has_final_xform, variation set).
        Pipelines are cached per-tuple — first-call cost is a shader
        compile + driver-side SPIR-V→ISA, subsequent calls are
        dict-lookup-cheap.

        keep_vars is the set of variation indices used anywhere in the
        genome (across all transforms' active_vars + pre_vars). The
        source transform strips the 127-case switch in variations.glsl
        down to only these cases. Universal shader has 1929:3326
        spill/fill ops; trimmed-to-~15-cases should land in the
        SIMD8 no-spill regime per measure_spill_threshold.py."""
        key = (int(n_transforms), int(has_final_xform),
               keep_vars)
        pipe = self._chaos_pipes.get(key)
        if pipe is None:
            # Log the spec-const set + variation count BEFORE compile —
            # the spill threshold per measure_spill_threshold.py says
            # ≤15 cases keeps SIMD8 no-spill; over that Mesa-Xe pays
            # 1929:3326 spill/fill ops vs 0 for the untrimmed shader.
            # If chaos perf is mysteriously slow with the warm-gate
            # active, this tells us whether the surviving pipeline is
            # actually narrow enough to escape spillage.
            import time as _time
            _t0 = _time.perf_counter()
            sorted_vars = sorted(keep_vars)
            log.info(
                f'[chaos-compile] compiling pipeline: '
                f'n_transforms={n_transforms} '
                f'has_final={has_final_xform} '
                f'|vars|={len(keep_vars)} '
                f'vars={sorted_vars} '
                f'(spill threshold ~15)')
            pipe = ComputePipeline(
                self.ctx, SHADER_DIR / 'flame_chaos_specconst.comp',
                buffers=[self.histogram, self.walkers, self.affines,
                         self.active_vars, self.colors, self.weights,
                         self.color_speeds, self.transform_hits,
                         self.post_affines, self.pre_vars],
                push_constant_size=40,  # see frame() for layout
                source_transform=_make_chaos_source_transform(
                    keep_vars, scoring_mode=self._scoring_mode),
                specialization={
                    0: key[0],          # SC_n_transforms
                    1: key[1],          # SC_has_final_xform
                    2: self.canvas_w,   # SC_width
                    3: self.canvas_h,   # SC_height
                })
            compile_ms = (_time.perf_counter() - _t0) * 1000
            log.info(
                f'[chaos-compile] done in {compile_ms:.0f}ms '
                f'(|vars|={len(keep_vars)}; '
                f'{"COLD" if compile_ms > 50 else "warm-cache hit"})')
            self._chaos_pipes[key] = pipe
            # Mark this key warm on disk so the wallpaper (or a future
            # session, or a peer process) can know the Mesa shader cache
            # has been populated and a re-create will be ~3ms instead of
            # 200-400ms. Both wallpaper + precompile_worker go through
            # this method, so the marker is written by whichever process
            # compiled first.
            pipeline_warm.mark_warm(n_transforms, has_final_xform,
                                      keep_vars)
        return pipe

    # ----- genome wiring ------------------------------------------------

    def set_genome(self, *,
                   affines: np.ndarray, post_affines: np.ndarray,
                   active_vars: np.ndarray, pre_vars: np.ndarray,
                   colors: np.ndarray, weights: np.ndarray,
                   color_speeds: np.ndarray,
                   n_transforms: int, has_final_xform: bool,
                   require_warm: bool = False) -> bool:
        """Upload a genome's buffer contents.

        Each array must have the shape the shader expects (e.g. affines
        = (MAX_TRANSFORMS+1, 6) float32). n_transforms and
        has_final_xform are remembered for later render_frame() calls.

        require_warm=True: if the (n_transforms, has_final_xform,
        variation-set) pipeline cache key is not already in-memory AND
        not marked warm on disk (see pipeline_warm), keep the CURRENT
        pipeline bound — but still upload all buffer contents. Caller's
        next render will use the previously-bound shader against the
        new affines/colors/etc. Rotation animates correctly; if the
        old and new variation sets differ, those frames render
        slightly off (transforms whose variation isn't in the old
        switch fall through to the default case) until precompile
        warms the new key.

        Why this split: render thread NEVER blocks on a cold compile
        (200-400ms), but rotation animation continues without freezing.
        Caller should still enqueue a precompile to get the right
        pipeline bound on a subsequent frame.

        Returns True iff the pipeline was actually swapped in (or
        already matched the current binding). False = cold-rejected;
        old pipeline still bound; new buffers uploaded.
        """
        T = MAX_TRANSFORMS + 1
        SL = MAX_ACTIVE_VARS * SLOT_SIZE

        def _check(name, arr, expected_shape):
            if arr.shape != expected_shape:
                raise ValueError(
                    f'{name}: expected shape {expected_shape}, got {arr.shape}')

        _check('affines', affines, (T, 6))
        _check('post_affines', post_affines, (T, 6))
        _check('active_vars', active_vars, (T, MAX_ACTIVE_VARS, SLOT_SIZE))
        _check('pre_vars', pre_vars, (T, MAX_ACTIVE_VARS, SLOT_SIZE))
        _check('colors', colors, (T,))
        _check('color_speeds', color_speeds, (T,))
        _check('weights', weights, (MAX_TRANSFORMS,))

        # Buffer uploads always happen — rotation animation lives
        # in the affine matrices (Genome.rotated() bakes phase into
        # each transform's 2x3 affine). Skipping uploads on a cold
        # pipeline swap would freeze rotation; we don't want that.
        self.ctx.upload(self.affines, affines.astype(np.float32).ravel())
        self.ctx.upload(self.post_affines,
                         post_affines.astype(np.float32).ravel())
        self.ctx.upload(self.active_vars,
                         active_vars.astype(np.float32).ravel())
        self.ctx.upload(self.pre_vars, pre_vars.astype(np.float32).ravel())
        self.ctx.upload(self.colors, colors.astype(np.float32))
        self.ctx.upload(self.color_speeds, color_speeds.astype(np.float32))
        self.ctx.upload(self.weights, weights.astype(np.float32))
        self._n_transforms = int(n_transforms)
        self._has_final_xform = 1 if has_final_xform else 0
        # Variation set = union of var_idx across active_vars + pre_vars,
        # all transforms. The buffers' first element of each slot is
        # var_idx (-1 = unused). We collect only var_idx >= 0; that's
        # exactly the set of cases the trimmed switch needs to keep.
        var_ids = np.concatenate([
            active_vars[..., 0].ravel(),
            pre_vars[..., 0].ravel(),
        ])
        keep_vars = frozenset(int(v) for v in var_ids if v >= 0)
        # Pipeline swap. If the key is already in our in-memory cache
        # this is dict-lookup-cheap. If it's a fresh key, the cost is
        # the SPIR-V→ISA compile (200-400ms cold, ~3ms warm via Mesa
        # cache). require_warm=True refuses the swap when both caches
        # would miss, keeping the render thread off cold compiles.
        key = (self._n_transforms, self._has_final_xform, keep_vars)
        if key in self._chaos_pipes:
            self._chaos_pipe = self._chaos_pipes[key]
            return True
        if require_warm and not pipeline_warm.is_warm(
                self._n_transforms, bool(self._has_final_xform),
                keep_vars):
            # Cold; keep previously-bound pipeline. Buffers already
            # uploaded above so rotation stays alive.
            return False
        self._chaos_pipe = self._get_chaos_pipe(
            self._n_transforms, self._has_final_xform, keep_vars)
        return True

    def reset_walkers(self, seed: int | None = None):
        """Random walker positions in [-1, 1]. Call after a genome swap
        so walkers from the prior genome's attractor don't pollute the
        new one."""
        rng = np.random.default_rng(seed)
        data = rng.uniform(-1, 1, (self.n_walkers, 3)).astype(np.float32)
        self.ctx.upload(self.walkers, data)

    # ----- per-frame ops ------------------------------------------------

    def clear_histogram(self, decay: float = 0.0):
        """Zero (decay=0) or decay (multiply by decay/256) the histogram.
        Always clears the whole buffer from offset 0 — compare-mode
        handling will come later."""
        n_pixels = self.canvas_w * self.canvas_h
        size = n_pixels * 2  # hits + colors regions
        push = struct.pack('3I', size, int(decay * 256), 0)
        groups = (size + 63) // 64
        self.ctx.dispatch_compute(self._clear_pipe, groups_x=groups,
                                    push_constants=push)

    def render_frame(self, iterations: int = 30,
                     zoom: tuple[float, float] = (1.0, 1.0),
                     rotation: float = 0.0,
                     center: tuple[float, float] = (0.0, 0.0)):
        """One chaos-game pass: n_walkers walkers, each running
        `iterations` steps. Histogram is mutated in place (caller
        should clear_histogram() first if they want a fresh frame)."""
        import math
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)
        n_pixels = self.canvas_w * self.canvas_h
        # Push layout matches flame_chaos_specconst.comp's PushConstants:
        #   vec2 zoom; float cos, sin; vec2 center;
        #   int iters; uint seed, offset, stride
        # n_transforms / has_final_xform / w / h moved to specialization
        # constants (driver constant-folds at pipeline-creation time).
        push = struct.pack(
            '2f 2f 2f i I 2I',
            zoom[0], zoom[1],
            cos_r, sin_r,
            center[0], center[1],
            iterations,
            self._frame_seed, 0, n_pixels,
        )
        self._frame_seed += 1
        groups = self.n_walkers // WORKGROUP_WALKERS
        self.ctx.dispatch_compute(self._chaos_pipe, groups_x=groups,
                                    push_constants=push)

    def density_estimate(self, max_radius: int = 5, curve: float = 0.5):
        """Adaptive density-estimation post-process. Writes scattered
        result to histogram_scratch, then swaps buffers so subsequent
        ops see the smoothed histogram.

        Note: ping-pong via swap means histogram_scratch holds the OLD
        unfiltered histogram after this call. clear_histogram() targets
        the live histogram, so it'll clear the smoothed-result side —
        callers running DE every frame want to clear scratch instead
        (TODO when needed)."""
        self.ctx.zero_buffer(self.histogram_scratch)
        push = struct.pack('3if', self.canvas_w, self.canvas_h,
                            int(max_radius), float(curve))
        gx = (self.canvas_w + 15) // 16
        gy = (self.canvas_h + 15) // 16
        self.ctx.dispatch_compute(self._de_pipe, groups_x=gx, groups_y=gy,
                                    push_constants=push)
        # Swap so the live histogram is the smoothed one.
        self.histogram, self.histogram_scratch = (
            self.histogram_scratch, self.histogram)

    def reduce_max_hits(self) -> int:
        """Compute the max value across the hit-count region (first
        n_pixels of the histogram). Returns the value. Used by tonemap
        for brightness normalization."""
        self.ctx.zero_buffer(self.max_buf)
        n_pixels = self.canvas_w * self.canvas_h
        push = struct.pack('2I', n_pixels, 0)
        groups = (n_pixels + 255) // 256
        self.ctx.dispatch_compute(self._reduce_pipe, groups_x=groups,
                                    push_constants=push)
        return int(self.ctx.download(self.max_buf, np.uint32, 1)[0])

    # -----------------------------------------------------------------
    # Per-frame batched dispatch — clear + chaos + reduce_max in ONE
    # command buffer with memory barriers between. Per-call cost was
    # 3× fence-wait round-trips with the unbatched version, which on
    # the wallpaper's busy frame pattern measured ~40 ms (vs the GL
    # backend's much lower per-frame cost). Batching collapses the
    # round-trips to one.
    #
    # We don't read max_buf from the GPU after this returns — the
    # tonemap fragment shader reads it directly via its descriptor
    # set, no CPU readback needed during a frame. The
    # reduce_max_hits() method above is retained for callers that
    # actually want the value back on the CPU.
    # -----------------------------------------------------------------

    def frame(self, iterations: int = 30,
              zoom: tuple[float, float] = (1.0, 1.0),
              rotation: float = 0.0,
              center: tuple[float, float] = (0.0, 0.0),
              decay: float = 0.0,
              synchronous: bool = True):
        """Run the per-frame chaos-game cycle in a single command-buffer
        submit:
          1. clear / decay the histogram
          2. n_walkers × `iterations` chaos game steps
          3. reduce the histogram to a single max value in max_buf
        Memory barriers between dispatches make the writes from the
        prior dispatch visible to the next.

        synchronous=True (default): the call vkWaitForFences before
        returning. Bench script depends on this for accurate per-call
        timing — each invocation measures complete GPU work.

        synchronous=False: submit and return immediately. The CPU is
        at most 1 frame ahead — we wait on the PREVIOUS call's fence
        at the START of this call (so the cb is safe to reuse).
        Caller is responsible for ensuring anything downstream that
        reads the histogram (e.g. tonemap pipelines) issues an
        explicit memory barrier from COMPUTE_SHADER/SHADER_WRITE to
        its read stage — the fence-implicit ordering goes away in
        async mode. Async halves per-frame fence overhead and lets
        the GPU run chaos + tonemap back-to-back.
        """
        import cffi
        import math
        ffi = cffi.FFI()

        n_pixels = self.canvas_w * self.canvas_h
        size = n_pixels * 2
        # ---- precompute per-dispatch push constants -----------------
        clear_push = struct.pack('3I', size, int(decay * 256), 0)
        cos_r = math.cos(rotation); sin_r = math.sin(rotation)
        # See render_frame() for layout; n_transforms / has_final_xform
        # / w / h are spec consts on the current pipeline.
        chaos_push = struct.pack(
            '2f 2f 2f i I 2I',
            zoom[0], zoom[1],
            cos_r, sin_r,
            center[0], center[1],
            iterations,
            self._frame_seed, 0, n_pixels,
        )
        self._frame_seed += 1
        reduce_push = struct.pack('2I', n_pixels, 0)

        # ---- the actual command buffer ------------------------------
        # Sync: fresh cb + fence each call (caller takes the wait).
        # Async: reuse persistent cb + fence; wait on previous frame's
        # fence to ensure the cb is safe to overwrite.
        if synchronous:
            cb = self.ctx.allocate_command_buffers(1)[0]
            fence = vk.vkCreateFence(
                self.ctx.device, vk.VkFenceCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO), None)
        else:
            if self._async_fence is None:
                self._async_cb = self.ctx.allocate_command_buffers(1)[0]
                self._async_fence = vk.vkCreateFence(
                    self.ctx.device, vk.VkFenceCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                        flags=vk.VK_FENCE_CREATE_SIGNALED_BIT),  # so first
                                                                    # wait is
                                                                    # a no-op
                    None)
            # Wait on previous frame's fence — CPU is at most 1 frame ahead.
            vk.vkWaitForFences(self.ctx.device, 1, [self._async_fence],
                                vk.VK_TRUE, 0xFFFFFFFFFFFFFFFF)
            vk.vkResetFences(self.ctx.device, 1, [self._async_fence])
            vk.vkResetCommandBuffer(self._async_cb, 0)
            cb = self._async_cb
            fence = self._async_fence
        vk.vkBeginCommandBuffer(cb, vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        ))

        # In async mode, the previous frame's per-output tonemap may
        # still be reading histogram + max_buf when this frame's
        # writes start. Submission-order does NOT imply
        # execution-order on a Vulkan queue — that's exactly what
        # this barrier provides: chaos's writes (compute) and
        # max_buf's fill (transfer, below) are deferred until any
        # prior FRAGMENT_SHADER reads (the tonemap pipelines) have
        # completed. In sync mode this is a no-op (the prior frame's
        # chaos.frame() already fence-waited so no in-flight work
        # exists).
        prev_reads_done = vk.VkMemoryBarrier(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            srcAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
            dstAccessMask=(vk.VK_ACCESS_SHADER_WRITE_BIT
                            | vk.VK_ACCESS_TRANSFER_WRITE_BIT),
        )
        vk.vkCmdPipelineBarrier(
            cb,
            vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            (vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
             | vk.VK_PIPELINE_STAGE_TRANSFER_BIT),
            0, 1, [prev_reads_done], 0, None, 0, None)

        def _bind_and_dispatch(pipeline, push_bytes, gx, gy=1, gz=1):
            vk.vkCmdBindPipeline(cb, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline.pipeline)
            vk.vkCmdBindDescriptorSets(
                cb, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline.layout, 0, 1, [pipeline.descriptor_set], 0, None)
            pc = ffi.new('char[]', push_bytes)
            vk.vkCmdPushConstants(
                cb, pipeline.layout,
                vk.VK_SHADER_STAGE_COMPUTE_BIT,
                0, len(push_bytes), ffi.cast('void*', pc))
            vk.vkCmdDispatch(cb, gx, gy, gz)

        def _barrier():
            # Global memory barrier between compute dispatches —
            # makes prior shader writes visible before next reads.
            mb = vk.VkMemoryBarrier(
                sType=vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
                dstAccessMask=(vk.VK_ACCESS_SHADER_READ_BIT
                                | vk.VK_ACCESS_SHADER_WRITE_BIT),
            )
            vk.vkCmdPipelineBarrier(
                cb,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, [mb], 0, None, 0, None)

        # 1. clear / decay
        _bind_and_dispatch(self._clear_pipe, clear_push,
                            gx=(size + 63) // 64)
        _barrier()

        # 2. chaos game
        _bind_and_dispatch(self._chaos_pipe, chaos_push,
                            gx=self.n_walkers // WORKGROUP_WALKERS)
        _barrier()

        # 3. reduce max. The shader does atomicMax into max_buf so the
        # buffer must be zeroed first. Use vkCmdFillBuffer (a
        # GPU-side transfer op) instead of the previous host-side
        # zero — host-zero races against the previous frame's tonemap
        # still reading max_buf via FRAGMENT_SHADER (visible as torn-
        # pixel artifacts in async mode; latent in sync mode too).
        # Barrier after the fill so the reduce dispatch reads zeros.
        vk.vkCmdFillBuffer(cb, self.max_buf.buffer, 0,
                            self.max_buf.size, 0)
        fill_to_compute = vk.VkMemoryBarrier(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT
                            | vk.VK_ACCESS_SHADER_WRITE_BIT,
        )
        vk.vkCmdPipelineBarrier(
            cb,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, [fill_to_compute], 0, None, 0, None)
        _bind_and_dispatch(self._reduce_pipe, reduce_push,
                            gx=(n_pixels + 255) // 256)

        vk.vkEndCommandBuffer(cb)

        # ---- submit ------------------------------------------------
        submit = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1, pCommandBuffers=[cb],
        )
        vk.vkQueueSubmit(self.ctx.graphics_queue, 1, [submit], fence)
        if synchronous:
            # Sync: wait + free this call's resources.
            vk.vkWaitForFences(self.ctx.device, 1, [fence], vk.VK_TRUE,
                                0xFFFFFFFFFFFFFFFF)
            vk.vkDestroyFence(self.ctx.device, fence, None)
            vk.vkFreeCommandBuffers(
                self.ctx.device, self.ctx.command_pool, 1, [cb])
        # else: async — fence will be waited on at the start of the
        # next frame() call; cb is persistent (self._async_cb).

    # ----- readback -----------------------------------------------------

    def download_transform_hits(self) -> np.ndarray:
        """Returns (H, W, MAX_TRANSFORMS) uint32 — per-pixel per-transform
        hit counts. Used by the offline scorer to compute cluster /
        symmetry / balance metrics that need per-transform attribution."""
        n_pixels = self.canvas_w * self.canvas_h
        flat = self.ctx.download(self.transform_hits, np.uint32,
                                  n_pixels * MAX_TRANSFORMS)
        return flat.reshape(self.canvas_h, self.canvas_w, MAX_TRANSFORMS)

    def clear_transform_hits(self) -> None:
        """Zero the transform_hits buffer. Called between scoring
        renders that need a clean per-pixel attribution map."""
        self.ctx.zero_buffer(self.transform_hits)

    def download_histogram(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (hits, color_acc) — each (H, W) uint32. Both arrays
        share the same indexing: hits[y, x] = histogram[y*W + x],
        color_acc[y, x] = histogram[n_pixels + y*W + x].
        Color_idx for a pixel = color_acc / (hits * COLOR_SCALE)."""
        n_pixels = self.canvas_w * self.canvas_h
        flat = self.ctx.download(self.histogram, np.uint32, n_pixels * 2)
        hits = flat[:n_pixels].reshape(self.canvas_h, self.canvas_w)
        color_acc = flat[n_pixels:].reshape(self.canvas_h, self.canvas_w)
        return hits, color_acc

    def cleanup(self):
        # Wait for any in-flight async submit before destroying its
        # fence — otherwise vkDestroyFence is undefined behavior.
        if self._async_fence is not None:
            vk.vkWaitForFences(self.ctx.device, 1, [self._async_fence],
                                vk.VK_TRUE, 0xFFFFFFFFFFFFFFFF)
            vk.vkDestroyFence(self.ctx.device, self._async_fence, None)
            vk.vkFreeCommandBuffers(
                self.ctx.device, self.ctx.command_pool, 1,
                [self._async_cb])
            self._async_fence = None
            self._async_cb = None
        for p in (self._clear_pipe, self._de_pipe, self._reduce_pipe):
            p.cleanup()
        for p in self._chaos_pipes.values():
            p.cleanup()
        self._chaos_pipes.clear()
        # buffers get auto-cleaned by ctx.cleanup()
