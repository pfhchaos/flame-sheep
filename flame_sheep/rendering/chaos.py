"""ChaosGame — flame-fractal chaos game dispatch + histogram management.

Owns the per-frame compute-shader dispatch that runs `n_walkers` walkers
through `iterations` steps each, accumulating hits into the histogram
SSBO. Also owns histogram side-effects: clear/decay, dual-mode
offsetting for compare mode, walker re-randomization, and the
max-reduction that feeds the tonemap brightness divisor.

Sub-component of FlameRenderer. Holds a back-reference (`self._r`) for
shared resource access — doesn't own the buffers itself (FlameRenderer
allocates them at __init__ and exposes them as attributes). The split
is for clarity-of-purpose: this module is "everything that mutates the
histogram via compute dispatch."

Stage 0 of `docs/reorg_plan.md`: separate FlameRenderer's units so
each one passes the "describe in one sentence" test. This unit is
"runs the chaos game and manages the histogram buffer it writes into."
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..genome import MAX_TRANSFORMS

if TYPE_CHECKING:
    from .renderer import FlameRenderer


class ChaosGame:
    """Chaos-game dispatch + histogram-buffer ops, backed by a
    FlameRenderer's shared resources (compute_shader, clear_shader,
    reduce_max_shader, histogram_buf, walker_buf, transform_hits_buf,
    _max_buf, and the genome-data SSBOs)."""

    def __init__(self, renderer: 'FlameRenderer') -> None:
        self._r = renderer

    def reset_walkers(self) -> None:
        """Re-randomize walker positions. Call after a genome swap to
        avoid stuck walkers that escaped to infinity under a degenerate
        genome."""
        r = self._r
        walker_data = np.random.uniform(-1, 1, (r.n_walkers, 3)).astype(np.float32)
        r.walker_buf.write(walker_data.tobytes())

    def ensure_double_histogram(self) -> None:
        """Double the histogram buffer for compare mode.

        Layout: [hits_L(n_px), hits_R(n_px), colors_L(n_px), colors_R(n_px)]
        Stride (hits → colors) = 2*n_px instead of n_px.
        Left offset = 0, right offset = n_px.
        """
        r = self._r
        n_pixels = r.canvas_w * r.canvas_h
        if r.histogram_buf.size >= n_pixels * 4 * 4:
            return  # already doubled
        new_buf = r.gpu.allocate_storage_buffer(n_pixels * 4 * 4)  # uint32 zero-init
        new_buf.bind_to_storage_buffer(0)
        r.histogram_buf = new_buf
        # Update stride to account for doubled hits region. Routes
        # through the sticky-uniform mechanism so the per-genome
        # shader cache re-applies it on every swap.
        stride = n_pixels * 2
        r._hist_stride = stride
        r._set_sticky_chaos_uniform('u_hist_stride', stride)
        # Non-swapped programs get set directly — they have one
        # instance for the renderer's whole lifetime.
        r.clear_shader['u_hist_offset'] = 0
        r.reduce_max_shader['u_hist_offset'] = 0
        r.tonemap_program['u_hist_stride'] = stride

    def set_histogram_offset(self, offset: int) -> None:
        """Set histogram offset for compare mode (0 = left/normal,
        n_pixels = right)."""
        r = self._r
        r._hist_offset = offset
        r._set_sticky_chaos_uniform('u_hist_offset', offset)
        # Non-swapped programs get set directly.
        r.clear_shader['u_hist_offset'] = offset
        r.reduce_max_shader['u_hist_offset'] = offset
        r.tonemap_program['u_hist_offset'] = offset

    def clear_histogram(self, decay: float = 0.0) -> None:
        """Clear or decay the histogram.

        decay: 0.0 = full clear, 0.9 = keep 90% of previous frame's hits.

        Clears the entire allocated histogram in one dispatch. In single
        mode the buffer holds [hits | colors] = 2*n_px uint32s; in dual
        mode (compare) it holds [hits_L | hits_R | colors_L | colors_R]
        = 4*n_px uint32s. Caller-set u_hist_offset is ignored — we always
        clear from index 0 across the full buffer.
        """
        r = self._r
        r._decay = decay
        n_px = r.canvas_w * r.canvas_h
        is_dual = r.histogram_buf.size >= n_px * 4 * 4
        size = n_px * 4 if is_dual else n_px * 2
        r.clear_shader['u_size'] = size
        r.clear_shader['u_hist_offset'] = 0
        r.clear_shader['u_decay_num'] = int(decay * 256)
        groups = (size + 63) // 64
        r.clear_shader.run(group_x=groups)
        r.ctx.memory_barrier()

    def clear_transform_hits(self) -> None:
        """Zero the per-transform hit counts buffer."""
        r = self._r
        n = r.canvas_w * r.canvas_h * MAX_TRANSFORMS
        r.transform_hits_buf.write(np.zeros(n, dtype=np.uint32).tobytes())
        r.ctx.memory_barrier()

    def bind_buffers(self) -> None:
        """Rebind all SSBOs to their binding points.

        Required when multiple FlameRenderer instances share a GL context
        (e.g. compare mode), since bind_to_storage_buffer is global state.
        """
        r = self._r
        r.histogram_buf.bind_to_storage_buffer(0)
        r.walker_buf.bind_to_storage_buffer(1)
        r.affines_buf.bind_to_storage_buffer(2)
        r.active_vars_buf.bind_to_storage_buffer(3)
        r.colors_buf.bind_to_storage_buffer(4)
        r.weights_buf.bind_to_storage_buffer(5)
        r.color_speeds_buf.bind_to_storage_buffer(6)
        r.transform_hits_buf.bind_to_storage_buffer(7)
        r._max_buf.bind_to_storage_buffer(8)
        r.post_affines_buf.bind_to_storage_buffer(9)
        r.pre_vars_buf.bind_to_storage_buffer(10)

    def dispatch(self, iterations: int) -> None:
        """Run one chaos-game pass: n_walkers walkers, each doing
        `iterations` plot iterations, accumulating into the histogram."""
        r = self._r
        r.compute_shader['u_iterations'] = iterations
        r.compute_shader['u_rng_seed'] = r._rng_frame_counter
        r._rng_frame_counter += 1
        groups = r.n_walkers // 64
        r.compute_shader.run(group_x=groups)

    def reduce_histogram_max(self) -> None:
        """Compute max hit count from histogram via GPU reduction.
        Result is stored in _max_buf SSBO (binding=8) for tonemap to read."""
        r = self._r
        n_pixels = r.canvas_w * r.canvas_h
        # Reset max to 0 before reduction
        r._max_buf.write(np.zeros(1, dtype=np.uint32).tobytes())
        r.reduce_max_shader['u_n_pixels'] = n_pixels
        groups = (n_pixels + 255) // 256
        r.reduce_max_shader.run(group_x=groups)
        r.ctx.memory_barrier()
