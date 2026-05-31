"""Property test for FlameRenderer's sticky-chaos-uniforms mechanism.

The per-genome trimmed-shader cache swaps `self.compute_shader` to
different moderngl Program objects each time the genome changes its
variation set. Each Program has independent uniform storage — values
set on program A don't transfer to program B. Compare mode hit this
twice (u_hist_offset, u_hist_stride) as silent half-blank rendering
because the swapped-in program kept its default uniform values.

The fix: FlameRenderer.set_histogram_offset() and similar APIs go
through _set_sticky_chaos_uniform() which updates a dict on the
renderer; _sync_chaos_program_uniforms() re-applies the dict to any
swapped-in program. Adding a new sticky uniform should "just work"
without per-uniform sync code.

This test asserts the property — set a sticky uniform, swap to a
different per-genome shader via upload_genome, verify the new
active program reads back the sticky value.

Requires a GL context (moderngl standalone). Skipped if unavailable.
"""
from __future__ import annotations

import pytest

try:
    import moderngl
    _ctx = moderngl.create_standalone_context()
    HAS_GL = True
except Exception:
    _ctx = None
    HAS_GL = False


pytestmark = pytest.mark.skipif(not HAS_GL,
                                 reason='No standalone moderngl context')


def _make_renderer():
    from flame_sheep.rendering import FlameRenderer, GpuContext
    # Small canvas — we never dispatch, just compile.
    gpu = GpuContext(_ctx, canvas_w=64, canvas_h=64, ppmm=1.0)
    return FlameRenderer(gpu)


def _genome_with_keep_vars(var_idx: int):
    """Build a minimal Genome that hits one specific variation slot,
    so its keep_vars frozenset differs from the universal startup
    shader's. upload_genome will compile a new trimmed program for
    it — that's the shader-swap event the test exercises."""
    from flame_sheep.genome import Genome, Transform
    import numpy as np
    from flame_sheep.variations import NUM_VARIATIONS

    g = Genome()
    tr = Transform()
    tr.affine = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    tr.variations = np.zeros(NUM_VARIATIONS, dtype=np.float32)
    tr.variations[var_idx] = 1.0
    tr.color = 0.5
    tr.weight = 1.0
    g.transforms = [tr]
    g.palette = np.linspace(0, 1, 256 * 3, dtype=np.float32).reshape(256, 3)
    g.zoom = 1.0
    g.rotation = 0.0
    g.center = np.array([0.0, 0.0], dtype=np.float32)
    return g


def test_sticky_uniforms_persist_across_shader_swap():
    """After set_histogram_offset + ensure_double_histogram + a
    genome upload that triggers a per-genome shader compile, the
    new active compute_shader must have the same uniform values
    the renderer was told."""
    r = _make_renderer()
    # Both sticky uniforms set to non-default values.
    r.ensure_double_histogram()           # sets u_hist_stride = 2 * canvas
    r.set_histogram_offset(123)           # sets u_hist_offset = 123
    assert r._sticky_chaos_uniforms['u_hist_offset'] == 123
    n_px = r.canvas_w * r.canvas_h
    assert r._sticky_chaos_uniforms['u_hist_stride'] == 2 * n_px

    # Force a shader swap by uploading a genome whose keep_vars is
    # different from whatever's currently active. Variation index
    # doesn't matter as long as it triggers a fresh compile.
    g = _genome_with_keep_vars(var_idx=5)  # SWIRL — common, cheap to compile
    r.upload_genome(g)

    # New active compute_shader must carry the sticky uniform values.
    # moderngl exposes uniforms via Uniform.value.
    assert r.compute_shader['u_hist_offset'].value == 123, (
        'u_hist_offset did not propagate to the swapped-in shader')
    assert r.compute_shader['u_hist_stride'].value == 2 * n_px, (
        'u_hist_stride did not propagate to the swapped-in shader')


def test_sticky_uniform_dict_is_source_of_truth():
    """_set_sticky_chaos_uniform updates the dict AND the current
    program; later swaps read from the dict. Test that the dict
    matches the current program after each set."""
    r = _make_renderer()
    r._set_sticky_chaos_uniform('u_hist_offset', 42)
    assert r._sticky_chaos_uniforms['u_hist_offset'] == 42
    assert r.compute_shader['u_hist_offset'].value == 42

    # Update and verify both move together.
    r._set_sticky_chaos_uniform('u_hist_offset', 7)
    assert r._sticky_chaos_uniforms['u_hist_offset'] == 7
    assert r.compute_shader['u_hist_offset'].value == 7


def test_unknown_sticky_uniform_does_not_crash():
    """Sticky-uniforms dict entries that don't exist in the shader
    should be silently skipped (some uniforms get optimized out of
    trimmed shader variants when no variation needs them)."""
    r = _make_renderer()
    # Inject a name the shader doesn't have. _sync_* must not raise.
    r._sticky_chaos_uniforms['u_nonexistent_uniform'] = 999
    # Trigger a sync via shader swap.
    g = _genome_with_keep_vars(var_idx=10)
    r.upload_genome(g)
    # If we got here, the missing-uniform path didn't crash. Good.
    assert r.compute_shader is not None
