"""Vulkan port of test_gpu_golden.py — variations.glsl correctness vs CPU.

Mirror of the existing GL test. Same shared fixtures (variation_fixtures.py),
same CPU reference (apply_variation_cpu), same point sets / param sets /
tolerances. Only difference: runs the variation through Vulkan via
viz_authoring.vk.shaders.test_variation.comp instead of the GL test
shader.

What this catches:
  - Drift between the GL and Vulkan ports of variations.glsl
  - SPIR-V compiler choices that diverge from GLSL semantics
  - Vulkan binding / descriptor-set bugs in test_variation.comp itself

Pairs with test_gpu_golden.py. Either both should pass or the divergence
points at a real shader port bug.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from flame_sheep.variations import (
    Variation, NUM_VARIATIONS, apply_variation_cpu,
)
from flame_sheep.variations._registry import VAR_PARAMS_SPEC, SLOT_SIZE
from flame_sheep.genome import MAX_ACTIVE_VARS
import flame_sheep.variations._cpu as cpu_mod

from variation_fixtures import (
    BASE_POINTS,
    VAR_POINTS,
    TEST_AFFINE,
    AFFINE_VARIATIONS,
    RANDOM_VARIATIONS,
    VAR_PARAM_SETS,
)

# Re-derive the test-case enumeration locally rather than import from
# the sibling test_gpu_golden — pytest doesn't add tests/variations/
# to sys.path and a deeper cross-test dependency would be brittle.
# The derivation is mechanical (same as test_gpu_golden's), keyed off
# the shared VAR_PARAM_SETS fixture.
VAR_NAMES: dict[int, str] = {}
for _name in dir(Variation):
    if _name.startswith('_'):
        continue
    _val = getattr(Variation, _name)
    if isinstance(_val, int) and 0 <= _val < NUM_VARIATIONS:
        VAR_NAMES[_val] = _name

MAX_PARAMS_PER_VAR = SLOT_SIZE - 2

VAR_TEST_CASES: list[tuple[int, str]] = []
VAR_TEST_IDS: list[str] = []
for _vi in range(NUM_VARIATIONS):
    _psets = VAR_PARAM_SETS.get(_vi, {'default': {}})
    for _pname in _psets:
        VAR_TEST_CASES.append((_vi, _pname))
        VAR_TEST_IDS.append(
            f'{VAR_NAMES.get(_vi, f"var_{_vi}")}-{_pname}')


SHADER_DIR = (Path(__file__).resolve().parents[2]
              / 'viz_authoring' / 'src' / 'viz_authoring' / 'vk' / 'shaders')
TEST_SHADER = SHADER_DIR / 'test_variation.comp'

MAX_POINTS = 256   # generous upper bound on per-variation point counts


def _build_active_vars(var_idx: int, params: dict) -> np.ndarray:
    """Mirror of test_gpu_golden._build_active_vars — single variation at
    weight 1.0, params packed starting at slot offset PARAM_OFFSET=2."""
    buf = np.full(MAX_ACTIVE_VARS * SLOT_SIZE, -1.0, dtype=np.float32)
    buf[0] = float(var_idx)
    buf[1] = 1.0
    spec = VAR_PARAMS_SPEC.get(var_idx, [])
    for k, param_name in enumerate(spec[:MAX_PARAMS_PER_VAR]):
        buf[2 + k] = params.get(param_name, 0.0)
    return buf


def _inject_symmetry(src: str) -> str:
    """Replace the {{SYMMETRY_GROUPS}} placeholder inside the included
    variations.glsl with the generated constant tables."""
    from flame_sheep.variations._symmetry_groups import generate_glsl
    return src.replace('// {{SYMMETRY_GROUPS}}', generate_glsl())


# ---------------------------------------------------------------------------
# Module-scoped Vulkan setup. Created once for all parametrized cases.
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def vk_ctx():
    """Vulkan context, no surface (compute-only). Pipeline cache
    disabled so test runs are reproducible — caching shouldn't change
    output but the determinism guarantee makes the suite less
    suspect when something does go wrong."""
    try:
        from viz_authoring.vk.context import VkContext
    except ImportError:
        pytest.skip('viz_authoring.vk not available')
    try:
        ctx = VkContext(instance_extensions=[],
                         app_name='vk-golden-test',
                         pipeline_cache_path=None)
        ctx.select_device()
    except Exception as e:
        pytest.skip(f'No Vulkan device: {e}')
    yield ctx
    ctx.cleanup()


@pytest.fixture(scope='module')
def vk_pipeline(vk_ctx):
    """Test pipeline + the four buffers it binds. Reused across all
    test cases; per-test we upload new data + push the point count."""
    from viz_authoring.vk.pipeline import ComputePipeline

    in_buf = vk_ctx.create_buffer(MAX_POINTS * 2 * 4, 'storage')
    out_buf = vk_ctx.create_buffer(MAX_POINTS * 2 * 4, 'storage')
    av_buf = vk_ctx.create_buffer(
        MAX_ACTIVE_VARS * SLOT_SIZE * 4, 'storage')
    aff_buf = vk_ctx.create_buffer(6 * 4, 'storage')
    pre_buf = vk_ctx.create_buffer(
        MAX_ACTIVE_VARS * SLOT_SIZE * 4, 'storage')
    # pre-variations always empty in this test — fill with -1 so the
    # "var_idx < 0" early-break in apply_pre_variations fires immediately.
    vk_ctx.upload(pre_buf, np.full(
        MAX_ACTIVE_VARS * SLOT_SIZE, -1.0, dtype=np.float32))

    pipeline = ComputePipeline(
        vk_ctx, TEST_SHADER,
        buffers=[in_buf, out_buf, av_buf, aff_buf, pre_buf],
        push_constant_size=4,
        source_transform=_inject_symmetry,
    )
    yield pipeline, in_buf, out_buf, av_buf, aff_buf
    pipeline.cleanup()


def _run_variation_vk(vk_ctx, pipeline_bundle, var_idx: int, params: dict,
                       points: list[tuple[float, float]],
                       affine: np.ndarray | None = None
                       ) -> list[tuple[float, float]]:
    """Vulkan equivalent of test_gpu_golden._run_variation_gpu."""
    pipeline, in_buf, out_buf, av_buf, aff_buf = pipeline_bundle
    n = len(points)
    assert n <= MAX_POINTS, f'point set ({n}) exceeds MAX_POINTS ({MAX_POINTS})'

    in_data = np.array(points, dtype=np.float32).flatten()
    # Pad to full buffer so we don't have to track sizes — driver only
    # reads up to n entries via the push-constant count.
    in_padded = np.zeros(MAX_POINTS * 2, dtype=np.float32)
    in_padded[:n * 2] = in_data
    vk_ctx.upload(in_buf, in_padded)

    affine_data = (affine.copy() if affine is not None
                    else np.array([1, 0, 0, 0, 1, 0], dtype=np.float32))
    vk_ctx.upload(aff_buf, affine_data)
    vk_ctx.upload(av_buf, _build_active_vars(var_idx, params))
    vk_ctx.zero_buffer(out_buf)

    push = struct.pack('i', n)
    groups = (n + 63) // 64
    vk_ctx.dispatch_compute(pipeline, groups_x=groups, push_constants=push)

    out_data = vk_ctx.download(
        out_buf, np.float32, count=n * 2).reshape(-1, 2)
    return [(float(out_data[i, 0]), float(out_data[i, 1])) for i in range(n)]


@pytest.mark.parametrize('var_idx,param_set_name',
                          VAR_TEST_CASES, ids=VAR_TEST_IDS)
def test_variation_vk_matches_cpu(var_idx, param_set_name, vk_ctx, vk_pipeline):
    """Vulkan variation output should match CPU reference within
    floating-point tolerance — same contract as test_gpu_golden."""
    param_sets = VAR_PARAM_SETS.get(var_idx, {'default': {}})
    params = param_sets[param_set_name]
    cpu_mod._current_var_params = params
    uses_rng = var_idx in RANDOM_VARIATIONS
    points = VAR_POINTS.get(var_idx, BASE_POINTS)

    affine = TEST_AFFINE if var_idx in AFFINE_VARIATIONS else None

    vk_results = _run_variation_vk(
        vk_ctx, vk_pipeline, var_idx, params, points, affine=affine)

    for i, (px, py) in enumerate(points):
        # CPU reference — for RNG variations, seed Xorshift32 per-point
        # the same way the shader does (idx * 17 + 1).
        if uses_rng:
            cpu_mod._rng = cpu_mod.Xorshift32(seed=i * 17 + 1)
        cx, cy = apply_variation_cpu(var_idx, px, py, 1.0, affine)
        if uses_rng:
            cpu_mod._rng = None
        vx, vy = vk_results[i]

        # Mirror test_gpu_golden's skip conditions exactly so any test
        # that passes there must pass here (and vice versa).
        if abs(px) < 1e-8 and abs(py) < 1e-8:
            continue
        if (abs(cx) < 1e-8 and abs(cy) < 1e-8
                and abs(vx) < 1e-8 and abs(vy) < 1e-8):
            continue
        if not (np.isfinite(cx) and np.isfinite(cy)):
            continue
        if not (np.isfinite(vx) and np.isfinite(vy)):
            pytest.fail(
                f'Vulkan returned non-finite for {VAR_NAMES.get(var_idx)} '
                f'at ({px}, {py}): ({vx}, {vy})')
        if (abs(cx) > 1000 or abs(cy) > 1000
                or abs(vx) > 1000 or abs(vy) > 1000):
            continue

        np.testing.assert_allclose(
            [vx, vy], [cx, cy], atol=1e-3, rtol=1e-3,
            err_msg=(f'{VAR_NAMES.get(var_idx, var_idx)} '
                     f'[{param_set_name}] at ({px}, {py})'))
