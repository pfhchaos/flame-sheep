"""GPU variation correctness tests — compare shader output to CPU reference.

Feeds known input points through the GPU variation shader and compares
results to apply_variation_cpu(). This verifies the GLSL math matches
the Python reference implementation.

Requires a GPU (headless GL context). Skipped if no GL available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

try:
    import moderngl
    _HAS_GL = True
except ImportError:
    _HAS_GL = False

from flame_sheep.variations import (
    Variation, NUM_VARIATIONS, apply_variation_cpu,
)
from flame_sheep.variations._registry import VAR_PARAMS_SPEC, SLOT_SIZE
from flame_sheep.genome import MAX_ACTIVE_VARS
import flame_sheep.variations._cpu as cpu_mod

SHADER_DIR = Path(__file__).parent.parent / 'flame_sheep' / 'shaders'

# Test points — cover different quadrants and magnitudes
TEST_POINTS = [
    (1.0, 0.5),
    (0.0, 0.0),
    (-0.3, 0.7),
    (0.5, -0.5),
    (2.0, 1.0),
    (-1.0, -1.0),
    (0.1, 0.1),
    (0.8, -0.2),
    (-0.5, 0.3),
    (1.5, -0.8),
    (0.3, 1.2),
    (-0.7, -0.4),
]

# Same PARAM_FIXTURES as CPU golden masters
PARAM_FIXTURES = {
    Variation.WAVES: {'waves_freq_x': 0.5, 'waves_freq_y': 0.3,
                      'waves_amp_x': 0.8, 'waves_amp_y': 0.6},
    Variation.POPCORN: {'popcorn_cx': 0.3, 'popcorn_cy': 0.5},
    Variation.RINGS: {'rings_c': 0.4},
    Variation.FAN: {'fan_c': 0.3, 'fan_f': 0.5},
    23: {'blob_low': 0.3, 'blob_high': 1.2, 'blob_waves': 6.0},
    24: {'pdj_a': 1.4, 'pdj_b': 2.3, 'pdj_c': 2.4, 'pdj_d': 2.2},
    25: {'fan2_x': 0.5, 'fan2_y': 1.2},
    26: {'rings2_val': 0.5},
    Variation.JULIAN: {'julian_power': 4.0, 'julian_dist': 1.0},
    Variation.JULIASCOPE: {'julian_power': 4.0, 'julian_dist': 1.0},
    Variation.SPLITS: {'splits_x': 0.5, 'splits_y': 0.3},
    Variation.CURL: {'curl_c1': 0.5, 'curl_c2': -0.3},
    Variation.RECTANGLES: {'rect_x': 0.8, 'rect_y': 0.6},
    Variation.CHECKS: {'check_size': 1.5, 'check_x': 0.2, 'check_y': 0.1},
    Variation.HEX_MODULUS: {'hex_size': 1.0},
    Variation.KALEIDOSCOPE: {'kal_pull': 0.3, 'kal_rotate': 1.0, 'kal_n': 6.0},
    Variation.ICON: {'icon_degree': 4.0, 'icon_lambda': 1.5,
                     'icon_alpha': 0.5, 'icon_beta': 0.3,
                     'icon_gamma': 0.1, 'icon_omega': 0.2},
    Variation.SATTRACTOR: {'sat_m': 6.0},
    Variation.WALLPAPER: {'wallpaper_group': 0.0},
    Variation.FRIEZE: {'frieze_group': 0.0},
    47: {'mobius_re_a': 0.1, 'mobius_re_b': 0.2, 'mobius_re_c': -0.15,
         'mobius_re_d': 0.21, 'mobius_im_a': 0.2, 'mobius_im_b': -0.12,
         'mobius_im_c': -0.15, 'mobius_im_d': 0.1},
    48: {'cpow_r': 1.0, 'cpow_i': 0.1, 'cpow_power': 1.5},
    49: {'ngon_circle': 1.0, 'ngon_corners': 2.0, 'ngon_power': 3.0, 'ngon_sides': 5.0},
    52: {'epispiral_n': 6.0, 'epispiral_thickness': 0.0, 'epispiral_holes': 1.0},
    53: {'waves3_scalex': 0.05, 'waves3_scaley': 0.05, 'waves3_freqx': 7.0,
         'waves3_freqy': 13.0, 'waves3_sx_freq': 0.0, 'waves3_sy_freq': 2.0},
}

# Variations that use RNG — results will differ between CPU and GPU
# so we skip strict comparison for these
RANDOM_VARIATIONS = {Variation.JULIA, Variation.SATTRACTOR, Variation.WALLPAPER,
                     Variation.FRIEZE, Variation.JULIAN, Variation.JULIASCOPE,
                     Variation.ICON, Variation.CPOW}

# Variation names for readable output
VAR_NAMES: dict[int, str] = {}
for _name in dir(Variation):
    if _name.startswith('_'):
        continue
    _val = getattr(Variation, _name)
    if isinstance(_val, int) and 0 <= _val < NUM_VARIATIONS:
        VAR_NAMES[_val] = _name

MAX_PARAMS_PER_VAR = SLOT_SIZE - 2


def _resolve_includes(source: str) -> str:
    """Resolve #include directives from the shader directory."""
    import re
    def _replace(m):
        inc_path = SHADER_DIR / m.group(1)
        return inc_path.read_text()
    return re.sub(r'#include\s+"(.+?)"', _replace, source)


def _inject_symmetry_data(source: str) -> str:
    """Inject WALLPAPER_DATA/FRIEZE_DATA constants into shader source."""
    from flame_sheep.variations._symmetry_groups import generate_glsl
    return source.replace('// {{SYMMETRY_GROUPS}}', generate_glsl())


def _build_active_vars(var_idx: int, params: dict) -> np.ndarray:
    """Build the active_vars buffer for a single variation at weight 1.0."""
    buf = np.full(MAX_ACTIVE_VARS * SLOT_SIZE, -1.0, dtype=np.float32)
    buf[0] = float(var_idx)  # var index
    buf[1] = 1.0             # weight
    # Pack params
    spec = VAR_PARAMS_SPEC.get(var_idx, [])
    for k, param_name in enumerate(spec[:MAX_PARAMS_PER_VAR]):
        buf[2 + k] = params.get(param_name, 0.0)
    return buf


@pytest.fixture(scope='module')
def gpu_ctx():
    if not _HAS_GL:
        pytest.skip('No moderngl available')
    try:
        ctx = moderngl.create_context(standalone=True)
    except Exception:
        pytest.skip('No GL context available')
    yield ctx
    ctx.release()


@pytest.fixture(scope='module')
def variation_shader(gpu_ctx):
    """Compile the test variation shader."""
    src = (SHADER_DIR / 'test_variation.comp').read_text()
    src = _inject_symmetry_data(src)
    src = _resolve_includes(src)
    prog = gpu_ctx.compute_shader(src)
    return prog


def _run_variation_gpu(gpu_ctx, shader, var_idx: int,
                       points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Run a variation on the GPU and return output points."""
    n = len(points)
    params = PARAM_FIXTURES.get(var_idx, {})

    # Input points buffer
    in_data = np.array(points, dtype=np.float32).flatten()
    in_buf = gpu_ctx.buffer(in_data.tobytes())
    in_buf.bind_to_storage_buffer(6)

    # Output points buffer
    out_buf = gpu_ctx.buffer(reserve=n * 2 * 4)
    out_buf.bind_to_storage_buffer(7)

    # Active vars buffer (single variation at tidx=0)
    active_vars = _build_active_vars(var_idx, params)
    av_buf = gpu_ctx.buffer(active_vars.tobytes())
    av_buf.bind_to_storage_buffer(3)

    # Run
    shader['u_n_points'].value = n
    groups = (n + 63) // 64
    shader.run(group_x=groups)
    gpu_ctx.memory_barrier()

    # Read back
    out_data = np.frombuffer(out_buf.read(), dtype=np.float32).reshape(-1, 2)

    # Cleanup
    in_buf.release()
    out_buf.release()
    av_buf.release()

    return [(float(out_data[i, 0]), float(out_data[i, 1])) for i in range(n)]


@pytest.mark.parametrize('var_idx', range(NUM_VARIATIONS),
                         ids=[VAR_NAMES.get(i, f'var_{i}') for i in range(NUM_VARIATIONS)])
def test_variation_gpu_matches_cpu(var_idx, gpu_ctx, variation_shader):
    """GPU variation output should match CPU reference within floating point tolerance."""
    if var_idx in RANDOM_VARIATIONS:
        pytest.xfail('RNG synchronization between CPU/GPU not yet verified')

    params = PARAM_FIXTURES.get(var_idx, {})
    cpu_mod._current_var_params = params

    # Run GPU
    gpu_results = _run_variation_gpu(gpu_ctx, variation_shader, var_idx, TEST_POINTS)

    # Run CPU
    for i, (px, py) in enumerate(TEST_POINTS):
        cx, cy = apply_variation_cpu(var_idx, px, py, 1.0)
        gx, gy = gpu_results[i]

        # Skip origin — many variations are undefined at (0,0)
        if abs(px) < 1e-8 and abs(py) < 1e-8:
            continue

        # Skip if both are near-zero (degenerate output)
        if abs(cx) < 1e-8 and abs(cy) < 1e-8 and abs(gx) < 1e-8 and abs(gy) < 1e-8:
            continue

        # Skip if diverged to infinity or very large (singularities)
        if not (np.isfinite(cx) and np.isfinite(cy)):
            continue
        if not (np.isfinite(gx) and np.isfinite(gy)):
            pytest.fail(f'GPU returned non-finite for {VAR_NAMES.get(var_idx)} '
                        f'at ({px}, {py}): ({gx}, {gy})')
        if abs(cx) > 1000 or abs(cy) > 1000 or abs(gx) > 1000 or abs(gy) > 1000:
            continue  # near singularity, epsilon handling diverges

        np.testing.assert_allclose(
            [gx, gy], [cx, cy], atol=1e-3, rtol=1e-3,
            err_msg=f'{VAR_NAMES.get(var_idx, var_idx)} at ({px}, {py})')
