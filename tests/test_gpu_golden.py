"""GPU variation correctness tests — compare shader output to CPU reference.

Feeds known input points through the GPU variation shader and compares
results to apply_variation_cpu(). This verifies the GLSL math matches
the Python reference implementation.

Each variation gets:
- Per-variation test points targeting its specific edge cases (or BASE_POINTS fallback)
- Multiple parameter sets to exercise conditional branches

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

# ---------------------------------------------------------------------------
# Base test points — good spread of quadrants, magnitudes, near-origin/unit circle
# Used as fallback for variations without specific edge-case points.
# ---------------------------------------------------------------------------
BASE_POINTS = [
    (1.0, 0.5),
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
    (0.05, 0.02),       # near-zero
    (0.95, 0.05),       # r ≈ 1 from inside
    (1.05, 0.05),       # r ≈ 1 from outside
    (3.0, -2.0),        # large r
    (0.5, 0.0),         # on x-axis
]

# ---------------------------------------------------------------------------
# Per-variation edge-case points
# ---------------------------------------------------------------------------

# Reusable point sets for common edge-case groups
_NEAR_ZERO = [(0.05, 0.03), (0.03, -0.05), (-0.04, 0.02), (0.02, 0.07)]
_UNIT_CIRCLE = [(0.9, 0.3), (1.1, 0.3), (0.6, 0.6), (0.0, 0.9),
                (0.0, 1.1), (-0.6, 0.7), (0.85, 0.4), (1.15, 0.4)]
_GRID_BOUNDARY = [(0.49, 0.49), (0.51, 0.51), (1.49, 0.5), (-0.49, -0.51),
                  (0.99, 0.01), (1.01, -0.01), (0.25, 0.75), (-1.5, 0.5)]
# 24 points for RNG variations — enough to cover all branches with deterministic seeds
_RNG_COVERAGE = [
    (1.0, 0.5), (-0.3, 0.7), (0.5, -0.5), (2.0, 1.0),
    (-1.0, -1.0), (0.1, 0.1), (0.8, -0.2), (-0.5, 0.3),
    (1.5, -0.8), (0.3, 1.2), (-0.7, -0.4), (0.4, 0.9),
    (1.3, -0.3), (-0.8, 0.6), (0.6, 0.6), (-0.2, -0.9),
    (0.9, -0.1), (-0.6, -0.6), (1.1, 0.4), (0.2, -0.7),
    (-0.4, 1.1), (0.7, 0.3), (-1.2, 0.5), (1.8, -0.6),
]
# Mid-sector angle points — avoid exact multiples of π/n
_ANGLE_SECTORS = [
    (1.0, 0.3), (-0.3, 1.0), (0.7, -0.7), (-1.0, -0.3),
    (0.2, 0.9), (-0.9, 0.2), (0.6, -0.4), (-0.4, -0.8),
    (0.9, 0.5), (-0.5, 0.9), (0.3, -1.0), (-0.8, -0.5),
]

VAR_POINTS: dict[int, list[tuple[float, float]]] = {
    # --- r → 0 singularity group ---
    Variation.SPHERICAL:    BASE_POINTS + _NEAR_ZERO,
    Variation.SPIRAL:       BASE_POINTS + _NEAR_ZERO,
    Variation.FISHEYE:      BASE_POINTS + _NEAR_ZERO,
    Variation.EYEFISH:      BASE_POINTS + _NEAR_ZERO,
    Variation.BUTTERFLY:    BASE_POINTS + _NEAR_ZERO,
    Variation.LOONIE:       BASE_POINTS + _NEAR_ZERO + _UNIT_CIRCLE,
    Variation.SCRY:         BASE_POINTS + _NEAR_ZERO,
    Variation.FLOWER:       _RNG_COVERAGE + _NEAR_ZERO,
    Variation.BLADE:        _RNG_COVERAGE + _NEAR_ZERO,
    Variation.SPIRALWING:   BASE_POINTS + _NEAR_ZERO,
    Variation.HYPERBOLIC:   BASE_POINTS + _NEAR_ZERO,

    # --- r = 1 boundary group ---
    Variation.WHORL:        BASE_POINTS + _UNIT_CIRCLE,
    Variation.FLIPCIRCLE:   BASE_POINTS + _UNIT_CIRCLE,
    Variation.ECLIPSE:      BASE_POINTS + [(0.0, 0.8), (0.0, 1.2), (0.5, 0.95),
                                           (0.5, 1.05), (-0.3, 0.9), (-0.3, 1.1)],

    # --- Integer grid boundary group ---
    Variation.BOARDERS:     _RNG_COVERAGE + _GRID_BOUNDARY,
    Variation.CELL:         BASE_POINTS + _GRID_BOUNDARY,
    Variation.STRIPES:      BASE_POINTS + _GRID_BOUNDARY,
    Variation.RECTANGLES:   BASE_POINTS + _GRID_BOUNDARY,
    Variation.CHECKS:       BASE_POINTS + _GRID_BOUNDARY,
    Variation.HEX_MODULUS:  BASE_POINTS + _GRID_BOUNDARY,

    # --- Angle sector group ---
    Variation.COLLIDEOSCOPE: _ANGLE_SECTORS + [(0.5, -0.3), (-0.6, 0.4), (1.2, 0.1)],
    Variation.KALEIDOSCOPE:  _ANGLE_SECTORS + [(0.5, -0.3), (-0.6, 0.4), (1.2, 0.1)],
    Variation.FAN:           BASE_POINTS + _ANGLE_SECTORS,
    Variation.FAN2:          BASE_POINTS + _ANGLE_SECTORS,

    # --- RNG branch coverage (need 24+ points) ---
    Variation.JULIA:        _RNG_COVERAGE,
    Variation.JULIAN:       _RNG_COVERAGE,
    Variation.JULIASCOPE:   _RNG_COVERAGE,
    Variation.CPOW:         _RNG_COVERAGE,
    Variation.SATTRACTOR:   _RNG_COVERAGE,
    Variation.WALLPAPER:    _RNG_COVERAGE,
    Variation.FRIEZE:       _RNG_COVERAGE,
    Variation.LISSAJOUS:    _RNG_COVERAGE,

    # --- Division hazard group ---
    Variation.RINGS:        BASE_POINTS + _NEAR_ZERO,
    Variation.RINGS2:       BASE_POINTS + _NEAR_ZERO,
    Variation.RINGS3:       BASE_POINTS + _NEAR_ZERO,
    Variation.CURL:         BASE_POINTS + _NEAR_ZERO,
    Variation.NGON:         BASE_POINTS + _NEAR_ZERO,
    Variation.EPISPIRAL:    BASE_POINTS + _NEAR_ZERO,
    Variation.HYPERTILE:    BASE_POINTS + _NEAR_ZERO,
    Variation.MOBIUS:        BASE_POINTS + _NEAR_ZERO,
}

# ---------------------------------------------------------------------------
# Per-variation parameter sets — multiple configs to exercise branches
# ---------------------------------------------------------------------------
VAR_PARAM_SETS: dict[int, dict[str, dict]] = {
    # --- Non-parametric (no params, single default entry) ---
    Variation.LINEAR:       {'default': {}},
    Variation.SINUSOIDAL:   {'default': {}},
    Variation.SPHERICAL:    {'default': {}},
    Variation.SWIRL:        {'default': {}},
    Variation.HORSESHOE:    {'default': {}},
    Variation.POLAR:        {'default': {}},
    Variation.HANDKERCHIEF: {'default': {}},
    Variation.HEART:        {'default': {}},
    Variation.DISK:         {'default': {}},
    Variation.SPIRAL:       {'default': {}},
    Variation.HYPERBOLIC:   {'default': {}},
    Variation.DIAMOND:      {'default': {}},
    Variation.EX:           {'default': {}},
    Variation.BENT:         {'default': {}},
    Variation.FISHEYE:      {'default': {}},
    Variation.EXPONENTIAL:  {'default': {}},
    Variation.POWER:        {'default': {}},
    Variation.COSINE:       {'default': {}},
    Variation.EYEFISH:      {'default': {}},
    Variation.BUBBLE:       {'default': {}},
    Variation.CYLINDER:     {'default': {}},
    Variation.CLOVERLEAF:   {'default': {}},
    Variation.TANGENT:      {'default': {}},
    Variation.CROSS:        {'default': {}},
    Variation.BUTTERFLY:    {'default': {}},
    Variation.LOONIE:       {'default': {}},
    Variation.SCRY:         {'default': {}},
    Variation.SPIRALWING:   {'default': {}},
    Variation.FLIPCIRCLE:   {'default': {}},
    Variation.BLADE:        {'default': {}},

    # --- Parametric: 2+ param sets ---
    Variation.WAVES: {
        'default': {'waves_freq_x': 0.5, 'waves_freq_y': 0.3,
                    'waves_amp_x': 0.8, 'waves_amp_y': 0.6},
        'high_freq': {'waves_freq_x': 5.0, 'waves_freq_y': 7.0,
                      'waves_amp_x': 0.2, 'waves_amp_y': 0.1},
    },
    Variation.POPCORN: {
        'default': {'popcorn_cx': 0.3, 'popcorn_cy': 0.5},
        'strong': {'popcorn_cx': 1.5, 'popcorn_cy': 1.5},
    },
    Variation.RINGS: {
        'default': {'rings_c': 0.4},
        'tight': {'rings_c': 0.1},
    },
    Variation.FAN: {
        'default': {'fan_c': 0.3, 'fan_f': 0.5},
        'narrow': {'fan_c': 0.1, 'fan_f': 0.8},
    },
    Variation.BLOB: {
        'default': {'blob_low': 0.3, 'blob_high': 1.2, 'blob_waves': 6.0},
        'tight': {'blob_low': 0.8, 'blob_high': 1.0, 'blob_waves': 3.0},
    },
    Variation.PDJ: {
        'default': {'pdj_a': 1.4, 'pdj_b': 2.3, 'pdj_c': 2.4, 'pdj_d': 2.2},
        'small': {'pdj_a': 0.5, 'pdj_b': 0.3, 'pdj_c': 0.7, 'pdj_d': 0.4},
    },
    Variation.FAN2: {
        'default': {'fan2_x': 0.5, 'fan2_y': 1.2},
        'wide': {'fan2_x': 2.0, 'fan2_y': 0.3},
    },
    Variation.RINGS2: {
        'default': {'rings2_val': 0.5},
        'small': {'rings2_val': 0.1},
    },
    Variation.JULIA: {
        'default': {},
    },
    Variation.JULIAN: {
        'default': {'julian_power': 4.0, 'julian_dist': 1.0},
        'high_power': {'julian_power': 8.0, 'julian_dist': 1.0},
        'negative': {'julian_power': -3.0, 'julian_dist': 1.0},
    },
    Variation.JULIASCOPE: {
        'default': {'julian_power': 4.0, 'julian_dist': 1.0},
        'high_power': {'julian_power': 6.0, 'julian_dist': 1.5},
    },
    Variation.SPLITS: {
        'default': {'splits_x': 0.5, 'splits_y': 0.3},
        'large': {'splits_x': 2.0, 'splits_y': 1.5},
    },
    Variation.CURL: {
        'default': {'curl_c1': 0.5, 'curl_c2': -0.3},
        'strong': {'curl_c1': 2.0, 'curl_c2': 1.0},
    },
    Variation.RECTANGLES: {
        'default': {'rect_x': 0.8, 'rect_y': 0.6},
        'small': {'rect_x': 0.2, 'rect_y': 0.3},
    },
    Variation.CHECKS: {
        'default': {'check_size': 1.5, 'check_x': 0.2, 'check_y': 0.1},
        'fine': {'check_size': 0.5, 'check_x': 0.5, 'check_y': 0.3},
    },
    Variation.HEX_MODULUS: {
        'default': {'hex_size': 1.0},
    },
    Variation.KALEIDOSCOPE: {
        'default': {'kal_pull': 0.3, 'kal_rotate': 1.0, 'kal_n': 6.0},
        'tri': {'kal_pull': 0.5, 'kal_rotate': 0.5, 'kal_n': 3.0},
    },
    Variation.ICON: {
        'default': {'icon_degree': 4.0, 'icon_lambda': 1.5,
                    'icon_alpha': 0.5, 'icon_beta': 0.3,
                    'icon_gamma': 0.1, 'icon_omega': 0.2},
        'high_sym': {'icon_degree': 6.0, 'icon_lambda': 1.2,
                     'icon_alpha': 0.3, 'icon_beta': 0.1,
                     'icon_gamma': 0.0, 'icon_omega': 0.5},
    },
    Variation.SATTRACTOR: {
        'default': {'sat_m': 6.0},
        'low': {'sat_m': 3.0},
    },
    Variation.WALLPAPER: {
        'default': {'wallpaper_group': 0.0},
        'group3': {'wallpaper_group': 3.0},
    },
    Variation.FRIEZE: {
        'default': {'frieze_group': 0.0},
        'group2': {'frieze_group': 2.0},
    },
    Variation.RINGS3: {
        'default': {'rings3_val': 0.5, 'rings3_n': 4.0},
        'tight': {'rings3_val': 0.2, 'rings3_n': 8.0},
    },
    Variation.MOBIUS: {
        'default': {'mobius_re_a': 0.1, 'mobius_re_b': 0.2, 'mobius_re_c': -0.15,
                    'mobius_re_d': 0.21, 'mobius_im_a': 0.2, 'mobius_im_b': -0.12,
                    'mobius_im_c': -0.15, 'mobius_im_d': 0.1},
    },
    Variation.CPOW: {
        'default': {'cpow_r': 1.0, 'cpow_i': 0.1, 'cpow_power': 1.5},
        'high_power': {'cpow_r': 0.8, 'cpow_i': 0.3, 'cpow_power': 3.0},
    },
    Variation.NGON: {
        'default': {'ngon_circle': 1.0, 'ngon_corners': 2.0, 'ngon_power': 3.0, 'ngon_sides': 5.0},
        'tri': {'ngon_circle': 0.5, 'ngon_corners': 1.0, 'ngon_power': 2.0, 'ngon_sides': 3.0},
    },
    Variation.EPISPIRAL: {
        'default': {'epispiral_n': 6.0, 'epispiral_thickness': 0.0, 'epispiral_holes': 1.0},
        'thick': {'epispiral_n': 4.0, 'epispiral_thickness': 0.5, 'epispiral_holes': 0.5},
    },
    Variation.WAVES3: {
        'default': {'waves3_scalex': 0.05, 'waves3_scaley': 0.05, 'waves3_freqx': 7.0,
                    'waves3_freqy': 13.0, 'waves3_sx_freq': 0.0, 'waves3_sy_freq': 2.0},
        'fast': {'waves3_scalex': 0.1, 'waves3_scaley': 0.1, 'waves3_freqx': 15.0,
                 'waves3_freqy': 15.0, 'waves3_sx_freq': 3.0, 'waves3_sy_freq': 3.0},
    },
    Variation.BOARDERS: {
        'default': {'boarders_c': 0.5, 'boarders_cl': 0.25, 'boarders_cr': 0.75},
        'always_border': {'boarders_c': 0.5, 'boarders_cl': 0.25, 'boarders_cr': 0.0},
        'never_border': {'boarders_c': 0.5, 'boarders_cl': 0.25, 'boarders_cr': 1.1},
    },
    Variation.HYPERTILE: {
        'default': {'hypertile_re': 0.3, 'hypertile_im': 0.0},
        'complex': {'hypertile_re': 0.2, 'hypertile_im': 0.3},
    },
    Variation.CELL: {
        'default': {'cell_size': 1.0},
        'small': {'cell_size': 0.5},
    },
    Variation.WHORL: {
        'default': {'whorl_inside': 0.5, 'whorl_outside': 0.5},
        'strong': {'whorl_inside': 3.0, 'whorl_outside': 3.0},
        'asymmetric': {'whorl_inside': 0.1, 'whorl_outside': 2.0},
    },
    Variation.DISC2: {
        'default': {'disc2_twist': 1.0, 'disc2_cosadd': 0.0, 'disc2_sinadd': 0.0},
        'shifted': {'disc2_twist': 2.0, 'disc2_cosadd': 0.3, 'disc2_sinadd': -0.2},
    },
    Variation.FLOWER: {
        'default': {'flower_holes': 0.5, 'flower_petals': 6.0},
        'many_petals': {'flower_holes': 0.3, 'flower_petals': 12.0},
    },
    Variation.COLLIDEOSCOPE: {
        'default': {'collide_a': 0.5, 'collide_num': 5.0},
        'high_fold': {'collide_a': 0.3, 'collide_num': 7.0},
        'low_fold': {'collide_a': 0.8, 'collide_num': 3.0},
    },
    Variation.AUGER: {
        'default': {'auger_freq': 3.0, 'auger_weight': 0.5,
                    'auger_sym': 0.5, 'auger_scale': 0.5},
        'strong': {'auger_freq': 7.0, 'auger_weight': 1.0,
                   'auger_sym': 1.0, 'auger_scale': 1.0},
    },
    Variation.ECLIPSE: {
        'default': {'eclipse_shift': 0.5},
        'large': {'eclipse_shift': 1.5},
    },
    Variation.LAYERED_SPIRAL: {
        'default': {'layered_spiral_radius': 1.0},
        'tight': {'layered_spiral_radius': 3.0},
    },
    Variation.STRIPES: {
        'default': {'stripes_space': 0.5, 'stripes_warp': 0.5},
        'tight': {'stripes_space': 0.8, 'stripes_warp': 2.0},
    },
    Variation.LISSAJOUS: {
        'default': {'liss_tmin': -3.14159265, 'liss_tmax': 3.14159265,
                    'liss_a': 3.0, 'liss_b': 2.0, 'liss_c': 0.0,
                    'liss_d': 0.0, 'liss_e': 0.0},
        'with_drift': {'liss_tmin': -3.14159265, 'liss_tmax': 3.14159265,
                       'liss_a': 5.0, 'liss_b': 3.0, 'liss_c': 0.1,
                       'liss_d': 0.5, 'liss_e': 0.3},
    },
    Variation.RIPPLE: {
        'default': {'ripple_freq': 5.0, 'ripple_vel': 0.0, 'ripple_amp': 0.1,
                    'ripple_cx': 0.0, 'ripple_cy': 0.0, 'ripple_phase': 0.0,
                    'ripple_scale': 1.0, 'ripple_fixd': 1.0},
        'product_dist': {'ripple_freq': 5.0, 'ripple_vel': 0.0, 'ripple_amp': 0.1,
                         'ripple_cx': 0.0, 'ripple_cy': 0.0, 'ripple_phase': 0.0,
                         'ripple_scale': 1.0, 'ripple_fixd': 0.0},
    },
}

# Variations that use RNG — need deterministic seeding to match GPU
RANDOM_VARIATIONS = {Variation.JULIA, Variation.SATTRACTOR, Variation.WALLPAPER,
                     Variation.FRIEZE, Variation.JULIAN, Variation.JULIASCOPE,
                     Variation.CPOW, Variation.BOARDERS, Variation.FLOWER,
                     Variation.BLADE, Variation.LISSAJOUS, Variation.EPISPIRAL}

# Variation names for readable output
VAR_NAMES: dict[int, str] = {}
for _name in dir(Variation):
    if _name.startswith('_'):
        continue
    _val = getattr(Variation, _name)
    if isinstance(_val, int) and 0 <= _val < NUM_VARIATIONS:
        VAR_NAMES[_val] = _name

MAX_PARAMS_PER_VAR = SLOT_SIZE - 2

# ---------------------------------------------------------------------------
# Build parametrized test cases: (var_idx, param_set_name) pairs
# ---------------------------------------------------------------------------
VAR_TEST_CASES: list[tuple[int, str]] = []
VAR_TEST_IDS: list[str] = []
for _vi in range(NUM_VARIATIONS):
    _psets = VAR_PARAM_SETS.get(_vi, {'default': {}})
    for _pname in _psets:
        VAR_TEST_CASES.append((_vi, _pname))
        VAR_TEST_IDS.append(f'{VAR_NAMES.get(_vi, f"var_{_vi}")}-{_pname}')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _run_variation_gpu(gpu_ctx, shader, var_idx: int, params: dict,
                       points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Run a variation on the GPU and return output points."""
    n = len(points)

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


@pytest.mark.parametrize('var_idx,param_set_name', VAR_TEST_CASES, ids=VAR_TEST_IDS)
def test_variation_gpu_matches_cpu(var_idx, param_set_name, gpu_ctx, variation_shader):
    """GPU variation output should match CPU reference within floating point tolerance."""
    param_sets = VAR_PARAM_SETS.get(var_idx, {'default': {}})
    params = param_sets[param_set_name]
    cpu_mod._current_var_params = params
    uses_rng = var_idx in RANDOM_VARIATIONS
    points = VAR_POINTS.get(var_idx, BASE_POINTS)

    # Run GPU
    gpu_results = _run_variation_gpu(gpu_ctx, variation_shader, var_idx, params, points)

    # Run CPU — for RNG variations, seed xorshift32 per-point to match GPU
    for i, (px, py) in enumerate(points):
        if uses_rng:
            cpu_mod._rng = cpu_mod.Xorshift32(seed=i * 17 + 1)
        cx, cy = apply_variation_cpu(var_idx, px, py, 1.0)
        if uses_rng:
            cpu_mod._rng = None
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
            err_msg=f'{VAR_NAMES.get(var_idx, var_idx)} [{param_set_name}] at ({px}, {py})')
