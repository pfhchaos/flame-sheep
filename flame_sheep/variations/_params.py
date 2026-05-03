"""Random parameter generation for parametric variations.

Ranges from JWildfire's randomize() methods:
- Most variations use the base class default: uniform [-2.5, 2.5]
- Julian/Juliascope: custom multimodal (power 2-12, dist bimodal)
- Icon: preset table with ±0.1 jitter (17 known-good parameter sets)
- Kaleidoscope: we override kal_n to [3, 8] (JWildfire's default allows
  meaningless negative values)
"""
from __future__ import annotations

import numpy as np

from ._registry import Variation


# Icon attractor presets from JWildfire IconAttractorFunc.java
# 17 known-good parameter sets, empirically tuned over 20 years
_ICON_PRESETS = {
    'degree': [5, 3, 5, 3, 3, 9, 6, 23, 7, 5, 5, 5, 4, 3, 3, 3, 16],
    'lambda': [-2.5, 1.56, -1.806, -2.195, 2.5, -2.05, -2.7, 2.409,
               -2.08, -2.32, 2.6, -2.34, -1.86, 1.56, 1.5, 1.455, 2.39],
    'alpha':  [5.0, -1.0, 1.806, 10.0, -2.5, 3.0, 5.0, -2.5,
               1.0, 2.32, -2.0, 2.0, 2.0, -1.0, -1.0, -1.0, -2.5],
    'beta':   [-1.9, 0.1, 0.0, -12.0, 0.0, -16.79, 1.5, 0.0,
               -0.1, 0.0, 0.0, 0.2, 0.0, 0.1, 0.1, 0.03, -0.1],
    'gamma':  [1.0, -0.82, 1.0, 1.0, 0.9, 1.0, 1.0, 0.9,
               0.167, 0.75, -0.5, 0.1, 1.0, -0.82, -0.805, -0.80, 0.90],
    'omega':  [0.188, 0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, -0.15],
}


def random_var_params(var_idx: int, rng: np.random.Generator) -> dict[str, float]:
    """Generate random parameters for a parametric variation.

    Returns an empty dict for non-parametric variations.
    """
    if var_idx == Variation.WAVES:
        return {
            'waves_freq_x': float(rng.uniform(-2.5, 2.5)),
            'waves_freq_y': float(rng.uniform(-2.5, 2.5)),
            'waves_amp_x': float(rng.uniform(-2.5, 2.5)),
            'waves_amp_y': float(rng.uniform(-2.5, 2.5)),
        }

    elif var_idx == Variation.POPCORN:
        return {
            'popcorn_cx': float(rng.uniform(-2.5, 2.5)),
            'popcorn_cy': float(rng.uniform(-2.5, 2.5)),
        }

    elif var_idx == Variation.RINGS:
        return {
            'rings_c': float(rng.uniform(-2.5, 2.5)),
        }

    elif var_idx == Variation.FAN:
        return {
            'fan_c': float(rng.uniform(-2.5, 2.5)),
            'fan_f': float(rng.uniform(-2.5, 2.5)),
        }

    elif var_idx in (Variation.JULIAN, Variation.JULIASCOPE):
        # Power 2-12, 50% chance negative (inverts symmetry)
        power = float(rng.integers(2, 13))
        if rng.random() < 0.5:
            power = -power
        # Dist is multi-modal: 40% conservative, 40% wild, 20% exactly 1.0
        r = rng.random()
        if r < 0.4:
            dist = float(rng.uniform(0.75, 1.25))
        elif r < 0.8:
            dist = float(rng.uniform(0.2, 3.5))
        else:
            dist = 1.0
        if rng.random() < 0.4:
            dist = -dist
        return {'julian_power': power, 'julian_dist': dist}

    elif var_idx == Variation.BLOB:
        # From JWildfire BlobFunc.randomize():
        # 75%: low = rand*rand, high = low + rand
        # 25%: low = rand*2-1, high = low + rand*2
        # waves: 50% [3,9], 50% [1,31], 75% rounded to int
        if rng.random() < 0.75:
            low = float(rng.random() * rng.random())
            high = float(low + rng.random())
        else:
            low = float(rng.uniform(-1.0, 1.0))
            high = float(low + rng.uniform(0.0, 2.0))
        if rng.random() < 0.5:
            waves = float(rng.uniform(3.0, 9.0))
        else:
            waves = float(rng.uniform(1.0, 31.0))
        if rng.random() < 0.75:
            waves = round(waves)
        return {'blob_low': low, 'blob_high': high, 'blob_waves': waves}

    elif var_idx == Variation.PDJ:
        return {
            'pdj_a': float(rng.uniform(-2.5, 2.5)),
            'pdj_b': float(rng.uniform(-2.5, 2.5)),
            'pdj_c': float(rng.uniform(-2.5, 2.5)),
            'pdj_d': float(rng.uniform(-2.5, 2.5)),
        }

    elif var_idx == Variation.FAN2:
        return {
            'fan2_x': float(rng.uniform(-2.5, 2.5)),
            'fan2_y': float(rng.uniform(-2.5, 2.5)),
        }

    elif var_idx == Variation.RINGS2:
        return {
            'rings2_val': float(rng.uniform(-2.5, 2.5)),
        }

    elif var_idx == Variation.SPLITS:
        return {
            'splits_x': float(rng.uniform(-2.5, 2.5)),
            'splits_y': float(rng.uniform(-2.5, 2.5)),
        }

    elif var_idx == Variation.CURL:
        return {
            'curl_c1': float(rng.uniform(-2.5, 2.5)),
            'curl_c2': float(rng.uniform(-2.5, 2.5)),
        }

    elif var_idx == Variation.RECTANGLES:
        return {
            'rect_x': float(rng.uniform(-2.5, 2.5)),
            'rect_y': float(rng.uniform(-2.5, 2.5)),
        }

    elif var_idx == Variation.CHECKS:
        return {
            'check_size': float(rng.uniform(-2.5, 2.5)),
            'check_x': float(rng.uniform(-2.5, 2.5)),
            'check_y': float(rng.uniform(-2.5, 2.5)),
        }

    elif var_idx == Variation.HEX_MODULUS:
        return {
            'hex_size': float(rng.uniform(-2.5, 2.5)),
        }

    elif var_idx == Variation.KALEIDOSCOPE:
        # kal_n: integer sectors, must be >= 2 (our override, JWildfire allows negative)
        n = float(rng.integers(3, 9))
        return {
            'kal_pull': float(rng.uniform(-2.5, 2.5)),
            'kal_rotate': float(rng.uniform(-2.5, 2.5)),
            'kal_n': n,
        }

    elif var_idx == Variation.ICON:
        # Pick a preset and jitter ±0.1 (from JWildfire randomize())
        idx = int(rng.integers(0, len(_ICON_PRESETS['degree'])))
        return {
            'icon_degree': float(_ICON_PRESETS['degree'][idx]),
            'icon_lambda': float(_ICON_PRESETS['lambda'][idx] + rng.uniform(-0.1, 0.1)),
            'icon_alpha': float(_ICON_PRESETS['alpha'][idx] + rng.uniform(-0.1, 0.1)),
            'icon_beta': float(_ICON_PRESETS['beta'][idx] + rng.uniform(-0.1, 0.1)),
            'icon_gamma': float(_ICON_PRESETS['gamma'][idx] + rng.uniform(-0.1, 0.1)),
            'icon_omega': float(_ICON_PRESETS['omega'][idx] + rng.uniform(-0.1, 0.1)),
        }

    elif var_idx == Variation.SATTRACTOR:
        return {
            'sat_m': float(rng.integers(2, 13)),
        }

    elif var_idx == Variation.WALLPAPER:
        return {
            'wallpaper_group': float(rng.integers(0, 17)),
        }

    elif var_idx == Variation.FRIEZE:
        return {
            'frieze_group': float(rng.integers(0, 7)),
        }

    elif var_idx == Variation.RINGS3:
        return {
            'rings3_val': float(rng.uniform(-2.5, 2.5)),
            'rings3_n': float(rng.uniform(-2.5, 2.5)),
        }

    elif var_idx == Variation.MOBIUS:
        # Complex 2x2 matrix (az+b)/(cz+d) — small values to keep bounded
        return {
            'mobius_re_a': float(rng.uniform(-0.5, 0.5)),
            'mobius_re_b': float(rng.uniform(-0.5, 0.5)),
            'mobius_re_c': float(rng.uniform(-0.5, 0.5)),
            'mobius_re_d': float(rng.uniform(-0.5, 0.5)),
            'mobius_im_a': float(rng.uniform(-0.5, 0.5)),
            'mobius_im_b': float(rng.uniform(-0.5, 0.5)),
            'mobius_im_c': float(rng.uniform(-0.5, 0.5)),
            'mobius_im_d': float(rng.uniform(-0.5, 0.5)),
        }

    elif var_idx == Variation.CPOW:
        return {
            'cpow_r': float(rng.uniform(0.5, 2.0)),
            'cpow_i': float(rng.uniform(-0.5, 0.5)),
            'cpow_power': float(rng.integers(2, 8)),
        }

    elif var_idx == Variation.NGON:
        return {
            'ngon_circle': float(rng.uniform(0.5, 2.0)),
            'ngon_corners': float(rng.uniform(0.5, 4.0)),
            'ngon_power': float(rng.uniform(1.0, 5.0)),
            'ngon_sides': float(rng.integers(3, 9)),
        }

    elif var_idx == Variation.EPISPIRAL:
        # From JWildfire randomize()
        if rng.random() < 0.5:
            n = float(int(rng.uniform(3, 20)))
        else:
            n = float(rng.uniform(2.0, 50.0))
        thickness = 0.0 if rng.random() < 0.2 else float(rng.uniform(-2.5, 2.5))
        return {
            'epispiral_n': n,
            'epispiral_thickness': thickness,
            'epispiral_holes': float(rng.uniform(-5.0, 5.0)),
        }

    elif var_idx == Variation.WAVES3:
        return {
            'waves3_scalex': float(rng.uniform(0.01, 0.2)),
            'waves3_scaley': float(rng.uniform(0.01, 0.2)),
            'waves3_freqx': float(rng.uniform(2.0, 15.0)),
            'waves3_freqy': float(rng.uniform(2.0, 15.0)),
            'waves3_sx_freq': float(rng.uniform(0.0, 4.0)),
            'waves3_sy_freq': float(rng.uniform(0.0, 4.0)),
        }

    elif var_idx == Variation.BOARDERS:
        return {
            'boarders_c': float(rng.uniform(0.3, 0.7)),    # border width scale
            'boarders_cl': float(rng.uniform(0.1, 0.4)),   # border offset
            'boarders_cr': float(rng.uniform(0.5, 0.9)),   # randomization threshold
        }

    elif var_idx == Variation.HYPERTILE:
        # Möbius transform params — need to be inside unit disk for convergence
        angle = float(rng.uniform(0, 2 * np.pi))
        radius = float(rng.uniform(0.1, 0.8))
        return {
            'hypertile_re': radius * np.cos(angle),
            'hypertile_im': radius * np.sin(angle),
        }

    elif var_idx == Variation.CELL:
        return {
            'cell_size': float(rng.uniform(0.5, 2.5)),
        }

    elif var_idx == Variation.WHORL:
        return {
            'whorl_inside': float(rng.uniform(-1.0, 1.0)),
            'whorl_outside': float(rng.uniform(-1.0, 1.0)),
        }

    elif var_idx == Variation.DISC2:
        return {
            'disc2_twist': float(rng.uniform(0.5, 3.0)),
            'disc2_cosadd': float(rng.uniform(-0.5, 0.5)),
            'disc2_sinadd': float(rng.uniform(-0.5, 0.5)),
        }

    elif var_idx == Variation.FLOWER:
        return {
            'flower_holes': float(rng.uniform(0.0, 1.0)),
            'flower_petals': float(rng.integers(3, 12)),
        }

    elif var_idx == Variation.COLLIDEOSCOPE:
        return {
            'collide_a': float(rng.uniform(0.0, 2.0)),
            'collide_num': float(rng.integers(2, 10)),
        }

    elif var_idx == Variation.AUGER:
        return {
            'auger_freq': float(rng.uniform(1.0, 8.0)),
            'auger_weight': float(rng.uniform(0.1, 1.0)),
            'auger_sym': float(rng.uniform(0.0, 1.0)),
            'auger_scale': float(rng.uniform(0.1, 1.0)),
        }

    return {}


# Parameter ranges for jitter mutation — (min, max) per param name.
# Derived from the random_var_params() ranges above.
_PARAM_RANGES: dict[str, tuple[float, float]] = {
    # Default range (most JWildfire params)
    'waves_freq_x': (-2.5, 2.5), 'waves_freq_y': (-2.5, 2.5),
    'waves_amp_x': (-2.5, 2.5), 'waves_amp_y': (-2.5, 2.5),
    'popcorn_cx': (-2.5, 2.5), 'popcorn_cy': (-2.5, 2.5),
    'rings_c': (-2.5, 2.5),
    'fan_c': (-2.5, 2.5), 'fan_f': (-2.5, 2.5),
    'julian_power': (-12.0, 12.0), 'julian_dist': (-3.5, 3.5),
    'blob_low': (-1.0, 1.0), 'blob_high': (0.0, 3.0), 'blob_waves': (1.0, 31.0),
    'pdj_a': (-2.5, 2.5), 'pdj_b': (-2.5, 2.5),
    'pdj_c': (-2.5, 2.5), 'pdj_d': (-2.5, 2.5),
    'fan2_x': (-2.5, 2.5), 'fan2_y': (-2.5, 2.5),
    'rings2_val': (-2.5, 2.5),
    'splits_x': (-2.5, 2.5), 'splits_y': (-2.5, 2.5),
    'curl_c1': (-2.5, 2.5), 'curl_c2': (-2.5, 2.5),
    'rect_x': (-2.5, 2.5), 'rect_y': (-2.5, 2.5),
    'check_size': (-2.5, 2.5), 'check_x': (-2.5, 2.5), 'check_y': (-2.5, 2.5),
    'hex_size': (-2.5, 2.5),
    'kal_pull': (-2.5, 2.5), 'kal_rotate': (-2.5, 2.5), 'kal_n': (3.0, 8.0),
    'icon_degree': (3.0, 23.0), 'icon_lambda': (-2.7, 2.6),
    'icon_alpha': (-2.5, 10.0), 'icon_beta': (-16.79, 1.5),
    'icon_gamma': (-0.82, 1.0), 'icon_omega': (-0.15, 0.188),
    'sat_m': (2.0, 12.0),
    'wallpaper_group': (0.0, 16.0), 'frieze_group': (0.0, 6.0),
    'rings3_val': (-2.5, 2.5), 'rings3_n': (-2.5, 2.5),
    'mobius_re_a': (-0.5, 0.5), 'mobius_re_b': (-0.5, 0.5),
    'mobius_re_c': (-0.5, 0.5), 'mobius_re_d': (-0.5, 0.5),
    'mobius_im_a': (-0.5, 0.5), 'mobius_im_b': (-0.5, 0.5),
    'mobius_im_c': (-0.5, 0.5), 'mobius_im_d': (-0.5, 0.5),
    'cpow_r': (0.5, 2.0), 'cpow_i': (-0.5, 0.5), 'cpow_power': (2.0, 7.0),
    'ngon_circle': (0.5, 2.0), 'ngon_corners': (0.5, 4.0),
    'ngon_power': (1.0, 5.0), 'ngon_sides': (3.0, 8.0),
    'epispiral_n': (2.0, 50.0), 'epispiral_thickness': (-2.5, 2.5),
    'epispiral_holes': (-5.0, 5.0),
    'waves3_scalex': (0.01, 0.2), 'waves3_scaley': (0.01, 0.2),
    'waves3_freqx': (2.0, 15.0), 'waves3_freqy': (2.0, 15.0),
    'waves3_sx_freq': (0.0, 4.0), 'waves3_sy_freq': (0.0, 4.0),
    'boarders_c': (0.3, 0.7), 'boarders_cl': (0.1, 0.4), 'boarders_cr': (0.5, 0.9),
    'hypertile_re': (-0.8, 0.8), 'hypertile_im': (-0.8, 0.8),
    'cell_size': (0.5, 2.5),
    'whorl_inside': (-1.0, 1.0), 'whorl_outside': (-1.0, 1.0),
    'disc2_twist': (0.5, 3.0), 'disc2_cosadd': (-0.5, 0.5), 'disc2_sinadd': (-0.5, 0.5),
    'flower_holes': (0.0, 1.0), 'flower_petals': (3.0, 11.0),
    'collide_a': (0.0, 2.0), 'collide_num': (2.0, 9.0),
    'auger_freq': (1.0, 8.0), 'auger_weight': (0.1, 1.0),
    'auger_sym': (0.0, 1.0), 'auger_scale': (0.1, 1.0),
}

# Integer params that should be rounded after jitter
_INTEGER_PARAMS = {
    'julian_power', 'blob_waves', 'kal_n', 'icon_degree', 'sat_m',
    'wallpaper_group', 'frieze_group', 'cpow_power', 'ngon_sides',
    'flower_petals', 'collide_num',
}


def jitter_var_params(params: dict[str, float], rng: np.random.Generator,
                      scale: float = 0.1) -> dict[str, float]:
    """Apply gaussian noise to variation parameters.

    Each parameter is jittered by gaussian noise scaled to its valid range.
    The result is clamped to the valid range. Integer parameters are rounded.

    Args:
        params: existing parameter dict
        rng: numpy random generator
        scale: noise magnitude as fraction of parameter range (default 10%)

    Returns:
        New dict with jittered values (original is not modified).
    """
    result = {}
    for name, value in params.items():
        lo, hi = _PARAM_RANGES.get(name, (-2.5, 2.5))
        param_range = hi - lo
        noise = rng.normal(0, scale * param_range)
        new_val = float(np.clip(value + noise, lo, hi))
        if name in _INTEGER_PARAMS:
            new_val = float(round(new_val))
        result[name] = new_val
    return result
