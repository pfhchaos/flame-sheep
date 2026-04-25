"""Random parameter generation for parametric variations.

Ranges from JWildfire's randomize() methods:
- Most variations use the base class default: uniform [-2.5, 2.5]
- Julian/Juliascope: custom multimodal (power 2-12, dist bimodal)
- Icon: preset table with ±0.1 jitter (17 known-good parameter sets)
- Kaleidoscope: we override kal_n to [3, 8] (JWildfire's default allows
  meaningless negative values)
"""

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

    return {}
