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

    return {}
