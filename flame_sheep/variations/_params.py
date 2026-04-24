"""Random parameter generation for parametric variations.

Ranges derived from JWildfire's randomize() methods where available.
"""

import numpy as np

from ._registry import Variation


def random_var_params(var_idx: int, rng: np.random.Generator) -> dict[str, float]:
    """Generate random parameters for a parametric variation.

    Returns an empty dict for non-parametric variations.
    """
    if var_idx in (Variation.JULIAN, Variation.JULIASCOPE):
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
            'splits_x': float(rng.uniform(-1.0, 1.0)),
            'splits_y': float(rng.uniform(-1.0, 1.0)),
        }

    elif var_idx == Variation.CURL:
        return {
            'curl_c1': float(rng.uniform(-1.0, 1.0)),
            'curl_c2': float(rng.uniform(-1.0, 1.0)),
        }

    elif var_idx == Variation.RECTANGLES:
        return {
            'rect_x': float(rng.uniform(0.2, 1.5)),
            'rect_y': float(rng.uniform(0.2, 1.5)),
        }

    elif var_idx == Variation.CHECKS:
        return {
            'check_size': float(rng.uniform(0.5, 3.0)),
            'check_x': float(rng.uniform(-0.5, 0.5)),
            'check_y': float(rng.uniform(-0.5, 0.5)),
        }

    elif var_idx == Variation.HEX_MODULUS:
        return {
            'hex_size': float(rng.uniform(0.3, 2.0)),
        }

    elif var_idx == Variation.KALEIDOSCOPE:
        n = float(rng.integers(3, 9))  # 3 to 8 sectors
        return {
            'kal_pull': float(rng.uniform(0.0, 1.0)),
            'kal_rotate': float(rng.uniform(0.0, 6.283)),
            'kal_n': n,
        }

    elif var_idx == Variation.ICON:
        # From JWildfire IconAttractorFunc.java randomize()
        degree = float(rng.integers(2, 13))
        return {
            'icon_degree': degree,
            'icon_lambda': float(rng.uniform(-3.0, 3.0)),
            'icon_alpha': float(rng.uniform(-3.0, 3.0)),
            'icon_beta': float(rng.uniform(-3.0, 3.0)),
            'icon_gamma': float(rng.uniform(-1.0, 1.0)),
            'icon_omega': float(rng.uniform(-1.0, 1.0)),
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
