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

    return {}
