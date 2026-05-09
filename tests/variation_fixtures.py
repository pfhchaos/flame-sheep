"""Shared test fixtures for variation testing.

Single source of truth for test points, parameter sets, affine fixtures,
and variation classification (RNG, affine-reading, etc). Imported by:
  - tests/test_golden_masters.py
  - tests/test_gpu_golden.py
  - tests/test_gpu_pipeline.py
  - tests/test_flam3_oracle.py
  - tools/generate_golden_masters.py
"""

import numpy as np

from flame_sheep.variations._registry import Variation

# ---------------------------------------------------------------------------
# Test points
# ---------------------------------------------------------------------------

# Generic points exercising conditional branches:
#   - All four quadrants (bent, bent2, separation sign checks)
#   - Inside and outside unit circle (loonie, whorl, flipcircle)
#   - Near grid boundaries (rectangles, cell, modulus, checks)
#   - Near origin (spherical, spiral, scry singularities)
#   - Large radius (exponential blowup guards)
BASE_POINTS = [
    ( 1.0,   0.5),    # Q1, r>1
    (-0.3,   0.7),    # Q2
    ( 0.5,  -0.5),    # Q4, r≈0.71 (inside unit circle)
    ( 2.0,   1.0),    # Q1, large r
    (-1.0,  -1.0),    # Q3
    ( 0.1,   0.1),    # near origin but nonzero
    ( 0.8,  -0.2),    # Q4
    (-0.5,   0.3),    # Q2
    ( 1.5,  -0.8),    # Q4, large r
    ( 0.3,   1.2),    # Q1
    (-0.7,  -0.4),    # Q3
    ( 0.05,  0.02),   # very near origin (singularity probes)
    ( 0.95,  0.05),   # r≈1 from inside
    ( 1.05,  0.05),   # r≈1 from outside
    ( 3.0,  -2.0),    # large r, Q4
    ( 0.5,   0.0),    # on x-axis
]

# Golden master points (subset used for regression snapshots)
GOLDEN_POINTS = [
    ( 1.0,   0.5),
    ( 0.0,   0.0),
    (-0.3,   0.7),
    ( 0.5,  -0.5),
    ( 2.0,   1.0),
    (-0.8,  -0.4),
    ( 0.05,  0.03),
    ( 0.95,  0.31),
    ( 1.05,  0.31),
    ( 0.49,  0.49),
    ( 0.51,  0.51),
    (-1.5,   0.8),
]

# Per-variation edge-case point sets
_NEAR_ZERO = [(0.05, 0.03), (0.03, -0.05), (-0.04, 0.02), (0.02, 0.07)]
_UNIT_CIRCLE = [(0.9, 0.3), (1.1, 0.3), (0.6, 0.6), (0.0, 0.9),
                (0.0, 1.1), (-0.6, 0.7), (0.85, 0.4), (1.15, 0.4)]
_GRID_BOUNDARY = [(0.49, 0.49), (0.51, 0.51), (1.49, 0.5), (-0.49, -0.51),
                  (0.99, 0.01), (1.01, -0.01), (0.25, 0.75), (-1.5, 0.5)]
_RNG_COVERAGE = [
    (1.0, 0.5), (-0.3, 0.7), (0.5, -0.5), (2.0, 1.0),
    (-1.0, -1.0), (0.1, 0.1), (0.8, -0.2), (-0.5, 0.3),
    (1.5, -0.8), (0.3, 1.2), (-0.7, -0.4), (0.4, 0.9),
    (1.3, -0.3), (-0.8, 0.6), (0.6, 0.6), (-0.2, -0.9),
    (0.9, -0.1), (-0.6, -0.6), (1.1, 0.4), (0.2, -0.7),
    (-0.4, 1.1), (0.7, 0.3), (-1.2, 0.5), (1.8, -0.6),
]
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
    Variation.FAN_PARAM:     BASE_POINTS + _ANGLE_SECTORS,
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

    # --- Points avoiding float precision boundaries ---
    # SPLIT: cos(val*π)=0 at half-integers → f32/f64 sign ambiguity.
    # Avoid x or y where x*xsize or y*ysize is half-integer for any test param.
    # Test xsize values: 0.5, 2.0. Test ysize values: 0.5, 3.0.
    # Bad y: ±0.5 (with ysize=3.0 → cos(1.5π)=0), ±1.0 (with ysize=0.5 → cos(0.5π)=0)
    Variation.SPLIT:        [
        (1.1, 0.4), (-0.3, 0.7), (0.6, -0.6), (2.0, 0.9),
        (-1.1, -1.1), (0.1, 0.1), (0.8, -0.2), (-0.4, 0.3),
        (1.4, -0.8), (0.3, 1.2), (-0.7, -0.4), (0.95, 0.05),
        (1.05, 0.05), (0.6, 0.1), (1.7, -1.3), (0.37, 0.83),
    ],
    # FOCI: division-by-near-zero when exp(x)+exp(-x) ≈ cos(y) near origin
    Variation.FOCI:         [
        (1.0, 0.5), (-0.3, 0.7), (0.5, -0.5), (2.0, 1.0),
        (-1.0, -1.0), (0.8, -0.2), (-0.5, 0.3),
        (1.5, -0.8), (0.3, 1.2), (-0.7, -0.4),
        (0.95, 0.05), (1.05, 0.05), (3.0, -2.0), (0.5, 0.0),
    ],
}

# ---------------------------------------------------------------------------
# Fixed affine for affine-reading variations
# ---------------------------------------------------------------------------

TEST_AFFINE = np.array([0.8, 0.5, 0.3, -0.2, 0.6, 0.5], dtype=np.float32)

AFFINE_VARIATIONS = {Variation.WAVES, Variation.POPCORN,
                     Variation.RINGS, Variation.FAN}

# ---------------------------------------------------------------------------
# RNG variations — need deterministic xorshift seeding for GPU/CPU comparison
# ---------------------------------------------------------------------------

RANDOM_VARIATIONS = {
    Variation.JULIA, Variation.SATTRACTOR, Variation.WALLPAPER,
    Variation.FRIEZE, Variation.JULIAN, Variation.JULIASCOPE,
    Variation.ICON, Variation.CPOW,
    Variation.BOARDERS, Variation.FLOWER, Variation.BLADE,
    Variation.LISSAJOUS, Variation.EPISPIRAL,
    # batch 2
    Variation.BLUR, Variation.GAUSSIAN_BLUR, Variation.RADIAL_BLUR,
    Variation.SUPER_SHAPE, Variation.NOISE, Variation.PIE,
    Variation.ARCH, Variation.PARABOLA, Variation.RAYS,
    Variation.CONIC, Variation.SQUARE, Variation.TWINTRIAN,
    Variation.WEDGE_JULIA,
}

# ---------------------------------------------------------------------------
# Parameter fixtures for parametric variations
# ---------------------------------------------------------------------------

PARAM_FIXTURES: dict[int, dict[str, float]] = {
    # waves/popcorn/rings/fan (15/17/21/22) read from affine — no params
    Variation.WAVES_PARAM: {'waves_freq_x': 0.5, 'waves_freq_y': 0.3,
                            'waves_amp_x': 0.8, 'waves_amp_y': 0.6},
    Variation.POPCORN_PARAM: {'popcorn_cx': 0.3, 'popcorn_cy': 0.5},
    Variation.RINGS_PARAM: {'rings_c': 0.4},
    Variation.FAN_PARAM: {'fan_c': 0.3, 'fan_f': 0.5},
    Variation.WAVES2: {'waves2_scalex': 0.05, 'waves2_scaley': 0.05,
                       'waves2_freqx': 7.0, 'waves2_freqy': 13.0},
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
    Variation.BLOB: {'blob_low': 0.3, 'blob_high': 1.2, 'blob_waves': 6.0},
    Variation.PDJ: {'pdj_a': 1.4, 'pdj_b': 2.3, 'pdj_c': 2.4, 'pdj_d': 2.2},
    Variation.FAN2: {'fan2_x': 0.5, 'fan2_y': 1.2},
    Variation.RINGS2: {'rings2_val': 0.5},
    Variation.RINGS3: {'rings3_val': 0.5, 'rings3_n': 4.0},
    Variation.MOBIUS: {'mobius_re_a': 0.1, 'mobius_re_b': 0.2, 'mobius_re_c': -0.15,
                       'mobius_re_d': 0.21, 'mobius_im_a': 0.2, 'mobius_im_b': -0.12,
                       'mobius_im_c': -0.15, 'mobius_im_d': 0.1},
    Variation.CPOW: {'cpow_r': 1.0, 'cpow_i': 0.1, 'cpow_power': 1.5},
    Variation.NGON: {'ngon_circle': 1.0, 'ngon_corners': 2.0,
                     'ngon_power': 3.0, 'ngon_sides': 5.0},
    Variation.EPISPIRAL: {'epispiral_n': 6.0, 'epispiral_thickness': 0.0,
                          'epispiral_holes': 1.0},
    Variation.WAVES3: {'waves3_scalex': 0.05, 'waves3_scaley': 0.05,
                       'waves3_freqx': 7.0, 'waves3_freqy': 13.0,
                       'waves3_sx_freq': 0.0, 'waves3_sy_freq': 2.0},
    Variation.BOARDERS: {'boarders_c': 0.5, 'boarders_cl': 0.25, 'boarders_cr': 0.75},
    Variation.HYPERTILE: {'hypertile_re': 0.3, 'hypertile_im': 0.0},
    Variation.CELL: {'cell_size': 1.0},
    Variation.WHORL: {'whorl_inside': 0.5, 'whorl_outside': 0.5},
    Variation.DISC2: {'disc2_twist': 1.0, 'disc2_cosadd': 0.0, 'disc2_sinadd': 0.0},
    Variation.FLOWER: {'flower_holes': 0.5, 'flower_petals': 6.0},
    Variation.COLLIDEOSCOPE: {'collide_a': 0.5, 'collide_num': 5.0},
    Variation.AUGER: {'auger_freq': 3.0, 'auger_weight': 0.5,
                      'auger_sym': 0.5, 'auger_scale': 0.5},
    Variation.ECLIPSE: {'eclipse_shift': 0.5},
    Variation.LAYERED_SPIRAL: {'layered_spiral_radius': 1.0},
    Variation.STRIPES: {'stripes_space': 0.5, 'stripes_warp': 0.5},
    Variation.LISSAJOUS: {'liss_tmin': -3.14159265, 'liss_tmax': 3.14159265,
                          'liss_a': 3.0, 'liss_b': 2.0, 'liss_c': 0.0,
                          'liss_d': 0.0, 'liss_e': 0.0},
    Variation.RIPPLE: {'ripple_freq': 5.0, 'ripple_vel': 0.0, 'ripple_amp': 0.1,
                       'ripple_cx': 0.0, 'ripple_cy': 0.0, 'ripple_phase': 0.0,
                       'ripple_scale': 1.0, 'ripple_fixd': 1.0},
    # batch 2 parametric
    Variation.RADIAL_BLUR: {'radial_blur_angle': 0.5},
    Variation.PERSPECTIVE: {'perspective_angle': 0.8, 'perspective_dist': 2.0},
    Variation.SUPER_SHAPE: {'super_shape_rnd': 0.3, 'super_shape_m': 4.0,
                            'super_shape_n1': 2.0, 'super_shape_n2': 2.0,
                            'super_shape_n3': 2.0, 'super_shape_holes': 0.0},
    Variation.PIE: {'pie_slices': 6.0, 'pie_rotation': 0.0, 'pie_thickness': 0.5},
    Variation.PARABOLA: {'parabola_height': 1.0, 'parabola_width': 1.0},
    Variation.CONIC: {'conic_eccentricity': 1.0, 'conic_holes': 0.0},
    Variation.ESCHER: {'escher_beta': 0.3},
    Variation.OSCILLOSCOPE: {'osc_separation': 1.0, 'osc_frequency': 3.14159,
                             'osc_amplitude': 1.0, 'osc_damping': 0.1},
    Variation.CURVE: {'curve_xamp': 0.5, 'curve_yamp': 0.3,
                      'curve_xlength': 1.0, 'curve_ylength': 1.0},
    Variation.WEDGE_JULIA: {'wedge_julia_angle': 0.3, 'wedge_julia_count': 2.0,
                            'wedge_julia_power': 4.0, 'wedge_julia_dist': 1.0},
    Variation.WEDGE: {'wedge_angle': 0.3, 'wedge_hole': 0.0,
                      'wedge_count': 2.0, 'wedge_swirl': 0.5},
    Variation.WEDGE_SPH: {'wedge_sph_angle': 0.3, 'wedge_sph_hole': 0.0,
                          'wedge_sph_count': 2.0, 'wedge_sph_swirl': 0.5},
    Variation.LAZYSUSAN: {'lazysusan_x': 0.1, 'lazysusan_y': 0.1,
                          'lazysusan_spin': 0.5, 'lazysusan_space': 0.2,
                          'lazysusan_twist': 0.3},
    Variation.MODULUS_FUNC: {'modulus_x': 0.5, 'modulus_y': 0.5},
    Variation.BENT2: {'bent2_x': 1.5, 'bent2_y': -0.5},
    Variation.BIPOLAR: {'bipolar_shift': 0.3},
    Variation.FLUX: {'flux_spread': 0.5},
    Variation.SPLIT: {'split_xsize': 0.5, 'split_ysize': 0.5},
    Variation.SEPARATION: {'separation_x': 0.5, 'separation_y': 0.5,
                           'separation_xinside': 0.2, 'separation_yinside': 0.3},
    Variation.POPCORN2: {'popcorn2_x': 0.1, 'popcorn2_y': 0.1, 'popcorn2_c': 3.0},
}

# GPU golden test uses multiple param sets per variation for branch coverage
VAR_PARAM_SETS: dict[int, dict[str, dict]] = {}

def _build_param_sets():
    """Build GPU test param sets from PARAM_FIXTURES + extra configs."""
    # Start with single 'default' from PARAM_FIXTURES
    for var_idx, params in PARAM_FIXTURES.items():
        VAR_PARAM_SETS[var_idx] = {'default': params}

    # Non-parametric get empty default
    from flame_sheep.variations._registry import NUM_VARIATIONS, PARAMETRIC_VARIATIONS
    for i in range(NUM_VARIATIONS):
        if i not in VAR_PARAM_SETS:
            VAR_PARAM_SETS[i] = {'default': {}}

    # Extra param configs for thorough branch coverage
    _extras = {
        Variation.RINGS2: {'small': {'rings2_val': 0.1}},
        Variation.JULIAN: {'high_power': {'julian_power': 8.0, 'julian_dist': 1.0},
                           'negative': {'julian_power': -3.0, 'julian_dist': 1.0}},
        Variation.JULIASCOPE: {'high_power': {'julian_power': 6.0, 'julian_dist': 1.5}},
        Variation.SPLITS: {'large': {'splits_x': 2.0, 'splits_y': 1.5}},
        Variation.CURL: {'strong': {'curl_c1': 2.0, 'curl_c2': 1.0}},
        Variation.RECTANGLES: {'small': {'rect_x': 0.2, 'rect_y': 0.3}},
        Variation.CHECKS: {'fine': {'check_size': 0.5, 'check_x': 0.5, 'check_y': 0.3}},
        Variation.KALEIDOSCOPE: {'tri': {'kal_pull': 0.5, 'kal_rotate': 0.5, 'kal_n': 3.0}},
        Variation.ICON: {'high_sym': {'icon_degree': 6.0, 'icon_lambda': 1.2,
                                      'icon_alpha': 0.3, 'icon_beta': 0.1,
                                      'icon_gamma': 0.0, 'icon_omega': 0.5}},
        Variation.SATTRACTOR: {'low': {'sat_m': 3.0}},
        Variation.WALLPAPER: {'group3': {'wallpaper_group': 3.0}},
        Variation.FRIEZE: {'group2': {'frieze_group': 2.0}},
        Variation.WAVES_PARAM: {'high_freq': {'waves_freq_x': 5.0, 'waves_freq_y': 7.0,
                                              'waves_amp_x': 0.2, 'waves_amp_y': 0.1}},
        Variation.POPCORN_PARAM: {'strong': {'popcorn_cx': 1.5, 'popcorn_cy': 1.5}},
        Variation.RINGS_PARAM: {'tight': {'rings_c': 0.1}},
        Variation.FAN_PARAM: {'narrow': {'fan_c': 0.1, 'fan_f': 0.8}},
        Variation.BLOB: {'tight': {'blob_low': 0.8, 'blob_high': 1.0, 'blob_waves': 3.0}},
        Variation.PDJ: {'small': {'pdj_a': 0.5, 'pdj_b': 0.3, 'pdj_c': 0.7, 'pdj_d': 0.4}},
        Variation.FAN2: {'wide': {'fan2_x': 2.0, 'fan2_y': 0.3}},
        Variation.NGON: {'tri': {'ngon_circle': 0.5, 'ngon_corners': 1.0,
                                 'ngon_power': 2.0, 'ngon_sides': 3.0}},
        Variation.CPOW: {'high_power': {'cpow_r': 0.8, 'cpow_i': 0.3, 'cpow_power': 3.0}},
        Variation.EPISPIRAL: {'thick': {'epispiral_n': 4.0, 'epispiral_thickness': 0.5,
                                        'epispiral_holes': 0.5}},
        Variation.WAVES3: {'fast': {'waves3_scalex': 0.1, 'waves3_scaley': 0.1,
                                    'waves3_freqx': 15.0, 'waves3_freqy': 15.0,
                                    'waves3_sx_freq': 3.0, 'waves3_sy_freq': 3.0}},
        Variation.BOARDERS: {'always_border': {'boarders_c': 0.5, 'boarders_cl': 0.25,
                                               'boarders_cr': 0.0},
                             'never_border': {'boarders_c': 0.5, 'boarders_cl': 0.25,
                                              'boarders_cr': 1.1}},
        Variation.HYPERTILE: {'complex': {'hypertile_re': 0.2, 'hypertile_im': 0.3}},
        Variation.CELL: {'small': {'cell_size': 0.5}},
        Variation.WHORL: {'strong': {'whorl_inside': 3.0, 'whorl_outside': 3.0},
                          'asymmetric': {'whorl_inside': 0.1, 'whorl_outside': 2.0}},
        Variation.DISC2: {'shifted': {'disc2_twist': 2.0, 'disc2_cosadd': 0.3,
                                      'disc2_sinadd': -0.2}},
        Variation.FLOWER: {'many_petals': {'flower_holes': 0.3, 'flower_petals': 12.0}},
        Variation.COLLIDEOSCOPE: {'high_fold': {'collide_a': 0.3, 'collide_num': 7.0},
                                  'low_fold': {'collide_a': 0.8, 'collide_num': 3.0}},
        Variation.AUGER: {'strong': {'auger_freq': 7.0, 'auger_weight': 1.0,
                                     'auger_sym': 1.0, 'auger_scale': 1.0}},
        Variation.ECLIPSE: {'large': {'eclipse_shift': 1.5}},
        Variation.LAYERED_SPIRAL: {'tight': {'layered_spiral_radius': 3.0}},
        Variation.STRIPES: {'tight': {'stripes_space': 0.8, 'stripes_warp': 2.0}},
        Variation.LISSAJOUS: {'with_drift': {'liss_tmin': -3.14159265, 'liss_tmax': 3.14159265,
                                             'liss_a': 5.0, 'liss_b': 3.0, 'liss_c': 0.1,
                                             'liss_d': 0.5, 'liss_e': 0.3}},
        Variation.RIPPLE: {'product_dist': {'ripple_freq': 5.0, 'ripple_vel': 0.0,
                                           'ripple_amp': 0.1, 'ripple_cx': 0.0,
                                           'ripple_cy': 0.0, 'ripple_phase': 0.0,
                                           'ripple_scale': 1.0, 'ripple_fixd': 0.0}},
        Variation.WAVES2: {'strong': {'waves2_scalex': 0.2, 'waves2_scaley': 0.2,
                                      'waves2_freqx': 3.0, 'waves2_freqy': 5.0}},
        # batch 2 extra param configs for branch coverage
        Variation.BENT2: {'inverted': {'bent2_x': -2.0, 'bent2_y': 3.0}},
        Variation.OSCILLOSCOPE: {'no_damping': {'osc_separation': 0.5, 'osc_frequency': 2.0,
                                                'osc_amplitude': 2.0, 'osc_damping': 0.0}},
        Variation.SPLIT: {'large': {'split_xsize': 2.0, 'split_ysize': 3.0}},
        Variation.SEPARATION: {'no_inside': {'separation_x': 1.0, 'separation_y': 1.0,
                                             'separation_xinside': 0.0, 'separation_yinside': 0.0}},
        Variation.LAZYSUSAN: {'outside': {'lazysusan_x': 0.0, 'lazysusan_y': 0.0,
                                          'lazysusan_spin': 1.0, 'lazysusan_space': 0.5,
                                          'lazysusan_twist': 0.0}},
        Variation.MODULUS_FUNC: {'large': {'modulus_x': 2.0, 'modulus_y': 1.5}},
        Variation.BIPOLAR: {'strong': {'bipolar_shift': 1.5}},
        Variation.WEDGE_JULIA: {'neg_dist': {'wedge_julia_angle': 0.1, 'wedge_julia_count': 3.0,
                                             'wedge_julia_power': 6.0, 'wedge_julia_dist': -1.25}},
        Variation.PERSPECTIVE: {'steep': {'perspective_angle': 1.2, 'perspective_dist': 0.8}},
        Variation.SUPER_SHAPE: {'high_m': {'super_shape_rnd': 0.0, 'super_shape_m': 8.0,
                                           'super_shape_n1': 1.0, 'super_shape_n2': 1.0,
                                           'super_shape_n3': 1.0, 'super_shape_holes': 0.5}},
        Variation.POPCORN2: {'strong': {'popcorn2_x': 0.5, 'popcorn2_y': 0.5, 'popcorn2_c': 1.0}},
    }
    for var_idx, extras in _extras.items():
        if var_idx in VAR_PARAM_SETS:
            VAR_PARAM_SETS[var_idx].update(extras)
        else:
            VAR_PARAM_SETS[var_idx] = extras

_build_param_sets()
