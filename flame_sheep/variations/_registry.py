"""Variation registry — enum indices matching flame.comp shader order."""


class Variation:
    """Variation function indices — must match order in flame.comp."""
    LINEAR      = 0
    SINUSOIDAL  = 1
    SPHERICAL   = 2
    SWIRL       = 3
    HORSESHOE   = 4
    POLAR       = 5
    HANDKERCHIEF= 6
    HEART       = 7
    DISK        = 8
    SPIRAL      = 9
    HYPERBOLIC  = 10
    DIAMOND     = 11
    EX          = 12
    JULIA       = 13
    BENT        = 14
    WAVES       = 15
    FISHEYE     = 16
    POPCORN     = 17
    EXPONENTIAL = 18
    POWER       = 19
    COSINE      = 20
    RINGS       = 21
    FAN         = 22
    BLOB        = 23
    PDJ         = 24
    FAN2        = 25
    RINGS2      = 26
    EYEFISH     = 27
    BUBBLE      = 28
    CYLINDER    = 29
    # --- extended variations (JWildfire / flam3 inspired) ---
    SPLITS      = 30
    CLOVERLEAF  = 31
    JULIAN      = 32
    JULIASCOPE  = 33
    TANGENT     = 34
    CROSS       = 35
    BUTTERFLY   = 36
    CURL        = 37
    # --- tiling variations ---
    RECTANGLES  = 38
    CHECKS      = 39
    HEX_MODULUS = 40
    KALEIDOSCOPE= 41
    # --- symmetry-generating variations ---
    ICON        = 42
    SATTRACTOR  = 43
    WALLPAPER   = 44
    FRIEZE      = 45
    RINGS3      = 46
    # --- new batch ---
    MOBIUS      = 47
    CPOW        = 48
    NGON        = 49
    LOONIE      = 50
    SCRY        = 51
    EPISPIRAL   = 52
    WAVES3      = 53


NUM_VARIATIONS = 54

# Variations that require per-transform parameters (var_params dict)
PARAMETRIC_VARIATIONS = {
    Variation.WAVES, Variation.POPCORN, Variation.RINGS, Variation.FAN,
    Variation.BLOB, Variation.PDJ, Variation.FAN2, Variation.RINGS2,
    Variation.JULIAN, Variation.JULIASCOPE,
    Variation.SPLITS, Variation.CURL,
    Variation.RECTANGLES, Variation.CHECKS,
    Variation.HEX_MODULUS, Variation.KALEIDOSCOPE,
    Variation.ICON, Variation.SATTRACTOR,
    Variation.WALLPAPER, Variation.FRIEZE,
    Variation.RINGS3,
    Variation.MOBIUS, Variation.CPOW, Variation.NGON,
    Variation.EPISPIRAL, Variation.WAVES3,
}

# Per-variation parameter spec: ordered list of param names.
# Params are packed alongside each active variation in the GPU buffer.
# Max 6 params per variation (icon has 6, the most).
MAX_PARAMS_PER_VAR = 8
SLOT_SIZE = 2 + MAX_PARAMS_PER_VAR  # var_idx, weight, p0..p7

VAR_PARAMS_SPEC = {
    Variation.WAVES:        ['waves_freq_x', 'waves_freq_y',
                             'waves_amp_x', 'waves_amp_y'],
    Variation.POPCORN:      ['popcorn_cx', 'popcorn_cy'],
    Variation.RINGS:        ['rings_c'],
    Variation.FAN:          ['fan_c', 'fan_f'],
    Variation.BLOB:         ['blob_low', 'blob_high', 'blob_waves'],
    Variation.PDJ:          ['pdj_a', 'pdj_b', 'pdj_c', 'pdj_d'],
    Variation.FAN2:         ['fan2_x', 'fan2_y'],
    Variation.RINGS2:       ['rings2_val'],
    Variation.JULIAN:       ['julian_power', 'julian_dist'],
    Variation.JULIASCOPE:   ['julian_power', 'julian_dist'],
    Variation.SPLITS:       ['splits_x', 'splits_y'],
    Variation.CURL:         ['curl_c1', 'curl_c2'],
    Variation.RECTANGLES:   ['rect_x', 'rect_y'],
    Variation.CHECKS:       ['check_size', 'check_x', 'check_y'],
    Variation.HEX_MODULUS:  ['hex_size'],
    Variation.KALEIDOSCOPE: ['kal_pull', 'kal_rotate', 'kal_n'],
    Variation.ICON:         ['icon_degree', 'icon_lambda', 'icon_alpha',
                             'icon_beta', 'icon_gamma', 'icon_omega'],
    Variation.SATTRACTOR:   ['sat_m'],
    Variation.WALLPAPER:    ['wallpaper_group'],
    Variation.FRIEZE:       ['frieze_group'],
    Variation.RINGS3:       ['rings3_val', 'rings3_n'],
    Variation.MOBIUS:        ['mobius_re_a', 'mobius_re_b', 'mobius_re_c', 'mobius_re_d',
                              'mobius_im_a', 'mobius_im_b', 'mobius_im_c', 'mobius_im_d'],
    Variation.CPOW:         ['cpow_r', 'cpow_i', 'cpow_power'],
    Variation.NGON:         ['ngon_circle', 'ngon_corners', 'ngon_power', 'ngon_sides'],
    Variation.EPISPIRAL:    ['epispiral_n', 'epispiral_thickness', 'epispiral_holes'],
    Variation.WAVES3:       ['waves3_scalex', 'waves3_scaley', 'waves3_freqx',
                              'waves3_freqy', 'waves3_sx_freq', 'waves3_sy_freq'],
}

# Backwards compat — old code may reference this
MAX_VAR_PARAMS = MAX_PARAMS_PER_VAR
