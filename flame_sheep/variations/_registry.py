"""Variation registry — enum indices matching flame.comp shader order."""
from __future__ import annotations


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
    RINGS3      = 46   # note: not the same as RINGS_PARAM
    # --- new batch ---
    MOBIUS      = 47
    CPOW        = 48
    NGON        = 49
    LOONIE      = 50
    SCRY        = 51
    EPISPIRAL   = 52
    WAVES3      = 53
    # --- tiling batch 2 ---
    BOARDERS    = 54
    HYPERTILE   = 55
    CELL        = 56
    WHORL       = 57
    DISC2       = 58
    # --- shape & spiral batch ---
    FLOWER      = 59
    BLADE       = 60
    SPIRALWING  = 61
    COLLIDEOSCOPE = 62
    AUGER       = 63
    # --- reflective & misc batch ---
    FLIPCIRCLE  = 64
    ECLIPSE     = 65
    LAYERED_SPIRAL = 66
    STRIPES     = 67
    LISSAJOUS   = 68
    RIPPLE      = 69
    # --- displaced parameterized variants (originals at 15/17/21/22 now read from affine) ---
    WAVES_PARAM   = 70   # waves formula with explicit params (was idx 15)
    POPCORN_PARAM = 71   # popcorn formula with explicit params (was idx 17)
    RINGS_PARAM   = 72   # rings formula with explicit param (was idx 21)
    FAN_PARAM     = 73   # fan formula with explicit params (was idx 22)
    # --- flam3 standard additions ---
    WAVES2        = 74   # x + scalex*sin(y*freqx), distinct formula from waves
    # --- flam3 standard variations (batch 2) ---
    BLUR          = 75
    GAUSSIAN_BLUR = 76
    RADIAL_BLUR   = 77
    PERSPECTIVE   = 78
    SUPER_SHAPE   = 79
    NOISE         = 80
    SECANT2       = 81
    PIE           = 82
    ARCH          = 83
    PARABOLA      = 84
    RAYS          = 85
    CONIC         = 86
    ESCHER        = 87
    ELLIPTIC      = 88
    OSCILLOSCOPE  = 89
    EDISC         = 90
    SQUARE        = 91
    CURVE         = 92
    TWINTRIAN     = 93
    WEDGE_JULIA   = 94
    WEDGE         = 95
    WEDGE_SPH     = 96
    LAZYSUSAN     = 97
    MODULUS_FUNC  = 98   # 'modulus' in flam3 XML
    BENT2         = 99
    BIPOLAR       = 100
    FLUX          = 101
    SPLIT         = 102  # different from SPLITS (idx 30)
    SEPARATION    = 103
    POLAR2        = 104
    FOCI          = 105
    POPCORN2      = 106
    SECANT_FUNC   = 107  # original secant (before secant2 improvement)
    # --- complex trig variations (z = x + iy) ---
    SIN_FUNC      = 108
    COS_FUNC      = 109
    TAN_FUNC      = 110
    SEC_FUNC      = 111
    CSC_FUNC      = 112
    COT_FUNC      = 113
    SINH_FUNC     = 114
    COSH_FUNC     = 115
    TANH_FUNC     = 116
    SECH_FUNC     = 117
    CSCH_FUNC     = 118
    COTH_FUNC     = 119
    EXP_FUNC      = 120  # complex exp, different from EXPONENTIAL (idx 18)
    LOG_FUNC      = 121


NUM_VARIATIONS = 122

# Variations that require per-transform parameters (var_params dict).
# Note: WAVES/POPCORN/RINGS/FAN (idx 15/17/21/22) read from the affine
# coefficients at render time — they are NOT parametric.  Their parameterized
# counterparts are WAVES_PARAM/POPCORN_PARAM/RINGS_PARAM/FAN_PARAM.
PARAMETRIC_VARIATIONS = {
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
    Variation.BOARDERS, Variation.HYPERTILE, Variation.CELL,
    Variation.WHORL, Variation.DISC2,
    Variation.FLOWER, Variation.COLLIDEOSCOPE, Variation.AUGER,
    Variation.ECLIPSE, Variation.LAYERED_SPIRAL, Variation.STRIPES,
    Variation.LISSAJOUS, Variation.RIPPLE,
    # displaced parameterized variants
    Variation.WAVES_PARAM, Variation.POPCORN_PARAM,
    Variation.RINGS_PARAM, Variation.FAN_PARAM,
    Variation.WAVES2,
    # flam3 standard batch 2
    Variation.RADIAL_BLUR, Variation.PERSPECTIVE, Variation.SUPER_SHAPE,
    Variation.PIE, Variation.PARABOLA, Variation.CONIC, Variation.ESCHER,
    Variation.OSCILLOSCOPE, Variation.CURVE, Variation.WEDGE_JULIA,
    Variation.WEDGE, Variation.WEDGE_SPH, Variation.LAZYSUSAN,
    Variation.MODULUS_FUNC, Variation.BENT2, Variation.BIPOLAR,
    Variation.FLUX, Variation.SPLIT, Variation.SEPARATION,
    Variation.POPCORN2,
}

# Per-variation parameter spec: ordered list of param names.
# Params are packed alongside each active variation in the GPU buffer.
# Max 6 params per variation (icon has 6, the most).
MAX_PARAMS_PER_VAR = 8
SLOT_SIZE = 2 + MAX_PARAMS_PER_VAR  # var_idx, weight, p0..p7

VAR_PARAMS_SPEC = {
    # Note: WAVES (15), POPCORN (17), RINGS (21), FAN (22) read from the
    # affine at render time — no param spec needed for them.
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
    Variation.BOARDERS:     ['boarders_c', 'boarders_cl', 'boarders_cr'],
    Variation.HYPERTILE:    ['hypertile_re', 'hypertile_im'],
    Variation.CELL:         ['cell_size'],
    Variation.WHORL:        ['whorl_inside', 'whorl_outside'],
    Variation.DISC2:        ['disc2_rot', 'disc2_twist'],
    Variation.FLOWER:       ['flower_holes', 'flower_petals'],
    Variation.COLLIDEOSCOPE: ['collide_a', 'collide_num'],
    Variation.AUGER:        ['auger_freq', 'auger_weight', 'auger_sym', 'auger_scale'],
    Variation.ECLIPSE:      ['eclipse_shift'],
    Variation.LAYERED_SPIRAL: ['layered_spiral_radius'],
    Variation.STRIPES:      ['stripes_space', 'stripes_warp'],
    Variation.LISSAJOUS:    ['liss_tmin', 'liss_tmax', 'liss_a', 'liss_b',
                             'liss_c', 'liss_d', 'liss_e'],
    Variation.RIPPLE:       ['ripple_freq', 'ripple_vel', 'ripple_amp',
                             'ripple_cx', 'ripple_cy', 'ripple_phase',
                             'ripple_scale', 'ripple_fixd'],
    # displaced parameterized variants (same formulas as originals, explicit params)
    Variation.WAVES_PARAM:  ['waves_freq_x', 'waves_freq_y',
                             'waves_amp_x', 'waves_amp_y'],
    Variation.POPCORN_PARAM: ['popcorn_cx', 'popcorn_cy'],
    Variation.RINGS_PARAM:  ['rings_c'],
    Variation.FAN_PARAM:    ['fan_c', 'fan_f'],
    # flam3 waves2 — different formula from waves
    Variation.WAVES2:       ['waves2_scalex', 'waves2_scaley',
                             'waves2_freqx', 'waves2_freqy'],
    # flam3 standard batch 2
    Variation.RADIAL_BLUR:  ['radial_blur_angle'],
    Variation.PERSPECTIVE:  ['perspective_angle', 'perspective_dist'],
    Variation.SUPER_SHAPE:  ['super_shape_rnd', 'super_shape_m',
                             'super_shape_n1', 'super_shape_n2',
                             'super_shape_n3', 'super_shape_holes'],
    Variation.PIE:          ['pie_slices', 'pie_rotation', 'pie_thickness'],
    Variation.PARABOLA:     ['parabola_height', 'parabola_width'],
    Variation.CONIC:        ['conic_eccentricity', 'conic_holes'],
    Variation.ESCHER:       ['escher_beta'],
    Variation.OSCILLOSCOPE: ['osc_separation', 'osc_frequency',
                             'osc_amplitude', 'osc_damping'],
    Variation.CURVE:        ['curve_xamp', 'curve_yamp',
                             'curve_xlength', 'curve_ylength'],
    Variation.WEDGE_JULIA:  ['wedge_julia_angle', 'wedge_julia_count',
                             'wedge_julia_power', 'wedge_julia_dist'],
    Variation.WEDGE:        ['wedge_angle', 'wedge_hole',
                             'wedge_count', 'wedge_swirl'],
    Variation.WEDGE_SPH:    ['wedge_sph_angle', 'wedge_sph_hole',
                             'wedge_sph_count', 'wedge_sph_swirl'],
    Variation.LAZYSUSAN:    ['lazysusan_x', 'lazysusan_y', 'lazysusan_spin',
                             'lazysusan_space', 'lazysusan_twist'],
    Variation.MODULUS_FUNC: ['modulus_x', 'modulus_y'],
    Variation.BENT2:        ['bent2_x', 'bent2_y'],
    Variation.BIPOLAR:      ['bipolar_shift'],
    Variation.FLUX:         ['flux_spread'],
    Variation.SPLIT:        ['split_xsize', 'split_ysize'],
    Variation.SEPARATION:   ['separation_x', 'separation_y',
                             'separation_xinside', 'separation_yinside'],
    Variation.POPCORN2:     ['popcorn2_x', 'popcorn2_y', 'popcorn2_c'],
}

# Backwards compat — old code may reference this
MAX_VAR_PARAMS = MAX_PARAMS_PER_VAR
