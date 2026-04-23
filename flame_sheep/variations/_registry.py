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


NUM_VARIATIONS = 42

# Variations that require per-transform parameters (var_params dict)
PARAMETRIC_VARIATIONS = {
    Variation.JULIAN, Variation.JULIASCOPE,
    Variation.SPLITS, Variation.CURL,
    Variation.RECTANGLES, Variation.CHECKS,
    Variation.HEX_MODULUS, Variation.KALEIDOSCOPE,
}

# GPU var_params layout: 16 floats per transform
# [0] julian_power  [1] julian_dist  [2] splits_x   [3] splits_y
# [4] curl_c1       [5] curl_c2      [6] rect_x     [7] rect_y
# [8] check_size    [9] check_x      [10] check_y   [11] hex_size
# [12] kal_pull     [13] kal_rotate  [14] kal_n      [15] reserved
MAX_VAR_PARAMS = 16
