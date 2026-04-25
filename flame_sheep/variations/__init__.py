"""Variation definitions for flame fractals.

This package defines the nonlinear variation functions that warp points
in the IFS chaos game. Each variation is identified by an integer index
that matches its position in the GPU shader switch statement (flame.comp).

Modules:
    _registry  — Variation enum, NUM_VARIATIONS, constants
    _params    — random parameter generation (from JWildfire randomize())
    _cpu       — CPU approximation for viability testing
"""

from ._registry import (
    Variation,
    NUM_VARIATIONS,
    MAX_VAR_PARAMS,
    MAX_PARAMS_PER_VAR,
    SLOT_SIZE,
    PARAMETRIC_VARIATIONS,
    VAR_PARAMS_SPEC,
)

from ._params import random_var_params

from ._cpu import apply_variation_cpu, apply_variations_cpu
