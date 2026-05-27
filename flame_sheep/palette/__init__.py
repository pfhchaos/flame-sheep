"""Palette domain — generation, scoring."""

from .generation import _random_palette, _lerp_arr
from .scoring import score_palette

__all__ = ['_random_palette', '_lerp_arr', 'score_palette']
