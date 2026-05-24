"""Palette domain — generation, scoring.

Just `score_palette` lives here for now. Generation helpers
(`_random_palette` and friends) move out of `genome.py` in Stage 4.
"""

from .scoring import score_palette

__all__ = ['score_palette']
