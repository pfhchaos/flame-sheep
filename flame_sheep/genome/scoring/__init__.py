"""Genome aesthetic scoring.

Multiple scoring paths share this package:
  - histogram.py:      _score_from_histogram — fast tier-1 metrics shared
                        by CPU (64x64 coarse) and GPU (full-res) paths
  - symmetry_score.py: _score_symmetry — tier-2 metrics (moderate cost)
  - scoring_channels.py: 4-channel HSL+A standardized CNN input pipeline
  - cnn_scorer.py:     CNN model load/predict
  - cluster_scorer.py, image_scorer.py, experimental_metrics.py:
                        alternative metric implementations
  - scorer.py:         subprocess entrypoint
  - gpu.py:            GPU-side scorer worker
"""

from .histogram import _score_from_histogram
from .symmetry_score import _score_symmetry

__all__ = ['_score_from_histogram', '_score_symmetry']
