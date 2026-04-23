"""
Audio analysis package for flame-sheep.

Re-exports all public symbols so existing imports like
`from flame_sheep.audio import AudioProcessor, BeatEvent` continue to work.
"""

# Constants
from ._constants import (
    SAMPLE_RATE, DEFAULT_DEVICE, BLOCK_SIZE, FFT_SIZE, HOP_SIZE, N_BINS,
    HISTORY_LEN, FREQS,
)

# Types
from ._types import BeatEvent, AudioState, AudioSnapshot
from ._spectrum import SpectrumEngine, SpectrumFrame

# Band definitions and utilities
from ._bands import (
    AdaptiveBand, make_mask, make_weights, a_weight_curve, A_WEIGHTS,
    ALLOWED_RANGES, DEFAULT_RANGES,
    ADAPT_ALPHA, ADAPT_FAST_ALPHA, ADAPT_INTERVAL, ADAPT_ANCHOR,
    SECTION_THRESHOLD, FAST_ADAPT_FRAMES,
)

# Beat detection
from .beat_detector import FluxBeatDetector
from .drop_detector import DropDetector

# Energy analysis
from .energy import EnergyAnalyzer

# Signal sources
from .source import PipeWireSource, FeedSource

# Processor classes
from .processor import AudioProcessor, SyntheticAudioProcessor

# Backward-compat aliases for underscore-prefixed names
_FREQS = FREQS
_make_mask = make_mask
