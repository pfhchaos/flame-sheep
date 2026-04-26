"""
Transition shim — re-exports from the standalone flame_sheep_audio package.

All imports like `from flame_sheep.audio import X` continue to work.
New code should import from `flame_sheep_audio` directly.
"""

# Re-export everything from the standalone package
from flame_sheep_audio import *  # noqa: F401,F403
from flame_sheep_audio import (
    # Constants
    SAMPLE_RATE, DEFAULT_DEVICE, FFT_SIZE, HOP_SIZE, N_BINS, HISTORY_LEN, FREQS,
    # Types
    BeatEvent, BandState, AudioState, AudioSnapshot,
    SpectrumEngine, SpectrumFrame,
    # Band configuration
    BandConfig, EnergyBandDef, DetectionBandDef, default_band_config,
    make_mask, make_weights, A_WEIGHTS,
    # Beat detection
    FluxBeatDetector,
    # Energy
    EnergyAnalyzer,
    # Sources
    PipeWireSource, FeedSource,
    # Processors
    AudioProcessor, SyntheticAudioProcessor, list_monitor_devices,
)
