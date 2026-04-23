"""Shared types for audio analysis."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BeatEvent:
    """A discrete audio event detected by the audio engine.

    Axes should ignore event kinds they don't recognize — new kinds
    can be added without updating every consumer.
    """
    kind: str       # 'kick' | 'snare' | 'clap' | 'hihat' | 'drop' | 'song_start'
    energy: float   # normalized 0..1, how strong the onset was


@dataclass
class AudioState:
    """Per-frame audio state passed to visual axes.

    Bundles discrete events with continuous features so axes get
    everything through one interface.
    """
    events: list[BeatEvent] = field(default_factory=list)
    rms: float = 0.0
    percussiveness: float = 0.5
    centroid: float = 1000.0
    centroid_delta: float = 0.0
    centroid_rms: float = 0.0
    bpm: float = 0.0
    drifting: bool = False


@dataclass
class AudioSnapshot:
    """Atomic audio state snapshot from the audio thread.

    Returned by AudioProcessor.drain() — contains all accumulated beat events
    since the last drain, plus the latest spectrum/RMS/waveform/continuous features.
    """
    events: list[BeatEvent] = field(default_factory=list)
    spectrum: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    rms: float = 0.0
    waveform: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    centroid: float = 1000.0
    centroid_delta: float = 0.0
    centroid_rms: float = 0.0
    percussiveness: float = 0.5
    band_rms: dict = field(default_factory=lambda: {'kick': 0.0, 'snare': 0.0, 'clap': 0.0, 'hihat': 0.0})
