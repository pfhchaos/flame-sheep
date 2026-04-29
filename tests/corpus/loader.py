"""Unified audio loader — load from any corpus source, resample to 48kHz mono."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from flame_sheep_audio import SAMPLE_RATE

log = logging.getLogger(__name__)


@dataclass
class AudioClip:
    """Loaded audio clip with metadata."""
    name: str
    audio: np.ndarray     # mono float32, 48kHz
    sample_rate: int      # always SAMPLE_RATE (48000)
    source: str           # 'osu', 'youtube', 'musdb18'
    tags: list[str]
    duration: float       # seconds

    @property
    def n_samples(self) -> int:
        return len(self.audio)


def resample_to_48k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample to 48kHz using linear interpolation."""
    if orig_sr == SAMPLE_RATE:
        return audio
    ratio = SAMPLE_RATE / orig_sr
    n_out = int(len(audio) * ratio)
    indices = np.clip((np.arange(n_out) / ratio).astype(int), 0, len(audio) - 1)
    return audio[indices]


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load audio file, convert to mono float32. Returns (audio, sample_rate)."""
    try:
        audio, sr = sf.read(str(path), dtype='float32')
    except Exception:
        # Fallback for MP3 via soundfile's underlying libsndfile
        # If that fails too, try with ffmpeg via subprocess
        audio, sr = _load_with_ffmpeg(path)

    # Convert to mono if needed
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio.astype(np.float32), sr


def _load_with_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    """Fallback audio loader using ffmpeg subprocess."""
    import subprocess
    import struct

    cmd = [
        'ffmpeg', '-i', str(path),
        '-f', 'f32le',        # raw float32 little-endian
        '-acodec', 'pcm_f32le',
        '-ac', '1',           # mono
        '-ar', str(SAMPLE_RATE),
        '-v', 'quiet',
        'pipe:1'
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f'ffmpeg failed for {path}')

    audio = np.frombuffer(result.stdout, dtype=np.float32)
    return audio, SAMPLE_RATE


def load_clip(path: Path, name: str, source: str = 'unknown',
              tags: list[str] | None = None) -> AudioClip:
    """Load an audio file as a corpus AudioClip."""
    audio, sr = load_audio(path)
    audio = resample_to_48k(audio, sr)
    return AudioClip(
        name=name,
        audio=audio,
        sample_rate=SAMPLE_RATE,
        source=source,
        tags=tags or [],
        duration=len(audio) / SAMPLE_RATE,
    )
