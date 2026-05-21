"""Tests for the beat-detector eval interface and CurrentSystemDetector
wrapper."""
from __future__ import annotations

import numpy as np
import pytest

from flame_sheep.eval.detectors import (
    BeatDetector,
    CurrentSystemDetector,
)


def _click_track(beat_times_sec: list[float],
                 duration_sec: float = 5.0,
                 sr: int = 48000) -> np.ndarray:
    """Make audio with a sharp click at each beat time. Each click is a
    short burst (~5 ms) of random noise — enough broadband energy to
    fire any reasonable onset detector."""
    n = int(duration_sec * sr)
    audio = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(0)
    click_len = int(0.005 * sr)  # 5 ms
    envelope = np.exp(-np.linspace(0, 6, click_len))  # fast decay
    for t in beat_times_sec:
        start = int(t * sr)
        if start + click_len > n:
            continue
        click = rng.standard_normal(click_len).astype(np.float32) * envelope
        audio[start:start + click_len] += click
    return audio


def test_interface_contract():
    """BeatDetector must be abstract; subclasses must implement detect."""
    with pytest.raises(TypeError):
        BeatDetector()
    d = CurrentSystemDetector()
    assert d.name == 'current_system'
    assert d.version.startswith('current_system_')


def test_version_is_stable_within_run():
    """Repeated construction yields the same version string (used as
    cache key — drift across runs would invalidate cache without code
    actually changing)."""
    v1 = CurrentSystemDetector().version
    v2 = CurrentSystemDetector().version
    assert v1 == v2


def test_returns_sorted_seconds():
    """detect() output must be a 1D float array of beat times in
    seconds, sorted ascending."""
    d = CurrentSystemDetector()
    audio = _click_track([0.5, 1.0, 1.5, 2.0], duration_sec=3.0)
    beats = d.detect(audio, sr=48000)
    assert beats.ndim == 1
    assert beats.dtype.kind == 'f'
    assert np.all(np.diff(beats) >= 0)


def test_detects_regular_clicks():
    """Heuristic check: on a clean click track at known times the
    detector should find SOMETHING near each click. We don't assert
    high F1 here (that's the harness's job) — just that the wrapper
    isn't a no-op."""
    d = CurrentSystemDetector()
    expected = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    audio = _click_track(expected, duration_sec=4.0)
    beats = d.detect(audio, sr=48000)
    # At least one detected beat must land within ±100ms of each click,
    # for most of the clicks (allow a couple of misses for the very
    # first one while the engine warms up).
    hits = 0
    for t in expected:
        if np.any(np.abs(beats - t) < 0.1):
            hits += 1
    assert hits >= 4, f'only {hits}/{len(expected)} clicks detected: {beats}'


def test_handles_other_sample_rates():
    """44.1 kHz audio should be resampled to the engine's native rate
    without crashing and without losing all clicks."""
    d = CurrentSystemDetector()
    audio = _click_track([0.5, 1.0, 1.5, 2.0], duration_sec=3.0, sr=44100)
    beats = d.detect(audio, sr=44100)
    # Should detect at least one click after resampling.
    assert len(beats) >= 1


def test_empty_audio_returns_empty():
    d = CurrentSystemDetector()
    beats = d.detect(np.zeros(48000, dtype=np.float32), sr=48000)
    # Silence shouldn't produce beats.
    assert len(beats) == 0


def test_mono_conversion():
    """Stereo input is silently downmixed to mono."""
    d = CurrentSystemDetector()
    mono = _click_track([0.5, 1.0, 1.5], duration_sec=2.5)
    stereo = np.stack([mono, mono])  # (2, N)
    beats = d.detect(stereo, sr=48000)
    assert len(beats) >= 1
