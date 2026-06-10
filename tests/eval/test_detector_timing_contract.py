"""Detector instrumentation contract (§7): every BeatDetector subclass
must populate `last_timing` with at least a `total` key after each
detect*() call.

Decomposed-component detectors (current_system, madmom) emit named
components whose sum approximates `total`; black-box wrappers
(BeatNet) emit `{'total': ...}` only. We don't enforce the sum
constraint here — components include overlap and the test would be
brittle against measurement noise. We DO enforce: total is a finite
positive number, and at least one timing key is populated.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


_GTZAN_ROOT = Path.home() / 'datasets' / 'GTZAN'
_HAS_GTZAN_AUDIO = (_GTZAN_ROOT / 'genres' / 'pop' / 'pop.00010.wav').exists()


def _load_short_clip(sr: int = 22050) -> tuple[np.ndarray, int]:
    """3 seconds of pop.00010 — enough to exercise each detector once."""
    import librosa
    audio, real_sr = librosa.load(
        str(_GTZAN_ROOT / 'genres' / 'pop' / 'pop.00010.wav'),
        sr=sr, mono=True, duration=3.0)
    return audio.astype(np.float32), real_sr


def _assert_valid_timing(timing: dict, allow_components: bool = True) -> None:
    assert 'total' in timing, f'last_timing missing `total`: {timing}'
    assert timing['total'] > 0, f'total must be positive: {timing}'
    assert timing['total'] < 60, f'total looks impossibly large: {timing}'
    if not allow_components:
        assert list(timing.keys()) == ['total'], (
            f'expected total-only timing, got {timing}')


@pytest.mark.skipif(not _HAS_GTZAN_AUDIO, reason='GTZAN audio not installed')
@pytest.mark.slow
def test_current_system_detector_emits_components():
    from flame_sheep.eval.detectors import CurrentSystemDetector
    det = CurrentSystemDetector()
    audio, sr = _load_short_clip()
    det.detect(audio, sr)
    _assert_valid_timing(det.last_timing)
    # CurrentSystemDetector decomposes into cqt + csd_transform + peak_pick.
    for k in ('cqt', 'csd_transform', 'peak_pick'):
        assert k in det.last_timing, (
            f'missing component {k!r} in {det.last_timing}')


@pytest.mark.skipif(not _HAS_GTZAN_AUDIO, reason='GTZAN audio not installed')
@pytest.mark.slow
def test_beatnet_detector_emits_total_only():
    try:
        from flame_sheep.eval.detectors import BeatNetDetector
        det = BeatNetDetector(model_index=1)
    except (ImportError, FileNotFoundError) as e:
        pytest.skip(f'BeatNet stack unavailable: {e}')
    audio, sr = _load_short_clip()
    det.detect(audio, sr)
    # BeatNet is a black-box wrap — total-only is the contract.
    _assert_valid_timing(det.last_timing, allow_components=False)
    # Same for downbeat path.
    det.detect_downbeats(audio, sr)
    _assert_valid_timing(det.last_timing, allow_components=False)


@pytest.mark.skipif(not _HAS_GTZAN_AUDIO, reason='GTZAN audio not installed')
@pytest.mark.slow
def test_madmom_detector_emits_rnn_dbn_components():
    try:
        from flame_sheep.eval.detectors import MadmomDetector
        det = MadmomDetector()
    except (ImportError, FileNotFoundError) as e:
        pytest.skip(f'madmom unavailable: {e}')
    audio, sr = _load_short_clip(sr=44100)
    det.detect(audio, sr)
    _assert_valid_timing(det.last_timing)
    for k in ('rnn', 'dbn'):
        assert k in det.last_timing, (
            f'beat: missing component {k!r} in {det.last_timing}')
    det.detect_downbeats(audio, sr)
    _assert_valid_timing(det.last_timing)
    for k in ('rnn', 'dbn'):
        assert k in det.last_timing, (
            f'downbeat: missing component {k!r} in {det.last_timing}')
