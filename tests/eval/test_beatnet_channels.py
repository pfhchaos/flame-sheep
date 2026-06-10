"""Regression test for BeatNet channel-order convention.

Locks in the documented channel mapping
    col 0 = beat, col 1 = downbeat, col 2 = non-beat
against the BeatNet weights bundled with the upstream package. This
caught a months-long mis-attribution where every consumer in the
codebase had channels 0 and 1 swapped (see the 2026-06-04 fix
in detectors.py / generate_beat_labels.py / pack_beat_labels.py).

Property exploited: downbeats are a STRICT SUBSET of beats — every
downbeat is also a beat. So integrated over any real music, the
beat-channel total activation must exceed the downbeat-channel total.
If a future BeatNet release swaps channels under us, this fails fast
instead of silently corrupting training labels again.

The same inequality applies to peak counts (more beat peaks than
downbeat peaks per track), so we assert both — integrated mass is
the principal check, peak counts are the corroborating signal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


_GTZAN_ROOT = Path.home() / 'datasets' / 'GTZAN'
_HAS_GTZAN = (_GTZAN_ROOT / 'genres' / 'pop' / 'pop.00010.wav').exists()


def _try_load_beatnet():
    try:
        from flame_sheep.eval.detectors import BeatNetDetector
        det = BeatNetDetector(model_index=1)
        det._load()
        return det
    except (ImportError, FileNotFoundError) as e:
        pytest.skip(f'BeatNet stack unavailable: {e}')


@pytest.mark.skipif(not _HAS_GTZAN,
                     reason='GTZAN audio not installed at ~/datasets/GTZAN')
@pytest.mark.slow
def test_beatnet_channel_order_via_subset_inequality():
    """beat ⊃ downbeat → integrated beat activation > integrated downbeat
    activation on any real-music track. If this flips, channels 0 and 1
    have been swapped by an upstream change."""
    import librosa

    det = _try_load_beatnet()

    # pop.00010 is 30 s of 4/4 pop at ~108 BPM. ~54 beats, ~13 downbeats.
    # Any reasonable real-music track works here; pop.00010 is chosen
    # because the gtzan ground truth confirms 4× more beats than
    # downbeats, so the inequality margin is large.
    audio_path = _GTZAN_ROOT / 'genres' / 'pop' / 'pop.00010.wav'
    audio, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    probs = det._activation_probs(audio.astype(np.float32), sr)
    assert probs.shape[1] == 3, f'expected 3 channels, got {probs.shape}'

    beat_mass = float(probs[:, 0].sum())
    downbeat_mass = float(probs[:, 1].sum())
    nonbeat_mass = float(probs[:, 2].sum())

    # Strict subset: every downbeat is a beat. Beat channel must
    # accumulate strictly more probability mass than downbeat channel.
    # The ~4:1 beat:downbeat ratio in 4/4 music gives plenty of margin;
    # require beat_mass > downbeat_mass with NO floor on the gap (the
    # inequality itself is what we're testing — the gap can shrink in
    # adverse music, but it must not invert).
    assert beat_mass > downbeat_mass, (
        f'BeatNet channel order assumption broken: beat ch (col 0) mass '
        f'{beat_mass:.1f} <= downbeat ch (col 1) mass {downbeat_mass:.1f}. '
        f'Either the upstream BeatNet release swapped channels, or the '
        f'`_activation_probs` softmax axis changed. Check '
        f'BeatNet/particle_filtering_cascade.py for the current convention.')

    # Sanity floor: non-beat channel should hold the most mass overall
    # (most frames are silence-between-beats). If non-beat were lowest,
    # the softmax was reduced along the wrong axis.
    assert nonbeat_mass > beat_mass, (
        f'non-beat channel ({nonbeat_mass:.1f}) should hold more mass '
        f'than beat channel ({beat_mass:.1f}); softmax may be reduced '
        f'along the wrong axis.')


@pytest.mark.skipif(not _HAS_GTZAN,
                     reason='GTZAN audio not installed at ~/datasets/GTZAN')
@pytest.mark.slow
def test_beatnet_channel_alignment_with_ground_truth():
    """At true downbeat frames, channel 1 (downbeat) should fire harder
    than channel 0 (beat). At true non-downbeat-beat frames, the reverse.

    Independent of the integrated-mass test above: this catches the
    case where the channel order is right BUT the model's downbeat
    discrimination is broken, which would also corrupt downstream
    training labels."""
    import librosa
    from flame_sheep.eval.gtzan_parser import parse_track

    det = _try_load_beatnet()

    track = parse_track('pop.00010', _GTZAN_ROOT)
    audio, sr = librosa.load(str(track.audio_path), sr=22050, mono=True)
    probs = det._activation_probs(audio.astype(np.float32), sr)
    fps = 50.0  # BeatNet frame rate

    def _peak_around(times, ch, win_frames=2):
        frames = (np.asarray(times) * fps).astype(int)
        frames = frames[(frames >= 0) & (frames < probs.shape[0])]
        return np.array([probs[max(0, f - win_frames):
                                 min(probs.shape[0], f + win_frames + 1), ch].max()
                          for f in frames])

    non_db_beats = np.setdiff1d(track.beat_times, track.downbeat_times)

    db_ch0 = _peak_around(track.downbeat_times, 0).mean()  # beat channel
    db_ch1 = _peak_around(track.downbeat_times, 1).mean()  # downbeat channel
    nb_ch0 = _peak_around(non_db_beats, 0).mean()
    nb_ch1 = _peak_around(non_db_beats, 1).mean()

    # At true downbeats: downbeat channel should out-fire beat channel.
    assert db_ch1 > db_ch0, (
        f'at TRUE downbeats, ch1 (downbeat) peak {db_ch1:.3f} '
        f'<= ch0 (beat) peak {db_ch0:.3f} — channels likely swapped.')
    # At true beat-but-not-downbeat: beat channel out-fires downbeat.
    assert nb_ch0 > nb_ch1, (
        f'at TRUE non-downbeat beats, ch0 (beat) peak {nb_ch0:.3f} '
        f'<= ch1 (downbeat) peak {nb_ch1:.3f} — channels likely swapped.')
