"""Tests for the osu parser + beat derivation."""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from flame_sheep.eval.osu_parser import parse_osu_file, OsuMap


def _write_osu(tmp_path: Path, body: str) -> Path:
    p = tmp_path / 'test.osu'
    p.write_text(textwrap.dedent(body).lstrip())
    return p


def test_single_uninherited_140_bpm(tmp_path):
    """140 BPM = 60000/140 ≈ 428.571 ms/beat. First beat at offset_ms,
    subsequent every 428.571 ms."""
    body = """
        osu file format v14

        [General]
        AudioFilename: audio.mp3
        AudioLeadIn: 0

        [TimingPoints]
        725,428.571428571429,4,2,1,80,1,0
    """
    m = parse_osu_file(_write_osu(tmp_path, body))
    assert m.audio_filename == 'audio.mp3'
    assert m.audio_lead_in_ms == 0
    assert len(m.uninherited_points) == 1

    beats, downbeats = m.beat_times(audio_duration_sec=5.0)
    # First beat at 0.725s, 428.57ms spacing.
    np.testing.assert_allclose(beats[0], 0.725, atol=1e-6)
    np.testing.assert_allclose(np.diff(beats), 0.428571, atol=1e-3)
    # 4/4 meter: first, fifth, ninth ... are downbeats.
    np.testing.assert_array_equal(downbeats, beats[::4])


def test_inherited_points_are_ignored(tmp_path):
    """Inherited timing points (beat_length < 0) must NOT contribute beats."""
    body = """
        osu file format v14

        [General]
        AudioFilename: audio.mp3
        AudioLeadIn: 0

        [TimingPoints]
        725,428.571428571429,4,2,1,80,1,0
        1582,-100,4,2,1,80,0,0
        2224,-100,4,2,1,80,0,0
    """
    m = parse_osu_file(_write_osu(tmp_path, body))
    assert len(m.timing_points) == 3
    assert len(m.uninherited_points) == 1
    # Beats should be identical to single-uninherited case.
    beats, _ = m.beat_times(audio_duration_sec=5.0)
    np.testing.assert_allclose(beats[0], 0.725, atol=1e-6)
    np.testing.assert_allclose(np.diff(beats), 0.428571, atol=1e-3)


def test_tempo_change_midsong(tmp_path):
    """Two uninherited points → tempo switches partway."""
    body = """
        osu file format v14

        [General]
        AudioFilename: audio.mp3
        AudioLeadIn: 0

        [TimingPoints]
        0,500,4,2,1,80,1,0
        5000,250,4,2,1,80,1,0
    """
    m = parse_osu_file(_write_osu(tmp_path, body))
    assert len(m.uninherited_points) == 2
    beats, downbeats = m.beat_times(audio_duration_sec=7.0)
    # First segment: 0..5s, 500ms spacing → 10 beats at 0, .5, 1, ..., 4.5
    # Second segment: 5..7s, 250ms spacing → 8 beats at 5, 5.25, ..., 6.75
    seg1 = beats[beats < 5.0]
    seg2 = beats[beats >= 5.0]
    np.testing.assert_allclose(seg1, np.arange(10) * 0.5, atol=1e-9)
    np.testing.assert_allclose(seg2, 5.0 + np.arange(8) * 0.25, atol=1e-9)
    # Downbeats reset on the tempo change: first beat of each segment counts.
    assert 5.0 in downbeats  # second segment's first beat


def test_lead_in_subtraction(tmp_path):
    """AudioLeadIn > 0 must subtract from beat times (osu times are
    game-time, beats are audio-time)."""
    body = """
        osu file format v14

        [General]
        AudioFilename: audio.mp3
        AudioLeadIn: 1000

        [TimingPoints]
        2000,500,4,2,1,80,1,0
    """
    m = parse_osu_file(_write_osu(tmp_path, body))
    beats, _ = m.beat_times(audio_duration_sec=5.0)
    # offset=2000ms game time, lead_in=1000ms → first beat at 1000ms audio time.
    np.testing.assert_allclose(beats[0], 1.0, atol=1e-9)


def test_lead_in_clips_negative_beats(tmp_path):
    """If lead-in pushes a beat into negative audio time, drop it."""
    body = """
        osu file format v14

        [General]
        AudioFilename: audio.mp3
        AudioLeadIn: 5000

        [TimingPoints]
        2000,500,4,2,1,80,1,0
    """
    m = parse_osu_file(_write_osu(tmp_path, body))
    beats, _ = m.beat_times(audio_duration_sec=10.0)
    assert beats.min() >= 0


def test_non_four_four_meter(tmp_path):
    """3/4 meter: every 3rd beat is a downbeat."""
    body = """
        osu file format v14

        [General]
        AudioFilename: audio.mp3
        AudioLeadIn: 0

        [TimingPoints]
        0,500,3,2,1,80,1,0
    """
    m = parse_osu_file(_write_osu(tmp_path, body))
    beats, downbeats = m.beat_times(audio_duration_sec=6.0)
    # 12 beats total; downbeats at indices 0, 3, 6, 9.
    np.testing.assert_array_equal(downbeats, beats[::3])


def test_no_uninherited_raises(tmp_path):
    body = """
        osu file format v14

        [General]
        AudioFilename: audio.mp3

        [TimingPoints]
        100,-100,4,2,1,80,0,0
    """
    with pytest.raises(ValueError, match='no uninherited'):
        parse_osu_file(_write_osu(tmp_path, body))


def test_no_audio_filename_raises(tmp_path):
    body = """
        osu file format v14

        [General]
        AudioLeadIn: 0

        [TimingPoints]
        0,500,4,2,1,80,1,0
    """
    with pytest.raises(ValueError, match='no AudioFilename'):
        parse_osu_file(_write_osu(tmp_path, body))


def test_real_corpus_sample(tmp_path):
    """Parse one real .osu file from the corpus, sanity-check structure."""
    corpus = Path.home() / '.cache' / 'flame-sheep' / 'corpus' / 'osu'
    if not corpus.exists():
        pytest.skip('osu corpus not present')
    candidates = list(corpus.glob('*/*.osu'))
    if not candidates:
        pytest.skip('no .osu files in corpus')
    m = parse_osu_file(candidates[0])
    assert m.audio_filename  # non-empty
    assert m.uninherited_points  # at least one
    beats, downbeats = m.beat_times(audio_duration_sec=180.0)  # 3 min upper bound
    assert len(beats) > 10
    # Beats sorted ascending
    assert np.all(np.diff(beats) > 0)
    # Downbeats ⊆ beats
    assert set(downbeats.tolist()).issubset(set(beats.tolist()))
