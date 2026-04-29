"""Tests for corpus infrastructure — osu! parser, metrics, loader."""

from __future__ import annotations

import pytest

import numpy as np

from tests.corpus.osu_parser import parse_osu, OsuBeatmap, TimingPoint, HitObject
from tests.corpus.metrics import onset_precision_recall, tempo_accuracy, hpss_similarity


# Sample .osu content (minimal valid beatmap)
SAMPLE_OSU = """osu file format v14

[General]
AudioFilename: audio.mp3
Mode: 0

[Metadata]
Title:Test Song
Artist:Test Artist
Version:Normal

[TimingPoints]
0,500,4,1,0,100,1,0
10000,333.33,4,1,0,100,1,0
15000,-50,4,1,0,100,0,0

[HitObjects]
256,192,0,1,0,0:0:0:0:
256,192,500,1,2,0:0:0:0:
256,192,1000,1,0,0:0:0:0:
256,192,1500,2,0,L|400:0,1,0:0:0:0:
256,192,2000,1,4,0:0:0:0:
256,192,3000,8,0,6000,0:0:0:0:
"""


class TestOsuParser:

    def test_parse_metadata(self):
        bm = parse_osu(SAMPLE_OSU)
        assert bm.title == 'Test Song'
        assert bm.artist == 'Test Artist'
        assert bm.audio_filename == 'audio.mp3'

    def test_parse_timing_points(self):
        bm = parse_osu(SAMPLE_OSU)
        assert len(bm.timing_points) == 3
        # First: 120 BPM (500ms per beat)
        assert bm.timing_points[0].uninherited is True
        assert abs(bm.timing_points[0].bpm - 120.0) < 0.1
        assert bm.timing_points[0].meter == 4
        # Second: 180 BPM (333.33ms per beat)
        assert bm.timing_points[1].uninherited is True
        assert abs(bm.timing_points[1].bpm - 180.0) < 0.1
        # Third: inherited (velocity modifier)
        assert bm.timing_points[2].uninherited is False
        assert bm.timing_points[2].bpm == 0.0

    def test_bpm_from_first_uninherited(self):
        bm = parse_osu(SAMPLE_OSU)
        assert abs(bm.bpm - 120.0) < 0.1

    def test_parse_hit_objects(self):
        bm = parse_osu(SAMPLE_OSU)
        assert len(bm.hit_objects) == 6

    def test_hit_object_types(self):
        bm = parse_osu(SAMPLE_OSU)
        assert bm.hit_objects[0].is_circle
        assert bm.hit_objects[3].is_slider
        assert bm.hit_objects[5].is_spinner

    def test_hit_sounds(self):
        bm = parse_osu(SAMPLE_OSU)
        assert bm.hit_objects[0].hit_sound == 0  # normal
        assert bm.hit_objects[1].hit_sound == 2  # whistle
        assert bm.hit_objects[4].hit_sound == 4  # finish

    def test_onset_times(self):
        bm = parse_osu(SAMPLE_OSU)
        times = bm.onset_times
        # Spinner (at 3000ms) should be excluded
        assert len(times) == 5
        assert times[0] == 0.0
        assert times[1] == 0.5
        assert times[-1] == 2.0

    def test_has_tempo_changes(self):
        bm = parse_osu(SAMPLE_OSU)
        assert bm.has_tempo_changes is True

    def test_no_tempo_changes(self):
        osu = """osu file format v14

[TimingPoints]
0,500,4,1,0,100,1,0

[HitObjects]
256,192,0,1,0,0:0:0:0:
"""
        bm = parse_osu(osu)
        assert bm.has_tempo_changes is False

    def test_empty_beatmap(self):
        bm = parse_osu("")
        assert bm.bpm == 0.0
        assert bm.onset_times == []
        assert bm.title == ''

    def test_malformed_lines_skipped(self):
        osu = """osu file format v14

[TimingPoints]
not,a,valid,line
0,500,4,1,0,100,1,0

[HitObjects]
bad line
256,192,1000,1,0,0:0:0:0:
"""
        bm = parse_osu(osu)
        assert len(bm.timing_points) == 1
        assert len(bm.hit_objects) == 1


# ----------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------

class TestOnsetPrecisionRecall:

    def test_perfect_match(self):
        gt = [0.0, 0.5, 1.0, 1.5]
        est = [0.0, 0.5, 1.0, 1.5]
        p, r, f1 = onset_precision_recall(est, gt, window=0.1)
        assert p == 1.0
        assert r == 1.0
        assert f1 == 1.0

    def test_within_window(self):
        gt = [0.0, 0.5, 1.0]
        est = [0.05, 0.48, 1.09]  # all within 100ms
        p, r, f1 = onset_precision_recall(est, gt, window=0.1)
        assert p == 1.0
        assert r == 1.0

    def test_outside_window(self):
        gt = [0.0, 0.5, 1.0]
        est = [0.2, 0.8]  # both miss by >100ms
        p, r, f1 = onset_precision_recall(est, gt, window=0.1)
        assert p == 0.0
        assert r == 0.0

    def test_extra_detections(self):
        gt = [0.0, 1.0]
        est = [0.0, 0.5, 1.0]  # one extra
        p, r, f1 = onset_precision_recall(est, gt, window=0.1)
        assert r == 1.0
        assert abs(p - 2/3) < 0.01

    def test_missed_detections(self):
        gt = [0.0, 0.5, 1.0]
        est = [0.0]  # missed two
        p, r, f1 = onset_precision_recall(est, gt, window=0.1)
        assert p == 1.0
        assert abs(r - 1/3) < 0.01

    def test_empty_both(self):
        p, r, f1 = onset_precision_recall([], [])
        assert p == 1.0 and r == 1.0

    def test_empty_est(self):
        p, r, f1 = onset_precision_recall([], [0.0, 0.5])
        assert r == 0.0


class TestTempoAccuracy:

    def test_exact_match(self):
        result = tempo_accuracy(120.0, 120.0)
        assert result['exact'] is True
        assert result['octave'] is True
        assert result['error'] < 0.001

    def test_within_tolerance(self):
        result = tempo_accuracy(123.0, 120.0)  # 2.5% error
        assert result['exact'] is True

    def test_outside_tolerance(self):
        result = tempo_accuracy(130.0, 120.0)  # 8.3% error
        assert result['exact'] is False

    def test_octave_double(self):
        result = tempo_accuracy(240.0, 120.0)
        assert result['exact'] is False
        assert result['octave'] is True

    def test_octave_half(self):
        result = tempo_accuracy(60.0, 120.0)
        assert result['exact'] is False
        assert result['octave'] is True


class TestHpssSimilarity:

    def test_identical(self):
        h = np.random.rand(100, 50)
        p = np.random.rand(100, 50)
        result = hpss_similarity(h, p, h, p)
        assert result['harmonic_sim'] > 0.999
        assert result['percussive_sim'] > 0.999

    def test_orthogonal(self):
        h1 = np.array([[1, 0], [0, 0]], dtype=float)
        p1 = np.array([[0, 0], [0, 1]], dtype=float)
        h2 = np.array([[0, 1], [0, 0]], dtype=float)
        p2 = np.array([[0, 0], [1, 0]], dtype=float)
        result = hpss_similarity(h1, p1, h2, p2)
        assert result['harmonic_sim'] == 0.0
