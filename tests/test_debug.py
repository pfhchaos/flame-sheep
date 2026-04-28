"""Tests for the debug overlay data structures and panels."""

from __future__ import annotations

import time

import numpy as np
import pytest

from flame_sheep_audio._types import BeatEvent, AudioSnapshot, BandState
from flame_sheep.orchestrator import TimestampedEvent
from flame_sheep.debug._timeline import TimelineBuffer, OnsetMark
from flame_sheep.debug._panels import (
    HeaderPanel, TimelinePanel, SpectrumPanel, BandMetricsPanel,
    BreakPanel, build_panels,
)
from flame_sheep_audio._band_config import default_band_config


@pytest.fixture
def band_config():
    return default_band_config()


@pytest.fixture
def timeline(band_config):
    return TimelineBuffer(band_names=band_config.detection_band_names,
                          max_seconds=4.0)


def _make_event(kind: str, energy: float, t: float) -> TimestampedEvent:
    return TimestampedEvent(event=BeatEvent(kind=kind, energy=energy), timestamp=t)


# ----------------------------------------------------------------
# TimelineBuffer
# ----------------------------------------------------------------

class TestTimelineBuffer:

    def test_push_and_retrieve(self, timeline):
        now = 1000.0
        timeline.push(_make_event('kick', 0.8, now))
        marks = timeline.marks_in_window(now)
        assert len(marks['kick']) == 1
        assert marks['kick'][0].energy == 0.8

    def test_ignores_unknown_bands(self, timeline):
        timeline.push(_make_event('cowbell', 1.0, 1000.0))
        marks = timeline.marks_in_window(1000.0)
        assert 'cowbell' not in marks

    def test_window_filtering(self, timeline):
        # Push events at different times
        timeline.push(_make_event('kick', 0.5, 990.0))  # 10s ago
        timeline.push(_make_event('kick', 0.8, 997.0))  # 3s ago
        timeline.push(_make_event('kick', 1.0, 999.5))  # 0.5s ago
        marks = timeline.marks_in_window(1000.0)
        # Window is 4s, so 990.0 should be excluded
        assert len(marks['kick']) == 2

    def test_clear(self, timeline):
        timeline.push(_make_event('kick', 1.0, 1000.0))
        timeline.clear()
        marks = timeline.marks_in_window(1000.0)
        assert len(marks['kick']) == 0

    def test_multiple_bands(self, timeline):
        now = 1000.0
        timeline.push(_make_event('kick', 1.0, now))
        timeline.push(_make_event('snare', 0.7, now))
        timeline.push(_make_event('hihat', 0.5, now))
        marks = timeline.marks_in_window(now)
        assert len(marks['kick']) == 1
        assert len(marks['snare']) == 1
        assert len(marks['hihat']) == 1

    def test_band_names_from_config(self, band_config):
        tl = TimelineBuffer(band_names=band_config.detection_band_names)
        assert set(tl.band_names) == set(band_config.detection_band_names)

    def test_max_seconds_property(self):
        tl = TimelineBuffer(band_names=['kick'], max_seconds=6.0)
        assert tl.max_seconds == 6.0

    def test_high_volume_doesnt_crash(self, timeline):
        """Push more events than maxlen — should not error."""
        for i in range(3000):
            timeline.push(_make_event('kick', 0.5, 1000.0 + i * 0.01))
        marks = timeline.marks_in_window(1000.0 + 3000 * 0.01)
        assert len(marks['kick']) <= 2000  # maxlen


# ----------------------------------------------------------------
# Panels (update logic only — no GL rendering)
# ----------------------------------------------------------------

def _snap(**kwargs) -> AudioSnapshot:
    """Build an AudioSnapshot with overrides."""
    defaults = dict(
        mode='beat',
        effective_bpm=120.0,
        tempo_confidence=0.7,
        percussiveness=0.5,
        break_intensity=0.0,
        section_change=0.0,
        bands={'kick': BandState(rms=0.1, onset_density=2.0, density_delta=0.1),
               'snare': BandState(rms=0.05, onset_density=1.0),
               'hihat': BandState(rms=0.02, onset_density=4.0),
               'subbass': BandState(rms=0.3)},
    )
    defaults.update(kwargs)
    return AudioSnapshot(**defaults)


class TestHeaderPanel:

    def test_update_reads_state(self):
        panel = HeaderPanel()
        snap = _snap(mode='energy', effective_bpm=140.0, tempo_confidence=0.9)
        panel.update(snap, TimelineBuffer(['kick']), 0.016)
        assert panel._mode == 'energy'
        assert panel._bpm == 140.0
        assert panel._confidence == 0.9


class TestBandMetricsPanel:

    def test_reads_all_bands(self, band_config):
        panel = BandMetricsPanel(band_config)
        snap = _snap()
        panel.update(snap, TimelineBuffer(['kick']), 0.016)
        assert 'kick' in panel._bands
        assert 'subbass' in panel._bands


class TestBreakPanel:

    def test_reads_break_intensity(self):
        panel = BreakPanel()
        snap = _snap(break_intensity=0.75)
        panel.update(snap, TimelineBuffer(['kick']), 0.016)
        assert panel._break_intensity == 0.75


class TestBuildPanels:

    def test_default_builds_all(self, band_config):
        panels = build_panels(band_config)
        assert len(panels) == 5

    def test_filtered_by_enabled(self, band_config):
        panels = build_panels(band_config, enabled=['header', 'spectrum'])
        assert len(panels) == 2

    def test_ignores_unknown_names(self, band_config):
        panels = build_panels(band_config, enabled=['header', 'nonexistent'])
        assert len(panels) == 1
