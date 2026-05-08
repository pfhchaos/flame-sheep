"""Unit tests for the morph lifecycle state machine."""

import pytest
from flame_sheep.axes._morph_lifecycle import MorphLifecycle, MorphState


class TestMorphLifecycle:

    def test_starts_in_dwell(self):
        lc = MorphLifecycle(clock=0.0, dwell_beats=4, morph_beats=2)
        assert lc.state == MorphState.DWELL
        assert lc.morph_t == 0.0

    # --- DWELL → READY ---

    def test_dwell_to_ready(self):
        lc = MorphLifecycle(clock=0.0, dwell_beats=4, morph_beats=2)
        # 4 beats at 120 BPM = 2 seconds
        lc.tick(1.0, bpm=120.0)
        assert lc.state == MorphState.DWELL  # not yet
        lc.tick(2.1, bpm=120.0)
        assert lc.state == MorphState.READY

    def test_dwell_adapts_to_tempo(self):
        lc = MorphLifecycle(clock=0.0, dwell_beats=4, morph_beats=2)
        # At 60 BPM, 4 beats = 4 seconds
        lc.tick(3.0, bpm=60.0)
        assert lc.state == MorphState.DWELL
        lc.tick(4.1, bpm=60.0)
        assert lc.state == MorphState.READY

    def test_tick_returns_false_during_dwell(self):
        lc = MorphLifecycle(clock=0.0, dwell_beats=4, morph_beats=2)
        assert lc.tick(1.0, bpm=120.0) is False

    # --- READY → MORPHING ---

    def test_on_beat_in_ready_starts_morph(self):
        lc = MorphLifecycle(clock=0.0, dwell_beats=4, morph_beats=2)
        lc.tick(2.1, bpm=120.0)  # → READY
        assert lc.on_beat(2.5) is True
        assert lc.state == MorphState.MORPHING

    def test_on_beat_in_dwell_ignored(self):
        lc = MorphLifecycle(clock=0.0, dwell_beats=4, morph_beats=2)
        assert lc.on_beat(0.5) is False
        assert lc.state == MorphState.DWELL

    def test_on_beat_in_morphing_ignored(self):
        lc = MorphLifecycle(clock=0.0, dwell_beats=4, morph_beats=2)
        lc.tick(2.1, bpm=120.0)
        lc.on_beat(2.5)
        assert lc.on_beat(2.8) is False  # already morphing

    # --- MORPHING → DWELL ---

    def test_morph_advances_per_frame(self):
        lc = MorphLifecycle(clock=0.0, dwell_beats=4, morph_beats=2)
        lc.tick(2.1, bpm=120.0)
        lc.on_beat(2.5)
        # morph_beats=2 at 120 BPM = 1 second morph
        lc.tick(3.0, bpm=120.0)  # 0.5s into 1s morph
        assert 0.4 < lc.morph_t < 0.6

    def test_morph_completes(self):
        lc = MorphLifecycle(clock=0.0, dwell_beats=4, morph_beats=2)
        lc.tick(2.1, bpm=120.0)
        lc.on_beat(2.5)
        completed = lc.tick(3.6, bpm=120.0)  # > 1s past morph start
        assert completed is True
        assert lc.state == MorphState.DWELL
        assert lc.morph_t == 0.0

    def test_morph_complete_resets_dwell_start(self):
        lc = MorphLifecycle(clock=0.0, dwell_beats=4, morph_beats=2)
        lc.tick(2.1, bpm=120.0)
        lc.on_beat(2.5)
        lc.tick(3.6, bpm=120.0)
        # New dwell starts from completion time
        # Should NOT be ready immediately
        lc.tick(3.7, bpm=120.0)
        assert lc.state == MorphState.DWELL

    # --- Full cycle ---

    def test_full_cycle(self):
        lc = MorphLifecycle(clock=0.0, dwell_beats=4, morph_beats=2)
        # DWELL
        lc.tick(2.1, bpm=120.0)  # → READY
        lc.on_beat(2.5)          # → MORPHING
        completed = lc.tick(3.6, bpm=120.0)  # → DWELL
        assert completed is True
        assert lc.state == MorphState.DWELL
        # Second cycle
        lc.tick(5.7, bpm=120.0)  # 3.6 + 2.0 + 0.1 → READY
        assert lc.state == MorphState.READY

    # --- Reset ---

    def test_reset_to_dwell(self):
        lc = MorphLifecycle(clock=0.0, dwell_beats=4, morph_beats=2)
        lc.tick(2.1, bpm=120.0)
        lc.on_beat(2.5)
        assert lc.state == MorphState.MORPHING
        lc.reset(3.0)
        assert lc.state == MorphState.DWELL
        assert lc.morph_t == 0.0

    # --- Properties ---

    def test_is_morphing(self):
        lc = MorphLifecycle(clock=0.0, dwell_beats=4, morph_beats=2)
        assert not lc.is_morphing
        lc.tick(2.1, bpm=120.0)
        lc.on_beat(2.5)
        assert lc.is_morphing

    def test_is_ready(self):
        lc = MorphLifecycle(clock=0.0, dwell_beats=4, morph_beats=2)
        assert not lc.is_ready
        lc.tick(2.1, bpm=120.0)
        assert lc.is_ready
        lc.on_beat(2.5)
        assert not lc.is_ready
