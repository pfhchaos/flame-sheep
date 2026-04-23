"""
Unit tests for individual components in isolation.

Tests each module with minimal dependencies — no FlameSheepCore, no
AudioProcessor, just the component under test with direct input.
"""

import numpy as np
import pytest

from flame_sheep.audio._spectrum import SpectrumEngine, SpectrumFrame
from flame_sheep.audio._constants import FFT_SIZE, HOP_SIZE, N_BINS, SAMPLE_RATE
from flame_sheep.audio._types import BeatEvent, AudioState
from flame_sheep.audio.beat_detector import FluxBeatDetector
from flame_sheep.audio.drop_detector import DropDetector
from flame_sheep.audio.energy import EnergyAnalyzer
from flame_sheep.axes.zoom_axis import ZoomAxis
from flame_sheep.axes.brightness_axis import BrightnessAxis
from flame_sheep.axes.detail_axis import DetailAxis
from flame_sheep.axes.drift_axis import DriftAxis
from flame_sheep.axes.genome_axis import GenomeAxis

from .conftest import trivial_genome


# -------------------------------------------------------------------
# SpectrumEngine
# -------------------------------------------------------------------

class TestSpectrumEngine:

    def test_output_shape(self):
        engine = SpectrumEngine()
        pcm = np.zeros(FFT_SIZE, dtype=np.float32)
        frame = engine.compute(pcm)
        assert frame.magnitude.shape == (N_BINS,)
        assert frame.flux.shape == (N_BINS,)
        assert frame.waveform.shape == (FFT_SIZE,)

    def test_silence_produces_zero_magnitude(self):
        engine = SpectrumEngine()
        pcm = np.zeros(FFT_SIZE, dtype=np.float32)
        frame = engine.compute(pcm)
        assert frame.magnitude.max() < 1e-10

    def test_first_frame_has_zero_flux(self):
        engine = SpectrumEngine()
        pcm = np.random.randn(FFT_SIZE).astype(np.float32) * 0.5
        frame = engine.compute(pcm)
        assert frame.flux.max() == 0.0  # no prev spectrum yet

    def test_second_frame_has_nonzero_flux(self):
        engine = SpectrumEngine()
        engine.compute(np.zeros(FFT_SIZE, dtype=np.float32))
        pcm = np.random.randn(FFT_SIZE).astype(np.float32) * 0.5
        frame = engine.compute(pcm)
        assert frame.flux.max() > 0.0

    def test_identical_frames_have_zero_flux(self):
        engine = SpectrumEngine()
        pcm = np.random.randn(FFT_SIZE).astype(np.float32) * 0.5
        engine.compute(pcm)
        frame = engine.compute(pcm.copy())
        assert frame.flux.max() < 1e-10

    def test_sine_peak_at_correct_freq(self):
        engine = SpectrumEngine()
        t = np.arange(FFT_SIZE) / SAMPLE_RATE
        pcm = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        frame = engine.compute(pcm)
        freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
        peak_bin = np.argmax(frame.magnitude)
        peak_freq = freqs[peak_bin]
        assert abs(peak_freq - 440) < 50, f"Peak at {peak_freq}Hz, expected ~440Hz"

    def test_reset_clears_prev(self):
        engine = SpectrumEngine()
        pcm = np.random.randn(FFT_SIZE).astype(np.float32) * 0.5
        engine.compute(pcm)
        engine.reset()
        frame = engine.compute(pcm)
        assert frame.flux.max() == 0.0  # no prev after reset


# -------------------------------------------------------------------
# FluxBeatDetector
# -------------------------------------------------------------------

class TestFluxBeatDetector:

    def _make_onset_frame(self, band: str) -> SpectrumFrame:
        """Create a SpectrumFrame with strong flux in a specific band."""
        flux = np.zeros(N_BINS, dtype=np.float32)
        freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
        if band == 'kick':
            flux[(freqs >= 50) & (freqs < 100)] = 1.0
        elif band == 'snare':
            flux[(freqs >= 300) & (freqs < 1000)] = 0.5
            flux[(freqs >= 1000) & (freqs < 3000)] = 0.5  # snare confirm
        elif band == 'hihat':
            flux[freqs >= 8000] = 0.3
        return SpectrumFrame(
            magnitude=flux,  # magnitude doesn't matter for detection
            flux=flux,
            waveform=np.zeros(FFT_SIZE, dtype=np.float32),
        )

    def _silence_frame(self) -> SpectrumFrame:
        return SpectrumFrame(
            magnitude=np.zeros(N_BINS, dtype=np.float32),
            flux=np.zeros(N_BINS, dtype=np.float32),
            waveform=np.zeros(FFT_SIZE, dtype=np.float32),
        )

    def _warmup(self, detector: FluxBeatDetector, n: int = 15):
        for _ in range(n):
            detector.detect(self._silence_frame())

    def test_no_events_without_warmup(self):
        det = FluxBeatDetector()
        events = det.detect(self._make_onset_frame('kick'))
        assert len(events) == 0  # needs 10 frames of history

    def test_kick_detected_after_warmup(self):
        det = FluxBeatDetector()
        self._warmup(det)
        events = det.detect(self._make_onset_frame('kick'))
        kinds = [e.kind for e in events]
        assert 'kick' in kinds

    def test_cooldown_prevents_refire(self):
        det = FluxBeatDetector()
        self._warmup(det)
        det.detect(self._make_onset_frame('kick'))  # fires
        events = det.detect(self._make_onset_frame('kick'))  # should be cooled down
        kicks = [e for e in events if e.kind == 'kick']
        assert len(kicks) == 0

    def test_reset_bands(self):
        det = FluxBeatDetector(adaptive=True)
        self._warmup(det)
        det.detect(self._make_onset_frame('kick'))
        det.reset_bands()
        assert det.adaptive_bands is not None
        for ab in det.adaptive_bands.values():
            np.testing.assert_array_equal(ab.flux_accum,
                                          np.zeros(N_BINS, dtype=np.float32))

    def test_energy_in_range(self):
        det = FluxBeatDetector()
        self._warmup(det)
        events = det.detect(self._make_onset_frame('kick'))
        for e in events:
            assert 0.0 <= e.energy <= 1.0


# -------------------------------------------------------------------
# EnergyAnalyzer
# -------------------------------------------------------------------

class TestEnergyAnalyzer:

    def test_silence_returns_zero(self):
        ea = EnergyAnalyzer()
        spectrum = np.zeros(N_BINS, dtype=np.float32)
        rms = ea.update(spectrum)
        assert rms == 0.0

    def test_energy_increases_with_bass(self):
        ea = EnergyAnalyzer()
        spectrum = np.zeros(N_BINS, dtype=np.float32)
        freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
        spectrum[(freqs >= 50) & (freqs < 150)] = 1.0
        rms = ea.update(spectrum)
        assert rms > 0.0

    def test_ema_smoothing(self):
        ea = EnergyAnalyzer(alpha=0.9)
        spectrum = np.zeros(N_BINS, dtype=np.float32)
        freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
        spectrum[(freqs >= 50) & (freqs < 150)] = 1.0

        # First update
        r1 = ea.update(spectrum)
        # Second update with silence — should decay
        r2 = ea.update(np.zeros(N_BINS, dtype=np.float32))
        assert r2 < r1  # EMA decays

    def test_property_matches_return(self):
        ea = EnergyAnalyzer()
        spectrum = np.ones(N_BINS, dtype=np.float32)
        rms = ea.update(spectrum)
        assert ea.rms == rms


# -------------------------------------------------------------------
# ZoomAxis
# -------------------------------------------------------------------

class TestZoomAxisUnit:

    def test_hihat_adds_boost(self):
        axis = ZoomAxis()
        axis.tick(AudioState(events=[BeatEvent('hihat', 1.0)], rms=0.0), 1/60, 0.0)
        assert axis.zoom_boost > 0

    def test_non_hihat_ignored(self):
        axis = ZoomAxis()
        axis.tick(AudioState(events=[BeatEvent('kick', 1.0)], rms=0.0), 1/60, 0.0)
        assert axis.zoom_boost == 0.0

    def test_boost_decays(self):
        axis = ZoomAxis()
        axis.tick(AudioState(events=[BeatEvent('hihat', 1.0)], rms=0.0), 1/60, 0.0)
        peak = axis.zoom_boost
        axis.tick(AudioState(rms=0.0), 1/60, 0.0)  # no events, just decay
        assert axis.zoom_boost < peak

    def test_boost_bounded(self):
        axis = ZoomAxis()
        for _ in range(100):
            axis.tick(AudioState(events=[BeatEvent('hihat', 1.0)], rms=0.0), 1/60, 0.0)
        assert axis.zoom_boost <= axis.ZOOM_BOOST_MAX


# -------------------------------------------------------------------
# BrightnessAxis
# -------------------------------------------------------------------

class TestBrightnessAxisUnit:

    def test_silence_returns_floor(self):
        axis = BrightnessAxis(floor=0.7, ceiling=12.0)
        axis.tick(AudioState(rms=0.0), 1/60, 0.0)
        assert axis.brightness == 0.7

    def test_loud_returns_ceiling(self):
        axis = BrightnessAxis(floor=0.7, ceiling=12.0, rms_scale=0.01)
        axis.tick(AudioState(rms=1.0), 1/60, 0.0)  # rms >> scale
        assert axis.brightness == pytest.approx(12.0)

    def test_mid_rms_between_floor_and_ceiling(self):
        axis = BrightnessAxis(floor=0.7, ceiling=12.0, rms_scale=0.01)
        axis.tick(AudioState(rms=0.005), 1/60, 0.0)
        assert 0.7 < axis.brightness < 12.0


# -------------------------------------------------------------------
# DetailAxis
# -------------------------------------------------------------------

class TestDetailAxisUnit:

    def test_silence_returns_min(self):
        axis = DetailAxis(min_iters=100, max_iters=500)
        axis.tick(AudioState(rms=0.0), 1/60, 0.0)
        assert axis.iterations == 100

    def test_loud_returns_max(self):
        axis = DetailAxis(min_iters=100, max_iters=500, rms_scale=0.01)
        axis.tick(AudioState(rms=1.0), 1/60, 0.0)
        assert axis.iterations == 500


# -------------------------------------------------------------------
# GenomeAxis
# -------------------------------------------------------------------

class TestGenomeAxisUnit:

    def _make_axis(self):
        _seed = iter(range(1000))
        return GenomeAxis(
            genome_factory=lambda: trivial_genome(next(_seed)))

    def test_starts_with_genomes(self):
        axis = self._make_axis()
        assert axis.current_genome is not None
        assert axis.target_genome is not None

    def test_morph_advances(self):
        axis = self._make_axis()
        axis.tick(AudioState(rms=0.0), 1/60, 0.0)
        assert axis.morph_t > 0.0

    def test_kick_increments_counter(self):
        axis = self._make_axis()
        axis.tick(AudioState(events=[BeatEvent('kick', 0.5)], rms=0.0), 1/60, 1.0)
        assert axis._kick_count == 1

    def test_four_kicks_trigger_swap(self):
        axis = self._make_axis()
        initial_target = id(axis.target_genome)
        for i in range(4):
            axis.tick(AudioState(events=[BeatEvent('kick', 0.5)], rms=0.0), 1/60, float(i))
        assert id(axis.target_genome) != initial_target

    def test_force_swap(self):
        axis = self._make_axis()
        initial_target = id(axis.target_genome)
        axis.force_swap()
        assert id(axis.target_genome) != initial_target
        assert axis.needs_walker_reset


class FakeLib:
    """Minimal loop library stub for testing loop selection."""

    def __init__(self, n_loops: int = 20):
        self._loops = {}
        for i in range(1, n_loops + 1):
            self._loops[i] = {
                'fitness': 1.0 / i,  # descending fitness
                'mean_coherence': 0.5, 'min_coherence': 0.3,
                'diversity': 0.5, 'palette_flow': 0.5, 'smoothness': 0.5,
            }

    def loop_count(self):
        return len(self._loops)

    def top_loops(self, n=10):
        ranked = sorted(self._loops.items(), key=lambda x: -x[1]['fitness'])
        return [(lid, info) for lid, info in ranked[:n]]

    def load_loop(self, loop_id):
        g = trivial_genome(loop_id)
        return [(loop_id, g, 0)]


class TestNextLoopSelection:

    def _make_axis(self, n_loops=20, seed=42):
        _genseed = iter(range(1000))
        return GenomeAxis(
            genome_factory=lambda: trivial_genome(next(_genseed)),
            lib=FakeLib(n_loops),
            rng=np.random.default_rng(seed),
        )

    def test_next_loop_avoids_current(self):
        axis = self._make_axis()
        for _ in range(50):
            prev = axis.active_loop_id
            axis.next_loop()
            assert axis.active_loop_id != prev

    def test_next_loop_avoids_recent_history(self):
        axis = self._make_axis()
        for _ in range(20):
            axis.next_loop()
            # Current loop should never be in history minus the just-added entry
            history = list(axis._loop_history)
            # No duplicates within the history window
            assert len(history) == len(set(history)) or len(history) > axis.LOOP_HISTORY_SIZE

    def test_next_loop_uses_variety(self):
        """Over many calls, should visit more than 2 loops."""
        axis = self._make_axis()
        visited = set()
        for _ in range(30):
            axis.next_loop()
            visited.add(axis.active_loop_id)
        assert len(visited) >= 5

    def test_next_loop_favors_high_fitness(self):
        """Higher fitness loops should appear more often."""
        axis = self._make_axis(seed=0)
        counts = {}
        for _ in range(200):
            axis.next_loop()
            lid = axis.active_loop_id
            counts[lid] = counts.get(lid, 0) + 1
        # Top loop (id=1, fitness=1.0) should appear more than bottom (id=20, fitness=0.05)
        assert counts.get(1, 0) > counts.get(20, 0)

    def test_next_loop_no_repeat_within_history_window(self):
        """A loop should not reappear within LOOP_HISTORY_SIZE calls."""
        axis = self._make_axis()
        recent = []
        for _ in range(40):
            axis.next_loop()
            lid = axis.active_loop_id
            window = recent[-axis.LOOP_HISTORY_SIZE:]
            assert lid not in window, f'Loop {lid} repeated within history window'
            recent.append(lid)

    def test_fallback_when_few_loops(self):
        """With fewer loops than history size, should still work."""
        axis = self._make_axis(n_loops=3)
        for _ in range(10):
            axis.next_loop()
            assert axis.active_loop_id is not None

    def test_history_tracks_initial_load(self):
        """The initial loop loaded at startup should be in history."""
        axis = self._make_axis()
        assert axis.active_loop_id in axis._loop_history


# -------------------------------------------------------------------
# DriftAxis
# -------------------------------------------------------------------

class TestDriftAxisUnit:

    def _make_axis(self):
        _seed = iter(range(1000))
        return GenomeAxis(
            genome_factory=lambda: trivial_genome(next(_seed)))

    def test_no_drift_when_loud(self):
        genome_axis = self._make_axis()
        drift = DriftAxis(genome_axis)
        initial_target = id(genome_axis.target_genome)
        for i in range(600):
            drift.tick(AudioState(rms=0.1), 1/60, float(i))  # rms above threshold
        assert id(genome_axis.target_genome) == initial_target

    def test_drift_swaps_when_quiet(self):
        genome_axis = self._make_axis()
        drift = DriftAxis(genome_axis)
        initial_target = id(genome_axis.target_genome)
        # Tick enough quiet frames to trigger a swap
        for i in range(drift.SWAP_FRAMES + 10):
            drift.tick(AudioState(rms=0.0), 1/60, float(i))
        assert id(genome_axis.target_genome) != initial_target

    def test_loud_resets_quiet_counter(self):
        genome_axis = self._make_axis()
        drift = DriftAxis(genome_axis)
        # Almost enough quiet frames
        for i in range(drift.SWAP_FRAMES - 10):
            drift.tick(AudioState(rms=0.0), 1/60, float(i))
        # Loud frame resets
        drift.tick(AudioState(rms=0.1), 1/60, float(drift.SWAP_FRAMES))
        assert drift._quiet_frames == 0


# -------------------------------------------------------------------
# SpectrumEngine push_hop
# -------------------------------------------------------------------

class TestPushHop:

    def test_returns_spectrum_frame(self):
        engine = SpectrumEngine()
        hop = np.random.randn(HOP_SIZE).astype(np.float32) * 0.1
        frame = engine.push_hop(hop)
        assert isinstance(frame, SpectrumFrame)
        assert frame.magnitude.shape == (N_BINS,)
        assert frame.flux.shape == (N_BINS,)

    def test_successive_hops_produce_flux(self):
        engine = SpectrumEngine()
        silence = np.zeros(HOP_SIZE, dtype=np.float32)
        engine.push_hop(silence)
        # Sudden energy should produce flux
        burst = np.random.randn(HOP_SIZE).astype(np.float32) * 0.5
        frame = engine.push_hop(burst)
        assert frame.flux.sum() > 0

    def test_sliding_window_accumulates(self):
        engine = SpectrumEngine()
        # Fill the window with 4 hops (4 * 512 = 2048 = FFT_SIZE)
        for _ in range(4):
            engine.push_hop(np.ones(HOP_SIZE, dtype=np.float32) * 0.1)
        # The buffer should now be fully populated
        assert np.all(engine._buffer != 0)


# -------------------------------------------------------------------
# EnergyAnalyzer continuous features
# -------------------------------------------------------------------

class TestEnergyAnalyzerContinuous:

    def test_centroid_in_reasonable_range(self):
        ea = EnergyAnalyzer()
        freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
        spectrum = np.zeros(N_BINS, dtype=np.float32)
        # Energy at 1kHz
        spectrum[(freqs >= 900) & (freqs < 1100)] = 1.0
        ea.update(spectrum)
        assert 500 < ea.centroid < 2000

    def test_centroid_follows_energy(self):
        ea = EnergyAnalyzer()
        freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
        # Low energy
        low = np.zeros(N_BINS, dtype=np.float32)
        low[(freqs >= 100) & (freqs < 200)] = 1.0
        for _ in range(20):
            ea.update(low)
        centroid_low = ea.centroid
        # High energy
        high = np.zeros(N_BINS, dtype=np.float32)
        high[(freqs >= 5000) & (freqs < 6000)] = 1.0
        for _ in range(20):
            ea.update(high)
        centroid_high = ea.centroid
        assert centroid_high > centroid_low

    def test_centroid_rms_tracks_dominant(self):
        ea = EnergyAnalyzer()
        freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
        spectrum = np.zeros(N_BINS, dtype=np.float32)
        spectrum[(freqs >= 900) & (freqs < 1100)] = 1.0
        for _ in range(10):
            ea.update(spectrum)
        assert ea.centroid_rms > 0

    def test_percussiveness_high_on_transient(self):
        ea = EnergyAnalyzer()
        spectrum = np.ones(N_BINS, dtype=np.float32) * 0.01
        flux = np.ones(N_BINS, dtype=np.float32) * 0.5
        for _ in range(20):
            ea.update(spectrum, flux)
        # High flux relative to magnitude = percussive
        assert ea.percussiveness > 0.3

    def test_percussiveness_low_on_sustain(self):
        ea = EnergyAnalyzer()
        spectrum = np.ones(N_BINS, dtype=np.float32) * 0.5
        flux = np.ones(N_BINS, dtype=np.float32) * 0.001
        for _ in range(20):
            ea.update(spectrum, flux)
        # Low flux relative to magnitude = sustained
        assert ea.percussiveness < 0.1

    def test_band_rms_keys(self):
        ea = EnergyAnalyzer()
        spectrum = np.ones(N_BINS, dtype=np.float32) * 0.1
        ea.update(spectrum)
        band = ea.band_rms
        assert set(band.keys()) == {'kick', 'snare', 'clap', 'hihat'}
        assert all(v >= 0 for v in band.values())

    def test_centroid_delta(self):
        ea = EnergyAnalyzer()
        freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
        low = np.zeros(N_BINS, dtype=np.float32)
        low[(freqs >= 100) & (freqs < 200)] = 1.0
        for _ in range(20):
            ea.update(low)
        # Sudden jump
        high = np.zeros(N_BINS, dtype=np.float32)
        high[(freqs >= 5000) & (freqs < 6000)] = 1.0
        ea.update(high)
        assert ea.centroid_delta > 0


# -------------------------------------------------------------------
# DropDetector
# -------------------------------------------------------------------

class TestDropDetector:

    def test_no_drop_during_warmup(self):
        dd = DropDetector()
        kick = [BeatEvent('kick', 1.0)]
        # Even with quiet frames, warmup blocks detection
        for _ in range(200):
            dd.detect([], 0.0, 120.0, False, 1/60)
        result = dd.detect(kick, 0.0, 120.0, False, 1/60)
        assert result is None  # still in warmup (300 frames)

    def test_no_drop_on_first_kicks(self):
        dd = DropDetector()
        dd._warmup_frames = 999  # skip warmup
        kick = [BeatEvent('kick', 1.0)]
        # Only a few kicks — below MIN_KICKS_BEFORE_DROP (8)
        for _ in range(5):
            dd.detect(kick, 1.0, 120.0, False, 1/60)
        # Go quiet
        for _ in range(150):
            dd.detect([], 0.0, 120.0, False, 1/60)
        # Kick after quiet — but only 6 total kicks, below threshold
        result = dd.detect(kick, 0.5, 120.0, False, 1/60)
        assert result is None

    def test_drop_fires_after_quiet(self):
        dd = DropDetector()
        dd._warmup_frames = 999
        kick = [BeatEvent('kick', 1.0)]
        # Build up history
        for _ in range(20):
            dd.detect(kick, 1.0, 120.0, False, 1/60)
        # Go quiet
        for _ in range(150):
            dd.detect([], 0.0, 120.0, False, 1/60)
        # Kick after quiet
        result = dd.detect(kick, 0.5, 120.0, False, 1/60)
        assert result is not None
        assert result.kind == 'drop'

    def test_drop_cooldown(self):
        dd = DropDetector()
        dd._warmup_frames = 999
        kick = [BeatEvent('kick', 1.0)]
        # First drop
        for _ in range(20):
            dd.detect(kick, 1.0, 120.0, False, 1/60)
        for _ in range(150):
            dd.detect([], 0.0, 120.0, False, 1/60)
        result1 = dd.detect(kick, 0.5, 120.0, False, 1/60)
        assert result1 is not None
        # Second attempt — should be blocked by cooldown
        for _ in range(20):
            dd.detect(kick, 1.0, 120.0, False, 1/60)
        for _ in range(150):
            dd.detect([], 0.0, 120.0, False, 1/60)
        result2 = dd.detect(kick, 0.5, 120.0, False, 1/60)
        assert result2 is None  # cooldown active

    def test_no_drop_during_drift(self):
        dd = DropDetector()
        dd._warmup_frames = 999
        kick = [BeatEvent('kick', 1.0)]
        for _ in range(20):
            dd.detect(kick, 1.0, 120.0, False, 1/60)
        for _ in range(150):
            dd.detect([], 0.0, 120.0, False, 1/60)
        # drifting=True should block
        result = dd.detect(kick, 0.5, 120.0, True, 1/60)
        assert result is None

    def test_reset_clears_state(self):
        dd = DropDetector()
        dd._warmup_frames = 999
        dd._total_kicks = 100
        dd._quiet_frames = 200
        dd.reset()
        assert dd._total_kicks == 0
        assert dd._quiet_frames == 0
        assert dd._cooldown == 0.0

    def test_freeze_duration_uses_bpm(self):
        dd = DropDetector()
        dd._warmup_frames = 999
        kick = [BeatEvent('kick', 1.0)]
        for _ in range(20):
            dd.detect(kick, 1.0, 120.0, False, 1/60)
        for _ in range(150):
            dd.detect([], 0.0, 120.0, False, 1/60)
        result = dd.detect(kick, 0.5, 120.0, False, 1/60)
        assert result is not None
        # At 120 BPM: bar = 4 * 0.5s = 2s, 2 bars = 4s
        assert 3.5 < result.energy < 4.5


# -------------------------------------------------------------------
# GenomeAxis event handling
# -------------------------------------------------------------------

class TestGenomeAxisEvents:

    def _make_axis(self):
        _seed = iter(range(1000))
        return GenomeAxis(
            genome_factory=lambda: trivial_genome(next(_seed)))

    def test_ignores_unknown_events(self):
        axis = self._make_axis()
        audio = AudioState(events=[BeatEvent('alien_signal', 1.0)])
        # Should not raise
        axis.tick(audio, 1/60, 0.0)

    def test_drop_freezes_morph(self):
        axis = self._make_axis()
        # Fire a drop with 1s freeze
        audio = AudioState(events=[BeatEvent('drop', 1.0)])
        axis.tick(audio, 1/60, 0.0)
        assert axis._drop_freeze_remaining > 0
        # During freeze, morph shouldn't advance
        mt_before = axis.morph_t
        axis.tick(AudioState(), 1/60, 1.0)
        assert axis.morph_t == mt_before

    def test_song_start_resets(self):
        axis = self._make_axis()
        # Build some state
        for i in range(10):
            axis.tick(AudioState(events=[BeatEvent('kick', 0.5)]), 1/60, float(i))
        assert axis._kick_count > 0
        # Song start should reset
        axis.tick(AudioState(events=[BeatEvent('song_start', 0.0)]), 1/60, 20.0)
        assert axis._kick_count == 0

    def test_low_percussiveness_slows_morph(self):
        axis = self._make_axis()
        axis.morph_speed = 0.1
        # High percussiveness
        axis.tick(AudioState(percussiveness=0.8), 1/60, 0.0)
        mt_fast = axis.morph_t
        # Reset
        axis.morph_t = 0.0
        # Low percussiveness
        axis.tick(AudioState(percussiveness=0.1), 1/60, 1.0)
        mt_slow = axis.morph_t
        assert mt_slow < mt_fast

    def test_centroid_swap_on_low_percussiveness(self):
        axis = self._make_axis()
        axis.morph_t = 0.5  # must be > 0.3
        initial_target = id(axis.target_genome)
        audio = AudioState(
            percussiveness=0.1,
            centroid_delta=500.0,  # big shift
        )
        axis.tick(audio, 1/60, 0.0)
        assert id(axis.target_genome) != initial_target
