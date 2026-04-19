"""
Unit tests for individual components in isolation.

Tests each module with minimal dependencies — no FlameSheepCore, no
AudioProcessor, just the component under test with direct input.
"""

import numpy as np
import pytest

from flame_sheep.audio._spectrum import SpectrumEngine, SpectrumFrame
from flame_sheep.audio._constants import FFT_SIZE, N_BINS, SAMPLE_RATE
from flame_sheep.audio._types import BeatEvent
from flame_sheep.audio.beat_detector import FluxBeatDetector
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
        axis.tick([BeatEvent('hihat', 1.0)], 0.0, 1/60, 0.0)
        assert axis.zoom_boost > 0

    def test_non_hihat_ignored(self):
        axis = ZoomAxis()
        axis.tick([BeatEvent('kick', 1.0)], 0.0, 1/60, 0.0)
        assert axis.zoom_boost == 0.0

    def test_boost_decays(self):
        axis = ZoomAxis()
        axis.tick([BeatEvent('hihat', 1.0)], 0.0, 1/60, 0.0)
        peak = axis.zoom_boost
        axis.tick([], 0.0, 1/60, 0.0)  # no events, just decay
        assert axis.zoom_boost < peak

    def test_boost_bounded(self):
        axis = ZoomAxis()
        for _ in range(100):
            axis.tick([BeatEvent('hihat', 1.0)], 0.0, 1/60, 0.0)
        assert axis.zoom_boost <= axis.ZOOM_BOOST_MAX


# -------------------------------------------------------------------
# BrightnessAxis
# -------------------------------------------------------------------

class TestBrightnessAxisUnit:

    def test_silence_returns_floor(self):
        axis = BrightnessAxis(floor=0.7, ceiling=12.0)
        axis.tick([], 0.0, 1/60, 0.0)
        assert axis.brightness == 0.7

    def test_loud_returns_ceiling(self):
        axis = BrightnessAxis(floor=0.7, ceiling=12.0, rms_scale=0.01)
        axis.tick([], 1.0, 1/60, 0.0)  # rms >> scale
        assert axis.brightness == pytest.approx(12.0)

    def test_mid_rms_between_floor_and_ceiling(self):
        axis = BrightnessAxis(floor=0.7, ceiling=12.0, rms_scale=0.01)
        axis.tick([], 0.005, 1/60, 0.0)
        assert 0.7 < axis.brightness < 12.0


# -------------------------------------------------------------------
# DetailAxis
# -------------------------------------------------------------------

class TestDetailAxisUnit:

    def test_silence_returns_min(self):
        axis = DetailAxis(min_iters=100, max_iters=500)
        axis.tick([], 0.0, 1/60, 0.0)
        assert axis.iterations == 100

    def test_loud_returns_max(self):
        axis = DetailAxis(min_iters=100, max_iters=500, rms_scale=0.01)
        axis.tick([], 1.0, 1/60, 0.0)
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
        axis.tick([], 0.0, 1/60, 0.0)
        assert axis.morph_t > 0.0

    def test_kick_increments_counter(self):
        axis = self._make_axis()
        axis.tick([BeatEvent('kick', 0.5)], 0.0, 1/60, 1.0)
        assert axis._kick_count == 1

    def test_four_kicks_trigger_swap(self):
        axis = self._make_axis()
        initial_target = id(axis.target_genome)
        for i in range(4):
            axis.tick([BeatEvent('kick', 0.5)], 0.0, 1/60, float(i))
        assert id(axis.target_genome) != initial_target

    def test_force_swap(self):
        axis = self._make_axis()
        initial_target = id(axis.target_genome)
        axis.force_swap()
        assert id(axis.target_genome) != initial_target
        assert axis.needs_walker_reset


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
            drift.tick([], 0.1, 1/60, float(i))  # rms above threshold
        assert id(genome_axis.target_genome) == initial_target

    def test_drift_swaps_when_quiet(self):
        genome_axis = self._make_axis()
        drift = DriftAxis(genome_axis)
        initial_target = id(genome_axis.target_genome)
        # Tick enough quiet frames to trigger a swap
        for i in range(drift.SWAP_FRAMES + 10):
            drift.tick([], 0.0, 1/60, float(i))
        assert id(genome_axis.target_genome) != initial_target

    def test_loud_resets_quiet_counter(self):
        genome_axis = self._make_axis()
        drift = DriftAxis(genome_axis)
        # Almost enough quiet frames
        for i in range(drift.SWAP_FRAMES - 10):
            drift.tick([], 0.0, 1/60, float(i))
        # Loud frame resets
        drift.tick([], 0.1, 1/60, float(drift.SWAP_FRAMES))
        assert drift._quiet_frames == 0
