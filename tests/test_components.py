"""
Unit tests for individual components in isolation.

Tests each module with minimal dependencies — no FlameSheepCore, no
AudioProcessor, just the component under test with direct input.
"""

import numpy as np
import pytest

from flame_sheep_audio._spectrum import SpectrumEngine, SpectrumFrame
from flame_sheep_audio._constants import FFT_SIZE, HOP_SIZE, N_BINS, SAMPLE_RATE
from flame_sheep_audio._types import BeatEvent, BandState, AudioState, _default_bands
from flame_sheep_audio.beat_detector import FluxBeatDetector
from flame_sheep_audio.drop_detector import DropDetector
from flame_sheep_audio.bass_drop_detector import BassDropDetector
from flame_sheep_audio.onset_density import OnsetDensityTracker
from flame_sheep_audio.stability import MagnitudeStability
from flame_sheep_audio.energy import EnergyAnalyzer
from flame_sheep.axes.zoom_axis import ZoomAxis
from flame_sheep.axes.brightness_axis import BrightnessAxis
from flame_sheep.axes.detail_axis import DetailAxis
from flame_sheep.drift_mode import DriftMode
from flame_sheep.axes.genome_axis import GenomeAxis
from flame_sheep.role_mapper import RoleMapper
from flame_sheep_audio.mode import ModeDetector, Mode

from .conftest import trivial_genome

_default_role = RoleMapper()  # uses default mapping: low→downbeat, mid→backbeat, etc.


def _audio(events=None, rms=0.0, harmonic_rms=0.0, breaking=False,
           percussiveness=0.5, centroid_delta=0.0, onset_density=None,
           **kwargs):
    """Helper to construct AudioState with convenience kwargs."""
    bands = _default_bands()
    bands['subbass'] = BandState(rms=rms, harmonic_rms=harmonic_rms)
    if onset_density:
        for name, val in onset_density.items():
            if name in bands:
                bands[name] = BandState(
                    rms=bands[name].rms,
                    harmonic_rms=bands[name].harmonic_rms,
                    onset_density=val)
    return AudioState(
        events=events or [],
        bands=bands,
        percussiveness=percussiveness,
        centroid_delta=centroid_delta,
        centroid_rms=harmonic_rms,  # axes now read centroid_rms directly
        centroid_harmonic_rms=harmonic_rms,
        break_intensity=1.0 if breaking else 0.0,
        **kwargs,
    )



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
        if band == 'low':
            flux[(freqs >= 50) & (freqs < 100)] = 1.0
        elif band == 'mid':
            flux[(freqs >= 300) & (freqs < 1000)] = 0.5
            flux[(freqs >= 1000) & (freqs < 3000)] = 0.5  # mid-band confirm
        elif band == 'high':
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
        events = det.detect(self._make_onset_frame('low'))
        assert len(events) == 0  # needs 10 frames of history

    def test_kick_detected_after_warmup(self):
        det = FluxBeatDetector()
        self._warmup(det)
        events = det.detect(self._make_onset_frame('low'))
        kinds = [e.kind for e in events]
        assert 'low' in kinds

    def test_cooldown_prevents_refire(self):
        det = FluxBeatDetector()
        self._warmup(det)
        det.detect(self._make_onset_frame('low'))  # fires
        events = det.detect(self._make_onset_frame('low'))  # should be cooled down
        kicks = [e for e in events if e.kind == 'low']
        assert len(kicks) == 0

    def test_reset_bands(self):
        det = FluxBeatDetector(adaptive=True)
        self._warmup(det)
        det.detect(self._make_onset_frame('low'))
        det.reset_bands()
        assert det.adaptive_bands is not None
        for ab in det.adaptive_bands.values():
            np.testing.assert_array_equal(ab.flux_accum,
                                          np.zeros(N_BINS, dtype=np.float32))

    def test_energy_in_range(self):
        det = FluxBeatDetector()
        self._warmup(det)
        events = det.detect(self._make_onset_frame('low'))
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
# RoleMapper
# -------------------------------------------------------------------

class TestRoleMapper:

    def test_default_mapping(self):
        rm = RoleMapper()
        assert rm.band_for_role('downbeat') == 'low'
        assert rm.band_for_role('backbeat') == 'mid'
        assert rm.band_for_role('subdivision') == 'high'
        assert rm.band_for_role('energy') == 'subbass'

    def test_role_for_event(self):
        rm = RoleMapper()
        event = BeatEvent(kind='low', energy=1.0)
        assert rm.role_for_event(event) == 'downbeat'

    def test_custom_mapping(self):
        rm = RoleMapper({'downbeat': 'bass', 'backbeat': 'clap'})
        assert rm.band_for_role('downbeat') == 'bass'

    def test_unknown_band(self):
        rm = RoleMapper()
        event = BeatEvent(kind='cowbell', energy=1.0)
        assert rm.role_for_event(event) is None


# -------------------------------------------------------------------
# ZoomAxis
# -------------------------------------------------------------------

class TestZoomAxisUnit:

    def test_hihat_adds_boost(self):
        axis = ZoomAxis(role=_default_role)
        axis.tick(_audio(events=[BeatEvent('high', 1.0)], rms=0.0), 1/60, 0.0)
        assert axis.zoom_boost > 0

    def test_non_hihat_ignored(self):
        axis = ZoomAxis(role=_default_role)
        axis.tick(_audio(events=[BeatEvent('low', 1.0)], rms=0.0), 1/60, 0.0)
        assert axis.zoom_boost == 0.0

    def test_boost_decays(self):
        axis = ZoomAxis(role=_default_role)
        axis.tick(_audio(events=[BeatEvent('high', 1.0)], rms=0.0), 1/60, 0.0)
        peak = axis.zoom_boost
        axis.tick(_audio(rms=0.0), 1/60, 0.0)  # no events, just decay
        assert axis.zoom_boost < peak

    def test_boost_bounded(self):
        axis = ZoomAxis(role=_default_role)
        for _ in range(100):
            axis.tick(_audio(events=[BeatEvent('high', 1.0)], rms=0.0), 1/60, 0.0)
        assert axis.zoom_boost <= axis.ZOOM_BOOST_MAX


# -------------------------------------------------------------------
# BrightnessAxis
# -------------------------------------------------------------------

class TestBrightnessAxisUnit:

    def test_silence_returns_floor(self):
        axis = BrightnessAxis(role=_default_role, floor=0.7, ceiling=12.0)
        axis.tick(_audio(rms=0.0), 1/60, 0.0)
        assert axis.brightness == 0.7

    def test_loud_returns_ceiling(self):
        axis = BrightnessAxis(role=_default_role, floor=0.7, ceiling=12.0, rms_scale=0.01)
        axis.tick(_audio(harmonic_rms=1.0), 1/60, 0.0)
        assert axis.brightness == pytest.approx(12.0)

    def test_mid_rms_between_floor_and_ceiling(self):
        axis = BrightnessAxis(role=_default_role, floor=0.7, ceiling=12.0, rms_scale=0.01)
        axis.tick(_audio(harmonic_rms=0.005), 1/60, 0.0)
        assert 0.7 < axis.brightness < 12.0


# -------------------------------------------------------------------
# DetailAxis
# -------------------------------------------------------------------

class TestDetailAxisUnit:

    def test_silence_returns_min(self):
        axis = DetailAxis(role=_default_role, min_iters=100, max_iters=500)
        axis.tick(_audio(rms=0.0), 1/60, 0.0)
        assert axis.iterations == 100

    def test_loud_returns_max(self):
        axis = DetailAxis(role=_default_role, min_iters=100, max_iters=500, rms_scale=0.01)
        axis.tick(_audio(harmonic_rms=1.0), 1/60, 0.0)
        assert axis.iterations == 500


# -------------------------------------------------------------------
# GenomeAxis
# -------------------------------------------------------------------

class TestGenomeAxisUnit:

    def _make_axis(self):
        _seed = iter(range(1000))
        return GenomeAxis(
            genome_factory=lambda: trivial_genome(next(_seed)),
            role=_default_role)

    def test_starts_with_genomes(self):
        axis = self._make_axis()
        assert axis.current_genome is not None
        assert axis.target_genome is not None

    def test_morph_advances(self):
        axis = self._make_axis()
        axis.tick(_audio(rms=0.0), 1/60, 0.0)
        assert axis.morph_t > 0.0

    def test_strong_beat_triggers_swap(self):
        axis = self._make_axis()
        # Build up low energy average with quiet kicks
        for i in range(10):
            axis.tick(_audio(events=[BeatEvent('low', 0.2)]), 1/60, float(i))
        initial_target = id(axis.target_genome)
        # One loud low-band onset should trigger a swap
        axis.tick(_audio(events=[BeatEvent('low', 1.0)]), 1/60, 20.0)
        assert id(axis.target_genome) != initial_target

    def test_even_kicks_no_swap(self):
        """All kicks at same energy should not trigger swaps."""
        axis = self._make_axis()
        initial_target = id(axis.target_genome)
        # 10 kicks all at same energy
        for i in range(10):
            axis.tick(_audio(events=[BeatEvent('low', 0.5)]), 1/60, float(i))
        # No swap — energy never exceeds threshold
        assert id(axis.target_genome) == initial_target

    def test_density_drives_morph_speed(self):
        axis = self._make_axis()
        # Low density → slow baseline
        axis.tick(_audio(onset_density={'low': 1.0, 'mid': 0, 'high': 0}), 1/60, 0.0)
        slow_speed = axis.morph_speed
        # High density → faster baseline
        axis.tick(_audio(onset_density={'low': 5.0, 'mid': 0, 'high': 0}), 1/60, 1.0)
        fast_speed = axis.morph_speed
        assert fast_speed > slow_speed

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

    def loop_type(self, loop_id):
        return 'cyclic'


class TestNextLoopSelection:

    def _make_axis(self, n_loops=20, seed=42):
        _genseed = iter(range(1000))
        return GenomeAxis(
            genome_factory=lambda: trivial_genome(next(_genseed)),
            role=_default_role,
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

    def test_no_immediate_repeat(self):
        """next_loop() should not return the same loop twice in a row."""
        axis = self._make_axis()
        prev = axis.active_loop_id
        for _ in range(20):
            axis.next_loop()
            current = axis.active_loop_id
            assert current != prev, "next_loop() returned the same loop twice in a row"
            prev = current


# -------------------------------------------------------------------
# DriftMode
# -------------------------------------------------------------------

class TestDriftModeUnit:

    def _make(self):
        _seed = iter(range(1000))
        factory = lambda: trivial_genome(next(_seed))
        genome_axis = GenomeAxis(genome_factory=factory, role=_default_role)
        drift = DriftMode(genome_factory=factory)
        return drift, genome_axis

    def test_inactive_by_default(self):
        drift, _ = self._make()
        assert not drift.active

    def test_morph_advances_when_active(self):
        drift, genome_axis = self._make()
        drift.active = True
        drift.enter(genome_axis)
        drift.tick(_audio(rms=0.0), 1/60, 0.0)
        assert drift.morph_t > 0.0

    def test_morph_does_not_advance_when_inactive(self):
        drift, genome_axis = self._make()
        drift.tick(_audio(rms=0.0), 1/60, 0.0)
        assert drift.morph_t == 0.0

    def test_morph_completes_in_reasonable_time(self):
        """Fixed morph speed should complete within ~20 seconds."""
        from flame_sheep.config import cfg
        frames_to_complete = int(1.0 / cfg.drift.morph_speed) + 1
        assert frames_to_complete < 60 * 20, \
            f"Morph takes {frames_to_complete} frames ({frames_to_complete/60:.0f}s) — too slow"

    def test_enter_snapshots_genome(self):
        drift, genome_axis = self._make()
        # Advance genome_axis morph partway
        genome_axis.morph_t = 0.5
        expected = genome_axis.current_genome.lerp(
            genome_axis.target_genome, 0.5)
        drift.tick(_audio(rms=0.0), 1/60, 0.0)
        # Force activation for test
        drift.active = True
        drift.enter(genome_axis)
        assert drift.current_genome.distance(expected) < 1e-6

    def test_exit_returns_interpolated(self):
        drift, genome_axis = self._make()
        drift.active = True
        drift.enter(genome_axis)
        # Advance morph partway
        for i in range(100):
            drift.tick(_audio(rms=0.0), 1/60, float(i))
        expected = drift.current_genome.lerp(drift.target_genome, drift.morph_t)
        result = drift.exit()
        assert result.distance(expected) < 1e-6


# -------------------------------------------------------------------
# ModeDetector
# -------------------------------------------------------------------

class TestModeDetector:

    def test_starts_idle(self):
        md = ModeDetector()
        assert md.mode == Mode.IDLE

    def test_silence_stays_idle(self):
        md = ModeDetector()
        for _ in range(100):
            md.tick(_audio(rms=0.0, spectral_novelty=0.0))
        assert md.mode == Mode.IDLE

    def test_music_enters_beat(self):
        md = ModeDetector()
        # Low spectral novelty = music -> beat mode
        md.tick(_audio(rms=0.1, spectral_novelty=0.10))
        assert md.mode == Mode.BEAT

    def test_speech_transitions_to_energy(self):
        """Sustained high spectral novelty = speech -> eventually energy mode.
        Cold start from idle goes to beat first (EMA starts at 0), then
        transitions to energy as novelty EMA ramps up past threshold."""
        md = ModeDetector()
        # First frame: EMA is 0, enters beat from idle
        md.tick(_audio(rms=0.1, spectral_novelty=0.30))
        assert md.mode == Mode.BEAT  # cold start
        # Sustained speech: novelty EMA rises, eventually exits beat
        for _ in range(2000):
            md.tick(_audio(rms=0.1, spectral_novelty=0.30))
        assert md.mode == Mode.ENERGY

    def test_energy_to_beat_requires_sustained_music(self):
        md = ModeDetector()
        md.mode = Mode.ENERGY
        # One frame of low novelty isn't enough
        md.tick(_audio(rms=0.1, spectral_novelty=0.10))
        assert md.mode == Mode.ENERGY
        # Sustained low novelty transitions to beat
        for _ in range(100):
            md.tick(_audio(rms=0.1, spectral_novelty=0.10))
        assert md.mode == Mode.BEAT

    def test_beat_to_energy_on_sustained_speech(self):
        md = ModeDetector()
        md.mode = Mode.BEAT
        md._novelty_ema = 0.10  # was music
        # Sustained high novelty (speech) exits beat eventually
        for _ in range(1500):
            md.tick(_audio(rms=0.1, spectral_novelty=0.30))
        assert md.mode == Mode.ENERGY

    def test_brief_speech_doesnt_exit_beat(self):
        md = ModeDetector()
        md.mode = Mode.BEAT
        md._novelty_ema = 0.10
        # Brief speech section shouldn't exit beat
        for _ in range(60):
            md.tick(_audio(rms=0.1, spectral_novelty=0.30))
        assert md.mode == Mode.BEAT

    def test_acf_confidence_helps_enter_beat(self):
        """Moderate novelty + high tempo confidence = music."""
        md = ModeDetector()
        # Novelty in the ambiguous zone (0.16-0.20) but strong ACF
        md.tick(_audio(rms=0.1, spectral_novelty=0.18, tempo_confidence=0.8))
        assert md.mode == Mode.BEAT

    def test_reset_returns_to_idle(self):
        md = ModeDetector()
        md.mode = Mode.BEAT
        md.reset()
        assert md.mode == Mode.IDLE


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
        rng = np.random.default_rng(42)
        # Alternating spike/silence — shape changes radically each frame
        for i in range(40):
            if i % 2 == 0:
                spectrum = np.ones(N_BINS, dtype=np.float32) * 0.5
            else:
                spectrum = np.ones(N_BINS, dtype=np.float32) * 0.01
                spectrum[:100] = 0.5  # different shape
            flux = np.abs(spectrum - 0.01)  # non-zero flux
            ea.update(spectrum, flux)
        # Shape changes = percussive (both flux and shape methods)
        assert ea.percussiveness > 0.1
        assert ea.flux_percussiveness > 0.1

    def test_percussiveness_low_on_sustain(self):
        ea = EnergyAnalyzer()
        # Constant spectrum — no shape change, low flux
        spectrum = np.ones(N_BINS, dtype=np.float32) * 0.5
        flux = np.ones(N_BINS, dtype=np.float32) * 0.001
        for _ in range(50):  # enough warmup for EMA to decay from initial 0.5
            ea.update(spectrum, flux)
        # Stable shape + low flux = sustained
        assert ea.percussiveness < 0.15
        assert ea.flux_percussiveness < 0.1

    def test_band_rms_all_keys(self):
        ea = EnergyAnalyzer()
        spectrum = np.ones(N_BINS, dtype=np.float32) * 0.1
        ea.update(spectrum)
        band = ea.band_rms_all
        from flame_sheep_audio._band_config import default_band_config
        expected = set(default_band_config().all_band_names)
        assert set(band.keys()) == expected
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
# OnsetDensityTracker
# -------------------------------------------------------------------

class TestOnsetDensityTracker:

    def test_density_increases_with_kicks(self):
        dt = OnsetDensityTracker()
        # Steady kicks at ~10/s, update each hop (~10ms) to let EMA converge
        for i in range(100):
            t = i * 0.01
            if i % 10 == 0:
                dt.process_onset('low', t)
            dt.update(t)
        assert dt.densities['low'] > 5.0

    def test_density_decays_without_kicks(self):
        dt = OnsetDensityTracker()
        # Build up density
        for i in range(100):
            t = i * 0.01
            if i % 10 == 0:
                dt.process_onset('low', t)
            dt.update(t)
        high = dt.densities['low']
        # No more kicks, keep updating
        for i in range(200):
            dt.update(1.0 + i * 0.01)
        assert dt.densities['low'] < high * 0.5

    def test_per_band_independence(self):
        dt = OnsetDensityTracker()
        for i in range(10):
            dt.process_onset('low', i * 0.1)
        for i in range(5):
            dt.process_onset('high', i * 0.05)
        dt.update(1.0)
        assert dt.densities['low'] > 0
        assert dt.densities['high'] > 0
        assert dt.densities['mid'] == 0.0

    def test_delta_positive_during_accelerando(self):
        dt = OnsetDensityTracker()
        # Slow kicks for 2 seconds
        for i in range(4):
            dt.process_onset('low', i * 0.5)
        dt.update(2.0)
        # Fast kicks for next second
        for i in range(10):
            dt.process_onset('low', 2.0 + i * 0.1)
        dt.update(3.0)
        assert dt.density_deltas['low'] > 0

    def test_delta_near_zero_at_steady_rate(self):
        dt = OnsetDensityTracker()
        # Steady 4 kicks/second for 5 seconds (let EMA converge)
        for i in range(500):
            t = i * 0.01
            if i % 25 == 0:  # every 250ms = 4/s
                dt.process_onset('low', t)
            dt.update(t)
        assert abs(dt.density_deltas['low']) < 1.0

    def test_reset_clears_state(self):
        dt = OnsetDensityTracker()
        for i in range(10):
            dt.process_onset('low', i * 0.1)
        dt.update(1.0)
        assert dt.densities['low'] > 0
        dt.reset()
        assert dt.densities['low'] == 0.0
        assert dt.density_deltas['low'] == 0.0


# -------------------------------------------------------------------
# MagnitudeStability
# -------------------------------------------------------------------

class TestMagnitudeStability:

    def test_sustained_signal_high_stability(self):
        ms = MagnitudeStability()
        # Feed constant magnitude for many frames → high stability
        mag = np.zeros(N_BINS, dtype=np.float32)
        mag[2:5] = 0.5  # constant energy in low band
        for _ in range(100):
            ms.update(mag)
        mask = np.zeros(N_BINS, dtype=bool)
        mask[2:5] = True
        assert ms.band_stability(mask) > 0.8

    def test_transient_signal_low_stability(self):
        ms = MagnitudeStability()
        mask = np.ones(N_BINS, dtype=bool)  # broadband
        # Broadband transient spike as the LAST frame → percussive
        # For median: time median of mostly silence = 0, freq median of
        # broadband spike = large → h/(h+p) ≈ 0 → low stability
        # For EMA: high variance from spiky input → low stability
        for i in range(200):
            mag = np.zeros(N_BINS, dtype=np.float32)
            if i % 3 == 0:
                mag[:] = 1.0
            ms.update(mag)
        # End with a spike frame
        mag = np.ones(N_BINS, dtype=np.float32)
        ms.update(mag)
        assert ms.band_stability(mask) < 0.5

    def test_silence_is_stable(self):
        ms = MagnitudeStability()
        mag = np.zeros(N_BINS, dtype=np.float32)
        for _ in range(50):
            ms.update(mag)
        mask = np.zeros(N_BINS, dtype=bool)
        mask[2:5] = True
        assert ms.band_stability(mask) == 1.0

    def test_reset_clears_state(self):
        ms = MagnitudeStability()
        mag = np.ones(N_BINS, dtype=np.float32)
        for _ in range(50):
            ms.update(mag)
        ms.reset()
        # After reset, stability should return to default (stable)
        mask = np.ones(N_BINS, dtype=bool)
        assert ms.band_stability(mask) >= 0.5
        # After reset + one new frame, should behave like fresh start
        ms.update(mag)
        assert ms.band_stability(mask) >= 0.5


# -------------------------------------------------------------------
# DropDetector
# -------------------------------------------------------------------

class TestDropDetector:

    def test_no_break_during_warmup(self):
        dd = DropDetector()
        # Quiet frames during warmup shouldn't activate breaking
        for _ in range(200):
            dd.detect([], 0.0, 120.0, False, 1/60)
        assert not dd.breaking

    def test_no_break_on_first_lows(self):
        dd = DropDetector()
        dd._warmup_frames = 999
        low = [BeatEvent('low', 1.0)]
        # Only a few low-band onsets — below MIN_LOWS_BEFORE_DROP
        for _ in range(5):
            dd.detect(low, 1.0, 120.0, False, 1/60)
        for _ in range(150):
            dd.detect([], 0.0, 120.0, False, 1/60)
        assert not dd.breaking

    def test_break_activates_after_quiet(self):
        dd = DropDetector()
        dd._warmup_frames = 999
        low = [BeatEvent('low', 1.0)]
        for _ in range(20):
            dd.detect(low, 1.0, 120.0, False, 1/60)
        for _ in range(150):
            dd.detect([], 0.0, 120.0, False, 1/60)
        assert dd.breaking

    def test_break_ends_on_low(self):
        dd = DropDetector()
        dd._warmup_frames = 999
        low = [BeatEvent('low', 1.0)]
        for _ in range(20):
            dd.detect(low, 1.0, 120.0, False, 1/60)
        for _ in range(150):
            dd.detect([], 0.0, 120.0, False, 1/60)
        assert dd.breaking
        dd.detect(low, 0.5, 120.0, False, 1/60)
        assert not dd.breaking

    def test_break_cooldown(self):
        dd = DropDetector()
        dd._warmup_frames = 999
        low = [BeatEvent('low', 1.0)]
        # First break
        for _ in range(20):
            dd.detect(low, 1.0, 120.0, False, 1/60)
        for _ in range(150):
            dd.detect([], 0.0, 120.0, False, 1/60)
        assert dd.breaking
        # End it
        dd.detect(low, 0.5, 120.0, False, 1/60)
        assert not dd.breaking
        # Second attempt — should be blocked by cooldown
        for _ in range(10):
            dd.detect(low, 1.0, 120.0, False, 1/60)
        for _ in range(150):
            dd.detect([], 0.0, 120.0, False, 1/60)
        assert not dd.breaking  # cooldown active

    def test_no_break_during_drift(self):
        dd = DropDetector()
        dd._warmup_frames = 999
        low = [BeatEvent('low', 1.0)]
        for _ in range(20):
            dd.detect(low, 1.0, 120.0, False, 1/60)
        for _ in range(150):
            dd.detect([], 0.0, 120.0, True, 1/60)
        assert not dd.breaking

    def test_reset_clears_state(self):
        dd = DropDetector()
        dd._warmup_frames = 999
        dd._total_lows = 100
        dd._quiet_frames = 200
        dd.breaking = True
        dd.reset()
        assert dd._total_lows == 0
        assert dd._quiet_frames == 0
        assert dd._cooldown == 0.0
        assert not dd.breaking


# -------------------------------------------------------------------
# BassDropDetector
# -------------------------------------------------------------------

class TestBassDropDetector:

    def test_bass_break_activates_when_subbass_quiet(self):
        """Sub-bass dropout should activate breaking."""
        dd = BassDropDetector()
        dd._warmup_frames = 999
        low = [BeatEvent('low', 1.0)]
        for _ in range(20):
            dd.detect(low, 0.5, 120.0, False, 1/60)
        for _ in range(150):
            dd.detect([], 0.0, 120.0, False, 1/60)
        assert dd.breaking

    def test_no_break_when_subbass_present(self):
        """If sub-bass stays loud, no break."""
        dd = BassDropDetector()
        dd._warmup_frames = 999
        low = [BeatEvent('low', 1.0)]
        for _ in range(20):
            dd.detect(low, 0.5, 120.0, False, 1/60)
        for _ in range(150):
            dd.detect([], 0.3, 120.0, False, 1/60)
        assert not dd.breaking

    def test_bass_break_ends_on_low(self):
        dd = BassDropDetector()
        dd._warmup_frames = 999
        low = [BeatEvent('low', 1.0)]
        for _ in range(20):
            dd.detect(low, 0.5, 120.0, False, 1/60)
        for _ in range(150):
            dd.detect([], 0.0, 120.0, False, 1/60)
        assert dd.breaking
        dd.detect(low, 0.5, 120.0, False, 1/60)
        assert not dd.breaking

    def test_no_bass_break_during_drift(self):
        dd = BassDropDetector()
        dd._warmup_frames = 999
        low = [BeatEvent('low', 1.0)]
        for _ in range(20):
            dd.detect(low, 0.5, 120.0, False, 1/60)
        for _ in range(150):
            dd.detect([], 0.0, 120.0, True, 1/60)
        assert not dd.breaking

    def test_reset_clears_state(self):
        dd = BassDropDetector()
        dd._warmup_frames = 999
        dd._total_lows = 100
        dd._quiet_frames = 200
        dd.breaking = True
        dd.reset()
        assert dd._total_lows == 0
        assert dd._quiet_frames == 0
        assert dd._cooldown == 0.0
        assert not dd.breaking


# -------------------------------------------------------------------
# GenomeAxis event handling
# -------------------------------------------------------------------

class TestGenomeAxisEvents:

    def _make_axis(self):
        _seed = iter(range(1000))
        from flame_sheep_audio import FREQS
        return GenomeAxis(
            genome_factory=lambda: trivial_genome(next(_seed)),
            role=_default_role, freqs=FREQS)

    def test_ignores_unknown_events(self):
        axis = self._make_axis()
        audio = _audio(events=[BeatEvent('alien_signal', 1.0)])
        # Should not raise
        axis.tick(audio, 1/60, 0.0)

    def test_break_damps_morph(self):
        # Breaking axis — morph should advance slowly
        axis_break = self._make_axis()
        axis_break.morph_speed = 0.1
        for i in range(60):
            axis_break.tick(_audio(breaking=True), 1/60, float(i))
        mt_breaking = axis_break.morph_t

        # Normal axis — morph should advance freely
        axis_normal = self._make_axis()
        axis_normal.morph_speed = 0.1
        for i in range(60):
            axis_normal.tick(_audio(), 1/60, float(i))
        mt_normal = axis_normal.morph_t

        assert mt_normal > 0
        assert mt_breaking < mt_normal * 0.5, \
            "Break should slow morph to less than half normal speed"

    def test_brief_break_minimal_effect(self):
        """A 10-frame false positive should barely affect morph."""
        # Normal morph for 10 frames
        axis_normal = self._make_axis()
        axis_normal.morph_speed = 0.1
        for i in range(10):
            axis_normal.tick(_audio(), 1/60, float(i))
        mt_normal = axis_normal.morph_t

        # 10 frames of break
        axis_break = self._make_axis()
        axis_break.morph_speed = 0.1
        for i in range(10):
            axis_break.tick(_audio(breaking=True), 1/60, float(i))
        mt_break = axis_break.morph_t

        # Brief break should still allow most of the normal morph progress
        assert mt_break > mt_normal * 0.5, \
            "Brief break should barely slow morph"

    def test_damping_recovers_after_break(self):
        axis = self._make_axis()
        axis.morph_speed = 0.1
        # Deep break — morph should be very slow
        for i in range(60):
            axis.tick(_audio(breaking=True), 1/60, float(i))
        mt_after_break = axis.morph_t

        # Recovery — morph in a single frame after recovery should be fast
        for i in range(60):
            axis.tick(_audio(breaking=False), 1/60, float(60 + i))
        mt_after_recovery = axis.morph_t
        # The 60 recovery frames should add substantially more progress
        recovery_progress = mt_after_recovery - mt_after_break
        assert recovery_progress > mt_after_break, \
            "Morph should be faster after break recovery than during break"

    def test_song_start_resets(self):
        axis = self._make_axis()
        # Build some state
        for i in range(10):
            axis.tick(_audio(events=[BeatEvent('low', 0.5)]), 1/60, float(i))
        # Song start should reset energy tracking
        axis.tick(_audio(events=[BeatEvent('song_start', 0.0)]), 1/60, 20.0)
        assert axis._recent_downbeat_energy == 0.5

    def test_low_percussiveness_slows_morph(self):
        axis = self._make_axis()
        axis.morph_speed = 0.1
        # High percussiveness
        axis.tick(_audio(percussiveness=0.8), 1/60, 0.0)
        mt_fast = axis.morph_t
        # Reset
        axis.morph_t = 0.0
        # Low percussiveness
        axis.tick(_audio(percussiveness=0.1), 1/60, 1.0)
        mt_slow = axis.morph_t
        assert mt_slow < mt_fast

    def test_centroid_swap_on_low_percussiveness(self):
        axis = self._make_axis()
        axis.morph_t = 0.5  # must be > 0.3
        initial_target = id(axis.target_genome)
        # First tick: initialize mel centroid with energy at 200 Hz
        from flame_sheep_audio import N_BINS
        spec_low = np.zeros(N_BINS, dtype=np.float32)
        spec_low[5:10] = 1.0  # energy around 200 Hz
        axis.tick(_audio(percussiveness=0.1, spectrum=spec_low), 1/60, 0.0)
        axis.morph_t = 0.5  # reset after first tick consumed it
        # Second tick: shift energy to 5 kHz — big mel delta
        spec_high = np.zeros(N_BINS, dtype=np.float32)
        spec_high[200:220] = 1.0  # energy around 5 kHz
        audio = _audio(percussiveness=0.1, spectrum=spec_high)
        axis.tick(audio, 1/60, 0.1)
        assert id(axis.target_genome) != initial_target
