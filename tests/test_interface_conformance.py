"""Interface conformance tests for swappable components.

Every component that can be swapped (spectrum engine, stability method)
must conform to its interface contract regardless of bin count, input
size, or initialization order. These tests catch the class of bugs
where a component hardcodes N_BINS=1025 or assumes non-silent first input.
"""

import numpy as np
import pytest

from flame_sheep_audio._constants import SAMPLE_RATE, HOP_SIZE
from flame_sheep_audio._spectrum import SpectrumFrame, SpectrumEngineBase
from flame_sheep_audio.stability import (
    StabilityMethod,
    MagnitudeStability, _StabilityEMA, _StabilityMedian, _StabilityShape,
)
from flame_sheep_audio.energy import EnergyAnalyzer
from flame_sheep_audio.beat_detector import BeatDetectorBase, FluxBeatDetector
from flame_sheep_audio._octave_bank import OctaveBankEngine

# Try importing optional engines
try:
    from flame_sheep_audio._cqt_engine import CqtEngine
    _HAS_CQT = True
except ImportError:
    _HAS_CQT = False


# --- Bin counts to test ---
# 1025 = FFT engine, 108 = octave bank / CQT, 64 = hypothetical
BIN_COUNTS = [108, 1025, 64]


# ===================================================================
# Stability interface conformance
# ===================================================================

# All concrete StabilityMethod subclasses with their factory functions.
# Adding a new stability method? Add it here — the tests are automatic.
STABILITY_CLASSES = [
    ("ema", lambda: _StabilityEMA(alpha=0.95)),
    ("median", lambda: _StabilityMedian(kernel_time=15, kernel_freq=7)),
    ("shape", lambda: _StabilityShape(alpha=0.95, kernel=7)),
]

def test_all_stability_methods_registered():
    """Ensure every StabilityMethod subclass has a test entry."""
    concrete = {cls.__name__ for cls in StabilityMethod.__subclasses__()}
    tested = {factory().__class__.__name__ for _, factory in STABILITY_CLASSES}
    missing = concrete - tested
    assert not missing, f"StabilityMethod subclasses without test entries: {missing}"


class TestStabilityConformance:
    """Every stability implementation must handle arbitrary bin counts."""

    @pytest.mark.parametrize("name,factory", STABILITY_CLASSES)
    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_output_matches_input_size(self, name, factory, n_bins):
        """stability_per_bin() must return same size as input magnitude."""
        stab = factory()
        mag = np.random.randn(n_bins).astype(np.float32) * 0.01
        mag = np.abs(mag)
        stab.update(mag)
        result = stab.stability_per_bin()
        assert result.shape == (n_bins,), \
            f"{name}: expected ({n_bins},), got {result.shape}"

    @pytest.mark.parametrize("name,factory", STABILITY_CLASSES)
    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_silent_first_frame(self, name, factory, n_bins):
        """Must handle silence on the first frame without size mismatch."""
        stab = factory()
        silence = np.zeros(n_bins, dtype=np.float32)
        stab.update(silence)
        result = stab.stability_per_bin()
        assert result.shape == (n_bins,), \
            f"{name}: silent frame returned {result.shape}, expected ({n_bins},)"

    @pytest.mark.parametrize("name,factory", STABILITY_CLASSES)
    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_stability_per_bin_before_update(self, name, factory, n_bins):
        """stability_per_bin() before any update should not crash.

        May return a default size, but must not raise.
        """
        stab = factory()
        result = stab.stability_per_bin()
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1

    @pytest.mark.parametrize("name,factory", STABILITY_CLASSES)
    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_multiple_silent_then_signal(self, name, factory, n_bins):
        """Multiple silent frames followed by signal must work."""
        stab = factory()
        silence = np.zeros(n_bins, dtype=np.float32)
        for _ in range(5):
            stab.update(silence)
            result = stab.stability_per_bin()
            assert result.shape == (n_bins,)

        # Now feed signal
        mag = np.abs(np.random.randn(n_bins).astype(np.float32))
        stab.update(mag)
        result = stab.stability_per_bin()
        assert result.shape == (n_bins,)

    @pytest.mark.parametrize("name,factory", STABILITY_CLASSES)
    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_harmonic_rms_matches_input(self, name, factory, n_bins):
        """harmonic_rms() must work with masks matching input size."""
        stab = factory()
        mag = np.abs(np.random.randn(n_bins).astype(np.float32))
        stab.update(mag)
        mask = np.zeros(n_bins, dtype=bool)
        mask[:n_bins // 2] = True
        result = stab.harmonic_rms(mag, mask)
        assert isinstance(result, float)

    @pytest.mark.parametrize("name,factory", STABILITY_CLASSES)
    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_band_stability_matches_input(self, name, factory, n_bins):
        """band_stability() must work with masks matching input size."""
        stab = factory()
        mag = np.abs(np.random.randn(n_bins).astype(np.float32))
        stab.update(mag)
        mask = np.ones(n_bins, dtype=bool)
        result = stab.band_stability(mask)
        assert 0.0 <= result <= 1.0

    @pytest.mark.parametrize("name,factory", STABILITY_CLASSES)
    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_reset_then_different_size(self, name, factory, n_bins):
        """After reset, must accept a different bin count."""
        stab = factory()
        # Feed one size
        mag1 = np.abs(np.random.randn(64).astype(np.float32))
        stab.update(mag1)
        assert stab.stability_per_bin().shape == (64,)
        # Reset and feed different size
        stab.reset()
        mag2 = np.abs(np.random.randn(n_bins).astype(np.float32))
        stab.update(mag2)
        assert stab.stability_per_bin().shape == (n_bins,)


class TestMagnitudeStabilityConformance:
    """MagnitudeStability (composite) must respect bin counts for all methods."""

    @pytest.mark.parametrize("method", ["ema", "median", "shape"])
    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_composite_output_size(self, method, n_bins):
        stab = MagnitudeStability(method=method)
        mag = np.abs(np.random.randn(n_bins).astype(np.float32))
        stab.update(mag)
        assert stab.stability_per_bin().shape == (n_bins,)

    @pytest.mark.parametrize("method", ["ema", "median", "shape"])
    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_composite_silent_start(self, method, n_bins):
        stab = MagnitudeStability(method=method)
        stab.update(np.zeros(n_bins, dtype=np.float32))
        assert stab.stability_per_bin().shape == (n_bins,)


# ===================================================================
# Energy analyzer conformance
# ===================================================================

class TestEnergyConformance:
    """EnergyAnalyzer must work with any bin count."""

    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_update_with_stability(self, n_bins):
        """Energy update with stability must not crash on any bin count."""
        freqs = np.linspace(20, 20000, n_bins).astype(np.float32)
        energy = EnergyAnalyzer(freqs=freqs)
        stab = MagnitudeStability()
        mag = np.abs(np.random.randn(n_bins).astype(np.float32))
        flux = np.abs(np.random.randn(n_bins).astype(np.float32))
        stab.update(mag)
        energy.update(mag, flux, stability=stab)

    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_update_silent_start_with_stability(self, n_bins):
        """Energy + stability must work when audio starts with silence."""
        freqs = np.linspace(20, 20000, n_bins).astype(np.float32)
        energy = EnergyAnalyzer(freqs=freqs)
        stab = MagnitudeStability()
        silence = np.zeros(n_bins, dtype=np.float32)
        stab.update(silence)
        energy.update(silence, silence, stability=stab)


# ===================================================================
# Beat detector conformance
# ===================================================================

# All concrete BeatDetectorBase subclasses with factories.
DETECTOR_CLASSES = [
    ("flux", lambda freqs: FluxBeatDetector(freqs=freqs)),
]


def test_all_beat_detectors_registered():
    """Ensure every BeatDetectorBase subclass has a test entry."""
    concrete = {cls.__name__ for cls in BeatDetectorBase.__subclasses__()}
    tested = {factory(np.linspace(20, 20000, 108).astype(np.float32)).__class__.__name__
              for _, factory in DETECTOR_CLASSES}
    missing = concrete - tested
    assert not missing, f"BeatDetectorBase subclasses without test entries: {missing}"


class TestBeatDetectorConformance:
    """Every beat detector must handle arbitrary bin counts."""

    @pytest.mark.parametrize("name,factory", DETECTOR_CLASSES)
    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_detect_returns_list(self, name, factory, n_bins):
        """detect() must return a list of BeatEvents."""
        freqs = np.linspace(20, 20000, n_bins).astype(np.float32)
        det = factory(freqs)
        frame = SpectrumFrame(
            magnitude=np.abs(np.random.randn(n_bins).astype(np.float32)),
            flux=np.abs(np.random.randn(n_bins).astype(np.float32)),
            waveform=np.random.randn(HOP_SIZE).astype(np.float32),
        )
        events = det.detect(frame)
        assert isinstance(events, list)

    @pytest.mark.parametrize("name,factory", DETECTOR_CLASSES)
    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_detect_silent_input(self, name, factory, n_bins):
        """Silence must not crash and should produce no events."""
        freqs = np.linspace(20, 20000, n_bins).astype(np.float32)
        det = factory(freqs)
        frame = SpectrumFrame(
            magnitude=np.zeros(n_bins, dtype=np.float32),
            flux=np.zeros(n_bins, dtype=np.float32),
            waveform=np.zeros(HOP_SIZE, dtype=np.float32),
        )
        events = det.detect(frame)
        assert isinstance(events, list)
        assert len(events) == 0

    @pytest.mark.parametrize("name,factory", DETECTOR_CLASSES)
    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_reset_then_detect(self, name, factory, n_bins):
        """reset_bands() then detect() must not crash."""
        freqs = np.linspace(20, 20000, n_bins).astype(np.float32)
        det = factory(freqs)
        # Feed some frames
        for _ in range(20):
            frame = SpectrumFrame(
                magnitude=np.abs(np.random.randn(n_bins).astype(np.float32)),
                flux=np.abs(np.random.randn(n_bins).astype(np.float32)),
                waveform=np.random.randn(HOP_SIZE).astype(np.float32),
            )
            det.detect(frame)
        det.reset_bands()
        # Should work after reset
        events = det.detect(frame)
        assert isinstance(events, list)

    @pytest.mark.parametrize("name,factory", DETECTOR_CLASSES)
    def test_events_have_valid_kinds(self, name, factory):
        """BeatEvent.kind must be a string."""
        freqs = np.linspace(20, 20000, 108).astype(np.float32)
        det = factory(freqs)
        # Feed enough frames with signal to trigger events
        for _ in range(50):
            frame = SpectrumFrame(
                magnitude=np.abs(np.random.randn(108).astype(np.float32)) * 0.5,
                flux=np.abs(np.random.randn(108).astype(np.float32)) * 0.5,
                waveform=np.random.randn(HOP_SIZE).astype(np.float32),
            )
            events = det.detect(frame)
            for e in events:
                assert isinstance(e.kind, str)
                assert isinstance(e.energy, (int, float))


# ===================================================================
# Spectrum engine conformance
# ===================================================================

# All concrete SpectrumEngineBase subclasses.
# Adding a new engine? Add it here — the tests are automatic.
ENGINES = [("octave_bank", OctaveBankEngine)]
if _HAS_CQT:
    ENGINES.append(("cqt", CqtEngine))

def test_all_spectrum_engines_registered():
    """Ensure every SpectrumEngineBase subclass has a test entry."""
    from flame_sheep_audio._spectrum import SpectrumEngine
    # Exclude the FFT SpectrumEngine (it doesn't implement push_hop the same way)
    # and optional engines that aren't installed
    tested = {cls.__name__ for _, cls in ENGINES}
    concrete = set()
    for cls in SpectrumEngineBase.__subclasses__():
        if cls is SpectrumEngine:
            continue  # legacy FFT engine, separate interface
        try:
            # Only check if the class can be imported
            concrete.add(cls.__name__)
        except Exception:
            pass
    missing = concrete - tested
    assert not missing, f"SpectrumEngineBase subclasses without test entries: {missing}"


class TestSpectrumEngineConformance:
    """Every spectrum engine must produce consistent SpectrumFrame shapes."""

    @pytest.mark.parametrize("name,cls", ENGINES)
    def test_output_shape_consistency(self, name, cls):
        """All frame arrays must have the same length as n_bins."""
        engine = cls()
        hop = np.random.randn(HOP_SIZE).astype(np.float32) * 0.3
        frame = engine.push_hop(hop)
        assert frame.magnitude.shape == (engine.n_bins,), \
            f"{name}: magnitude {frame.magnitude.shape} != ({engine.n_bins},)"
        assert frame.flux.shape == (engine.n_bins,), \
            f"{name}: flux {frame.flux.shape} != ({engine.n_bins},)"
        assert frame.waveform.shape == (HOP_SIZE,)

    @pytest.mark.parametrize("name,cls", ENGINES)
    def test_silent_input(self, name, cls):
        """Engine must handle silence without crashing."""
        engine = cls()
        silence = np.zeros(HOP_SIZE, dtype=np.float32)
        frame = engine.push_hop(silence)
        assert frame.magnitude.shape == (engine.n_bins,)
        assert np.all(np.isfinite(frame.magnitude))

    @pytest.mark.parametrize("name,cls", ENGINES)
    def test_bin_centers_match_n_bins(self, name, cls):
        """bin_centers must have exactly n_bins elements."""
        engine = cls()
        assert len(engine.bin_centers) == engine.n_bins

    @pytest.mark.parametrize("name,cls", ENGINES)
    def test_compute_matches_push_hop(self, name, cls):
        """compute() must produce same shape as push_hop()."""
        engine = cls()
        pcm = np.random.randn(HOP_SIZE * 4).astype(np.float32) * 0.3
        frame = engine.compute(pcm)
        assert frame.magnitude.shape == (engine.n_bins,)

    @pytest.mark.parametrize("name,cls", ENGINES)
    def test_reset_allows_reuse(self, name, cls):
        """After reset, engine must work normally."""
        engine = cls()
        hop = np.random.randn(HOP_SIZE).astype(np.float32) * 0.3
        engine.push_hop(hop)
        engine.reset()
        frame = engine.push_hop(hop)
        assert frame.magnitude.shape == (engine.n_bins,)


# ===================================================================
# Integration: engine + stability + energy pipeline
# ===================================================================

class TestPipelineConformance:
    """Full pipeline must work for every engine × stability combination."""

    @pytest.mark.parametrize("eng_name,eng_cls", ENGINES)
    @pytest.mark.parametrize("stab_method", ["ema", "median", "shape"])
    def test_full_pipeline(self, eng_name, eng_cls, stab_method):
        """10 frames through engine → stability → energy must not crash."""
        engine = eng_cls()
        stab = MagnitudeStability(method=stab_method)
        energy = EnergyAnalyzer(freqs=engine.bin_centers)

        for i in range(10):
            hop = np.random.randn(HOP_SIZE).astype(np.float32) * 0.3
            frame = engine.push_hop(hop)
            stab.update(frame.magnitude)
            energy.update(frame.magnitude, frame.flux, stability=stab)

    @pytest.mark.parametrize("eng_name,eng_cls", ENGINES)
    @pytest.mark.parametrize("stab_method", ["ema", "median", "shape"])
    def test_pipeline_engine_hotswap(self, eng_name, eng_cls, stab_method):
        """Switching spectrum engine mid-stream must not crash.

        Stability and energy must adapt to the new bin count after
        the engine is swapped.
        """
        # Start with octave bank
        engine1 = OctaveBankEngine()
        stab = MagnitudeStability(method=stab_method)
        energy = EnergyAnalyzer(freqs=engine1.bin_centers)

        for _ in range(10):
            hop = np.random.randn(HOP_SIZE).astype(np.float32) * 0.3
            frame = engine1.push_hop(hop)
            stab.update(frame.magnitude)
            energy.update(frame.magnitude, frame.flux, stability=stab)

        # Swap to the test engine — stability and energy must be recreated
        # (they hold masks/freqs tied to bin count)
        engine2 = eng_cls()
        stab.reset()
        energy2 = EnergyAnalyzer(freqs=engine2.bin_centers)

        for _ in range(10):
            hop = np.random.randn(HOP_SIZE).astype(np.float32) * 0.3
            frame = engine2.push_hop(hop)
            stab.update(frame.magnitude)
            energy2.update(frame.magnitude, frame.flux, stability=stab)

    @pytest.mark.parametrize("eng_name,eng_cls", ENGINES)
    @pytest.mark.parametrize("stab_method", ["ema", "median", "shape"])
    def test_pipeline_silent_start(self, eng_name, eng_cls, stab_method):
        """Pipeline must survive 5 silent frames then 5 signal frames."""
        engine = eng_cls()
        stab = MagnitudeStability(method=stab_method)
        energy = EnergyAnalyzer(freqs=engine.bin_centers)

        # Silent frames
        for _ in range(5):
            frame = engine.push_hop(np.zeros(HOP_SIZE, dtype=np.float32))
            stab.update(frame.magnitude)
            energy.update(frame.magnitude, frame.flux, stability=stab)

        # Signal frames
        for _ in range(5):
            hop = np.random.randn(HOP_SIZE).astype(np.float32) * 0.3
            frame = engine.push_hop(hop)
            stab.update(frame.magnitude)
            energy.update(frame.magnitude, frame.flux, stability=stab)
