# flame-sheep Development Guide

## Architecture

```
Signal Source (PipeWire / test feed)
       │
  Spectrum Engine (shared FFT + flux)
       │
       ├─────────────┬──────────────┐
       ▼              ▼              ▼
  Beat Detector   Energy Analyzer  (future: Spectral Features)
  (spectral flux)  (sub-bass RMS)
       │              │
       ▼              │
  Rhythm Tracker      │
  (tempo gating)      │
       │              │
       ▼              ▼
  Visual Axes (composable, toggleable)
  ├─ GenomeAxis     kick → genome morph/swap
  ├─ PaletteAxis    snare → palette graph walk
  ├─ ZoomAxis       hihat → zoom pulse
  ├─ BrightnessAxis energy → display gamma
  ├─ DetailAxis     energy → iteration count
  └─ DriftAxis      silence → slow morph + loop cycling
       │
  Renderer (GPU compute shaders)
       │
  Display (Wayland layer-shell / GLFW)
```

## Module Layout

### `flame_sheep/audio/` — Audio Analysis Pipeline

| Module | Purpose |
|--------|---------|
| `_constants.py` | SAMPLE_RATE, FFT_SIZE, N_BINS, etc. |
| `_types.py` | `BeatEvent` dataclass |
| `_spectrum.py` | `SpectrumEngine`: FFT, windowing, spectral flux |
| `_bands.py` | `AdaptiveBand`, frequency masks, A-weighting curve |
| `beat_detector.py` | `FluxBeatDetector`: onset detection from spectral flux |
| `energy.py` | `EnergyAnalyzer`: sub-bass RMS for visceral loudness |
| `source.py` | `PipeWireSource`, `FeedSource`: signal input abstraction |
| `processor.py` | `AudioProcessor` (orchestrator), `SyntheticAudioProcessor` (test metronome) |
| `__init__.py` | Re-exports for backward-compatible `from flame_sheep.audio import ...` |

**Data flow:** Source → SpectrumEngine → SpectrumFrame → (BeatDetector, EnergyAnalyzer) → (BeatEvents, RMS)

### `flame_sheep/axes/` — Visual State Machines

Each axis implements: `tick(events, rms, dt, clock)` + `contribute(frame)`.
Each axis has an `enabled` flag.

| Module | Input | Output |
|--------|-------|--------|
| `genome_axis.py` | kick events | `frame.genome` (morphed genome) |
| `palette_axis.py` | snare events | `frame.palette` (interpolated colors) |
| `zoom_axis.py` | hihat events | `frame.genome.zoom` (multiplicative) |
| `brightness_axis.py` | RMS | `frame.brightness` |
| `detail_axis.py` | RMS | `frame.iterations` |
| `drift_axis.py` | RMS (quiet) | triggers GenomeAxis swaps |

**Axis order matters:** GenomeAxis must contribute before ZoomAxis (zoom modifies the genome).
DriftAxis must tick before GenomeAxis (drift triggers swaps that genome then morphs).

### Core Modules

| Module | Purpose |
|--------|---------|
| `main.py` | `FlameSheepCore` (thin orchestrator), CLI, render loops |
| `genome.py` | `Genome`, `Transform`, IFS parameters, viability checks |
| `tempo.py` | `TempoTracker`: IOI analysis, rhythm coherence gating |
| `renderer.py` | GPU flame fractal rendering (compute + tonemap) |
| `wayland_window.py` | Wayland layer-shell surfaces, frame callbacks, EGL |
| `control.py` | Named pipe control interface |
| `storage.py` | SQLite persistence (genomes, loops, ratings) |
| `loops.py` | Loop composition, breeding, evolution |

## Testing

### Test Files

| File | Scope | Speed |
|------|-------|-------|
| `test_components.py` | Individual components in isolation | < 1s |
| `test_core.py` | FlameSheepCore state machine (fake clock) | ~5s |
| `test_audio.py` | AudioProcessor spectrum/detection basics | < 1s |
| `test_beat_engine.py` | End-to-end PCM → events (synthetic drums) | ~12s |
| `test_genome.py` | Genome math, viability, lerp | ~5s |
| `test_tempo.py` | Tempo tracking, gating | < 1s |
| `test_storage.py` | SQLite I/O | ~2s |
| `test_loops.py` | Loop composition | ~5s |
| `eval_musdb.py` | MUSDB18 evaluation (not in test suite) | minutes |

### Test Infrastructure

- **`tests/conftest.py`** — `make_processor()` (uses `FeedSource`), `trivial_genome()`, `FakeClock`, PCM helpers
- **`tests/synths.py`** — Synthetic instruments (kick, snare, hihat, 808, clap, vocal, speech), `DrumPattern` builder, `PatternSpec` library, compression
- **Run tests:** `python -m pytest tests/ --ignore=tests/eval_musdb.py`
- **Fast subset:** `python -m pytest tests/test_components.py tests/test_core.py`

### Adding a New Drum Pattern

```python
# In tests/synths.py:
MY_PATTERN = PatternSpec(
    name='my pattern', bpm=120, bars=4, bar_length=4,
    kick_beats=[1, 3], snare_beats=[2, 4],
    hihat_beats=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],
    min_kicks=3, min_snares=2, min_hihats=3,
    kit=DrumPattern.KIT_ELECTRONIC,  # optional: use 808s
)

# Append to ALL_PATTERNS → gets 9 parametrized tests automatically
```

## Control Pipe

```bash
echo swap > ~/.local/share/flame-sheep/ctl    # force genome swap
echo like > ~/.local/share/flame-sheep/ctl    # upvote current loop
echo dislike > ~/.local/share/flame-sheep/ctl # downvote + switch loop
echo song > ~/.local/share/flame-sheep/ctl    # signal new song (reset tempo)
echo "tempo 120" > ~/.local/share/flame-sheep/ctl  # hint BPM
echo quit > ~/.local/share/flame-sheep/ctl    # clean shutdown
```

## Key Design Decisions

- **Spectral flux for beat detection** — volume-invariant, self-calibrating
- **Adaptive bands** — soft weights drift toward where energy lives (catches 808s, low hihats)
- **Sub-bass RMS for brightness/iterations** — tracks physical rumble, not perceptual loudness
- **Frame callbacks for display** — no vsync blocking, per-output independent refresh
- **Singleton via PID file** — prevents zombie instances on sway reload
- **Injectable clock + genome factory** — deterministic testing without real-time sleeps
