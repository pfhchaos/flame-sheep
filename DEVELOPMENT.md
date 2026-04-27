# flame-sheep Development Guide

## Architecture

### Audio subsystem

The audio engine runs in a daemon thread. All components receive a
`BandConfig` at construction time — the visualization declares what
bands it needs, the engine doesn't know what they're for.

```
PipeWire ─→ Signal Source ─→ SpectrumEngine
                                  │
                           magnitude + flux
                                  │
               ┌──────────────────┼──────────────────┐
               ▼                  ▼                   ▼
        MagnitudeStability   EnergyAnalyzer      A-weighted
        (per-bin variance,   (per-band RMS,      flux sum
         fast + slow EMA)    harmonic RMS)           │
               │                  │                   ▼
               │                  │          ACF Tempo Tracker
               │                  │          (autocorrelation of
               │                  │           onset strength,
               │                  │           independent of beat
               │                  │           detection)
               │                  │                   │
     stability │    harmonic      │            effective_bpm
      scaling  │    RMS weighting │                   │
               │                  │          tunes cooldowns
               ▼                  │                   │
        FluxBeatDetector ←────────┘───────────────────┘
        (per-band onset detection,
         stability-scaled thresholds,
         tempo-adaptive cooldown)
               │
          BeatEvents
               │
               ▼
        OnsetDensityTracker
        (per-band onset rate,
         per-band density delta)
               │
               ▼
        ┌──────────────────────────────────┐
        │         AudioSnapshot            │
        │  (all features under one lock)   │
        │                                  │
        │  bands[name].rms                 │
        │  bands[name].harmonic_rms        │
        │  bands[name].onset_density       │
        │  bands[name].density_delta       │
        │  centroid, centroid_delta         │
        │  centroid_rms, centroid_hrms      │
        │  percussiveness                  │
        │  section_change                  │
        │  bpm, effective_bpm              │
        │  tempo_confidence                │
        │  spectrum, waveform              │
        │  events[]                        │
        └──────────────────────────────────┘
```

Two paths through the audio engine:
- **Detection bands**: onset detection + density tracking + spring adaptation
- **Energy bands**: RMS + harmonic RMS tracking only

Tempo estimation is decoupled from beat detection — the ACF tracker
reads the continuous A-weighted flux sum, not discrete beat events.
The only feedback is tempo → beat detector cooldown tuning (one-way).

### Visual pipeline

```
AudioSnapshot
     │
drain() → render thread
     │
ModeDetector (beat / idle)
     │
AudioState → Visual Axes
├─ GenomeAxis     kick/snare density → genome morph/swap
│                 section_change → loop swap on next strong beat
├─ PaletteAxis    snare density → palette walk
├─ ZoomAxis       hihat density → zoom pulse
├─ BrightnessAxis slow_harmonic_rms → display gamma
├─ DetailAxis     slow_harmonic_rms → iteration count
└─ DriftMode      idle mode → slow autonomous morph
     │
Renderer (GPU compute shaders)
     │
Display (Wayland layer-shell)
```

### Key design principles

- **No audio analysis in render thread** — all analysis runs in the audio thread, render thread just reads snapshots
- **Density-driven, not event-counting** — morph speed is a continuous function of onset density, not a kick counter. Time-signature-agnostic
- **Independent tempo estimation** — autocorrelation of raw onset strength, decoupled from beat detection. No circular feedback.
- **Configurable bands** — BandConfig defines detection vs energy bands. The visualization declares what it needs.
- **Energy-based swaps** — genome direction changes on high-energy beats (strong beat detection), not fixed intervals
- **Harmonic/percussive separation** — per-bin magnitude variance (HPSS-lite) separates sustained content from transients. Brightness tracks harmonic energy only
- **Exponential break damping** — morph slows proportionally to break duration, brief false positives barely visible
- **Section change detection** — dual-EMA on centroid + energy, euclidean distance triggers loop swaps on next strong beat
- **Three-state mode machine** — beat (all axes) / idle (drift morph). Energy mode designed but disabled pending speech/music discrimination.
- **All constants configurable** — TOML config, hot-reloadable via control pipe

## Module Layout

### `flame_sheep_audio/` — Standalone Audio Analysis Package

| Module | Purpose |
|--------|---------|
| `_constants.py` | SAMPLE_RATE, FFT_SIZE, HOP_SIZE, FREQS |
| `_types.py` | `BeatEvent`, `BandState`, `AudioState`, `AudioSnapshot` |
| `_band_config.py` | `BandConfig`, `EnergyBandDef`, `DetectionBandDef`, `default_band_config()` |
| `_spectrum.py` | `SpectrumEngine`: FFT, windowing, spectral flux, onset_strength, ZCR |
| `_bands.py` | `SpringBand`, `AdaptiveBand`, `make_mask`, A-weighting |
| `config.py` | Audio-only config (detection, stability, energy, density, tempo thresholds) |
| `beat_detector.py` | `FluxBeatDetector`: onset detection with stability scaling |
| `energy.py` | `EnergyAnalyzer`: per-band RMS/harmonic RMS, slow envelopes, section change |
| `stability.py` | `MagnitudeStability`: per-bin EMA variance at two timescales |
| `onset_density.py` | `OnsetDensityTracker`: per-band onset rate + per-band density delta |
| `tempo_acf.py` | `AutocorrelationTempoTracker`: ACF-based tempo, confidence, BPM delta |
| `tempo_scaler.py` | `TempoScaler`: sigmoid BPM → constant value mapping |
| `drop_detector.py` | `DropDetector`: break detection from centroid energy dropout |
| `bass_drop_detector.py` | `BassDropDetector`: break detection from sub-bass dropout |
| `source.py` | `PipeWireSource`, `FeedSource`: signal input abstraction |
| `processor.py` | `AudioProcessor` (orchestrator), `SyntheticAudioProcessor` (test) |

### `flame_sheep/axes/` — Visual State Machines

Each axis follows the `VisualAxis` protocol: `tick(audio, dt, clock)` + `contribute(frame)`.

| Module | Input | Output |
|--------|-------|--------|
| `genome_axis.py` | beat events, onset density, section_change | `frame.genome` (morphed genome) |
| `palette_axis.py` | snare/mid events | `frame.palette` (interpolated colors) |
| `zoom_axis.py` | hihat/high events | `frame.genome.zoom` (multiplicative) |
| `brightness_axis.py` | harmonic RMS | `frame.brightness` |
| `detail_axis.py` | harmonic RMS | `frame.iterations` |

### Core Modules

| Module | Purpose |
|--------|---------|
| `main.py` | `FlameSheepCore` (compositor), CLI, render loop |
| `config.py` | Visualization TOML config (genome, zoom, palette, drift), hot-reloadable |
| `mode.py` | `ModeDetector`: three-state machine (beat/energy/idle) |
| `genome.py` | `Genome`, `Transform`, variation packing, IFS parameters |
| `drift_mode.py` | `DriftMode`: genome morph engine for idle mode |
| `tempo.py` | `TempoTracker`: legacy IOI tracker (replaced by `tempo_acf.py` in audio package) |
| `renderer.py` | GPU flame fractal rendering (compute shaders + tonemap) |
| `scorer.py` | `BackgroundScorer`: CPU symmetry/aesthetic scoring process |
| `symmetry.py` | Symmetry metrics: rotational, reflective, radial, self-similarity |
| `storage.py` | SQLite persistence (genomes, loops, palettes, ratings) |
| `loops.py` | Loop composition, motion fields, breeding, evolution |
| `control.py` | Named pipe control interface |

### `flame_sheep/variations/` — IFS Variation Functions

| Module | Purpose |
|--------|---------|
| `_registry.py` | `Variation` enum, `VAR_PARAMS_SPEC`, `NUM_VARIATIONS` |
| `_params.py` | Random parameter generation (from JWildfire randomize()) |
| `_cpu.py` | CPU approximation for viability testing and scoring |
| `_symmetry_groups.py` | Crystallographic group element tables (17 wallpaper + 7 frieze) |

46 variations including: linear, sinusoidal, spherical, swirl, julian/juliascope, curl, splits, rectangles, checks, hex, kaleidoscope, icon attractor, sattractor, wallpaper groups, frieze groups.

### GPU Shaders (`flame_sheep/shaders/`)

| File | Purpose |
|------|---------|
| `flame.comp` | Chaos game compute shader — IFS iteration + variation dispatch |
| `clear.comp` | Zero the histogram buffer |
| `tonemap.frag` | Log-density tonemap + palette coloring |
| `downsample_hist.comp` | Histogram downsampling for symmetry scoring |

Variation params packed inline with active variations: `(var_idx, weight, p0..p5)` per slot. Symmetry group data injected at shader load via `{{SYMMETRY_GROUPS}}` placeholder.

## Testing

```bash
# Full suite (~80s)
python -m pytest

# Fast subset (~10s)
python -m pytest tests/test_components.py tests/test_core.py

# Golden master regression
python tools/generate_golden_masters.py --verify
```

| File | Scope | Tests |
|------|-------|-------|
| `test_components.py` | Individual components in isolation | ~100 |
| `test_core.py` | FlameSheepCore state machine | ~15 |
| `test_pipeline.py` | End-to-end PCM → AudioState | 10 |
| `test_golden_masters.py` | Transform + symmetry metric regression | ~58 |
| `test_genome.py` | Genome math, packing, variations | ~57 |
| `test_beat_engine.py` | PCM → beat events (synthetic drums) | ~150 |
| `test_audio.py` | AudioProcessor basics | ~10 |
| `test_tempo.py` | Tempo tracking | ~19 |
| `test_symmetry.py` | Symmetry metrics | ~22 |
| `test_storage.py` | SQLite I/O | ~20 |
| `test_loops.py` | Loop composition | ~15 |

### Adding a new variation

1. Add to `Variation` enum in `_registry.py` with next index
2. Add to `VAR_PARAMS_SPEC` if parametric (list of param names)
3. Add to `PARAMETRIC_VARIATIONS` set
4. Add randomize in `_params.py`
5. Add GPU function in `flame.comp` reading `u_active_vars[slot + PARAM_OFFSET + N]`
6. Add switch case in `apply_single_variation`
7. Add CPU fallback in `_cpu.py`
8. Regenerate golden masters: `python tools/generate_golden_masters.py`

## Configuration

All tuning constants in `~/.config/flame-sheep/config.toml`. See `default_config.toml` for full reference. Hot-reload: `echo "config reload" > ~/.local/share/flame-sheep/ctl`

## Control Pipe

```bash
echo swap > ~/.local/share/flame-sheep/ctl      # force genome swap
echo like > ~/.local/share/flame-sheep/ctl       # upvote current loop
echo dislike > ~/.local/share/flame-sheep/ctl    # downvote + switch loop
echo song > ~/.local/share/flame-sheep/ctl       # signal new song (reset tempo)
echo "tempo 120" > ~/.local/share/flame-sheep/ctl # hint BPM
echo "config reload" > ~/.local/share/flame-sheep/ctl # hot-reload config
echo quit > ~/.local/share/flame-sheep/ctl       # clean shutdown
```

## Data paths

| Path | Purpose |
|------|---------|
| `~/.local/share/flame-sheep/library.db` | Genome/loop database |
| `~/.local/share/flame-sheep/ctl` | Control pipe (FIFO) |
| `~/.local/share/flame-sheep/pid` | Singleton PID file |
| `~/.config/flame-sheep/config.toml` | User config overrides |
