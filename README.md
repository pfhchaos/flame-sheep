# flame-sheep

Audio-reactive flame fractal wallpaper for Wayland compositors.

Renders IFS (Iterated Function System) flame fractals as your desktop wallpaper, morphing and pulsing in response to whatever audio is playing. Designed for sway and wlr-layer-shell compositors.

## What it does

- **Fractal wallpaper** — full-screen flame fractals rendered via OpenGL compute shaders, displayed as a Wayland layer-shell surface behind your windows
- **Audio-reactive** — captures system audio via PipeWire, analyzes it in real time, and drives the visual response
- **Beat detection** — spectral flux onset detection with per-bin harmonic/percussive separation. Low-frequency beats drive genome morphing, mid-range drives palette changes, high-frequency drives zoom pulses
- **Density-driven** — at high tempos, per-event effects smoothly scale down into continuous texture. Works with everything from ambient to speedcore
- **Adaptive** — band frequencies drift toward where the percussion actually lives (optional spring-model adaptive bands). Tempo-scaled constants adjust to the music's speed
- **Evolutionary** — genomes evolve via user feedback (like/dislike). Background CPU scorer evaluates symmetry, self-similarity, fractal dimension, and detail sensitivity
- **Multi-monitor** — renders independently on each output via wlr-layer-shell frame callbacks

## Requirements

- **GPU**: OpenGL 4.3+ compute shaders (tested on Intel Arc A770)
- **Compositor**: sway or any wlr-layer-shell compatible Wayland compositor
- **Audio**: PipeWire
- **Python**: 3.11+
- **Dependencies**: moderngl, numpy, scipy, sounddevice

## Quick start

```bash
git clone https://github.com/pfhchaos/flame-sheep.git
cd flame-sheep
python -m venv .venv
source .venv/bin/activate
pip install -e .

# First run auto-generates 200 genomes and composes loops
python -m flame_sheep
```

## Usage

```bash
# Run as wallpaper (default)
python -m flame_sheep

# Specify audio device
python -m flame_sheep --device "Monitor of Built-in Audio"

# Generate more genomes
python -m flame_sheep --generate-genomes 500

# Compose loops from existing genomes
python -m flame_sheep --compose-loops 50

# Run evolution (after accumulating likes/dislikes)
python -m flame_sheep --evolve

# Show library stats
python -m flame_sheep --stats
```

## Control pipe

Send commands while running:

```bash
echo "like" > ~/.local/share/flame-sheep/ctl     # like current genome
echo "dislike" > ~/.local/share/flame-sheep/ctl   # dislike current genome
echo "swap" > ~/.local/share/flame-sheep/ctl      # force genome swap
echo "next" > ~/.local/share/flame-sheep/ctl      # next loop
echo "config reload" > ~/.local/share/flame-sheep/ctl  # hot-reload config
```

## Configuration

Copy the default config and customize:

```bash
mkdir -p ~/.config/flame-sheep
cp flame_sheep/default_config.toml ~/.config/flame-sheep/config.toml
```

All tuning constants are configurable and hot-reloadable. See `default_config.toml` for descriptions of every parameter.

## Tools

```bash
# Visualize symmetry group transforms
python tools/view_symmetry_groups.py              # fractal view
python tools/view_symmetry_groups.py --matrix      # transform operations
python tools/view_symmetry_groups.py --variation sattractor --param sat_m=6

# Analyze a song's audio characteristics
python tools/generate_golden_masters.py --verify   # check transform regression
```

## License

GPL-3.0 — see [LICENSE](LICENSE).

## Acknowledgments

- Variation functions adapted from [JWildfire](https://github.com/thargor6/JWildfire) by Andreas Maschke
- Symmetry groups from McGregor & Watt, "The Art of Graphics for the IBM PC"
- Flame fractal algorithm by Scott Draves
