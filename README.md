# flame-sheep

Audio-reactive flame fractal wallpaper. Electric Sheep reimagined for modern hardware.

## Concept

Flame fractals (Scott Draves' algorithm) rendered in real time via OpenGL compute shaders.
Audio from PipeWire drives the visualization across three orthogonal axes:

- **Kick drum → geometry**: fractal structure morphs on the downbeat, with each genome
  in a pre-composed loop advancing every 4 kicks (one musical bar).
- **Snare → color**: palette shifts via graph traversal — energy controls jump distance
  (quiet snares = subtle color drift, loud = dramatic shift).
- **Hihat → zoom pulse**: brief zoom boost on hits, decays between them.
- **RMS → brightness**: overall audio loudness drives tonemap brightness.

A genetic algorithm evolves loops of genomes based on automated fitness metrics
(coverage, entropy, coherence, smoothness) and user feedback (like/dislike via
control pipe or sway hotkeys). Evolution runs in a subprocess to avoid blocking rendering.

A single Arc 770 can render what once required thousands of distributed machines.

## Architecture

```
PipeWire monitor sink
    → audio.py (capture + FFT + beat detection + tempo tracking)
    → FlameSheepCore (three orthogonal axes: genome/palette/zoom)
    → renderer.py (moderngl, compute + fragment shaders)
    → shaders/flame.comp (chaos game — GPU, 65k walkers, 30 variations)
    → shaders/tonemap.frag (log density + color + RMS brightness — GPU)

~/.local/share/flame-sheep/library.db (SQLite)
    → storage.py (genomes, loops, palettes, ratings, fitness)
    → loops.py (composition, crossover, mutation, evolution)

~/.local/share/flame-sheep/ctl (named pipe)
    → control.py (like, dislike, swap, song, tempo, quit)
```

## Usage

```bash
# Install (editable, in a venv)
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e .

# Seed the library
flame-sheep --generate-genomes 200
flame-sheep --generate-palettes 400
flame-sheep --compose-loops 10 --loop-length 6

# Run as wallpaper
flame-sheep --wallpaper

# Or in a window for development
flame-sheep --width 1920 --height 1080

# Check library state
flame-sheep --stats

# Evolve loops (can run while wallpaper is playing)
flame-sheep --evolve
```

### Sway integration

```
# ~/.config/sway/config
exec_always ~/.venv/bin/flame-sheep --wallpaper

mode "flame" {
    bindsym bracketright exec echo like > ~/.local/share/flame-sheep/ctl
    bindsym bracketleft exec echo dislike > ~/.local/share/flame-sheep/ctl
    bindsym backslash exec echo swap > ~/.local/share/flame-sheep/ctl
    bindsym Return mode "default"
    bindsym Escape mode "default"
}
bindsym $mod+g mode "flame"
```

## Control pipe

```bash
echo like    > ~/.local/share/flame-sheep/ctl   # rate current loop +1
echo dislike > ~/.local/share/flame-sheep/ctl   # rate -1, switch loop
echo swap    > ~/.local/share/flame-sheep/ctl   # force genome swap
echo song    > ~/.local/share/flame-sheep/ctl   # reset tempo tracker
echo "tempo 128" > ~/.local/share/flame-sheep/ctl  # hint BPM
echo quit    > ~/.local/share/flame-sheep/ctl   # clean shutdown
```

## Dependencies

- `dev-python/moderngl` (Guru, ~amd64)
- `dev-python/moderngl-window` (Guru, ~amd64)
- `dev-python/numpy`
- `dev-python/scipy`
- `dev-python/sounddevice` (personal-overlay)
- `media-libs/portaudio` (indirect, via sounddevice)

## Algorithm

See the flame fractal algorithm as described in Scott Draves' paper:
"The Fractal Flame Algorithm" — https://flam3.com/flame_draves.pdf
