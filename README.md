# flame-sheep

Audio-reactive flame fractal wallpaper. Electric Sheep reimagined for modern hardware.

## Concept

Flame fractals (Scott Draves' algorithm) rendered in real time via OpenGL compute shaders.
Audio from PipeWire drives beat detection — on a kick, the fractal genome mutates toward
a new attractor. On a snare, the color palette shifts. The fractal breathes with the music.

A single Arc 770 can render what once required thousands of distributed machines.

## Architecture

```
PipeWire monitor sink
    → audio.py (capture + FFT + beat detection)
    → genome.py (IFS parameter set + mutation)
    → renderer.py (moderngl, compute + fragment shaders)
    → shaders/flame.comp (chaos game — GPU)
    → shaders/tonemap.frag (log density + color — GPU)
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
