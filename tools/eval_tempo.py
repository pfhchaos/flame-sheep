#!/usr/bin/env python3
"""
Evaluate tempo estimation against MUSDB18 tracks.

Uses librosa.beat.beat_track on drum stems as reference BPM,
then compares our TempoTracker's estimate from the full mix.

Metrics:
  - Exact accuracy: within ±4% of reference
  - Octave accuracy: within ±4% of reference or 2x/0.5x
  - Mean absolute error (exact-match subset)

Usage:
  python -m tests.eval_tempo --root datasets/MUSDB18/MUSDB18-7
  python -m tests.eval_tempo --root datasets/MUSDB18/MUSDB18-7 --limit 20
"""

import argparse
import sys
import time
import numpy as np

sys.path.insert(0, '.')

from flame_sheep_audio import SAMPLE_RATE, FFT_SIZE, HOP_SIZE
from flame_sheep_audio._spectrum import SpectrumEngine
from flame_sheep_audio.beat_detector import FluxBeatDetector
from flame_sheep_audio.energy import EnergyAnalyzer
from flame_sheep_audio.stability import MagnitudeStability
from flame_sheep_audio.onset_density import OnsetDensityTracker
from flame_sheep_audio.tempo_acf import AutocorrelationTempoTracker


def resample_to_48k(audio: np.ndarray, orig_sr: int = 44100) -> np.ndarray:
    if orig_sr == SAMPLE_RATE:
        return audio
    ratio = SAMPLE_RATE / orig_sr
    n_out = int(len(audio) * ratio)
    indices = np.clip((np.arange(n_out) / ratio).astype(int), 0, len(audio) - 1)
    return audio[indices]


def reference_bpm(drums_audio: np.ndarray, sr: int) -> float:
    """Estimate reference BPM from drum stem using librosa."""
    import librosa
    # Mono, native sample rate
    if drums_audio.ndim > 1:
        mono = drums_audio.mean(axis=1)
    else:
        mono = drums_audio
    mono = mono.astype(np.float32)

    # librosa beat tracker
    tempo, _ = librosa.beat.beat_track(y=mono, sr=sr, units='time')
    if hasattr(tempo, '__len__'):
        # librosa >= 0.10 returns array
        return float(tempo[0]) if len(tempo) > 0 else 0.0
    return float(tempo)


def estimate_bpm_pipeline(mix_audio: np.ndarray, orig_sr: int) -> float:
    """Run our full pipeline on mix audio and return estimated BPM.

    Uses BTrack (production path) with raw audio feed. Falls back to
    ACF if BTrack is unavailable.
    """
    mix_48k = resample_to_48k(mix_audio, orig_sr)
    if mix_48k.ndim > 1:
        mono = mix_48k.mean(axis=1).astype(np.float32)
    else:
        mono = mix_48k.astype(np.float32)

    # Prefer BTrack (matches production)
    try:
        from flame_sheep_audio.tempo_btrack import BTrackTempoTracker
        tracker = BTrackTempoTracker(hop_size=HOP_SIZE, sample_rate=SAMPLE_RATE)
        for pos in range(0, len(mono) - HOP_SIZE, HOP_SIZE):
            tracker.feed_audio(mono[pos:pos + HOP_SIZE])
        return tracker.effective_bpm
    except ImportError:
        pass

    # ACF fallback
    engine = SpectrumEngine()
    stability = MagnitudeStability()
    detector = FluxBeatDetector(sharpness=True, stability=stability)
    density = OnsetDensityTracker()
    hop_dur = HOP_SIZE / SAMPLE_RATE
    tracker = AutocorrelationTempoTracker(hop_duration=hop_dur)

    from flame_sheep_audio.hpss import ComplexSpectralDiffTransform
    from flame_sheep_audio._bands import A_WEIGHTS, a_weight_curve
    csd = ComplexSpectralDiffTransform()

    now = 0.0
    for pos in range(0, len(mono) - HOP_SIZE, HOP_SIZE):
        chunk = mono[pos:pos + HOP_SIZE]
        frame = engine.push_hop(chunk)
        csd_frame = csd(frame)
        stability.update(frame.magnitude)
        events = detector.detect(csd_frame)
        now += hop_dur
        for e in events:
            density.process_onset(e.kind, now)
        density.update(now)
        a_w = A_WEIGHTS if len(csd_frame.flux) == len(A_WEIGHTS) else a_weight_curve(np.arange(len(csd_frame.flux)))
        onset_str = float(np.dot(csd_frame.flux, a_w))
        tracker.feed(onset_str, onset_density=sum(density.densities.values()))

    return tracker.effective_bpm


def bpm_exact(estimated: float, reference: float, tolerance_pct: float = 4.0) -> bool:
    """Within ±tolerance_pct of reference."""
    if reference <= 0:
        return False
    return abs(estimated - reference) / reference * 100 < tolerance_pct


def bpm_octave(estimated: float, reference: float, tolerance_pct: float = 4.0) -> bool:
    """Within ±tolerance_pct of reference, 2x, or 0.5x."""
    if reference <= 0:
        return False
    for ratio in [0.5, 1.0, 2.0]:
        ref = reference * ratio
        if ref > 0 and abs(estimated - ref) / ref * 100 < tolerance_pct:
            return True
    return False


def evaluate_track(track):
    """Evaluate one MUSDB18 track. Returns result dict."""
    t0 = time.time()

    # Reference BPM from drum stem
    drums = track.targets['drums'].audio
    ref = reference_bpm(drums, track.rate)

    # Our estimate from full mix
    est = estimate_bpm_pipeline(track.audio, track.rate)

    elapsed = time.time() - t0
    exact = bpm_exact(est, ref)
    octave = bpm_octave(est, ref)

    return {
        'track': track.name,
        'reference': ref,
        'estimated': est,
        'exact': exact,
        'octave': octave,
        'error': abs(est - ref) if ref > 0 else float('inf'),
        'duration': track.audio.shape[0] / track.rate,
        'elapsed': elapsed,
    }


def print_report(results):
    n = len(results)
    if n == 0:
        print("No results.")
        return

    exact_count = sum(1 for r in results if r['exact'])
    octave_count = sum(1 for r in results if r['octave'])

    # Filter out tracks where reference failed
    valid = [r for r in results if r['reference'] > 0]
    n_valid = len(valid)

    print(f'\n{"=" * 72}')
    print(f'MUSDB18 Tempo Estimation ({n} tracks, {n_valid} with valid reference)')
    print(f'{"=" * 72}')

    print(f'\n  Exact accuracy (±4%):  {exact_count}/{n_valid} ({100*exact_count/n_valid:.0f}%)')
    print(f'  Octave accuracy (±4%): {octave_count}/{n_valid} ({100*octave_count/n_valid:.0f}%)')

    # Error stats (exact matches only)
    exact_errors = [r['error'] for r in valid if r['exact']]
    if exact_errors:
        print(f'\n  Exact-match error: mean={np.mean(exact_errors):.1f} BPM, '
              f'max={max(exact_errors):.1f} BPM')

    # Octave-match error
    octave_errors = []
    for r in valid:
        if r['octave']:
            best = min(abs(r['estimated'] - r['reference'] * ratio)
                       for ratio in [0.5, 1.0, 2.0])
            octave_errors.append(best)
    if octave_errors:
        print(f'  Octave-match error: mean={np.mean(octave_errors):.1f} BPM, '
              f'max={max(octave_errors):.1f} BPM')

    # Distribution
    refs = [r['reference'] for r in valid]
    ests = [r['estimated'] for r in valid]
    print(f'\n  Reference BPM range: {min(refs):.0f} - {max(refs):.0f}')
    print(f'  Estimated BPM range: {min(ests):.0f} - {max(ests):.0f}')

    # Default-BPM count (tracker gave up)
    default_count = sum(1 for r in valid if 119 < r['estimated'] < 121)
    print(f'  Returned default (120): {default_count}/{n_valid} '
          f'({100*default_count/n_valid:.0f}%)')

    # Worst tracks
    worst = sorted(valid, key=lambda r: -r['error'])[:10]
    print(f'\n  WORST 10 TRACKS (by absolute error)')
    print(f'  {"Track":<40} {"Ref":>6} {"Est":>6} {"Err":>6} {"Oct":>4}')
    print(f'  {"-"*62}')
    for r in worst:
        oct_mark = '✓' if r['octave'] else '✗'
        print(f'  {r["track"][:40]:<40} {r["reference"]:>6.1f} '
              f'{r["estimated"]:>6.1f} {r["error"]:>6.1f} {oct_mark:>4}')

    # Best tracks
    best = sorted(valid, key=lambda r: r['error'])[:10]
    print(f'\n  BEST 10 TRACKS (by absolute error)')
    print(f'  {"Track":<40} {"Ref":>6} {"Est":>6} {"Err":>6}')
    print(f'  {"-"*58}')
    for r in best:
        print(f'  {r["track"][:40]:<40} {r["reference"]:>6.1f} '
              f'{r["estimated"]:>6.1f} {r["error"]:>6.1f}')

    total_time = sum(r['elapsed'] for r in results)
    total_audio = sum(r['duration'] for r in results)
    print(f'\n  Processing time: {total_time:.0f}s for {total_audio/60:.0f}min of audio '
          f'({total_audio/total_time:.1f}x realtime)')

    print(f'{"=" * 72}')


def main():
    parser = argparse.ArgumentParser(description='Evaluate tempo estimation on MUSDB18')
    parser.add_argument('--root', required=True, help='MUSDB18 root directory')
    parser.add_argument('--limit', type=int, default=None, help='Limit to N tracks')
    parser.add_argument('--wav', action='store_true', help='Use WAV format')
    args = parser.parse_args()

    import musdb
    kwargs = {'root': args.root}
    if args.wav:
        kwargs['is_wav'] = True
    mus = musdb.DB(**kwargs)
    tracks = list(mus)
    if args.limit:
        tracks = tracks[:args.limit]

    print(f'Evaluating tempo on {len(tracks)} tracks...')

    results = []
    for i, track in enumerate(tracks):
        print(f'  [{i+1}/{len(tracks)}] {track.name}...', end='', flush=True)
        r = evaluate_track(track)
        exact_mark = '✓' if r['exact'] else ('~' if r['octave'] else '✗')
        print(f'  ref={r["reference"]:.0f} est={r["estimated"]:.0f} {exact_mark} '
              f'({r["elapsed"]:.1f}s)')
        results.append(r)

    print_report(results)


if __name__ == '__main__':
    main()
