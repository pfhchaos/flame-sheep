#!/usr/bin/env python3
"""
Evaluate rhythm detection pipeline against MUSDB18 separated stems.

Uses stem separation as implicit ground truth:
  - drums onsets = true rhythmic events (should trigger visuals)
  - vocals/other onsets that don't coincide with drum hits = false positives

Usage:
  python -m tests.eval_musdb --root /path/to/musdb18
  python -m tests.eval_musdb --root /path/to/musdb18 --no-sharpness
  python -m tests.eval_musdb --root /path/to/musdb18 --limit 20 --jobs 4
"""

import argparse
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

import musdb

sys.path.insert(0, '.')

from flame_sheep_audio import SAMPLE_RATE, FFT_SIZE, HOP_SIZE

# Coincidence window: onsets within this of a drum hit are ignorable
COINCIDENCE_MS = 250


def resample_to_48k(audio: np.ndarray, orig_sr: int = 44100) -> np.ndarray:
    if orig_sr == SAMPLE_RATE:
        return audio
    ratio = SAMPLE_RATE / orig_sr
    n_out = int(len(audio) * ratio)
    indices = np.clip((np.arange(n_out) / ratio).astype(int), 0, len(audio) - 1)
    return audio[indices]


def collect_onset_times(pcm_mono: np.ndarray, adaptive: bool = False,
                        sharpness: bool = True,
                        fine_hop: bool = False,
                        ) -> list[float]:
    """Collect onset timestamps from the detector.

    fine_hop=True uses 512-sample hops via push_hop (new path).
    fine_hop=False uses 2048-sample chunks via process (old path).
    """
    if fine_hop:
        return _collect_fine_hop(pcm_mono, adaptive=adaptive, sharpness=sharpness)

    from tests.conftest import make_processor

    proc = make_processor(adaptive=adaptive, sharpness=sharpness)

    silence = np.zeros(FFT_SIZE, dtype=np.float32)
    for _ in range(15):
        proc.feed(silence)
        proc.process()

    times = []
    pos = 0
    while pos < len(pcm_mono):
        chunk = pcm_mono[pos:pos + FFT_SIZE]
        if len(chunk) < FFT_SIZE:
            chunk = np.pad(chunk, (0, FFT_SIZE - len(chunk)))
        proc.feed(chunk)
        ts = pos / SAMPLE_RATE
        for e in proc.process():
            times.append(ts)
        pos += FFT_SIZE

    return times


def _collect_fine_hop(pcm_mono: np.ndarray, adaptive: bool = False,
                      sharpness: bool = True,
                      ) -> list[float]:
    """512-hop path using push_hop for finer temporal resolution."""
    from flame_sheep_audio._spectrum import SpectrumEngine
    from flame_sheep_audio.beat_detector import FluxBeatDetector
    from flame_sheep_audio.energy import EnergyAnalyzer
    from flame_sheep_audio.stability import MagnitudeStability

    engine = SpectrumEngine()
    stability = MagnitudeStability()
    detector = FluxBeatDetector(adaptive=adaptive, sharpness=sharpness,
                                stability=stability)
    energy = EnergyAnalyzer()

    silence = np.zeros(HOP_SIZE, dtype=np.float32)
    for _ in range(40):
        frame = engine.push_hop(silence)
        stability.update(frame.magnitude)
        energy.update(frame.magnitude, stability=stability)
        detector.detect(frame)

    times = []
    pos = 0
    while pos < len(pcm_mono):
        chunk = pcm_mono[pos:pos + HOP_SIZE]
        if len(chunk) < HOP_SIZE:
            chunk = np.pad(chunk, (0, HOP_SIZE - len(chunk)))
        frame = engine.push_hop(chunk)
        stability.update(frame.magnitude)
        energy.update(frame.magnitude, frame.flux, stability=stability)
        ts = pos / SAMPLE_RATE
        for e in detector.detect(frame):
            times.append(ts)
        pos += HOP_SIZE

    return times


def count_isolated_fps(fp_times: list[float], tp_times: list[float]) -> int:
    """Count FP onsets that have no TP onset within COINCIDENCE_MS."""
    if not tp_times:
        return len(fp_times)
    if not fp_times:
        return 0
    tp = np.array(sorted(tp_times))
    window = COINCIDENCE_MS / 1000.0
    isolated = 0
    for t in fp_times:
        idx = np.searchsorted(tp, t)
        near = False
        if idx < len(tp) and tp[idx] - t <= window:
            near = True
        if not near and idx > 0 and t - tp[idx - 1] <= window:
            near = True
        if not near:
            isolated += 1
    return isolated


def evaluate_single_track(args):
    """Evaluate one track. Designed for use with ProcessPoolExecutor."""
    track_name, track_path, track_rate, stem_audios, mix_audio, adaptive, sharpness, fine_hop = args

    row = {'track': track_name}
    duration = mix_audio.shape[0] / track_rate

    stem_times = {}
    for stem_name in ['drums', 'bass', 'vocals', 'other']:
        audio = stem_audios[stem_name]
        audio_48k = resample_to_48k(audio, orig_sr=track_rate)
        mono = audio_48k.mean(axis=1).astype(np.float32) if audio_48k.ndim > 1 else audio_48k.astype(np.float32)
        stem_times[stem_name] = collect_onset_times(mono, adaptive=adaptive, sharpness=sharpness, fine_hop=fine_hop)
        row[stem_name] = len(stem_times[stem_name])

    # Mix
    audio_48k = resample_to_48k(mix_audio, orig_sr=track_rate)
    mono = audio_48k.mean(axis=1).astype(np.float32) if audio_48k.ndim > 1 else audio_48k.astype(np.float32)
    row['mix'] = len(collect_onset_times(mono, adaptive=adaptive, sharpness=sharpness, fine_hop=fine_hop))

    row['duration'] = duration

    # --- Legacy metric: per-stem TP/FP ---
    drum_times = stem_times['drums']
    fp_vocal = count_isolated_fps(stem_times['vocals'], drum_times)
    fp_other = count_isolated_fps(stem_times['other'], drum_times)

    tp = len(drum_times)
    fp = fp_vocal + fp_other
    total = tp + fp

    row['tp'] = tp
    row['fp'] = fp
    row['fp_vocal'] = fp_vocal
    row['fp_other'] = fp_other
    row['precision'] = tp / total if total > 0 else 0

    # --- Mix-vs-drums metric: detect on mix, check against drum onsets ---
    mix_48k = resample_to_48k(mix_audio, orig_sr=track_rate)
    mix_mono = mix_48k.mean(axis=1).astype(np.float32) if mix_48k.ndim > 1 else mix_48k.astype(np.float32)
    mix_times = collect_onset_times(mix_mono, adaptive=adaptive, sharpness=sharpness,
                                    fine_hop=fine_hop)

    window = COINCIDENCE_MS / 1000.0
    drum_arr = np.array(sorted(drum_times)) if drum_times else np.array([])
    mix_tp = 0
    mix_fp = 0
    for t in mix_times:
        if len(drum_arr) == 0:
            mix_fp += 1
            continue
        idx = np.searchsorted(drum_arr, t)
        near = False
        if idx < len(drum_arr) and drum_arr[idx] - t <= window:
            near = True
        if not near and idx > 0 and t - drum_arr[idx - 1] <= window:
            near = True
        if near:
            mix_tp += 1
        else:
            mix_fp += 1

    mix_total = mix_tp + mix_fp
    row['mix_tp'] = mix_tp
    row['mix_fp'] = mix_fp
    row['mix_precision'] = mix_tp / mix_total if mix_total > 0 else 0
    row['mix_total'] = mix_total

    return row


def evaluate_tracks(tracks, adaptive: bool = False, sharpness: bool = True,
                    fine_hop: bool = False,
                    jobs: int = 1):
    """Run evaluation across all tracks.

    Loads one track at a time to avoid OOM — each track's stems are ~0.5-1GB
    in float64, so pre-loading all 150 would require ~100GB.
    """
    results = []
    for i, track in enumerate(tracks):
        print(f'  [{i+1}/{len(tracks)}] {track.name}...', end='', flush=True)
        # Load stems on demand, process, then let GC free them
        stem_audios = {
            name: track.targets[name].audio
            for name in ['drums', 'bass', 'vocals', 'other']
        }
        args = (track.name, None, track.rate, stem_audios,
                track.audio, adaptive, sharpness, fine_hop)
        row = evaluate_single_track(args)
        print(f'  mixP={row["mix_precision"]:.2f} stemP={row["precision"]:.2f} mix={row["mix_tp"]}+{row["mix_fp"]}  stem={row["tp"]}+{row["fp"]}')
        results.append(row)
        # Explicitly free stem audio to keep RSS in check
        del stem_audios, args

    return results


def print_report(results, flags: str):
    """Print classification metrics report."""
    n = len(results)

    total_tp = sum(r['tp'] for r in results)
    total_fp = sum(r['fp'] for r in results)
    total_fp_vocal = sum(r['fp_vocal'] for r in results)
    total_fp_other = sum(r['fp_other'] for r in results)
    total = total_tp + total_fp

    precision = total_tp / total if total > 0 else 0

    total_dur = sum(r['duration'] for r in results)

    print(f'\n{"=" * 70}')
    print(f'MUSDB18 Onset Classification ({n} tracks)')
    print(f'  Config: {flags}')
    print(f'  Coincidence window: {COINCIDENCE_MS}ms')
    print(f'{"=" * 70}')

    # Mix-vs-drums metric
    total_mix_tp = sum(r.get('mix_tp', 0) for r in results)
    total_mix_fp = sum(r.get('mix_fp', 0) for r in results)
    total_mix = total_mix_tp + total_mix_fp
    mix_precision = total_mix_tp / total_mix if total_mix > 0 else 0

    print(f'\n  MIX VS DRUMS (does mix detection coincide with a drum hit?)')
    print(f'    Mix onsets near drums:           {total_mix_tp:6d}  ({total_mix_tp/total_dur:.1f}/sec)')
    print(f'    Mix onsets NOT near drums:       {total_mix_fp:6d}  ({total_mix_fp/total_dur:.1f}/sec)')
    print(f'    Mix precision:                    {mix_precision:.3f}')

    print(f'\n  PER-STEM CLASSIFICATION (legacy)')
    print(f'    TP (drum onsets):                {total_tp:6d}  ({total_tp/total_dur:.1f}/sec)')
    print(f'    FP (isolated non-drum onsets):   {total_fp:6d}  ({total_fp/total_dur:.1f}/sec)')
    print(f'      vocal:                         {total_fp_vocal:6d}  ({total_fp_vocal/total_dur:.1f}/sec)')
    print(f'      other:                         {total_fp_other:6d}  ({total_fp_other/total_dur:.1f}/sec)')
    print(f'    Stem precision:                   {precision:.3f}')

    # Per-stem raw event rates
    print(f'\n  RAW EVENT RATES (events/sec)')
    for stem in ['drums', 'bass', 'vocals', 'other', 'mix']:
        total_events = sum(r[stem] for r in results)
        rate = total_events / total_dur
        print(f'    {stem:8s}: {rate:5.2f}')

    # Per-track precision distribution (mix metric)
    mix_precs = [r.get('mix_precision', r['precision']) for r in results]
    print(f'\n  MIX PRECISION DISTRIBUTION')
    print(f'    min={min(mix_precs):.3f}  p25={np.percentile(mix_precs,25):.3f}  '
          f'median={np.median(mix_precs):.3f}  p75={np.percentile(mix_precs,75):.3f}  '
          f'max={max(mix_precs):.3f}')

    # Worst tracks by mix precision
    by_mix_prec = sorted(results, key=lambda r: r.get('mix_precision', r['precision']))
    print(f'\n  WORST 5 TRACKS (mix precision)')
    for r in by_mix_prec[:5]:
        mp = r.get('mix_precision', 0)
        print(f'    {r["track"][:45]:45s}  mixP={mp:.3f}  stemP={r["precision"]:.3f}  '
              f'mix_tp={r.get("mix_tp",0):4d}  mix_fp={r.get("mix_fp",0):4d}')

    print(f'{"=" * 70}')


def main():
    parser = argparse.ArgumentParser(description='Evaluate rhythm pipeline on MUSDB18')
    parser.add_argument('--root', default=None, help='MUSDB18 root directory')
    parser.add_argument('--wav', action='store_true', help='Use WAV (HQ) format')
    parser.add_argument('--adaptive', action='store_true', help='Use adaptive bands')
    parser.add_argument('--no-sharpness', action='store_true', help='Disable attack sharpness filter')
    parser.add_argument('--fine-hop', action='store_true', help='Use 512-sample hop (new path) instead of 2048')
    parser.add_argument('--limit', type=int, default=None, help='Limit to N tracks')
    parser.add_argument('--jobs', '-j', type=int, default=1, help='Parallel workers (default 1)')
    args = parser.parse_args()

    kwargs = {}
    if args.root:
        kwargs['root'] = args.root
        if args.wav:
            kwargs['is_wav'] = True
    else:
        kwargs['download'] = True

    mus = musdb.DB(**kwargs)
    tracks = list(mus)
    if args.limit:
        tracks = tracks[:args.limit]

    sharpness = not args.no_sharpness
    fine_hop = args.fine_hop
    flags = []
    if args.adaptive:
        flags.append('adaptive')
    flags.append(f'sharpness={sharpness}')
    flags.append(f'hop={"512" if fine_hop else "2048"}')
    flags.append(f'jobs={args.jobs}')
    flag_str = ', '.join(flags)

    print(f'Evaluating {len(tracks)} tracks ({flag_str})...')

    results = evaluate_tracks(tracks, adaptive=args.adaptive, sharpness=sharpness,
                              fine_hop=fine_hop, jobs=args.jobs)
    print_report(results, flags=flag_str)


if __name__ == '__main__':
    main()
