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

from flame_sheep.audio import SAMPLE_RATE, FFT_SIZE

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
                        sharpness: bool = True) -> list[float]:
    """Collect onset timestamps from the detector."""
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
    track_name, track_path, track_rate, stem_audios, mix_audio, adaptive, sharpness = args

    row = {'track': track_name}
    duration = mix_audio.shape[0] / track_rate

    stem_times = {}
    for stem_name in ['drums', 'bass', 'vocals', 'other']:
        audio = stem_audios[stem_name]
        audio_48k = resample_to_48k(audio, orig_sr=track_rate)
        mono = audio_48k.mean(axis=1).astype(np.float32) if audio_48k.ndim > 1 else audio_48k.astype(np.float32)
        stem_times[stem_name] = collect_onset_times(mono, adaptive=adaptive, sharpness=sharpness)
        row[stem_name] = len(stem_times[stem_name])

    # Mix
    audio_48k = resample_to_48k(mix_audio, orig_sr=track_rate)
    mono = audio_48k.mean(axis=1).astype(np.float32) if audio_48k.ndim > 1 else audio_48k.astype(np.float32)
    row['mix'] = len(collect_onset_times(mono, adaptive=adaptive, sharpness=sharpness))

    row['duration'] = duration

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

    return row


def evaluate_tracks(tracks, adaptive: bool = False, sharpness: bool = True,
                    jobs: int = 1):
    """Run evaluation across all tracks, optionally in parallel."""
    # Prepare args — extract audio data before forking
    track_args = []
    for track in tracks:
        stem_audios = {
            name: track.targets[name].audio
            for name in ['drums', 'bass', 'vocals', 'other']
        }
        track_args.append((
            track.name, None, track.rate, stem_audios,
            track.audio, adaptive, sharpness,
        ))

    results = []
    if jobs == 1:
        for i, args in enumerate(track_args):
            print(f'  [{i+1}/{len(tracks)}] {args[0]}...', end='', flush=True)
            row = evaluate_single_track(args)
            print(f'  P={row["precision"]:.2f} TP={row["tp"]} FP={row["fp"]} (v={row["fp_vocal"]} o={row["fp_other"]})')
            results.append(row)
    else:
        print(f'  Running {len(tracks)} tracks across {jobs} workers...')
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(evaluate_single_track, a): a[0] for a in track_args}
            done = 0
            for future in as_completed(futures):
                done += 1
                row = future.result()
                name = row['track']
                print(f'  [{done}/{len(tracks)}] {name[:45]:45s}  P={row["precision"]:.2f} TP={row["tp"]} FP={row["fp"]}')
                results.append(row)

    # Sort by original track order for consistent output
    name_order = {a[0]: i for i, a in enumerate(track_args)}
    results.sort(key=lambda r: name_order[r['track']])
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

    print(f'\n  CLASSIFICATION')
    print(f'    TP (drum onsets):                {total_tp:6d}  ({total_tp/total_dur:.1f}/sec)')
    print(f'    FP (isolated non-drum onsets):   {total_fp:6d}  ({total_fp/total_dur:.1f}/sec)')
    print(f'      vocal:                         {total_fp_vocal:6d}  ({total_fp_vocal/total_dur:.1f}/sec)')
    print(f'      other:                         {total_fp_other:6d}  ({total_fp_other/total_dur:.1f}/sec)')
    print(f'    Precision:                        {precision:.3f}')

    # Per-stem raw event rates
    print(f'\n  RAW EVENT RATES (events/sec)')
    for stem in ['drums', 'bass', 'vocals', 'other', 'mix']:
        total_events = sum(r[stem] for r in results)
        rate = total_events / total_dur
        print(f'    {stem:8s}: {rate:5.2f}')

    # Per-track precision distribution
    precisions = [r['precision'] for r in results]
    print(f'\n  PRECISION DISTRIBUTION')
    print(f'    min={min(precisions):.3f}  p25={np.percentile(precisions,25):.3f}  '
          f'median={np.median(precisions):.3f}  p75={np.percentile(precisions,75):.3f}  '
          f'max={max(precisions):.3f}')

    # Worst tracks
    by_precision = sorted(results, key=lambda r: r['precision'])
    print(f'\n  WORST 5 TRACKS')
    for r in by_precision[:5]:
        print(f'    {r["track"][:45]:45s}  P={r["precision"]:.3f}  TP={r["tp"]:4d}  FP={r["fp"]:4d}  (v={r["fp_vocal"]} o={r["fp_other"]})')

    print(f'{"=" * 70}')


def main():
    parser = argparse.ArgumentParser(description='Evaluate rhythm pipeline on MUSDB18')
    parser.add_argument('--root', default=None, help='MUSDB18 root directory')
    parser.add_argument('--wav', action='store_true', help='Use WAV (HQ) format')
    parser.add_argument('--adaptive', action='store_true', help='Use adaptive bands')
    parser.add_argument('--no-sharpness', action='store_true', help='Disable attack sharpness filter')
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
    flags = []
    if args.adaptive:
        flags.append('adaptive')
    flags.append(f'sharpness={sharpness}')
    flags.append(f'jobs={args.jobs}')
    flag_str = ', '.join(flags)

    print(f'Evaluating {len(tracks)} tracks ({flag_str})...')

    results = evaluate_tracks(tracks, adaptive=args.adaptive, sharpness=sharpness,
                              jobs=args.jobs)
    print_report(results, flags=flag_str)


if __name__ == '__main__':
    main()
