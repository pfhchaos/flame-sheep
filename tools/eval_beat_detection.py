#!/usr/bin/env python3
"""Evaluate a beat detector against osu-derived ground truth.

Walks the osu corpus, derives ground-truth beat times from each
beatmap's [TimingPoints], runs the named detector on the audio, and
reports F1@{25,50,70}ms plus cemgil — per track and aggregated.

Caches per (track_id, detector_name, detector_version) so repeated
runs skip inference. Cache invalidates automatically when a detector's
content hash changes (e.g. after retraining).

Usage:
    python tools/eval_beat_detection.py \\
        --corpus ~/.cache/flame-sheep/corpus/osu \\
        --detector current_system \\
        --tracks 10

    # JSON results sidecar
    python tools/eval_beat_detection.py ... -o results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Make project imports work without installing.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flame_sheep.eval.detectors import BeatDetector, CurrentSystemDetector
from flame_sheep.eval.metrics import evaluate
from flame_sheep.eval.osu_parser import parse_osu_file


CACHE_DIR = Path.home() / 'datasets' / 'beat-eval-cache'


def _build_detector(name: str) -> BeatDetector:
    if name == 'current_system':
        return CurrentSystemDetector()
    # BeatNetDetector + BeatRNNDetector go here when ready.
    raise SystemExit(f'Unknown detector: {name!r}')


def _collect_tracks(corpus_dir: Path) -> list[tuple[Path, Path]]:
    """Return (osu_file, audio_file) pairs. Dedupe by audio path —
    different difficulties of the same map all reference the same
    audio.mp3, so we pick one .osu per directory."""
    tracks = []
    for sub in sorted(corpus_dir.iterdir()):
        if not sub.is_dir():
            continue
        osu_files = sorted(sub.glob('*.osu'))
        if not osu_files:
            continue
        # Parse the first one to find the audio file. Skip the dir if
        # the .osu fails to parse.
        try:
            m = parse_osu_file(osu_files[0])
        except (ValueError, OSError):
            continue
        audio = sub / m.audio_filename
        if not audio.exists():
            continue
        tracks.append((osu_files[0], audio))
    return tracks


def _load_audio(path: Path, target_sr: int = 48000
                ) -> tuple[np.ndarray, float]:
    """Returns (mono audio float32, duration_sec)."""
    import librosa
    audio, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return audio.astype(np.float32), len(audio) / sr


def _cache_path(track_id: str, detector_name: str,
                detector_version: str) -> Path:
    safe_track = track_id.replace('/', '_').replace(' ', '_')
    return CACHE_DIR / detector_name / detector_version / f'{safe_track}.npy'


def _cached_detect(detector: BeatDetector, track_id: str,
                   audio: np.ndarray, sr: int,
                   force: bool = False) -> np.ndarray:
    cache_p = _cache_path(track_id, detector.name, detector.version)
    if cache_p.exists() and not force:
        return np.load(cache_p)
    beats = detector.detect(audio, sr)
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_p, beats)
    return beats


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a beat detector against osu ground truth.')
    parser.add_argument(
        '--corpus', type=Path,
        default=Path.home() / '.cache' / 'flame-sheep' / 'corpus' / 'osu',
        help='Directory of osu beatmap sets (one subdir per map)')
    parser.add_argument(
        '--detector', default='current_system',
        choices=['current_system'],
        help='Detector to evaluate')
    parser.add_argument(
        '--tracks', type=int, default=10,
        help='Number of tracks to evaluate (default 10)')
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for track selection')
    parser.add_argument(
        '--force', action='store_true',
        help='Ignore detector cache')
    parser.add_argument(
        '-o', '--output', type=Path,
        help='JSON sidecar of per-track + aggregate results')
    parser.add_argument(
        '--list-cache', action='store_true',
        help='Print cache stats and exit')
    args = parser.parse_args()

    if args.list_cache:
        if not CACHE_DIR.exists():
            print(f'Cache dir does not exist: {CACHE_DIR}')
            return
        for det_dir in sorted(CACHE_DIR.iterdir()):
            for ver_dir in sorted(det_dir.iterdir()):
                files = list(ver_dir.glob('*.npy'))
                print(f'  {det_dir.name}/{ver_dir.name}: {len(files)} tracks')
        return

    detector = _build_detector(args.detector)
    print(f'Detector: {detector.name} (version {detector.version})')

    tracks = _collect_tracks(args.corpus)
    if not tracks:
        print(f'No usable tracks found in {args.corpus}', file=sys.stderr)
        sys.exit(1)
    print(f'Found {len(tracks)} usable tracks in {args.corpus}')

    rng = np.random.default_rng(args.seed)
    if len(tracks) > args.tracks:
        idx = rng.choice(len(tracks), size=args.tracks, replace=False)
        tracks = [tracks[i] for i in sorted(idx)]
    print(f'Evaluating {len(tracks)} tracks\n')

    per_track = []
    aggregate_pred = []
    aggregate_true = []

    for i, (osu_path, audio_path) in enumerate(tracks, 1):
        track_id = audio_path.parent.name + '/' + audio_path.stem
        print(f'[{i}/{len(tracks)}] {audio_path.parent.name}')

        try:
            audio, duration = _load_audio(audio_path)
        except Exception as e:
            print(f'  audio load failed: {e}')
            continue

        try:
            osu_map = parse_osu_file(osu_path)
            true_beats, _ = osu_map.beat_times(audio_duration_sec=duration)
        except (ValueError, OSError) as e:
            print(f'  osu parse failed: {e}')
            continue

        t0 = time.monotonic()
        pred_beats = _cached_detect(detector, track_id, audio, sr=48000,
                                    force=args.force)
        dt = time.monotonic() - t0

        m = evaluate(pred_beats, true_beats)
        m['track_id'] = track_id
        m['duration_sec'] = duration
        m['detect_sec'] = dt
        per_track.append(m)

        # Aggregate by offsetting and concatenating — for whole-corpus F1.
        offset = sum(t.get('duration_sec', 0) for t in per_track[:-1])
        aggregate_pred.append(np.asarray(pred_beats) + offset)
        aggregate_true.append(np.asarray(true_beats) + offset)

        print(f'  duration={duration:.1f}s  detect={dt:.2f}s  '
              f'pred={m["n_pred"]} true={m["n_true"]}')
        print(f'  F1: 25ms={m["f1_25ms"]:.3f}  '
              f'50ms={m["f1_50ms"]:.3f}  70ms={m["f1_70ms"]:.3f}  '
              f'cemgil={m["cemgil"]:.3f}')

    if not per_track:
        print('No tracks were successfully evaluated.', file=sys.stderr)
        sys.exit(2)

    # Aggregate: mean of per-track F1s (equal weight per track).
    print('\n=== Aggregate (mean over tracks) ===')
    for ms in (25, 50, 70):
        f1s = [t[f'f1_{ms}ms'] for t in per_track]
        precs = [t[f'precision_{ms}ms'] for t in per_track]
        recs = [t[f'recall_{ms}ms'] for t in per_track]
        print(f'  F1@{ms}ms: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}  '
              f'(precision={np.mean(precs):.3f}, recall={np.mean(recs):.3f})')
    cemgils = [t['cemgil'] for t in per_track]
    print(f'  cemgil:   {np.mean(cemgils):.3f} ± {np.std(cemgils):.3f}')

    # Aggregate: F1 computed on concatenated predictions/ground truth
    # (length-weighted). Useful comparison to per-track mean.
    pred_concat = np.concatenate(aggregate_pred) if aggregate_pred else np.array([])
    true_concat = np.concatenate(aggregate_true) if aggregate_true else np.array([])
    pooled = evaluate(pred_concat, true_concat)
    print('\n=== Aggregate (length-weighted, pooled over corpus) ===')
    for ms in (25, 50, 70):
        print(f'  F1@{ms}ms: {pooled[f"f1_{ms}ms"]:.3f}')
    print(f'  cemgil:   {pooled["cemgil"]:.3f}')

    if args.output:
        result = {
            'detector': detector.name,
            'detector_version': detector.version,
            'corpus': str(args.corpus),
            'n_tracks': len(per_track),
            'per_track': per_track,
            'aggregate_mean_f1_25ms': float(np.mean([t['f1_25ms'] for t in per_track])),
            'aggregate_mean_f1_50ms': float(np.mean([t['f1_50ms'] for t in per_track])),
            'aggregate_mean_f1_70ms': float(np.mean([t['f1_70ms'] for t in per_track])),
            'aggregate_mean_cemgil': float(np.mean(cemgils)),
            'pooled_f1_50ms': pooled['f1_50ms'],
        }
        args.output.write_text(json.dumps(result, indent=2))
        print(f'\nResults written to {args.output}')


if __name__ == '__main__':
    main()
