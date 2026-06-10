#!/usr/bin/env python3
"""Evaluate a beat (or downbeat) detector against corpus ground truth.

Supports two corpora:
  * osu  — beat times derived from each beatmap's [TimingPoints].
  * gtzan — Marchand/Fresnel/Peeters 2015 hand-annotated beats +
            downbeats, layout under `~/datasets/GTZAN/`
            (audio + annotations canonicalized at install time).

Reports F1@{25,50,70}ms plus cemgil — per track and aggregated (both
mean-over-tracks and length-weighted pooled).

Caches per (track_id, detector_name, detector_version, kind) so
repeated runs skip inference. Cache invalidates when a detector's
content hash changes.

Usage:
    # Beat F1 on osu (default)
    python tools/eval_beat_detection.py --detector current_system

    # Beat F1 on gtzan
    python tools/eval_beat_detection.py \\
        --corpus-kind gtzan --detector beatnet

    # Downbeat F1 on gtzan
    python tools/eval_beat_detection.py \\
        --corpus-kind gtzan --kind downbeat --detector beatnet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

# Make project imports work without installing.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flame_sheep.eval.detectors import (
    BeatDetector, CurrentSystemDetector, BeatRNNDetector, BeatNetDetector,
    MultiDepthBeatRNNDetector, MadmomDetector, StreamingBeatNetDetector,
    SlidingBeatNetDetector, LiteBeatNetDetector,
)
from flame_sheep.eval.metrics import evaluate
from flame_sheep.eval.osu_parser import parse_osu_file
from flame_sheep.eval.gtzan_parser import iter_tracks as iter_gtzan_tracks


CACHE_DIR = Path.home() / 'datasets' / 'beat-eval-cache'

_DEFAULT_OSU_ROOT = Path.home() / '.cache' / 'flame-sheep' / 'corpus' / 'osu'
_DEFAULT_GTZAN_ROOT = Path.home() / 'datasets' / 'GTZAN'


# A track handle is corpus-agnostic from the main-loop's POV: the loop
# only needs (track_id, audio_path, callable→(beats, downbeats|None)).
# `get_ground_truth` accepts audio_duration_sec because osu beat
# derivation needs it; gtzan ignores it.
@dataclass
class TrackHandle:
    track_id: str
    audio_path: Path
    get_ground_truth: Callable[[float], tuple[np.ndarray, np.ndarray | None]]


def _build_detector(name: str, **kwargs) -> BeatDetector:
    if name == 'current_system':
        return CurrentSystemDetector()
    if name == 'beat_rnn':
        weights = kwargs.get('weights')
        if not weights:
            raise SystemExit('beat_rnn detector requires --weights PATH')
        return BeatRNNDetector(weights)
    if name == 'beat_rnn_multidepth':
        weights = kwargs.get('weights')
        if not weights:
            raise SystemExit('beat_rnn_multidepth detector requires --weights PATH')
        return MultiDepthBeatRNNDetector(weights)
    if name == 'beatnet':
        return BeatNetDetector(model_index=kwargs.get('beatnet_model', 1))
    if name == 'beatnet_streaming':
        return StreamingBeatNetDetector(model_index=kwargs.get('beatnet_model', 1))
    if name == 'beatnet_sliding':
        return SlidingBeatNetDetector(model_index=kwargs.get('beatnet_model', 1))
    if name == 'beatnet_lite':
        return LiteBeatNetDetector(model_index=kwargs.get('beatnet_model', 1))
    if name == 'madmom':
        return MadmomDetector()
    raise SystemExit(f'Unknown detector: {name!r}')


def _collect_osu(corpus_dir: Path) -> list[TrackHandle]:
    """Walk osu beatmap dirs. One handle per directory (different
    difficulties share audio)."""
    handles: list[TrackHandle] = []
    for sub in sorted(corpus_dir.iterdir()):
        if not sub.is_dir():
            continue
        osu_files = sorted(sub.glob('*.osu'))
        if not osu_files:
            continue
        try:
            osu_map = parse_osu_file(osu_files[0])
        except (ValueError, OSError):
            continue
        audio = sub / osu_map.audio_filename
        if not audio.exists():
            continue

        def _resolver(duration: float, m=osu_map):
            beats, downbeats = m.beat_times(audio_duration_sec=duration)
            # osu downbeat annotations are mapper-derived; the handoff
            # treats them as untrusted but still surfaces them so a
            # caller that wants them can opt in.
            return beats, downbeats

        handles.append(TrackHandle(
            track_id=sub.name + '/' + audio.stem,
            audio_path=audio,
            get_ground_truth=_resolver,
        ))
    return handles


def _collect_gtzan(corpus_dir: Path) -> list[TrackHandle]:
    handles: list[TrackHandle] = []
    for track in iter_gtzan_tracks(corpus_dir):
        beats = track.beat_times
        downbeats = track.downbeat_times

        def _resolver(_duration: float, b=beats, d=downbeats):
            return b, d

        handles.append(TrackHandle(
            track_id=track.track_id,
            audio_path=track.audio_path,
            get_ground_truth=_resolver,
        ))
    return handles


def _collect_tracks(corpus_kind: str, corpus_dir: Path) -> list[TrackHandle]:
    if corpus_kind == 'osu':
        return _collect_osu(corpus_dir)
    if corpus_kind == 'gtzan':
        return _collect_gtzan(corpus_dir)
    raise SystemExit(f'Unknown corpus kind: {corpus_kind!r}')


def _load_audio(path: Path, target_sr: int = 48000
                ) -> tuple[np.ndarray, float]:
    import librosa
    audio, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return audio.astype(np.float32), len(audio) / sr


def _cache_path(track_id: str, detector_name: str,
                detector_version: str, kind: str) -> Path:
    safe_track = track_id.replace('/', '_').replace(' ', '_')
    # Beat cache stays at the legacy `<track>.npy` path so existing
    # caches from prior runs remain valid. Downbeat goes alongside.
    suffix = '.npy' if kind == 'beat' else f'.{kind}.npy'
    return CACHE_DIR / detector_name / detector_version / f'{safe_track}{suffix}'


def _timing_sidecar_path(cache_p: Path) -> Path:
    """Sidecar JSON with last_timing dict for a cached prediction.

    Sidecar pattern (vs single combined file) keeps prediction npy
    backward-compatible — old cache entries that lack timing look
    like a miss and force regeneration, which is exactly what §7's
    'don't reuse cached predictions without their timing' demands.
    """
    return cache_p.with_suffix(cache_p.suffix + '.timing.json')


def _cached_detect(detector: BeatDetector, track_id: str,
                   audio: np.ndarray, sr: int, kind: str,
                   force: bool = False
                   ) -> tuple[np.ndarray | None, dict | None]:
    """Run detector.detect or detect_downbeats, cached. Returns
    (predictions, timing_dict). Either may be None: predictions when
    the detector doesn't support the kind, timing when only the
    legacy cache exists (caller decides what to do)."""
    import json
    cache_p = _cache_path(track_id, detector.name, detector.version, kind)
    timing_p = _timing_sidecar_path(cache_p)

    if cache_p.exists() and timing_p.exists() and not force:
        try:
            result = np.load(cache_p)
            timing = json.loads(timing_p.read_text())
            return result, timing
        except (OSError, ValueError, json.JSONDecodeError):
            # Corrupt sidecar — fall through and recompute.
            pass

    if kind == 'beat':
        result = detector.detect(audio, sr)
    elif kind == 'downbeat':
        result = detector.detect_downbeats(audio, sr)
        if result is None:
            return None, None
    else:
        raise SystemExit(f'Unknown kind: {kind!r}')

    timing = dict(detector.last_timing)
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_p, result)
    timing_p.write_text(json.dumps(timing))
    return result, timing


def _print_aggregate(per_track: list[dict], pooled: dict, label: str) -> None:
    """Emit the two aggregate sections the scorecard parser keys on.

    The label distinguishes beat vs downbeat aggregates in stdout but
    the section headers stay exactly as the registry's regex expects
    so the same parser handles both kinds.
    """
    print(f'\n=== {label} Aggregate (mean over tracks) ===')
    for ms in (25, 50, 70):
        f1s = [t[f'f1_{ms}ms'] for t in per_track]
        precs = [t[f'precision_{ms}ms'] for t in per_track]
        recs = [t[f'recall_{ms}ms'] for t in per_track]
        print(f'  F1@{ms}ms: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}  '
              f'(precision={np.mean(precs):.3f}, recall={np.mean(recs):.3f})')
    cemgils = [t['cemgil'] for t in per_track]
    print(f'  cemgil:   {np.mean(cemgils):.3f} ± {np.std(cemgils):.3f}')

    print(f'\n=== {label} Aggregate (length-weighted, pooled over corpus) ===')
    for ms in (25, 50, 70):
        print(f'  F1@{ms}ms: {pooled[f"f1_{ms}ms"]:.3f}')
    print(f'  cemgil:   {pooled["cemgil"]:.3f}')


def _print_latency(per_track: list[dict], label: str) -> None:
    """Latency table — same grep semantics as the F1 tables so the
    registry parser can pick out latency.* metrics.

    Per-track values: `detect_sec` (detector self-reported total) +
    `timing` dict (component breakdown). `realtime_factor` derived as
    detect_sec / duration_sec — sub-1.0 means the detector is faster
    than realtime on that track.
    """
    totals = np.asarray([t['detect_sec'] for t in per_track], dtype=np.float64)
    durations = np.asarray([t['duration_sec'] for t in per_track], dtype=np.float64)
    rtf = totals / np.clip(durations, 1e-6, None)

    print(f'\n=== {label} Latency (per-track, total) ===')
    print(f'  total_sec:         mean={float(np.mean(totals)):.3f}  '
          f'p50={float(np.percentile(totals, 50)):.3f}  '
          f'p95={float(np.percentile(totals, 95)):.3f}  '
          f'p99={float(np.percentile(totals, 99)):.3f}')
    print(f'  realtime_factor:   mean={float(np.mean(rtf)):.3f}  '
          f'p95={float(np.percentile(rtf, 95)):.3f}')

    # Per-component aggregates — union of keys across tracks (some
    # detectors only emit 'total', others emit decomposed components).
    comp_keys: set[str] = set()
    for t in per_track:
        comp_keys.update(t.get('timing', {}).keys())
    comp_keys.discard('total')  # already reported above

    if comp_keys:
        print(f'\n=== {label} Latency (per-track, components) ===')
        for k in sorted(comp_keys):
            vals = [t.get('timing', {}).get(k) for t in per_track]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            print(f'  {k}_sec: mean={float(np.mean(arr)):.4f}  '
                  f'p95={float(np.percentile(arr, 95)):.4f}  '
                  f'n={len(vals)}')


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a beat or downbeat detector against corpus ground truth.')
    parser.add_argument(
        '--corpus-kind', default='osu', choices=['osu', 'gtzan'],
        help='Which corpus format to use for ground truth (default osu)')
    parser.add_argument(
        '--corpus', type=Path, default=None,
        help='Corpus root directory (default depends on --corpus-kind: '
             f'osu={_DEFAULT_OSU_ROOT}, gtzan={_DEFAULT_GTZAN_ROOT})')
    parser.add_argument(
        '--kind', default='beat', choices=['beat', 'downbeat'],
        help='Whether to evaluate beat F1 or downbeat F1 (default beat)')
    parser.add_argument(
        '--detector', default='current_system',
        choices=['current_system', 'beat_rnn', 'beat_rnn_multidepth',
                 'beatnet', 'beatnet_streaming', 'beatnet_sliding',
                 'beatnet_lite', 'madmom'],
        help='Detector to evaluate')
    parser.add_argument(
        '--weights', type=Path, default=None,
        help='Path to beat_rnn checkpoint .npz (required for --detector beat_rnn*)')
    parser.add_argument(
        '--beatnet-model', type=int, default=1,
        help='BeatNet model index (1=GTZAN, 2=Ballroom, 3=Rock; default 1)')
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
                print(f'  {det_dir.name}/{ver_dir.name}: {len(files)} files')
        return

    if args.corpus is None:
        args.corpus = (_DEFAULT_OSU_ROOT if args.corpus_kind == 'osu'
                       else _DEFAULT_GTZAN_ROOT)

    detector = _build_detector(args.detector, weights=args.weights,
                                beatnet_model=args.beatnet_model)
    print(f'Detector: {detector.name} (version {detector.version})')
    print(f'Corpus:   {args.corpus_kind} @ {args.corpus}')
    print(f'Kind:     {args.kind}')

    handles = _collect_tracks(args.corpus_kind, args.corpus)
    if not handles:
        print(f'No usable tracks found in {args.corpus}', file=sys.stderr)
        sys.exit(1)
    print(f'Found {len(handles)} usable tracks')

    rng = np.random.default_rng(args.seed)
    if len(handles) > args.tracks:
        idx = rng.choice(len(handles), size=args.tracks, replace=False)
        handles = [handles[i] for i in sorted(idx)]
    print(f'Evaluating {len(handles)} tracks\n')

    per_track: list[dict] = []
    aggregate_pred: list[np.ndarray] = []
    aggregate_true: list[np.ndarray] = []
    skipped_no_downbeat_support = 0
    skipped_no_downbeat_truth = 0

    for i, h in enumerate(handles, 1):
        print(f'[{i}/{len(handles)}] {h.track_id}')
        try:
            audio, duration = _load_audio(h.audio_path)
        except Exception as e:
            print(f'  audio load failed: {e}')
            continue

        try:
            true_beats, true_downbeats = h.get_ground_truth(duration)
        except (ValueError, OSError) as e:
            print(f'  ground-truth resolve failed: {e}')
            continue

        if args.kind == 'downbeat':
            # None = corpus never has downbeat annotations for this track
            # type. Empty = this particular track shipped without bar
            # positions (jazz subset of gtzan). Treat both as skip rather
            # than scoring an empty ground truth (which would zero out
            # precision against any predictions).
            if true_downbeats is None or len(true_downbeats) == 0:
                skipped_no_downbeat_truth += 1
                print('  skipped: no downbeat ground truth')
                continue
            true_arr = np.asarray(true_downbeats)
        else:
            true_arr = np.asarray(true_beats)

        t0 = time.monotonic()
        pred_arr, timing = _cached_detect(detector, h.track_id, audio, sr=48000,
                                            kind=args.kind, force=args.force)
        dt = time.monotonic() - t0
        if pred_arr is None:
            skipped_no_downbeat_support += 1
            print(f'  skipped: {detector.name} does not produce {args.kind}s')
            continue

        m = evaluate(pred_arr, true_arr)
        m['track_id'] = h.track_id
        m['duration_sec'] = duration
        # Detector's own measurement: 'total' from last_timing. This
        # excludes cache + harness overhead and is what §7 reports.
        # Fall back to dt only for legacy caches that lack timing.
        m['detect_sec'] = float(timing.get('total', dt)) if timing else dt
        m['timing'] = timing or {}
        per_track.append(m)

        offset = sum(t.get('duration_sec', 0) for t in per_track[:-1])
        aggregate_pred.append(np.asarray(pred_arr) + offset)
        aggregate_true.append(true_arr + offset)

        print(f'  duration={duration:.1f}s  detect={dt:.2f}s  '
              f'pred={m["n_pred"]} true={m["n_true"]}')
        print(f'  F1: 25ms={m["f1_25ms"]:.3f}  '
              f'50ms={m["f1_50ms"]:.3f}  70ms={m["f1_70ms"]:.3f}  '
              f'cemgil={m["cemgil"]:.3f}')

    if not per_track:
        if skipped_no_downbeat_support == len(handles):
            print(f'\nDetector {detector.name!r} does not support '
                  f'{args.kind} detection — nothing to score.',
                  file=sys.stderr)
            sys.exit(3)
        if skipped_no_downbeat_truth == len(handles):
            print(f'\nCorpus has no {args.kind} ground truth — '
                  f'nothing to score.', file=sys.stderr)
            sys.exit(3)
        print('No tracks were successfully evaluated.', file=sys.stderr)
        sys.exit(2)

    pred_concat = np.concatenate(aggregate_pred) if aggregate_pred else np.array([])
    true_concat = np.concatenate(aggregate_true) if aggregate_true else np.array([])
    pooled = evaluate(pred_concat, true_concat)
    label = 'Downbeat' if args.kind == 'downbeat' else 'Beat'
    _print_aggregate(per_track, pooled, label=label)
    _print_latency(per_track, label=label)

    if args.output:
        result = {
            'detector': detector.name,
            'detector_version': detector.version,
            'corpus_kind': args.corpus_kind,
            'corpus': str(args.corpus),
            'kind': args.kind,
            'n_tracks': len(per_track),
            'per_track': per_track,
            'aggregate_mean_f1_25ms': float(np.mean([t['f1_25ms'] for t in per_track])),
            'aggregate_mean_f1_50ms': float(np.mean([t['f1_50ms'] for t in per_track])),
            'aggregate_mean_f1_70ms': float(np.mean([t['f1_70ms'] for t in per_track])),
            'aggregate_mean_cemgil': float(np.mean([t['cemgil'] for t in per_track])),
            'pooled_f1_50ms': pooled['f1_50ms'],
        }
        args.output.write_text(json.dumps(result, indent=2))
        print(f'\nResults written to {args.output}')


if __name__ == '__main__':
    main()
