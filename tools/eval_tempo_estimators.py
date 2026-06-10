#!/usr/bin/env python3
"""Evaluate a tempo estimator against GTZAN-rhythm ground truth.

Mirrors `tools/eval_beat_detection.py` in shape — per-track + aggregate
metrics, parseable output for the unified scorecard, on-disk cache of
predictions per (estimator, version). The eval is on the
swappable-interface side (`flame_sheep.eval.tempo_estimators`), so
swapping estimators is `--estimator acf` ↔ `--estimator madmom_comb` etc.

Metrics:
  * Exact accuracy (±4%)
  * OE1 — octave-tolerant accuracy (within ±4% of truth, ×2, or ×0.5)
  * Mean absolute BPM error (over correct-octave predictions)
  * Latency aggregates — total + per-component (mean, p50, p95, p99)

Usage:
    python tools/eval_tempo_estimators.py --estimator acf
    python tools/eval_tempo_estimators.py --estimator madmom_comb --tracks 50
    python tools/eval_tempo_estimators.py --estimator madmom_dbn -o out.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flame_sheep.eval.tempo_estimators import (
    TempoEstimator, AcfTempoEstimator, BTrackTempoEstimator,
    BeatNetPFTempoEstimator, MadmomTempoEstimator,
)
from flame_sheep.eval.gtzan_parser import iter_tracks as iter_gtzan


CACHE_DIR = Path.home() / 'datasets' / 'tempo-eval-cache'
GTZAN_ROOT = Path.home() / 'datasets' / 'GTZAN'
OSU_ROOT = Path.home() / '.cache' / 'flame-sheep' / 'corpus' / 'osu'

ESTIMATORS = {
    'acf':         lambda: AcfTempoEstimator(),
    'btrack':      lambda: BTrackTempoEstimator(),
    'beatnet_pf':  lambda: BeatNetPFTempoEstimator(),
    'madmom_comb': lambda: MadmomTempoEstimator(method='comb'),
    'madmom_acf':  lambda: MadmomTempoEstimator(method='acf'),
    'madmom_dbn':  lambda: MadmomTempoEstimator(method='dbn'),
}


def _collect_gtzan(root: Path) -> list[tuple[str, Path, float]]:
    """(track_id, audio_path, reference_bpm) for every usable gtzan track."""
    return [(t.track_id, t.audio_path, float(t.tempo_bpm))
            for t in iter_gtzan(root)]


def _collect_osu(root: Path) -> list[tuple[str, Path, float]]:
    """Same shape for osu beatmaps. Reference BPM = median across
    uninherited TimingPoints (multi-tempo charts get a centroid;
    single-tempo songs unaffected)."""
    from flame_sheep.eval.osu_parser import parse_osu_file
    out: list[tuple[str, Path, float]] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        osu_files = sorted(sub.glob('*.osu'))
        if not osu_files:
            continue
        try:
            m = parse_osu_file(osu_files[0])
        except (ValueError, OSError):
            continue
        audio = sub / m.audio_filename
        if not audio.exists():
            continue
        bpms = [60000.0 / tp.beat_length_ms
                for tp in m.uninherited_points if tp.beat_length_ms > 1.0]
        if not bpms:
            continue
        out.append((sub.name, audio, float(np.median(bpms))))
    return out


def _collect_tracks(corpus_kind: str, root: Path
                     ) -> list[tuple[str, Path, float]]:
    if corpus_kind == 'gtzan':
        return _collect_gtzan(root)
    if corpus_kind == 'osu':
        return _collect_osu(root)
    raise SystemExit(f'Unknown corpus kind: {corpus_kind!r}')


def _within(pred: float, truth: float, pct: float) -> bool:
    if truth <= 0.0:
        return False
    return abs(pred - truth) / truth * 100.0 < pct


def _octave_within(pred: float, truth: float, pct: float) -> bool:
    if truth <= 0.0:
        return False
    for ratio in (0.5, 1.0, 2.0):
        if _within(pred, truth * ratio, pct):
            return True
    return False


def _cache_path(track_id: str, est_name: str, est_version: str) -> Path:
    return CACHE_DIR / est_name / est_version / f'{track_id}.json'


def _cached_estimate(est: TempoEstimator, track_id: str,
                     audio: np.ndarray, sr: int,
                     force: bool = False) -> tuple[float, dict[str, float]]:
    cp = _cache_path(track_id, est.name, est.version)
    if cp.exists() and not force:
        d = json.loads(cp.read_text())
        return float(d['bpm']), {k: float(v) for k, v in d['timing'].items()}
    bpm = est.estimate(audio, sr)
    timing = dict(est.last_timing)
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps({'bpm': bpm, 'timing': timing}))
    return bpm, timing


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Loads GTZAN at native 22050 Hz mono. Estimator resamples as needed."""
    import librosa
    audio, sr = librosa.load(str(path), sr=22050, mono=True)
    return audio.astype(np.float32), sr


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--estimator', required=True, choices=sorted(ESTIMATORS),
                   help='Tempo estimator to evaluate')
    p.add_argument('--corpus-kind', default='gtzan', choices=['gtzan', 'osu'])
    p.add_argument('--corpus-root', type=Path, default=None,
                   help='Corpus root. Default depends on --corpus-kind: '
                        f'gtzan={GTZAN_ROOT}, osu={OSU_ROOT}')
    p.add_argument('--tracks', type=int, default=None,
                   help='Subset N tracks (default: all)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--tolerance-pct', type=float, default=4.0,
                   help='Accuracy tolerance percent (default 4)')
    p.add_argument('--force', action='store_true',
                   help='Ignore cache and re-run')
    p.add_argument('-o', '--output', type=Path,
                   help='Optional JSON sidecar of per-track + aggregate results')
    args = p.parse_args()

    if args.corpus_root is None:
        args.corpus_root = (GTZAN_ROOT if args.corpus_kind == 'gtzan'
                            else OSU_ROOT)

    est = ESTIMATORS[args.estimator]()
    print(f'Estimator: {est.name} (version {est.version})')
    print(f'Corpus:    {args.corpus_kind} @ {args.corpus_root}')

    tracks = _collect_tracks(args.corpus_kind, args.corpus_root)
    if not tracks:
        print(f'No tracks found at {args.corpus_root}', file=sys.stderr)
        sys.exit(1)
    print(f'Found {len(tracks)} usable tracks')

    if args.tracks is not None and len(tracks) > args.tracks:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(tracks), size=args.tracks, replace=False)
        tracks = [tracks[i] for i in sorted(idx)]
    print(f'Evaluating {len(tracks)} tracks\n')

    per_track: list[dict] = []

    for i, (track_id, audio_path, ref_bpm) in enumerate(tracks, 1):
        try:
            audio, sr = _load_audio(audio_path)
        except Exception as e:
            print(f'  [{i}] {track_id}: audio load failed: {e}')
            continue

        t0 = time.monotonic()
        bpm, timing = _cached_estimate(est, track_id, audio, sr,
                                       force=args.force)
        wall = time.monotonic() - t0

        exact = _within(bpm, ref_bpm, args.tolerance_pct)
        oe1 = _octave_within(bpm, ref_bpm, args.tolerance_pct)
        per_track.append({
            'track_id': track_id,
            'truth_bpm': ref_bpm,
            'pred_bpm': bpm,
            'exact': exact,
            'oe1': oe1,
            'duration_sec': len(audio) / sr,
            'detect_sec': wall,
            'timing': timing,
        })

        if i <= 5 or i % 100 == 0:
            print(f'  [{i:>3}/{len(tracks)}] {track_id:<28} '
                  f'truth={ref_bpm:6.1f} pred={bpm:6.1f}  '
                  f'{"OK" if exact else ("oe1" if oe1 else "miss")}  '
                  f'{wall:.2f}s')

    if not per_track:
        print('No tracks evaluated.', file=sys.stderr)
        sys.exit(2)

    # Aggregate accuracy
    n = len(per_track)
    n_exact = sum(1 for r in per_track if r['exact'])
    n_oe1 = sum(1 for r in per_track if r['oe1'])

    # Mean abs error among the OE1-correct subset (so octave errors don't
    # smear the precision number; report octave error rate separately).
    exact_errors = [abs(r['pred_bpm'] - r['truth_bpm'])
                    for r in per_track if r['exact']]

    # Latency aggregates
    totals = np.array([r['timing'].get('total', 0.0) for r in per_track])
    rt_factors = np.array([
        r['timing'].get('total', 0.0) / r['duration_sec']
        for r in per_track if r['duration_sec'] > 0])
    component_keys: set[str] = set()
    for r in per_track:
        component_keys.update(k for k in r['timing'] if k != 'total')

    # Match the section markers used by evals/builtin.py parsers
    # (`_parse_beat_report` looks for 'Aggregate (mean over tracks)' and
    # similar). Reuse that grammar so the tempo eval can register the
    # same way later.
    print('\n=== Aggregate (mean over tracks) ===')
    print(f'  exact:    {n_exact}/{n} ({100 * n_exact / n:.1f}%)')
    print(f'  oe1:      {n_oe1}/{n} ({100 * n_oe1 / n:.1f}%)')
    if exact_errors:
        print(f'  mae_bpm:  {np.mean(exact_errors):.2f} (over '
              f'{len(exact_errors)} correct-octave tracks)')

    print('\n=== Latency (across tracks) ===')
    print(f'  total.mean_sec:     {float(np.mean(totals)):.3f}')
    print(f'  total.p50_sec:      {float(np.percentile(totals, 50)):.3f}')
    print(f'  total.p95_sec:      {float(np.percentile(totals, 95)):.3f}')
    print(f'  total.p99_sec:      {float(np.percentile(totals, 99)):.3f}')
    if len(rt_factors) > 0:
        print(f'  realtime_factor.mean: {float(np.mean(rt_factors)):.4f}')
        print(f'  realtime_factor.p95:  {float(np.percentile(rt_factors, 95)):.4f}')
    for k in sorted(component_keys):
        vs = np.array([r['timing'].get(k, 0.0) for r in per_track])
        print(f'  {k}.mean_sec:        {float(np.mean(vs)):.3f}')
        print(f'  {k}.p95_sec:         {float(np.percentile(vs, 95)):.3f}')

    if args.output:
        args.output.write_text(json.dumps({
            'estimator': est.name,
            'estimator_version': est.version,
            'corpus_root': str(args.corpus_root),
            'n_tracks': n,
            'n_exact': n_exact,
            'n_oe1': n_oe1,
            'mae_bpm': float(np.mean(exact_errors)) if exact_errors else 0.0,
            'per_track': per_track,
            'latency': {
                'total_mean_sec': float(np.mean(totals)),
                'total_p50_sec': float(np.percentile(totals, 50)),
                'total_p95_sec': float(np.percentile(totals, 95)),
                'total_p99_sec': float(np.percentile(totals, 99)),
                'realtime_factor_mean':
                    float(np.mean(rt_factors)) if len(rt_factors) > 0 else 0.0,
                'realtime_factor_p95':
                    float(np.percentile(rt_factors, 95))
                    if len(rt_factors) > 0 else 0.0,
            },
        }, indent=2))
        print(f'\nResults written to {args.output}')


if __name__ == '__main__':
    main()
