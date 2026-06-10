#!/usr/bin/env python3
"""Per-track tempo comparison: BTrack vs ACF vs BeatNet PF.

For each tracker we sample its current tempo estimate every 1 second
across the track and ask "is this sample within tolerance of ground
truth?" Per-track fraction-of-time-correct rolls up to a per-tracker
percentage.

Two tolerance modes reported separately:
  - exact:  within 4% of truth (1× only)
  - octave: within 4% of truth × {0.5, 1, 2}

For a wallpaper viz, exact matters more than octave — an octave-doubled
tempo gives 2× visual twitch rate, halving gives sluggish/missed pulses.
Octave-acceptable on most beat-tracking benchmarks; visually wrong here.

A track's cold-start tempo (first ~2 seconds) is typically wrong for
all trackers — that wrongness is real and counts against the score,
which is what we want: the viz starts at song start and the user sees
that ramp-up.

Usage:
    python tools/eval_tempo_comparison.py
    python tools/eval_tempo_comparison.py --tracks 999 --skip-cold 2.0
    python tools/eval_tempo_comparison.py --sample-interval 0.5
"""
from __future__ import annotations

import argparse
import collections
import collections.abc
import sys
import time
from pathlib import Path

for _n in ('MutableSequence', 'MutableMapping', 'MutableSet',
           'Sequence', 'Mapping', 'Set', 'Iterable',
           'Container', 'Hashable', 'Callable', 'Sized'):
    if not hasattr(collections, _n):
        setattr(collections, _n, getattr(collections.abc, _n))

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / 'flame_sheep_audio' / 'src'))

from flame_sheep_audio._constants import HOP_SIZE, SAMPLE_RATE


_TOLERANCE_PCT = 4.0
_OCTAVE_RATIOS = (0.5, 1.0, 2.0)


def _bpm_within(est: float, ref: float, ratios: tuple[float, ...]) -> bool:
    if ref <= 0 or est <= 0:
        return False
    for r in ratios:
        target = ref * r
        if abs(est - target) / target * 100.0 < _TOLERANCE_PCT:
            return True
    return False


def _hops_per_sample(sample_interval_sec: float) -> int:
    """Convert a wall-clock sample interval to a daemon-hop count."""
    daemon_hop_sec = HOP_SIZE / SAMPLE_RATE     # ~10.7 ms
    return max(1, int(round(sample_interval_sec / daemon_hop_sec)))


def _sample_tempo_series(audio_48k: np.ndarray,
                          step_fn,
                          read_bpm_fn,
                          sample_every_hops: int) -> list[tuple[float, float]]:
    """Iterate hops, calling step_fn(hop) per hop and read_bpm_fn() every
    `sample_every_hops`. Returns list of (time_sec, bpm)."""
    samples: list[tuple[float, float]] = []
    hop_sec = HOP_SIZE / SAMPLE_RATE
    n_hops = (len(audio_48k) - HOP_SIZE) // HOP_SIZE
    for i in range(n_hops):
        step_fn(audio_48k[i * HOP_SIZE:(i + 1) * HOP_SIZE])
        if i % sample_every_hops == 0:
            try:
                bpm = float(read_bpm_fn())
            except Exception:
                bpm = 0.0
            samples.append((i * hop_sec, bpm))
    return samples


def _run_btrack(audio_48k, sample_every_hops):
    from flame_sheep_audio.tempo_btrack import BTrackTempoTracker
    t = BTrackTempoTracker(hop_size=HOP_SIZE, sample_rate=SAMPLE_RATE)
    return _sample_tempo_series(
        audio_48k,
        step_fn=t.feed_audio,
        read_bpm_fn=lambda: t.effective_bpm,
        sample_every_hops=sample_every_hops)


def _run_acf(audio_48k, sample_every_hops):
    from flame_sheep_audio.tempo_acf import AutocorrelationTempoTracker
    from flame_sheep_audio._cqt_engine import CqtEngine
    from flame_sheep_audio.hpss import ComplexSpectralDiffTransform
    from flame_sheep_audio._bands import A_WEIGHTS
    engine = CqtEngine()
    csd = ComplexSpectralDiffTransform()
    tracker = AutocorrelationTempoTracker()
    weights = A_WEIGHTS[:engine.bin_centers.shape[0]]

    def _step(chunk):
        raw = engine.push_hop(chunk)
        csd_frame = csd(raw)
        onset_str = float(np.dot(csd_frame.flux[:len(weights)], weights))
        tracker.feed(onset_str, onset_density=0.0)

    return _sample_tempo_series(
        audio_48k, _step,
        lambda: tracker.effective_bpm,
        sample_every_hops)


def _run_beatnet_pf(audio_48k, sample_every_hops):
    from flame_sheep_audio.beat_detector_beatnet import BeatNetLiveDetector
    det = BeatNetLiveDetector(use_particle_filter=True)

    def _step(chunk):
        det.feed_audio_hop(chunk)
        det.detect(None)

    def _read():
        snap = det.current_tempo()
        return float(snap['bpm']) if snap else 0.0

    return _sample_tempo_series(
        audio_48k, _step, _read, sample_every_hops)


def _accuracy_of_series(samples: list[tuple[float, float]],
                         ref_bpm: float,
                         ratios: tuple[float, ...],
                         skip_cold_sec: float) -> float:
    """Fraction of (post-cold-start) samples within tolerance."""
    if not samples or ref_bpm <= 0:
        return 0.0
    n_correct = 0
    n_total = 0
    for t, bpm in samples:
        if t < skip_cold_sec:
            continue
        n_total += 1
        if _bpm_within(bpm, ref_bpm, ratios):
            n_correct += 1
    return n_correct / n_total if n_total > 0 else 0.0


def _collect_tracks(corpus: str, root: Path) -> list[tuple[str, Path, float]]:
    if corpus == 'gtzan':
        from flame_sheep.eval.gtzan_parser import iter_tracks
        return [(t.track_id, t.audio_path, float(t.tempo_bpm))
                for t in iter_tracks(root)]
    if corpus == 'osu':
        from flame_sheep.eval.osu_parser import parse_osu_file
        tracks: list[tuple[str, Path, float]] = []
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
            # Reference BPM: median across uninherited timing points.
            # Same convention as tools/eval_tempo_reconcile.py:_OsuTrack.
            bpms = [60000.0 / tp.beat_length_ms
                    for tp in m.uninherited_points if tp.beat_length_ms > 1.0]
            if not bpms:
                continue
            ref = float(np.median(bpms))
            tracks.append((sub.name, audio, ref))
        return tracks
    raise SystemExit(f'unsupported corpus: {corpus}')


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus', default='gtzan', choices=['gtzan', 'osu'])
    parser.add_argument('--root', type=Path, default=None,
                        help='Corpus root. Defaults: gtzan=~/datasets/GTZAN, '
                             'osu=~/.cache/flame-sheep/corpus/osu')
    parser.add_argument('--tracks', type=int, default=100,
                        help='Number of tracks to sample (default 100; '
                             'pass 9999 for full corpus)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample-interval', type=float, default=1.0,
                        help='Seconds between tempo samples (default 1.0)')
    parser.add_argument('--skip-cold', type=float, default=2.0,
                        help='Drop the first N seconds of each track from '
                             'scoring (default 2.0 — typical PF/ACF cold-start)')
    parser.add_argument('--skip', nargs='*', default=[],
                        choices=['btrack', 'acf', 'beatnet'])
    args = parser.parse_args()

    if args.root is None:
        args.root = (Path.home() / 'datasets' / 'GTZAN'
                     if args.corpus == 'gtzan'
                     else Path.home() / '.cache' / 'flame-sheep' / 'corpus' / 'osu')
    handles = _collect_tracks(args.corpus, args.root)
    if not handles:
        raise SystemExit(f'no tracks found in {args.root}')
    print(f'corpus={args.corpus}  total usable tracks: {len(handles)}')

    rng = np.random.default_rng(args.seed)
    if len(handles) > args.tracks:
        idx = rng.choice(len(handles), size=args.tracks, replace=False)
        handles = [handles[i] for i in sorted(idx)]
    print(f'evaluating {len(handles)} tracks at {args.sample_interval}s '
          f'sampling (skip first {args.skip_cold}s)')
    print()

    trackers = []
    if 'btrack' not in args.skip:
        trackers.append(('btrack', _run_btrack))
    if 'acf' not in args.skip:
        trackers.append(('acf', _run_acf))
    if 'beatnet' not in args.skip:
        trackers.append(('beatnet', _run_beatnet_pf))

    sample_every = _hops_per_sample(args.sample_interval)

    per_track: dict[str, list[dict]] = {name: [] for name, _ in trackers}

    import librosa
    for i, (tid, audio_path, ref_bpm) in enumerate(handles, 1):
        try:
            audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
            audio = audio.astype(np.float32)
        except Exception as e:
            print(f'  [{i}/{len(handles)}] {tid}: load failed: {e}')
            continue
        print(f'[{i}/{len(handles)}] {tid}  ref={ref_bpm:6.1f}', end='', flush=True)
        for name, fn in trackers:
            t0 = time.monotonic()
            samples = fn(audio, sample_every)
            dt = time.monotonic() - t0
            exact_frac = _accuracy_of_series(samples, ref_bpm,
                                              (1.0,), args.skip_cold)
            octave_frac = _accuracy_of_series(samples, ref_bpm,
                                               _OCTAVE_RATIOS, args.skip_cold)
            # Jitter: per-track std-dev of bpm samples in the warm section.
            # High jitter even at high accuracy → tempo is flipping between
            # adjacent quantized tempi; viz would still need smoothing.
            warm = [bpm for t, bpm in samples if t >= args.skip_cold]
            warm_std = float(np.std(warm)) if warm else 0.0
            warm_mean = float(np.mean(warm)) if warm else 0.0
            per_track[name].append({
                'track': tid, 'ref': ref_bpm,
                'exact_frac': exact_frac,
                'octave_frac': octave_frac,
                'warm_mean': warm_mean,
                'warm_std': warm_std,
                'n_samples': len(samples),
                'dt_sec': dt})
            print(f'  {name}≈{warm_mean:5.1f}±{warm_std:4.1f}'
                  f'({int(exact_frac*100):3d}/{int(octave_frac*100):3d})',
                  end='')
        print(flush=True)

    print()
    print('=== Summary (averaged across tracks) ===')
    print(f'  {"tracker":<10s}  {"exact_pct":>10s}  {"octave_pct":>11s}  '
          f'{"jitter":>8s}  {"mean_sec":>10s}')
    print('  ' + '-' * 60)
    for name, _ in trackers:
        rows = per_track[name]
        if not rows:
            print(f'  {name:<10s}  no data')
            continue
        exact_mean = float(np.mean([r['exact_frac'] for r in rows])) * 100
        octave_mean = float(np.mean([r['octave_frac'] for r in rows])) * 100
        jitter_mean = float(np.mean([r['warm_std'] for r in rows]))
        mean_dt = float(np.mean([r['dt_sec'] for r in rows]))
        print(f'  {name:<10s}  {exact_mean:>8.1f}%   {octave_mean:>9.1f}%   '
              f'{jitter_mean:>5.1f} BPM   {mean_dt:>7.2f}')
    print()
    print(f'(exact = within {_TOLERANCE_PCT:.0f}% of truth; '
          f'octave = within {_TOLERANCE_PCT:.0f}% of truth × {{0.5, 1, 2}})')
    print(f'(skip first {args.skip_cold:.1f}s of each track from scoring)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
