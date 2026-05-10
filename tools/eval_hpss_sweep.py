#!/usr/bin/env python3
"""HPSS sweep: stability method × kernel size × spectrum engine.

Tests beat detection F1 across the full cached osu! corpus for
different HPSS configurations with the CQT engine.

Usage:
    python -m tests.eval_hpss_sweep
    python -m tests.eval_hpss_sweep --engines cqt octave_bank
    python -m tests.eval_hpss_sweep --kernels 3 5 7
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flame_sheep_audio._constants import SAMPLE_RATE, HOP_SIZE
from flame_sheep_audio._octave_bank import OctaveBankEngine
from flame_sheep_audio.beat_detector import FluxBeatDetector
from flame_sheep_audio.stability import MagnitudeStability
from flame_sheep_audio.energy import EnergyAnalyzer

from tests.corpus.downloader import download_osz, find_osu_files, find_audio_file
from tests.corpus.osu_parser import parse_osu_file
from tests.corpus.loader import load_audio, resample_to_48k
from tests.corpus.metrics import onset_precision_recall

try:
    from flame_sheep_audio._cqt_engine import CqtEngine
    _HAS_CQT = True
except ImportError:
    _HAS_CQT = False

CACHE = Path.home() / '.cache/flame-sheep/corpus'

# All osu! tracks with cached data
TRACKS = [
    (387700, 'Megalovania', ['electronic', 'game_ost']),
    (163112, 'My Love', ['pop', 'vocal', 'slow']),
    (522857, 'Shelter', ['pop', 'electronic', 'vocal']),
    (399358, 'Silhouette', ['rock', 'jpop', 'anime']),
    (158023, 'Everything Will Freeze', ['metal', 'blast_beats', 'fast']),
    (68893, 'River Flows In You', ['piano', 'slow', 'no_percussion']),
    (332532, 'Highscore', ['edm', 'dubstep', 'drops']),
    (372510, 'Kyouran Hey Kids', ['rock', 'energetic']),
    (503958, 'Whiplash', ['jazz', 'big_band', 'fast']),
    (813569, 'Sound Chimera', ['edm', 'hardcore', 'fast', 'complex']),
]


@dataclass
class SweepResult:
    engine: str
    method: str
    kernel: int
    track: str
    f1: float
    precision: float
    recall: float
    time_s: float


def run_track(engine, method: str, kernel: int, track_id: int) -> tuple[float, float, float] | None:
    """Run beat detection on one track. Returns (precision, recall, f1) or None."""
    osz_dir = download_osz(track_id, CACHE)
    osu_files = find_osu_files(osz_dir)
    if not osu_files:
        return None
    bm = parse_osu_file(osu_files[0])
    ap = find_audio_file(osz_dir, bm.audio_filename)
    if not ap:
        return None
    audio, sr = load_audio(ap)
    audio = resample_to_48k(audio, sr)

    # Build pipeline with specified HPSS config
    if method == 'shape':
        # Patch shape kernel via the config object
        from flame_sheep_audio.config import cfg
        old_kernel = getattr(cfg.stability, 'shape_kernel', 7)
        old_method = getattr(cfg.stability, 'method', 'ema')
        cfg.stability.shape_kernel = kernel
        cfg.stability.method = 'shape'

    stab = MagnitudeStability(method=method)
    det = FluxBeatDetector(freqs=engine.bin_centers)
    energy = EnergyAnalyzer(freqs=engine.bin_centers)

    if method == 'shape':
        cfg.stability.shape_kernel = old_kernel
        cfg.stability.method = old_method

    times = []
    n_hops = len(audio) // HOP_SIZE
    for i in range(n_hops):
        hop = audio[i * HOP_SIZE:(i + 1) * HOP_SIZE]
        if len(hop) < HOP_SIZE:
            hop = np.pad(hop, (0, HOP_SIZE - len(hop)))
        f = engine.push_hop(hop)
        stab.update(f.magnitude)
        energy.update(f.magnitude, f.flux, stability=stab)
        # Apply percussive weighting before detection (matches processor)
        perc_weight = np.sqrt(1.0 - stab.stability_per_bin())
        f.flux = f.flux * perc_weight
        if det.detect(f):
            times.append(i * HOP_SIZE / SAMPLE_RATE)

    gt = [h.time_ms / 1000.0 for h in bm.hit_objects]
    return onset_precision_recall(times, gt, window=0.05)


def main():
    parser = argparse.ArgumentParser(description='HPSS sweep evaluation')
    parser.add_argument('--engines', nargs='+', default=['cqt'],
                        choices=['cqt', 'octave_bank'])
    parser.add_argument('--methods', nargs='+', default=['ema', 'shape'],
                        choices=['ema', 'median', 'shape'])
    parser.add_argument('--kernels', nargs='+', type=int,
                        default=[3, 5, 7, 9, 11])
    args = parser.parse_args()

    engine_map = {'octave_bank': OctaveBankEngine}
    if _HAS_CQT:
        engine_map['cqt'] = CqtEngine

    # Results file — JSONL, one result per line, flushed immediately
    results_dir = Path.home() / '.local/share/flame-sheep/eval'
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_path = results_dir / f'hpss_sweep_{timestamp}.jsonl'
    results_fh = open(results_path, 'w')
    print(f'Writing results to {results_path}')

    results: list[SweepResult] = []

    for eng_name in args.engines:
        if eng_name not in engine_map:
            print(f'Skipping {eng_name} (not available)')
            continue
        eng_cls = engine_map[eng_name]

        for method in args.methods:
            kernels = args.kernels if method == 'shape' else [0]

            for kernel in kernels:
                label = f'{eng_name}/{method}'
                if method == 'shape':
                    label += f'/k={kernel}'

                f1s = []
                for tid, tname, tags in TRACKS:
                    engine = eng_cls()
                    t0 = time.perf_counter()
                    try:
                        r = run_track(engine, method, kernel, tid)
                    except Exception as e:
                        print(f'  {label:30s} {tname:25s} ERROR: {e}', flush=True)
                        continue
                    dt = time.perf_counter() - t0

                    if r:
                        p, rc, f1 = r
                        f1s.append(f1)
                        sr = SweepResult(
                            engine=eng_name, method=method, kernel=kernel,
                            track=tname, f1=f1, precision=p, recall=rc,
                            time_s=dt)
                        results.append(sr)
                        results_fh.write(json.dumps(asdict(sr)) + '\n')
                        results_fh.flush()
                        print(f'  {label:30s} {tname:25s} '
                              f'F1={f1:.3f} P={p:.3f} R={rc:.3f} ({dt:.1f}s)',
                              flush=True)

                if f1s:
                    mean_f1 = np.mean(f1s)
                    print(f'  {label:30s} {"MEAN":25s} F1={mean_f1:.3f}', flush=True)
                print(flush=True)

    # Summary table
    print('\n' + '=' * 70)
    print(f'{"Config":30s} {"Mean F1":>8s} {"Mean P":>8s} {"Mean R":>8s}')
    print('-' * 70)

    # Group by config
    configs = {}
    for r in results:
        key = (r.engine, r.method, r.kernel)
        configs.setdefault(key, []).append(r)

    for (eng, method, kernel), rs in sorted(configs.items(),
                                             key=lambda x: -np.mean([r.f1 for r in x[1]])):
        label = f'{eng}/{method}'
        if method == 'shape':
            label += f'/k={kernel}'
        mean_f1 = np.mean([r.f1 for r in rs])
        mean_p = np.mean([r.precision for r in rs])
        mean_r = np.mean([r.recall for r in rs])
        print(f'  {label:28s} {mean_f1:8.3f} {mean_p:8.3f} {mean_r:8.3f}')

    print('=' * 70)
    results_fh.close()
    print(f'\nResults saved to {results_path}')


if __name__ == '__main__':
    main()
