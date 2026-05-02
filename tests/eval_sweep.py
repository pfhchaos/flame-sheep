#!/usr/bin/env python3
"""Sweep evaluation across engine/stability/weighting combinations.

Runs beat detection (osu! corpus) and tempo estimation (MUSDB18) for
each combination and produces a comparison table.

Usage:
    python -m tests.eval_sweep
    python -m tests.eval_sweep --beat-only
    python -m tests.eval_sweep --tempo-only
    python -m tests.eval_sweep --engines fft octave_bank --stability median shape
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flame_sheep_audio import SAMPLE_RATE, FFT_SIZE, HOP_SIZE
from flame_sheep_audio._spectrum import SpectrumEngine
from flame_sheep_audio._octave_bank import OctaveBankEngine
from flame_sheep_audio.beat_detector import FluxBeatDetector
from flame_sheep_audio.stability import MagnitudeStability
from flame_sheep_audio.energy import EnergyAnalyzer
from flame_sheep_audio.onset_density import OnsetDensityTracker
from flame_sheep_audio.tempo_acf import AutocorrelationTempoTracker
from flame_sheep_audio._bands import a_weight_curve, A_WEIGHTS

from tests.corpus.downloader import download_osz, find_osu_files, find_audio_file
from tests.corpus.osu_parser import parse_osu_file
from tests.corpus.loader import load_audio, resample_to_48k
from tests.corpus.metrics import onset_precision_recall


# Available components
ENGINES = {
    'fft': SpectrumEngine,
    'octave_bank': OctaveBankEngine,
}

STABILITY_METHODS = ['ema', 'median', 'shape']

ONSET_WEIGHTS = ['raw', 'percussive']

# osu! tracks for beat detection eval
BEAT_TRACKS = [
    (387700, 'Megalovania'),
    (163112, 'My Love'),
    (522857, 'Shelter'),
    (399358, 'Silhouette'),
    (158023, 'Everything Will Freeze'),
    (68893, 'River Flows In You'),
]


@dataclass
class SweepResult:
    engine: str
    stability: str
    onset_weight: str
    # Beat detection
    beat_f1: float = 0.0
    beat_precision: float = 0.0
    beat_recall: float = 0.0
    # Tempo estimation
    tempo_exact: float = 0.0
    tempo_octave: float = 0.0
    # Timing
    time_seconds: float = 0.0


def _run_pipeline(audio: np.ndarray, engine_cls, stability_method: str,
                  onset_weight: str):
    """Run the full pipeline on audio. Returns (onset_times, effective_bpm)."""
    engine = engine_cls()
    freqs = getattr(engine, 'bin_centers', None)
    stab = MagnitudeStability(method=stability_method)
    det = FluxBeatDetector(sharpness=True, stability=stab, freqs=freqs)
    energy = EnergyAnalyzer(freqs=freqs)
    density = OnsetDensityTracker()
    hop_dur = HOP_SIZE / SAMPLE_RATE
    tempo = AutocorrelationTempoTracker(hop_duration=hop_dur)

    a_w = A_WEIGHTS if freqs is None else a_weight_curve(freqs)

    est_onsets = []
    now = 0.0
    for i in range(len(audio) // HOP_SIZE):
        chunk = audio[i * HOP_SIZE:(i + 1) * HOP_SIZE]
        if len(chunk) < HOP_SIZE:
            break
        frame = engine.push_hop(chunk)
        stab.update(frame.magnitude)
        energy.update(frame.magnitude, frame.flux, stability=stab)
        events = det.detect(frame)

        # Onset strength: raw or percussive-weighted
        if onset_weight == 'percussive':
            perc_w = np.sqrt(1.0 - stab.stability_per_bin())
            onset_str = float(np.dot(frame.flux * perc_w, a_w))
        else:
            onset_str = float(np.dot(frame.flux, a_w))

        now += hop_dur
        for e in events:
            if e.kind in density._band_names:
                density.process_onset(e.kind, now)
            est_onsets.append(now)
        density.update(now)

        total_density = sum(density.densities_slow.values())
        tempo.feed(onset_str, onset_density=total_density)

    return est_onsets, tempo.effective_bpm


def eval_beat_detection(engine_name: str, stability: str, onset_weight: str,
                        tracks: list[tuple[int, str]]) -> dict:
    """Run beat detection eval on osu! tracks."""
    all_p, all_r, all_f1 = [], [], []

    for tid, name in tracks:
        try:
            path = download_osz(tid)
            osu_files = find_osu_files(path)
            best = None
            for f in osu_files:
                bm = parse_osu_file(f)
                if best is None or len(bm.onset_times) > len(best.onset_times):
                    best = bm
            audio_path = find_audio_file(path, best.audio_filename)
            audio, sr = load_audio(audio_path)
            audio = resample_to_48k(audio, sr)

            est, _ = _run_pipeline(audio, ENGINES[engine_name], stability, onset_weight)
            p, r, f1 = onset_precision_recall(est, best.onset_times, window=0.1)
            all_p.append(p)
            all_r.append(r)
            all_f1.append(f1)
        except Exception as e:
            print(f'  WARNING: {name} failed: {e}', file=sys.stderr)

    return {
        'precision': float(np.mean(all_p)) if all_p else 0.0,
        'recall': float(np.mean(all_r)) if all_r else 0.0,
        'f1': float(np.mean(all_f1)) if all_f1 else 0.0,
    }


def eval_tempo(engine_name: str, stability: str, onset_weight: str,
               **kwargs) -> dict:
    """Run tempo estimation eval on osu! corpus (BPM from timing points)."""
    exact, octave, total = 0, 0, 0

    for tid, name in BEAT_TRACKS:
        try:
            path = download_osz(tid)
            osu_files = find_osu_files(path)
            bm = parse_osu_file(osu_files[0])
            ref_bpm = bm.bpm
            if ref_bpm <= 0:
                continue

            audio_path = find_audio_file(path, bm.audio_filename)
            audio, sr = load_audio(audio_path)
            audio = resample_to_48k(audio, sr)

            _, est_bpm = _run_pipeline(audio, ENGINES[engine_name],
                                       stability, onset_weight)

            total += 1
            error = abs(est_bpm - ref_bpm) / ref_bpm
            if error <= 0.04:
                exact += 1
                octave += 1
            else:
                for ratio in [0.5, 2.0, 1 / 3, 3.0]:
                    if abs(est_bpm - ref_bpm * ratio) / (ref_bpm * ratio) <= 0.04:
                        octave += 1
                        break
        except Exception:
            continue

    return {
        'exact': exact / total if total > 0 else 0.0,
        'octave': octave / total if total > 0 else 0.0,
        'total': total,
    }


def main():
    parser = argparse.ArgumentParser(description='Sweep audio pipeline combinations')
    parser.add_argument('--engines', nargs='+', default=list(ENGINES.keys()),
                        choices=list(ENGINES.keys()))
    parser.add_argument('--stability', nargs='+', default=STABILITY_METHODS,
                        choices=STABILITY_METHODS)
    parser.add_argument('--onset', nargs='+', default=ONSET_WEIGHTS,
                        choices=ONSET_WEIGHTS)
    parser.add_argument('--beat-only', action='store_true')
    parser.add_argument('--tempo-only', action='store_true')
    parser.add_argument('--musdb-root', default=None, help='(unused, tempo uses osu! corpus)')
    args = parser.parse_args()

    results: list[SweepResult] = []

    combos = [(e, s, o) for e in args.engines
              for s in args.stability for o in args.onset]

    print(f'Sweeping {len(combos)} combinations...\n')

    for i, (eng, stab, onset) in enumerate(combos, 1):
        label = f'{eng}/{stab}/{onset}'
        print(f'[{i}/{len(combos)}] {label}...', end=' ', flush=True)
        t0 = time.perf_counter()

        r = SweepResult(engine=eng, stability=stab, onset_weight=onset)

        if not args.tempo_only:
            beat = eval_beat_detection(eng, stab, onset, BEAT_TRACKS)
            r.beat_f1 = beat['f1']
            r.beat_precision = beat['precision']
            r.beat_recall = beat['recall']

        if not args.beat_only:
            tempo = eval_tempo(eng, stab, onset)
            r.tempo_exact = tempo.get('exact', 0.0)
            r.tempo_octave = tempo.get('octave', 0.0)

        r.time_seconds = time.perf_counter() - t0
        results.append(r)
        print(f'F1={r.beat_f1:.3f} T_exact={r.tempo_exact:.0%} '
              f'T_oct={r.tempo_octave:.0%} ({r.time_seconds:.0f}s)')

    # Summary table
    print(f'\n{"=" * 85}')
    print(f'{"Combination":30s} {"Beat F1":>8s} {"Beat P":>7s} {"Beat R":>7s} '
          f'{"T Exact":>8s} {"T Oct":>7s} {"Time":>6s}')
    print(f'{"-" * 85}')
    for r in sorted(results, key=lambda x: -(x.beat_f1 + x.tempo_octave)):
        label = f'{r.engine}/{r.stability}/{r.onset_weight}'
        print(f'{label:30s} {r.beat_f1:8.3f} {r.beat_precision:7.3f} '
              f'{r.beat_recall:7.3f} {r.tempo_exact:7.0%} {r.tempo_octave:7.0%} '
              f'{r.time_seconds:5.0f}s')


if __name__ == '__main__':
    main()
