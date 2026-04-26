#!/usr/bin/env python3
"""
Compare beat detection accuracy between 2048-hop (old) and 512-hop (new) paths.

Runs all 12 genre patterns through both analysis paths and reports:
  - Detection count per drum type
  - Timing accuracy (mean/max distance from nearest true beat)
  - False positive rate

Usage:
    python -m tests.bench_hop_comparison
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass

from flame_sheep_audio import SAMPLE_RATE, FFT_SIZE, HOP_SIZE, N_BINS
from flame_sheep_audio._spectrum import SpectrumEngine
from flame_sheep_audio.beat_detector import FluxBeatDetector
from flame_sheep_audio.energy import EnergyAnalyzer

from tests.synths import ALL_PATTERNS, PatternSpec


@dataclass
class DetectedEvent:
    kind: str
    energy: float
    sample: int

    @property
    def time(self) -> float:
        return self.sample / SAMPLE_RATE


def run_old_path(pcm: np.ndarray, warmup: int = 20) -> list[DetectedEvent]:
    """2048-hop path: feed full FFT_SIZE windows, advance by FFT_SIZE."""
    engine = SpectrumEngine()
    detector = FluxBeatDetector(sharpness=True)
    energy = EnergyAnalyzer()

    # Warmup
    silence = np.zeros(FFT_SIZE, dtype=np.float32)
    for _ in range(warmup):
        frame = engine.compute(silence)
        energy.update(frame.magnitude)
        detector.detect(frame)

    events = []
    pos = 0
    while pos < len(pcm):
        chunk = pcm[pos:pos + FFT_SIZE]
        if len(chunk) < FFT_SIZE:
            chunk = np.pad(chunk, (0, FFT_SIZE - len(chunk)))
        frame = engine.compute(chunk)
        energy.update(frame.magnitude)
        for e in detector.detect(frame):
            events.append(DetectedEvent(kind=e.kind, energy=e.energy, sample=pos))
        pos += FFT_SIZE
    return events


def run_new_path(pcm: np.ndarray, warmup: int = 40) -> list[DetectedEvent]:
    """512-hop path: feed HOP_SIZE windows, advance by HOP_SIZE."""
    engine = SpectrumEngine()
    detector = FluxBeatDetector(sharpness=True)
    energy = EnergyAnalyzer()

    # Warmup (more frames needed since hops are smaller)
    silence = np.zeros(HOP_SIZE, dtype=np.float32)
    for _ in range(warmup):
        frame = engine.push_hop(silence)
        energy.update(frame.magnitude)
        detector.detect(frame)

    events = []
    pos = 0
    while pos < len(pcm):
        chunk = pcm[pos:pos + HOP_SIZE]
        if len(chunk) < HOP_SIZE:
            chunk = np.pad(chunk, (0, HOP_SIZE - len(chunk)))
        frame = engine.push_hop(chunk)
        energy.update(frame.magnitude)
        for e in detector.detect(frame):
            events.append(DetectedEvent(kind=e.kind, energy=e.energy, sample=pos))
        pos += HOP_SIZE
    return events


def true_beat_times(spec: PatternSpec) -> dict[str, list[float]]:
    """Compute the expected beat times in seconds for each drum type."""
    beat_dur = 60.0 / spec.bpm
    result = defaultdict(list)
    for bar in range(spec.bars):
        offset = bar * spec.bar_length
        for b in spec.kick_beats:
            result['kick'].append((offset + b - 1) * beat_dur)
        for b in spec.snare_beats:
            result['snare'].append((offset + b - 1) * beat_dur)
        for b in spec.hihat_beats:
            result['hihat'].append((offset + b - 1) * beat_dur)
    return dict(result)


def timing_stats(detected: list[DetectedEvent], true_times: dict[str, list[float]],
                 window: float = 0.15) -> dict:
    """Compute timing accuracy metrics.

    For each true beat, find the nearest detected event of the same type.
    Returns per-type: hits (within window), misses, false_positives,
    mean_offset (of hits), max_offset.
    """
    stats = {}
    for kind in ('kick', 'snare', 'hihat'):
        true = sorted(true_times.get(kind, []))
        det = sorted(e.time for e in detected if e.kind == kind)

        if not true:
            stats[kind] = {'hits': 0, 'misses': 0, 'fp': len(det),
                           'mean_offset': 0, 'max_offset': 0, 'n_true': 0, 'n_det': len(det)}
            continue

        # Match detected to nearest true beat (greedy)
        matched_true = set()
        matched_det = set()
        offsets = []

        for i, t in enumerate(true):
            best_j = None
            best_dist = float('inf')
            for j, d in enumerate(det):
                if j in matched_det:
                    continue
                dist = abs(d - t)
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
            if best_j is not None and best_dist <= window:
                matched_true.add(i)
                matched_det.add(best_j)
                offsets.append(best_dist)

        hits = len(matched_true)
        misses = len(true) - hits
        fp = len(det) - len(matched_det)
        mean_off = np.mean(offsets) * 1000 if offsets else 0  # ms
        max_off = np.max(offsets) * 1000 if offsets else 0    # ms

        stats[kind] = {
            'hits': hits, 'misses': misses, 'fp': fp,
            'mean_offset': mean_off, 'max_offset': max_off,
            'n_true': len(true), 'n_det': len(det),
        }
    return stats


def main():
    print('Beat Detection: 2048-hop vs 512-hop Comparison')
    print('=' * 80)

    totals = {
        'old': defaultdict(lambda: defaultdict(int)),
        'new': defaultdict(lambda: defaultdict(int)),
    }
    offset_sums = {
        'old': defaultdict(list),
        'new': defaultdict(list),
    }

    for spec in ALL_PATTERNS:
        pcm = spec.build().render()
        truth = true_beat_times(spec)

        old_events = run_old_path(pcm)
        new_events = run_new_path(pcm)

        old_stats = timing_stats(old_events, truth)
        new_stats = timing_stats(new_events, truth)

        print(f'\n--- {spec.name} ({spec.bpm} BPM, {spec.bars} bars) ---')
        print(f'  {"type":<6}  {"path":<5}  {"hits":>4}/{" true":<4}  '
              f'{"miss":>4}  {"FP":>3}  {"mean_ms":>7}  {"max_ms":>6}')

        for kind in ('kick', 'snare', 'hihat'):
            o = old_stats[kind]
            n = new_stats[kind]
            print(f'  {kind:<6}  {"old":<5}  {o["hits"]:4d}/{o["n_true"]:<4d}  '
                  f'{o["misses"]:4d}  {o["fp"]:3d}  {o["mean_offset"]:7.1f}  {o["max_offset"]:6.1f}')
            print(f'  {"":6}  {"new":<5}  {n["hits"]:4d}/{n["n_true"]:<4d}  '
                  f'{n["misses"]:4d}  {n["fp"]:3d}  {n["mean_offset"]:7.1f}  {n["max_offset"]:6.1f}')

            # Accumulate totals
            for key in ('hits', 'misses', 'fp', 'n_true', 'n_det'):
                totals['old'][kind][key] += o[key]
                totals['new'][kind][key] += n[key]
            if o['mean_offset'] > 0:
                offset_sums['old'][kind].append(o['mean_offset'])
            if n['mean_offset'] > 0:
                offset_sums['new'][kind].append(n['mean_offset'])

    # Summary
    print('\n' + '=' * 80)
    print('TOTALS ACROSS ALL 12 PATTERNS')
    print('=' * 80)
    print(f'  {"type":<6}  {"path":<5}  {"hits":>4}/{" true":<4}  '
          f'{"recall":>6}  {"FP":>4}  {"prec":>5}  {"mean_ms":>7}')

    for kind in ('kick', 'snare', 'hihat'):
        for label in ('old', 'new'):
            t = totals[label][kind]
            recall = t['hits'] / max(t['n_true'], 1) * 100
            prec = t['hits'] / max(t['hits'] + t['fp'], 1) * 100
            mean_off = np.mean(offset_sums[label][kind]) if offset_sums[label][kind] else 0
            marker = ' <<' if label == 'new' else ''
            print(f'  {kind if label == "old" else "":<6}  {label:<5}  '
                  f'{t["hits"]:4d}/{t["n_true"]:<4d}  {recall:5.1f}%  '
                  f'{t["fp"]:4d}  {prec:4.1f}%  {mean_off:7.1f}{marker}')

    # Timing improvement
    print('\n--- Timing Accuracy Improvement ---')
    for kind in ('kick', 'snare', 'hihat'):
        old_off = np.mean(offset_sums['old'][kind]) if offset_sums['old'][kind] else 0
        new_off = np.mean(offset_sums['new'][kind]) if offset_sums['new'][kind] else 0
        if old_off > 0:
            pct = (1 - new_off / old_off) * 100
            print(f'  {kind:<6}: {old_off:.1f}ms -> {new_off:.1f}ms ({pct:+.1f}% {"better" if pct > 0 else "worse"})')
        else:
            print(f'  {kind:<6}: no data')


if __name__ == '__main__':
    main()
