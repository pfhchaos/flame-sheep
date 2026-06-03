#!/usr/bin/env python3
"""Diagnose whether the multidepth beat-RNN's three heads are actually
differentiating onset / beat / downbeat, or whether they're all firing
on the same underlying "rhythmic event" signal.

Two outputs:

  1. Per-head activation statistics (min/p25/median/p75/max) over a
     full track. If the distributions overlap heavily, the heads see
     similar things.

  2. Inter-head correlation matrix. r > 0.9 means the heads are
     essentially the same signal under different thresholds; r < 0.5
     means they're learning different features.

Run on an osu track (any music file with rhythmic structure works):

    python tools/diag_head_differentiation.py PATH/TO/audio.mp3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flame_sheep_audio import SAMPLE_RATE, HOP_SIZE
from flame_sheep_audio._cqt_engine import CqtEngine
from flame_sheep_audio.beat_rnn import load_beat_rnn


def _resample(audio, src_sr, dst_sr):
    if src_sr == dst_sr:
        return audio
    ratio = dst_sr / src_sr
    n = int(len(audio) * ratio)
    idx = np.clip((np.arange(n) / ratio).astype(int), 0, len(audio) - 1)
    return audio[idx]


def collect_activations(audio_path: Path, weights: Path
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (downbeat_acts, beat_acts, onset_acts) per frame
    over the full track."""
    import librosa
    audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
    audio = audio.astype(np.float32)
    audio = _resample(audio, sr, SAMPLE_RATE)

    cqt = CqtEngine()
    detector = load_beat_rnn(
        str(weights),
        threshold=0.0,
        downbeat_threshold=0.0,
        beat_threshold=0.0,
        onset_threshold=0.0,
    )

    d_acts, b_acts, o_acts = [], [], []
    for pos in range(0, len(audio) - HOP_SIZE, HOP_SIZE):
        chunk = audio[pos:pos + HOP_SIZE]
        frame = cqt.push_hop(chunk)
        # The detector's _forward_step returns the raw per-head
        # activations; tap directly into it (more reliable than
        # threshold-zero detection which is still subject to
        # peak-picking + refractory).
        mag = frame.magnitude
        # Match BeatRNNDetector.detect()'s input prep
        log_mag = np.log1p(mag * 770.0).astype(np.float32)
        if detector._prev_log_mag is None:
            diff = np.zeros_like(log_mag)
        else:
            diff = np.maximum(0.0, log_mag - detector._prev_log_mag).astype(np.float32)
        detector._prev_log_mag = log_mag
        x = np.concatenate([log_mag, diff])
        d, b, o = detector._forward_step(x)
        d_acts.append(d)
        b_acts.append(b)
        o_acts.append(o)
    return (np.asarray(d_acts), np.asarray(b_acts), np.asarray(o_acts))


def _stats(name: str, vals: np.ndarray) -> None:
    p = np.percentile(vals, [0, 25, 50, 75, 100])
    print(f'  {name:>9}  min={p[0]:.3f}  p25={p[1]:.3f}  '
          f'p50={p[2]:.3f}  p75={p[3]:.3f}  max={p[4]:.3f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('audio', type=Path, help='Audio file to analyze')
    parser.add_argument('--weights', type=Path,
                        default=Path('flame_sheep/data/beat_rnn_multidepth.npz'))
    parser.add_argument('--threshold-test', type=float, default=None,
                        help='If set, count cascade outputs with all three '
                             'thresholds at this value (uniform threshold)')
    args = parser.parse_args()

    print(f'Loading + running detector on {args.audio.name}...')
    d, b, o = collect_activations(args.audio, args.weights)
    n = len(d)
    print(f'  {n} frames ({n * HOP_SIZE / SAMPLE_RATE:.1f}s of audio)')

    print()
    print('=== Per-head activation distribution ===')
    _stats('downbeat', d)
    _stats('beat', b)
    _stats('onset', o)

    print()
    print('=== Inter-head correlation ===')
    print(f'  r(downbeat, beat) = {np.corrcoef(d, b)[0,1]:+.3f}')
    print(f'  r(downbeat, onset) = {np.corrcoef(d, o)[0,1]:+.3f}')
    print(f'  r(beat, onset)     = {np.corrcoef(b, o)[0,1]:+.3f}')

    if args.threshold_test is not None:
        th = args.threshold_test
        # Reproduce the cascade logic: emit kind per frame
        n_downbeat = int(np.sum(d >= th))
        # Beat: above threshold AND downbeat NOT above
        n_beat = int(np.sum((b >= th) & (d < th)))
        # Onset: above threshold AND neither downbeat nor beat
        n_onset = int(np.sum((o >= th) & (d < th) & (b < th)))
        n_emit = n_downbeat + n_beat + n_onset
        print()
        print(f'=== Cascade output at uniform threshold {th} ===')
        print(f'  downbeat (kind=low):  {n_downbeat} ({100*n_downbeat/max(n_emit,1):.0f}%)')
        print(f'  beat (kind=mid):      {n_beat} ({100*n_beat/max(n_emit,1):.0f}%)')
        print(f'  onset (kind=high):    {n_onset} ({100*n_onset/max(n_emit,1):.0f}%)')
        print(f'  total emitted:        {n_emit}')
        if n_onset < n_emit * 0.05:
            print('  ⚠ Onset kind is < 5% of total — likely indicates beat '
                  'head fires on the same frames as onset head, with beat '
                  'head simply being more strongly activated.')

        # Also: how often is onset activation > beat activation, AND
        # both > threshold? Those are frames where the onset head
        # "wins" but the cascade still classifies them as beats
        # because the cascade prefers more-specific tiers.
        n_onset_wins = int(np.sum((o > b) & (b >= th) & (o >= th)))
        print(f'  frames where o > b AND both >= {th}: {n_onset_wins}')
        print(f'    (those would be "beats" per the cascade even though '
              f'the onset head fires more strongly)')


if __name__ == '__main__':
    main()
