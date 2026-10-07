#!/usr/bin/env python3
"""Compare training-time CQT vs live-time CQT on the same audio.

Diagnostic for the beat-RNN's "activations below threshold" issue.
The model was trained on librosa.cqt output; the live daemon uses
prtcqt (rt-cqt SlidingCqt). If the two produce different magnitudes
for the same PCM input, the model sees out-of-distribution features
in production no matter what the AGC does.

This script:
1. Picks a training audio file (from a label .npz source).
2. Runs it through librosa.cqt with training-pipeline params.
3. Runs the SAME PCM through the daemon's CqtEngine, hop by hop.
4. Compares the magnitude streams: scale, correlation, ratio.
5. Optionally runs the trained model forward on both feature streams
   and compares activation distributions.

Tells you whether the discrepancy is upstream (CQT scale) or
downstream (AGC / threshold / user audio).

Usage:
    .venv/bin/python tools/diagnose_cqt_skew.py
    .venv/bin/python tools/diagnose_cqt_skew.py --duration 60 --no-model
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


SR = 48000
HOP = 512
N_BINS = 108
BINS_PER_OCTAVE = 12


def main():
    parser = argparse.ArgumentParser(
        description='Compare librosa.cqt (training) vs prtcqt (live) on the same audio')
    parser.add_argument('--labels-dir', type=Path,
                        default=Path.home() / 'datasets' / 'beat-labels')
    parser.add_argument('--source', type=Path,
                        help='Specific audio file (else random pick from labels)')
    parser.add_argument('--duration', type=float, default=30.0,
                        help='Seconds of audio to compare')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--weights', type=Path,
                        default=Path(__file__).resolve().parent.parent
                                / 'flame_sheep' / 'data'
                                / 'beat_rnn_continuous.npz')
    parser.add_argument('--no-model', action='store_true',
                        help='Skip the model-activation comparison')
    args = parser.parse_args()

    import librosa

    # Make flame_sheep_audio importable.
    project = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project / 'flame_sheep_audio' / 'src'))
    sys.path.insert(0, str(project))

    from flame_sheep_audio import CqtEngine

    # 1. Pick source audio.
    if args.source:
        src_path = args.source
    else:
        label_files = sorted(args.labels_dir.glob('*.npz'))
        if not label_files:
            print('No label files found', file=sys.stderr)
            sys.exit(1)
        rng = np.random.default_rng(args.seed)
        for _ in range(10):  # retry on missing source
            lf = label_files[int(rng.integers(0, len(label_files)))]
            with np.load(lf) as d:
                if 'source' in d.files:
                    src_path = Path(str(d['source']))
                    if src_path.exists():
                        break
        else:
            print('Could not find an existing source audio file', file=sys.stderr)
            sys.exit(2)
    print(f'Source: {src_path}')

    # 2. Load PCM (training-pipeline normalization: librosa.load defaults).
    audio, _ = librosa.load(str(src_path), sr=SR, mono=True,
                             duration=args.duration)
    print(f'Loaded {len(audio)} samples ({len(audio)/SR:.1f}s) at {SR} Hz')
    print(f'PCM RMS: {np.sqrt(np.mean(audio**2)):.4f}')

    # 3a. Training-pipeline CQT (librosa.cqt batched over whole audio).
    print('\nComputing librosa.cqt (training path)...')
    librosa_cqt = librosa.cqt(audio, sr=SR, hop_length=HOP,
                               n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE,
                               fmin=librosa.note_to_hz('C1'))
    librosa_mag = np.abs(librosa_cqt).T.astype(np.float32)  # (T, 108)
    print(f'  shape: {librosa_mag.shape}  range: [{librosa_mag.min():.5f}, {librosa_mag.max():.5f}]'
          f'  mean: {librosa_mag.mean():.5f}')

    # 3b. Live-pipeline CQT (daemon's CqtEngine, hop-by-hop streaming).
    print('Computing daemon CqtEngine (live path)...')
    engine = CqtEngine()
    n_hops = len(audio) // HOP
    daemon_mag = np.zeros((n_hops, N_BINS), dtype=np.float32)
    for i in range(n_hops):
        hop = audio[i * HOP:(i + 1) * HOP]
        frame = engine.push_hop(hop)
        daemon_mag[i] = frame.magnitude
    print(f'  shape: {daemon_mag.shape}  range: [{daemon_mag.min():.5f}, {daemon_mag.max():.5f}]'
          f'  mean: {daemon_mag.mean():.5f}')

    # Align to common length (different boundary handling between batched + streaming).
    n = min(len(librosa_mag), len(daemon_mag))
    librosa_mag = librosa_mag[:n]
    daemon_mag = daemon_mag[:n]

    # 4. Compare.
    print('\n=== Magnitude comparison ===')
    # Per-frame ratio (mean of daemon / mean of librosa) — detects scale offset.
    librosa_means = librosa_mag.mean(axis=1)
    daemon_means = daemon_mag.mean(axis=1)
    valid = librosa_means > 1e-8
    if valid.sum() > 0:
        ratios = daemon_means[valid] / librosa_means[valid]
        print(f'  per-frame mean-magnitude ratio (daemon/librosa):')
        print(f'    median: {np.median(ratios):.4f}')
        print(f'    p10/p90: {np.percentile(ratios, 10):.4f} / '
              f'{np.percentile(ratios, 90):.4f}')

    # Correlation: does the daemon track the librosa magnitudes' shape?
    flat_l = librosa_mag.flatten()
    flat_d = daemon_mag.flatten()
    corr = float(np.corrcoef(flat_l, flat_d)[0, 1])
    print(f'  pearson correlation (full): {corr:.4f}')

    # Per-bin correlation (different bins may align better/worse).
    per_bin_corr = []
    for b in range(N_BINS):
        if librosa_mag[:, b].std() > 1e-8 and daemon_mag[:, b].std() > 1e-8:
            c = float(np.corrcoef(librosa_mag[:, b], daemon_mag[:, b])[0, 1])
            per_bin_corr.append(c)
    if per_bin_corr:
        arr = np.array(per_bin_corr)
        print(f'  per-bin correlation (n={len(arr)} bins with variance):')
        print(f'    median: {np.median(arr):.4f}')
        print(f'    p10/p90: {np.percentile(arr, 10):.4f} / '
              f'{np.percentile(arr, 90):.4f}')

    # log1p(mag*10) — the transform the model actually sees.
    print('\n=== After log1p(mag*10) (model input) ===')
    lib_log = np.log1p(librosa_mag * 10.0)
    daem_log = np.log1p(daemon_mag * 10.0)
    print(f'  librosa log-mag mean: {lib_log.mean():.4f}  '
          f'std: {lib_log.std():.4f}  max: {lib_log.max():.4f}')
    print(f'  daemon  log-mag mean: {daem_log.mean():.4f}  '
          f'std: {daem_log.std():.4f}  max: {daem_log.max():.4f}')

    # 5. Optionally run the model forward and compare activations.
    if not args.no_model and args.weights.exists():
        print('\n=== Model activation comparison ===')
        from flame_sheep_audio.rnn_forward import (
            load_weights_npz, unpack_weights, forward_sequence, sigmoid,
        )
        flat = load_weights_npz(args.weights)
        weights = unpack_weights(flat, input_size=216, proj_size=32,
                                  hidden_size=48, n_classes=1)

        # Build (T, 216) feature: [log_mag(108), half-wave-rect-diff(108)]
        def build_features(log_mag):
            diff = np.zeros_like(log_mag)
            diff[1:] = np.maximum(0.0, log_mag[1:] - log_mag[:-1])
            return np.concatenate([log_mag, diff], axis=1).astype(np.float32)

        feats_lib = build_features(lib_log)
        feats_daem = build_features(daem_log)
        logits_lib = forward_sequence(feats_lib, weights, hidden_size=48, proj_size=32)
        logits_daem = forward_sequence(feats_daem, weights, hidden_size=48, proj_size=32)
        act_lib = sigmoid(logits_lib[:, 0])
        act_daem = sigmoid(logits_daem[:, 0])

        for name, a in [('librosa (training)', act_lib),
                         ('daemon  (live)   ', act_daem)]:
            print(f'  {name}: max={a.max():.3f} mean={a.mean():.3f} '
                  f'p90={np.percentile(a, 90):.3f} '
                  f'frames>0.3={int((a > 0.3).sum())}/{len(a)} '
                  f'({100*(a > 0.3).sum()/len(a):.1f}%)')
        # Activation correlation per frame
        act_corr = float(np.corrcoef(act_lib, act_daem)[0, 1])
        print(f'  per-frame activation correlation: {act_corr:.4f}')

    print('\n=== Interpretation guide ===')
    print('  - If per-frame ratio is far from 1.0 → magnitude scale mismatch.')
    print('  - If activations are similar AND both well below 0.3 → user audio')
    print('    just produces low confidence (genre mismatch with training).')
    print('  - If librosa activations >> daemon activations → CQT mismatch is')
    print('    the cause of below-threshold operation in production.')


if __name__ == '__main__':
    main()
