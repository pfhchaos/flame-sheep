#!/usr/bin/env python3
"""Export BeatNet model weights + LOG_SPECT filterbank to a torch-free
.npz consumed by `flame_sheep.eval.beatnet_lite.BeatNetLite`.

Usage:
    python tools/export_beatnet_lite_weights.py
    python tools/export_beatnet_lite_weights.py --model 2

This is a one-shot tool — only re-run if:
  * BeatNet upstream publishes new weights (currently model 1, 2, 3 are
    each frozen and ship inside the upstream package)
  * madmom changes its log-filterbank math (extremely unlikely; the
    filterbank-bit-exact regression test in tests/eval/test_beatnet_lite.py
    will catch any drift)
  * We change the LOG_SPECT parameters (num_bands, fmin, fmax, win, hop)

Output goes to `flame_sheep/data/beatnet_m{n}_lite.npz` — checked into
git so the lite detector is self-contained at install time.
"""
from __future__ import annotations

import argparse
import collections
import collections.abc
import os
import sys
from pathlib import Path

# madmom uses pre-3.10 collections imports.
for _n in ('MutableSequence', 'MutableMapping', 'MutableSet',
           'Sequence', 'Mapping', 'Set', 'Iterable',
           'Container', 'Hashable', 'Callable', 'Sized'):
    if not hasattr(collections, _n):
        setattr(collections, _n, getattr(collections.abc, _n))

import numpy as np


def export(model_index: int, out_dir: Path) -> Path:
    import torch
    from BeatNet.model import BDA
    from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
    from madmom.audio.stft import ShortTimeFourierTransformProcessor
    from madmom.audio.spectrogram import FilteredSpectrogramProcessor

    bn_dir = os.path.dirname(os.path.abspath(
        __import__('BeatNet').__file__))
    weight_path = os.path.join(bn_dir, 'models',
                                f'model_{model_index}_weights.pt')
    if not os.path.exists(weight_path):
        raise FileNotFoundError(weight_path)

    # Build BeatNet's BDA model and load weights.
    m = BDA(272, 150, 2, torch.device('cpu'))
    m.load_state_dict(torch.load(weight_path, map_location='cpu',
                                   weights_only=True))

    # Build the LOG_SPECT filterbank once. Parameters MUST match the
    # ones in `BeatNet.BeatNet.LOG_SPECT` instantiation (n_bands=24,
    # fmin=30, fmax=17000, norm_filters=True). The hop / win sizes
    # are BeatNet's published 20 ms hop / 64 ms window at 22050 Hz.
    SR, WIN, HOP = 22050, 1411, 441
    sig = SignalProcessor(num_channels=1, sample_rate=SR)
    fp = FramedSignalProcessor(frame_size=WIN, hop_size=HOP)
    stft_p = ShortTimeFourierTransformProcessor()
    filt_p = FilteredSpectrogramProcessor(
        num_bands=24, fmin=30, fmax=17000, norm_filters=True)
    dummy = np.zeros(WIN + 2 * HOP, dtype=np.float32)
    filterbank = np.asarray(
        filt_p(stft_p(fp(sig(dummy)))).filterbank, dtype=np.float32)

    # Pack the .npz. Cast everything to float32 explicitly so consumers
    # don't have to depend on torch's default-dtype matching numpy's.
    weights = {k: v.detach().cpu().numpy().astype(np.float32)
               for k, v in m.state_dict().items()}
    weights['filterbank'] = filterbank
    weights['hanning_window'] = np.hanning(WIN).astype(np.float32)

    # Architecture metadata for sanity-checking at load time.
    weights['_meta_dim_in'] = np.int32(272)
    weights['_meta_dim_hd'] = np.int32(150)
    weights['_meta_num_layers'] = np.int32(2)
    weights['_meta_n_out'] = np.int32(3)
    weights['_meta_sample_rate'] = np.int32(SR)
    weights['_meta_hop'] = np.int32(HOP)
    weights['_meta_win'] = np.int32(WIN)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'beatnet_m{model_index}_lite.npz'
    np.savez(out_path, **weights)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=int, default=None,
                        help='Single model index to export (1-3). '
                             'Defaults to all three.')
    parser.add_argument('--out-dir', type=Path,
                        default=Path(__file__).resolve().parent.parent
                                / 'flame_sheep' / 'data',
                        help='Output directory for the .npz files')
    args = parser.parse_args()

    indices = [args.model] if args.model else [1, 2, 3]
    for idx in indices:
        out = export(idx, args.out_dir)
        size_kb = out.stat().st_size / 1024
        print(f'wrote {out}  ({size_kb:.1f} KB)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
