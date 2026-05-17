#!/usr/bin/env python3
"""Generate beat detection training labels using BeatNet as oracle.

Requires Python 3.10 (madmom compatibility). Run with:
    python3.10 tools/generate_beat_labels.py ~/music/ -o ~/datasets/beat-labels/

For each audio file:
  1. Compute CQT spectrum at our native hop rate (48kHz, 512 hop)
  2. Run BeatNet to get per-frame [non-beat, beat, downbeat] soft labels
  3. Resample BeatNet labels to align with our frame rate
  4. Save as .npz: spectrum + diff + labels + metadata
"""

import argparse
import collections
import collections.abc
import sys
from pathlib import Path

# Monkey-patch for madmom on Python 3.10+ (removed in 3.9, deprecated since 3.3)
collections.MutableSequence = collections.abc.MutableSequence
collections.MutableMapping = collections.abc.MutableMapping
collections.MutableSet = collections.abc.MutableSet

import numpy as np
import librosa
import torch

from BeatNet.model import BDA
from BeatNet.log_spect import LOG_SPECT

# Our audio parameters
OUR_SAMPLE_RATE = 48000
OUR_HOP = 512
OUR_FPS = OUR_SAMPLE_RATE / OUR_HOP  # ~93.75

# BeatNet parameters
BN_SAMPLE_RATE = 22050
BN_HOP = int(20 * 0.001 * BN_SAMPLE_RATE)  # 441 samples = 20ms
BN_WIN = int(64 * 0.001 * BN_SAMPLE_RATE)  # 1411 samples = 64ms
BN_FPS = 50.0


def find_audio_files(root: Path) -> list:
    """Recursively find audio files."""
    exts = {'.flac', '.mp3', '.wav', '.ogg', '.opus', '.m4a', '.aac', '.wma'}
    files = []
    for f in sorted(root.rglob('*')):
        if f.suffix.lower() in exts and f.is_file():
            files.append(f)
    return files


def compute_cqt_frames(audio_48k: np.ndarray) -> np.ndarray:
    """Compute CQT magnitude at our native rate. Returns [T, 108] float32."""
    cqt = librosa.cqt(
        audio_48k, sr=OUR_SAMPLE_RATE, hop_length=OUR_HOP,
        n_bins=108, bins_per_octave=12, fmin=librosa.note_to_hz('C1'),
    )
    mag = np.abs(cqt).T.astype(np.float32)  # [T, 108]
    mag = np.log1p(mag * 10.0)  # log scale
    return mag


def compute_beatnet_activations(audio_path: str, model: BDA, proc: LOG_SPECT) -> np.ndarray:
    """Run BeatNet on audio file, return [T_bn, 3] softmax activations."""
    audio, _ = librosa.load(audio_path, sr=BN_SAMPLE_RATE)

    with torch.no_grad():
        feats = proc.process_audio(audio).T  # [T_bn, feat_dim]
        feats = torch.from_numpy(feats).unsqueeze(0)  # [1, T_bn, feat_dim]

        # Reset LSTM state for each file
        model.hidden = torch.zeros(2, 1, model.dim_hd)
        model.cell = torch.zeros(2, 1, model.dim_hd)

        preds = model(feats)[0]  # [3, T_bn]
        probs = torch.softmax(preds, dim=0)  # [3, T_bn]
        probs = probs.numpy().T  # [T_bn, 3]

    return probs


def resample_labels(bn_labels: np.ndarray, n_our_frames: int) -> np.ndarray:
    """Resample BeatNet labels (50fps) to our frame rate (~93.75fps).

    Nearest-neighbor interpolation. Returns [n_our_frames, 3] float32.
    """
    n_bn = len(bn_labels)
    our_times = np.arange(n_our_frames) / OUR_FPS
    bn_times = np.arange(n_bn) / BN_FPS
    indices = np.searchsorted(bn_times, our_times, side='right') - 1
    indices = np.clip(indices, 0, n_bn - 1)
    return bn_labels[indices].astype(np.float32)


def process_file(audio_path: Path, model: BDA, proc: LOG_SPECT,
                 output_dir: Path, max_duration=None) -> bool:
    """Process one audio file. Returns True on success."""
    try:
        # Load at our sample rate for CQT
        audio_48k, _ = librosa.load(str(audio_path), sr=OUR_SAMPLE_RATE,
                                     duration=max_duration)
        if len(audio_48k) < OUR_SAMPLE_RATE * 5:
            return False  # skip < 5s

        # CQT spectrum
        cqt_frames = compute_cqt_frames(audio_48k)
        n_frames = len(cqt_frames)

        # BeatNet activations
        bn_probs = compute_beatnet_activations(str(audio_path), model, proc)

        # Resample to our frame rate
        labels = resample_labels(bn_probs, n_frames)

        # First-order difference (half-wave rectified)
        diff = np.zeros_like(cqt_frames)
        diff[1:] = np.maximum(0, cqt_frames[1:] - cqt_frames[:-1])

        # Save
        stem = audio_path.stem[:80]
        parent = audio_path.parent.name[:30]
        out_name = f"{parent}__{stem}.npz"
        out_path = output_dir / out_name

        np.savez_compressed(
            out_path,
            spectrum=cqt_frames,   # [T, 108] float32
            diff=diff,             # [T, 108] float32
            labels=labels,         # [T, 3] float32 — soft probabilities
            sr=OUR_SAMPLE_RATE,
            hop=OUR_HOP,
            source=str(audio_path),
        )
        duration = len(audio_48k) / OUR_SAMPLE_RATE
        print(f"  {out_name} -- {n_frames} frames ({duration:.0f}s)")
        return True

    except Exception as e:
        print(f"  FAIL {audio_path.name}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description='Generate beat labels via BeatNet')
    parser.add_argument('input', type=Path, help='Audio file or directory')
    parser.add_argument('-o', '--output', type=Path,
                        default=Path.home() / 'datasets' / 'beat-labels',
                        help='Output directory for .npz files')
    parser.add_argument('--model', type=int, default=1, choices=[1, 2, 3],
                        help='BeatNet model (1=GTZAN, 2=Ballroom, 3=Rock)')
    parser.add_argument('--max-duration', type=float, default=None,
                        help='Max seconds per file (None = full)')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Max number of files to process')
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Find audio files
    if args.input.is_file():
        files = [args.input]
    else:
        files = find_audio_files(args.input)

    if args.max_files:
        files = files[:args.max_files]

    print(f"Found {len(files)} audio files")
    print(f"Output: {args.output}")
    print(f"BeatNet model: {args.model}")
    print()

    # Load BeatNet
    import os
    bn_dir = os.path.dirname(os.path.abspath(__import__('BeatNet').__file__))
    model = BDA(272, 150, 2, 'cpu')
    weights_path = os.path.join(bn_dir, f'models/model_{args.model}_weights.pt')
    model.load_state_dict(torch.load(weights_path, weights_only=True), strict=False)
    model.eval()

    proc = LOG_SPECT(
        sample_rate=BN_SAMPLE_RATE,
        win_length=BN_WIN,
        hop_size=BN_HOP,
        n_bands=[24],
        mode='online',
    )

    # Process
    success = 0
    for i, f in enumerate(files):
        rel = f.relative_to(args.input) if f.is_relative_to(args.input) else f.name
        print(f"[{i+1}/{len(files)}] {rel}")
        if process_file(f, model, proc, args.output, args.max_duration):
            success += 1

    print(f"\nDone: {success}/{len(files)} files")
    print(f"Output: {args.output}")


if __name__ == '__main__':
    main()
