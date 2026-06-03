#!/usr/bin/env python3
"""Hierarchy-aware test for whether the multidepth beat-RNN's three
heads carry class-specific information vs. just magnitude offsets on
the same underlying signal.

Method: take an osu beatmap with known beat + downbeat timestamps,
run the detector, bucket every frame by ground-truth class, then look
at how each head's activation distribution differs between buckets.

Three buckets per track:

  * **true_downbeat** — frames within ±tolerance_ms of an osu downbeat
  * **true_beat_only** — frames within ±tolerance_ms of an osu beat
    that is NOT a downbeat
  * **true_not_beat** — frames outside ±tolerance_ms of any osu beat
    (includes subdivisions, sustained tones, and silence; osu doesn't
    label these separately)

A properly-trained model shows clear separation between bucket
distributions for each head. A collapsed model shows similar
distributions across buckets with only a magnitude offset.

What the hierarchy predicts a well-trained model SHOULD show:

  | bucket            | downbeat_act    | beat_act       | onset_act |
  |-------------------|-----------------|----------------|-----------|
  | true_downbeat     | high            | high           | high      |
  | true_beat_only    | **low/medium**  | high           | high      |
  | true_not_beat     | low             | **low/medium** | varies    |

The bold cells are the discriminative ones. If true_beat_only has the
SAME downbeat activation as true_downbeat, the downbeat head isn't
actually distinguishing downbeats from beats — it's just firing on
beats generally. Same for beat-vs-non-beat on the beat head.

Usage:
    python tools/diag_head_vs_ground_truth.py PATH/TO/osu_track_dir
        [--weights flame_sheep/data/beat_rnn_multidepth.npz]
        [--tolerance-ms 70]
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
from flame_sheep.eval.osu_parser import parse_osu_file


FPS = SAMPLE_RATE / HOP_SIZE  # ~93.75 frames/sec at 48k/512


def _resample(audio, src_sr, dst_sr):
    if src_sr == dst_sr:
        return audio
    ratio = dst_sr / src_sr
    n = int(len(audio) * ratio)
    idx = np.clip((np.arange(n) / ratio).astype(int), 0, len(audio) - 1)
    return audio[idx]


def collect_activations(audio_path: Path, weights: Path
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Returns (downbeat_acts, beat_acts, onset_acts, duration_sec)."""
    import librosa
    audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
    audio = audio.astype(np.float32)
    audio = _resample(audio, sr, SAMPLE_RATE)
    duration = len(audio) / SAMPLE_RATE

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
        mag = frame.magnitude
        log_mag = np.log1p(mag * 770.0).astype(np.float32)
        if detector._prev_log_mag is None:
            diff = np.zeros_like(log_mag)
        else:
            diff = np.maximum(0.0, log_mag - detector._prev_log_mag).astype(np.float32)
        detector._prev_log_mag = log_mag
        x = np.concatenate([log_mag, diff])
        d, b, o = detector._forward_step(x)
        d_acts.append(d); b_acts.append(b); o_acts.append(o)
    return (np.asarray(d_acts), np.asarray(b_acts), np.asarray(o_acts),
            duration)


def find_osu_files(track_dir: Path) -> tuple[Path, Path]:
    """Pick the first .osu + audio file in the directory."""
    osu_files = sorted(track_dir.glob('*.osu'))
    if not osu_files:
        raise FileNotFoundError(f'No .osu in {track_dir}')
    osu = parse_osu_file(osu_files[0])
    audio = track_dir / osu.audio_filename
    if not audio.exists():
        raise FileNotFoundError(f'osu audio missing: {audio}')
    return audio, osu_files[0]


def bucket_frames(downbeat_times: np.ndarray, beat_times: np.ndarray,
                   n_frames: int, fps: float,
                   tolerance_sec: float) -> np.ndarray:
    """Returns an array of shape (n_frames,) with integer bucket labels:
        0 = true_not_beat (no beat or downbeat within tolerance)
        1 = true_beat_only (beat but not downbeat within tolerance)
        2 = true_downbeat (downbeat within tolerance)

    A downbeat is ALSO a beat (per the hierarchy and osu's encoding),
    so the bucket logic is: downbeat takes precedence over beat.
    """
    frame_times = np.arange(n_frames) / fps
    buckets = np.zeros(n_frames, dtype=np.int8)

    # Beat membership first (then downbeat overwrites)
    if beat_times.size:
        # For each frame, find nearest beat time. If within tolerance, bucket=1
        idx = np.searchsorted(beat_times, frame_times)
        # Candidate indices on left + right; pick closer.
        left = np.clip(idx - 1, 0, len(beat_times) - 1)
        right = np.clip(idx, 0, len(beat_times) - 1)
        d_left = np.abs(frame_times - beat_times[left])
        d_right = np.abs(frame_times - beat_times[right])
        dist = np.minimum(d_left, d_right)
        buckets[dist <= tolerance_sec] = 1

    if downbeat_times.size:
        idx = np.searchsorted(downbeat_times, frame_times)
        left = np.clip(idx - 1, 0, len(downbeat_times) - 1)
        right = np.clip(idx, 0, len(downbeat_times) - 1)
        d_left = np.abs(frame_times - downbeat_times[left])
        d_right = np.abs(frame_times - downbeat_times[right])
        dist = np.minimum(d_left, d_right)
        buckets[dist <= tolerance_sec] = 2

    return buckets


def _stats(label: str, vals: np.ndarray) -> None:
    if not len(vals):
        print(f'  {label:>22}  (no frames)')
        return
    p = np.percentile(vals, [10, 25, 50, 75, 90])
    print(f'  {label:>22}  N={len(vals):5d}  '
          f'p10={p[0]:.3f}  p25={p[1]:.3f}  p50={p[2]:.3f}  '
          f'p75={p[3]:.3f}  p90={p[4]:.3f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('track_dir', type=Path,
                        help='osu track directory (contains .osu + audio file)')
    parser.add_argument('--weights', type=Path,
                        default=Path('flame_sheep/data/beat_rnn_multidepth.npz'))
    parser.add_argument('--tolerance-ms', type=float, default=70.0,
                        help='Time tolerance for matching frames to ground-truth events')
    args = parser.parse_args()

    audio_path, osu_path = find_osu_files(args.track_dir)
    print(f'Track: {args.track_dir.name}')
    print(f'  audio: {audio_path.name}')
    print(f'  osu:   {osu_path.name}')

    print(f'\nRunning detector on full audio...')
    d, b, o, duration = collect_activations(audio_path, args.weights)
    print(f'  {len(d)} frames over {duration:.1f}s')

    osu = parse_osu_file(osu_path)
    beats, downbeats = osu.beat_times(duration)
    print(f'  osu ground truth: {len(beats)} beats, {len(downbeats)} downbeats')

    buckets = bucket_frames(
        downbeats, beats, len(d), FPS, args.tolerance_ms / 1000.0)
    n_db = int(np.sum(buckets == 2))
    n_bo = int(np.sum(buckets == 1))
    n_nb = int(np.sum(buckets == 0))

    print(f'\nFrame buckets (±{args.tolerance_ms:.0f}ms):')
    print(f'  true_downbeat:  {n_db}  ({100*n_db/len(d):.1f}%)')
    print(f'  true_beat_only: {n_bo}  ({100*n_bo/len(d):.1f}%)')
    print(f'  true_not_beat:  {n_nb}  ({100*n_nb/len(d):.1f}%)')

    for head_name, head_acts in [('DOWNBEAT', d), ('BEAT', b), ('ONSET', o)]:
        print(f'\n=== Activation distribution: {head_name} head ===')
        _stats('true_downbeat', head_acts[buckets == 2])
        _stats('true_beat_only', head_acts[buckets == 1])
        _stats('true_not_beat', head_acts[buckets == 0])

    # Diagnostic summary: report the median gaps that matter
    print()
    print('=== Discrimination scores (higher = better head differentiation) ===')
    def median_or_nan(a):
        return float(np.median(a)) if len(a) else float('nan')

    # The downbeat head should distinguish true_downbeat from true_beat_only.
    d_db = median_or_nan(d[buckets == 2])
    d_bo = median_or_nan(d[buckets == 1])
    d_nb = median_or_nan(d[buckets == 0])
    print(f'  DOWNBEAT head: median(downbeat)={d_db:.3f}  '
          f'median(beat_only)={d_bo:.3f}  median(not_beat)={d_nb:.3f}')
    print(f'    discrimination = downbeat - beat_only = {d_db - d_bo:+.3f}'
          f'  (positive = head IS specific to downbeats vs beats)')

    # The beat head should distinguish true_beat (downbeat OR beat_only)
    # from true_not_beat.
    is_beat = buckets >= 1
    b_beat = median_or_nan(b[is_beat])
    b_nb = median_or_nan(b[buckets == 0])
    print(f'  BEAT head: median(beat_or_downbeat)={b_beat:.3f}  '
          f'median(not_beat)={b_nb:.3f}')
    print(f'    discrimination = beat - not_beat = {b_beat - b_nb:+.3f}'
          f'  (positive = head fires more on beats than non-beats)')

    # The onset head should fire on everything that's an onset. osu
    # ground truth doesn't label non-beat onsets, so this one has the
    # weakest test, but onset > 0 on at least the beat frames should
    # be a baseline.
    o_beat = median_or_nan(o[is_beat])
    o_nb = median_or_nan(o[buckets == 0])
    print(f'  ONSET head: median(beat_or_downbeat)={o_beat:.3f}  '
          f'median(not_beat)={o_nb:.3f}')
    print(f'    discrimination = onset(beat) - onset(non-beat) = '
          f'{o_beat - o_nb:+.3f}'
          f'  (positive but small expected — non-beat includes onsets)')


if __name__ == '__main__':
    main()
