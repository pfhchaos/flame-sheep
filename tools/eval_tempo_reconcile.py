#!/usr/bin/env python3
"""Evaluate tempo reconciliation strategies against MUSDB18.

Compares three conditions per track:
  * baseline  — BTrack output as-is (current production behavior)
  * override  — strategy A: snap to beat_rate when off-octave, lose
                tracker precision in exchange for instant correction
  * shift     — strategy B: shift BTrack output by best octave ratio
                (1/3, 1/2, 1, 2, 3) so it matches beat_rate's octave,
                preserving tracker precision

Reference BPM comes from `librosa.beat.beat_track` on the drums stem
(matches tools/eval_tempo.py's reference convention). Accuracy is
reported as exact (±4%) and octave (±4% allowing 2x / 1x / 0.5x).

Usage:
    python tools/eval_tempo_reconcile.py --root ~/MUSDB18/MUSDB18-7
    python tools/eval_tempo_reconcile.py --root ~/MUSDB18/MUSDB18-7 --limit 20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flame_sheep_audio import SAMPLE_RATE, HOP_SIZE
from flame_sheep_audio._cqt_engine import CqtEngine
from flame_sheep_audio.beat_rnn import load_beat_rnn
from flame_sheep_audio.tempo_btrack import BTrackTempoTracker
from flame_sheep_audio.tempo_reconcile import (
    BeatRateEstimator,
    reconcile_tempo_override,
    reconcile_tempo_octave_shift,
    reconcile_tempo_octave_only,
)
from flame_sheep.eval.osu_parser import parse_osu_file


# Match the user's current deployed thresholds for the multidepth model
# (from ~/.config/flame-sheep/audio.toml as of 2026-06-02).
RNN_WEIGHTS = Path('flame_sheep/data/beat_rnn_multidepth.npz')
RNN_DOWNBEAT_TH = 0.80
RNN_BEAT_TH = 0.55
RNN_ONSET_TH = 0.15


def resample_to_48k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == SAMPLE_RATE:
        return audio
    ratio = SAMPLE_RATE / orig_sr
    n_out = int(len(audio) * ratio)
    indices = np.clip((np.arange(n_out) / ratio).astype(int), 0, len(audio) - 1)
    return audio[indices]


def reference_bpm(drums_audio: np.ndarray, sr: int) -> float:
    import librosa
    mono = drums_audio.mean(axis=1) if drums_audio.ndim > 1 else drums_audio
    mono = mono.astype(np.float32)
    tempo, _ = librosa.beat.beat_track(y=mono, sr=sr, units='time')
    if hasattr(tempo, '__len__'):
        return float(tempo[0]) if len(tempo) > 0 else 0.0
    return float(tempo)


def run_pipeline(mix_audio: np.ndarray, orig_sr: int
                 ) -> tuple[float, float, int]:
    """Process audio through BTrack + multidepth beat RNN.

    Returns (tracker_bpm, beat_rate_bpm_or_nan, n_beat_events).
    beat_rate_bpm is NaN if too few beat events were detected to estimate.
    """
    mix_48k = resample_to_48k(mix_audio, orig_sr)
    mono = (mix_48k.mean(axis=1) if mix_48k.ndim > 1
            else mix_48k).astype(np.float32)

    cqt = CqtEngine()
    tracker = BTrackTempoTracker(hop_size=HOP_SIZE, sample_rate=SAMPLE_RATE)
    detector = load_beat_rnn(
        str(RNN_WEIGHTS),
        threshold=0.3,
        downbeat_threshold=RNN_DOWNBEAT_TH,
        beat_threshold=RNN_BEAT_TH,
        onset_threshold=RNN_ONSET_TH,
    )
    estimator = BeatRateEstimator()

    hop_dur = HOP_SIZE / SAMPLE_RATE
    t = 0.0
    n_beats = 0
    for pos in range(0, len(mono) - HOP_SIZE, HOP_SIZE):
        chunk = mono[pos:pos + HOP_SIZE]
        tracker.feed_audio(chunk)
        frame = cqt.push_hop(chunk)
        events = detector.detect(frame)
        for ev in events:
            estimator.feed(ev.kind, t)
            if ev.kind in ('low', 'mid'):
                n_beats += 1
        t += hop_dur

    beat_rate = estimator.beat_rate_bpm
    return (tracker.effective_bpm,
            beat_rate if beat_rate is not None else float('nan'),
            n_beats)


def bpm_within(estimated: float, reference: float, tolerance_pct: float = 4.0,
               octave_ratios: tuple[float, ...] = (1.0,)) -> bool:
    if reference <= 0 or estimated <= 0:
        return False
    for ratio in octave_ratios:
        ref = reference * ratio
        if abs(estimated - ref) / ref * 100 < tolerance_pct:
            return True
    return False


def evaluate_track(track) -> dict:
    t0 = time.time()
    drums = track.targets['drums'].audio
    ref = reference_bpm(drums, track.rate)
    tracker_bpm, beat_rate, n_beats = run_pipeline(track.audio, track.rate)

    # Apply both reconciliation strategies post-hoc.
    if not np.isnan(beat_rate):
        # Re-build estimator with the final state so the reconciler can
        # check stability. We saved beat_rate_bpm only; reconstruct a
        # minimal estimator that's "stable" with that value.
        e = BeatRateEstimator()
        interval = 60.0 / beat_rate
        for i in range(6):
            e.feed('mid', i * interval)
        override_bpm = reconcile_tempo_override(tracker_bpm, e).bpm
        shift_bpm = reconcile_tempo_octave_shift(tracker_bpm, e).bpm
        only_bpm = reconcile_tempo_octave_only(tracker_bpm, e).bpm
    else:
        override_bpm = tracker_bpm
        shift_bpm = tracker_bpm
        only_bpm = tracker_bpm

    return {
        'track': track.name,
        'reference': ref,
        'tracker': tracker_bpm,
        'beat_rate': beat_rate,
        'n_beats': n_beats,
        'override': override_bpm,
        'shift': shift_bpm,
        'only': only_bpm,
        'duration': track.audio.shape[0] / track.rate,
        'elapsed': time.time() - t0,
    }


def _accuracy(results, field: str, octave: bool) -> tuple[int, int]:
    ratios = (0.5, 1.0, 2.0) if octave else (1.0,)
    valid = [r for r in results if r['reference'] > 0]
    n_correct = sum(1 for r in valid
                    if bpm_within(r[field], r['reference'],
                                  octave_ratios=ratios))
    return n_correct, len(valid)


def print_report(results) -> None:
    n_total = len(results)
    valid = [r for r in results if r['reference'] > 0]
    n = len(valid)

    print()
    print('=' * 80)
    print(f'TEMPO RECONCILIATION EVAL — {n_total} tracks ({n} with valid reference)')
    print('=' * 80)

    print()
    print(f'{"":>12}  {"exact (±4%)":>18}  {"octave (±4% of x0.5/x1/x2)":>30}')
    print('-' * 66)
    for field, label in [('tracker', 'baseline'),
                         ('override', 'A: override'),
                         ('shift', 'B: shift'),
                         ('only', 'C: octave-only')]:
        ex, nv = _accuracy(results, field, octave=False)
        oc, _ = _accuracy(results, field, octave=True)
        print(f'  {label:>12}  {ex}/{nv} ({100*ex/nv:>4.0f}%)        '
              f'{oc}/{nv} ({100*oc/nv:>4.0f}%)')

    print()
    print('--- Per-track comparisons (where A and B disagree) ---')
    disagreements = [r for r in valid
                     if abs(r['override'] - r['shift']) > 1.0][:20]
    if disagreements:
        print(f'  {"Track":<38} {"Ref":>5} {"Trk":>5} {"BR":>5} '
              f'{"A":>5} {"B":>5}')
        for r in disagreements:
            br = (f"{r['beat_rate']:>5.0f}" if not np.isnan(r['beat_rate'])
                  else f"{'--':>5}")
            print(f'  {r["track"][:38]:<38} {r["reference"]:>5.0f} '
                  f'{r["tracker"]:>5.0f} {br} '
                  f'{r["override"]:>5.0f} {r["shift"]:>5.0f}')
    else:
        print('  (A and B agree on all tracks)')

    print()
    print('--- Tracks where reconciliation helped vs. hurt ---')
    helped = sum(1 for r in valid
                 if not bpm_within(r['tracker'], r['reference'], octave_ratios=(0.5,1,2))
                 and bpm_within(r['shift'], r['reference'], octave_ratios=(0.5,1,2)))
    hurt = sum(1 for r in valid
               if bpm_within(r['tracker'], r['reference'], octave_ratios=(0.5,1,2))
               and not bpm_within(r['shift'], r['reference'], octave_ratios=(0.5,1,2)))
    print(f'  Strategy B helped:  {helped} (was wrong octave, now correct)')
    print(f'  Strategy B hurt:    {hurt} (was correct, now wrong)')
    helped_a = sum(1 for r in valid
                   if not bpm_within(r['tracker'], r['reference'], octave_ratios=(0.5,1,2))
                   and bpm_within(r['override'], r['reference'], octave_ratios=(0.5,1,2)))
    hurt_a = sum(1 for r in valid
                 if bpm_within(r['tracker'], r['reference'], octave_ratios=(0.5,1,2))
                 and not bpm_within(r['override'], r['reference'], octave_ratios=(0.5,1,2)))
    print(f'  Strategy A helped:  {helped_a}')
    print(f'  Strategy A hurt:    {hurt_a}')

    total_time = sum(r['elapsed'] for r in results)
    total_audio = sum(r['duration'] for r in results)
    print()
    print(f'  Processing: {total_time:.0f}s for {total_audio/60:.0f}min '
          f'audio ({total_audio/max(total_time,1):.1f}x realtime)')
    print('=' * 80)


class _OsuTrack:
    """Adapter so osu tracks fit the same interface MUSDB tracks use."""

    def __init__(self, track_dir: Path):
        self.track_dir = track_dir
        # Pick the FIRST .osu file in the directory — they all share the
        # same audio + tempo (different difficulties share the timing).
        osu_files = sorted(track_dir.glob('*.osu'))
        if not osu_files:
            raise ValueError(f'No .osu files in {track_dir}')
        self._osu = parse_osu_file(osu_files[0])
        self.name = track_dir.name
        # Defer audio load until needed.
        self._audio: np.ndarray | None = None
        self._sr: int | None = None

    def _load_audio(self):
        if self._audio is not None:
            return
        audio_path = self.track_dir / self._osu.audio_filename
        if not audio_path.exists():
            raise FileNotFoundError(f'osu audio missing: {audio_path}')
        # librosa is the existing dep already in eval pipeline
        import librosa
        self._audio, self._sr = librosa.load(str(audio_path), sr=None, mono=True)

    @property
    def audio(self) -> np.ndarray:
        self._load_audio()
        return self._audio

    @property
    def rate(self) -> int:
        self._load_audio()
        return self._sr

    def reference_bpm(self) -> float:
        """Median BPM across uninherited timing points (typically one
        for fixed-tempo songs; medianed for variable-tempo charts).
        Falls back to 0 if no timing points are usable."""
        ups = self._osu.uninherited_points
        if not ups:
            return 0.0
        bpms = [60000.0 / tp.beat_length_ms for tp in ups
                if tp.beat_length_ms > 1.0]
        if not bpms:
            return 0.0
        return float(np.median(bpms))


def _evaluate_osu_track(track: _OsuTrack) -> dict:
    """Same shape as evaluate_track() but uses osu's stored BPM
    instead of librosa-on-drum-stem."""
    t0 = time.time()
    ref = track.reference_bpm()
    tracker_bpm, beat_rate, n_beats = run_pipeline(track.audio, track.rate)

    if not np.isnan(beat_rate):
        e = BeatRateEstimator()
        interval = 60.0 / beat_rate
        for i in range(6):
            e.feed('mid', i * interval)
        override_bpm = reconcile_tempo_override(tracker_bpm, e).bpm
        shift_bpm = reconcile_tempo_octave_shift(tracker_bpm, e).bpm
        only_bpm = reconcile_tempo_octave_only(tracker_bpm, e).bpm
    else:
        override_bpm = tracker_bpm
        shift_bpm = tracker_bpm
        only_bpm = tracker_bpm

    return {
        'track': track.name,
        'reference': ref,
        'tracker': tracker_bpm,
        'beat_rate': beat_rate,
        'n_beats': n_beats,
        'override': override_bpm,
        'shift': shift_bpm,
        'only': only_bpm,
        'duration': len(track.audio) / track.rate,
        'elapsed': time.time() - t0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus', choices=['musdb', 'osu'], default='osu',
                        help='Corpus to evaluate against (default osu)')
    parser.add_argument('--root', type=Path,
                        default=Path.home() / '.cache/flame-sheep/corpus/osu',
                        help='Corpus root directory '
                             '(default ~/.cache/flame-sheep/corpus/osu)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit to N tracks for a quick run')
    parser.add_argument('--wav', action='store_true',
                        help='Use WAV format (MUSDB only)')
    args = parser.parse_args()

    if args.corpus == 'musdb':
        import musdb
        kwargs = {'root': str(args.root)}
        if args.wav:
            kwargs['is_wav'] = True
        mus = musdb.DB(**kwargs)
        tracks = list(mus)
        evaluator = evaluate_track
    else:  # osu
        track_dirs = sorted(d for d in args.root.iterdir() if d.is_dir())
        tracks = []
        for td in track_dirs:
            try:
                tracks.append(_OsuTrack(td))
            except (ValueError, FileNotFoundError) as e:
                print(f'  skip {td.name}: {e}')
        evaluator = _evaluate_osu_track

    if args.limit:
        tracks = tracks[:args.limit]
    print(f'Evaluating {len(tracks)} {args.corpus} tracks...')

    results = []
    for i, track in enumerate(tracks):
        name = track.name if hasattr(track, 'name') else str(track)
        print(f'  [{i+1}/{len(tracks)}] {name[:50]}... ', end='', flush=True)
        try:
            r = evaluator(track)
        except Exception as e:
            print(f'FAILED: {type(e).__name__}: {e}')
            continue
        br = (f'br={r["beat_rate"]:.0f}'
              if not np.isnan(r['beat_rate']) else 'br=--')
        print(f'ref={r["reference"]:.0f} trk={r["tracker"]:.0f} '
              f'{br} A={r["override"]:.0f} B={r["shift"]:.0f} '
              f'({r["elapsed"]:.1f}s)')
        results.append(r)

    print_report(results)


if __name__ == '__main__':
    main()
