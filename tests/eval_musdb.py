#!/usr/bin/env python3
"""
Evaluate rhythm detection pipeline against MUSDB18 separated stems.

Measures the full pipeline: AudioProcessor → TempoTracker → gated events.
Uses stem separation as implicit ground truth for rhythm quality, not drum
transcription accuracy.

Key metrics:
  - Tempo lock: does the tracker find a consistent tempo? At what BPM?
  - Gating ratio: what fraction of raw events survive tempo gating?
  - Rhythmic coherence: tracker's internal confidence (0..1)
  - Discrimination: rhythmic stems (drums, bass) should lock and gate well;
    non-rhythmic stems (vocals, other) should not lock or should gate heavily.

Usage:
  python -m tests.eval_musdb                    # run on 7s sample set
  python -m tests.eval_musdb --root /path/to/musdb18hq --wav  # full HQ set
  python -m tests.eval_musdb --adaptive         # test adaptive bands
"""

import argparse
import sys
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field

import musdb

sys.path.insert(0, '.')

from flame_sheep.audio import SAMPLE_RATE, FFT_SIZE
from flame_sheep.tempo import TempoTracker

# Import from test conftest if available, otherwise define locally
try:
    from tests.conftest import make_processor
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from conftest import make_processor


@dataclass
class StemResult:
    """Full pipeline evaluation for one stem of one track."""
    track: str
    stem: str
    duration: float
    # Raw detector output
    raw_events: int
    raw_by_kind: dict[str, int] = field(default_factory=dict)
    # After tempo gating
    gated_events: int = 0
    gated_by_kind: dict[str, int] = field(default_factory=dict)
    # Tempo tracker state
    locked: bool = False
    bpm: float = 0.0
    confidence: float = 0.0

    @property
    def raw_per_sec(self) -> float:
        return self.raw_events / self.duration if self.duration > 0 else 0.0

    @property
    def gated_per_sec(self) -> float:
        return self.gated_events / self.duration if self.duration > 0 else 0.0

    @property
    def gating_ratio(self) -> float:
        """Fraction of raw events that survived gating. Lower = more filtered."""
        return self.gated_events / self.raw_events if self.raw_events > 0 else 0.0



def resample_to_48k(audio: np.ndarray, orig_sr: int = 44100) -> np.ndarray:
    """Resample audio from orig_sr to 48kHz."""
    if orig_sr == SAMPLE_RATE:
        return audio
    ratio = SAMPLE_RATE / orig_sr
    n_out = int(len(audio) * ratio)
    indices = np.arange(n_out) / ratio
    indices = np.clip(indices.astype(int), 0, len(audio) - 1)
    return audio[indices]


def run_pipeline(pcm_mono: np.ndarray, adaptive: bool = False) -> StemResult:
    """Run full pipeline: detector → tempo tracker → gating. Returns results."""
    proc = make_processor(adaptive=adaptive)
    tracker = TempoTracker()

    # Warmup with silence
    silence = np.zeros(FFT_SIZE, dtype=np.float32)
    for _ in range(15):
        proc.feed(silence)
        proc.process()

    raw_events = []
    gated_events = []
    pos = 0
    while pos < len(pcm_mono):
        chunk = pcm_mono[pos:pos + FFT_SIZE]
        if len(chunk) < FFT_SIZE:
            chunk = np.pad(chunk, (0, FFT_SIZE - len(chunk)))
        proc.feed(chunk)

        timestamp = pos / SAMPLE_RATE
        for e in proc.process():
            raw_events.append((e.kind, e.energy, timestamp))
            # Run through tempo gating
            if tracker.process_onset(e.kind, timestamp):
                gated_events.append((e.kind, e.energy, timestamp))
        pos += FFT_SIZE

    raw_by_kind = defaultdict(int)
    for kind, _, _ in raw_events:
        raw_by_kind[kind] += 1

    gated_by_kind = defaultdict(int)
    for kind, _, _ in gated_events:
        gated_by_kind[kind] += 1

    return StemResult(
        track='', stem='',
        duration=len(pcm_mono) / SAMPLE_RATE,
        raw_events=len(raw_events),
        raw_by_kind=dict(raw_by_kind),
        gated_events=len(gated_events),
        gated_by_kind=dict(gated_by_kind),
        locked=tracker._locked,
        bpm=tracker._bpm,
        confidence=tracker._confidence,
    )


def evaluate_track(track, adaptive: bool = False) -> dict[str, StemResult]:
    """Run full pipeline on each stem of a track."""
    results = {}

    stems = {
        'drums':  track.targets['drums'].audio,
        'vocals': track.targets['vocals'].audio,
        'bass':   track.targets['bass'].audio,
        'other':  track.targets['other'].audio,
        'mix':    track.audio,
    }

    for stem_name, audio in stems.items():
        audio_48k = resample_to_48k(audio, orig_sr=track.rate)
        mono = audio_48k.mean(axis=1).astype(np.float32) if audio_48k.ndim > 1 else audio_48k.astype(np.float32)

        result = run_pipeline(mono, adaptive=adaptive)
        result.track = track.name
        result.stem = stem_name
        results[stem_name] = result

    return results


def print_report(all_results: list[dict[str, StemResult]]):
    """Print aggregate rhythm evaluation report."""
    n_tracks = len(all_results)

    # Aggregate per-stem
    agg = {}
    for stem_name in ['drums', 'bass', 'vocals', 'other', 'mix']:
        results = [r[stem_name] for r in all_results]
        agg[stem_name] = {
            'raw': sum(r.raw_events for r in results),
            'gated': sum(r.gated_events for r in results),
            'duration': sum(r.duration for r in results),
            'locked': sum(1 for r in results if r.locked),
            'avg_bpm': np.mean([r.bpm for r in results if r.bpm > 0]) if any(r.bpm > 0 for r in results) else 0,
            'avg_confidence': np.mean([r.confidence for r in results]),
            'gating_ratios': [r.gating_ratio for r in results],
        }

    print(f'\n{"=" * 78}')
    print(f'MUSDB18 Rhythm Pipeline Evaluation ({n_tracks} tracks)')
    print(f'{"=" * 78}')

    print(f'\n  {"STEM":8s}  {"RAW/s":>6s}  {"GATED/s":>7s}  {"GATE%":>6s}  '
          f'{"LOCKED":>7s}  {"BPM":>6s}  {"CONF":>5s}')
    print(f'  {"-"*8}  {"-"*6}  {"-"*7}  {"-"*6}  {"-"*7}  {"-"*6}  {"-"*5}')

    for stem_name in ['drums', 'bass', 'vocals', 'other', 'mix']:
        a = agg[stem_name]
        dur = a['duration']
        raw_ps = a['raw'] / dur if dur > 0 else 0
        gated_ps = a['gated'] / dur if dur > 0 else 0
        gate_pct = a['gated'] / a['raw'] * 100 if a['raw'] > 0 else 0
        lock_pct = a['locked'] / n_tracks * 100
        print(f'  {stem_name:8s}  {raw_ps:6.2f}  {gated_ps:7.2f}  {gate_pct:5.1f}%  '
              f'{lock_pct:5.1f}%  {a["avg_bpm"]:6.1f}  {a["avg_confidence"]:.3f}')

    # Rhythm discrimination metrics
    drums_a = agg['drums']
    bass_a = agg['bass']
    vocals_a = agg['vocals']
    other_a = agg['other']

    rhythmic_dur = drums_a['duration'] + bass_a['duration']
    non_rhythmic_dur = vocals_a['duration'] + other_a['duration']

    rhythmic_gated_ps = (drums_a['gated'] + bass_a['gated']) / rhythmic_dur if rhythmic_dur > 0 else 0
    non_rhythmic_gated_ps = (vocals_a['gated'] + other_a['gated']) / non_rhythmic_dur if non_rhythmic_dur > 0 else 0

    rhythmic_lock_rate = (drums_a['locked'] + bass_a['locked']) / (2 * n_tracks) * 100
    non_rhythmic_lock_rate = (vocals_a['locked'] + other_a['locked']) / (2 * n_tracks) * 100

    print(f'\n  DISCRIMINATION')
    print(f'    Rhythmic stems (drums+bass):')
    print(f'      Gated events/sec:  {rhythmic_gated_ps:.2f}')
    print(f'      Lock rate:         {rhythmic_lock_rate:.1f}%')
    print(f'    Non-rhythmic stems (vocals+other):')
    print(f'      Gated events/sec:  {non_rhythmic_gated_ps:.2f}')
    print(f'      Lock rate:         {non_rhythmic_lock_rate:.1f}%')

    if non_rhythmic_gated_ps > 0:
        ratio = rhythmic_gated_ps / non_rhythmic_gated_ps
        print(f'    Gated event ratio:   {ratio:.2f}x  (higher = better)')
    else:
        print(f'    Gated event ratio:   inf  (perfect)')

    # Gating effectiveness: how much does gating help discrimination?
    raw_rhythmic_ps = (drums_a['raw'] + bass_a['raw']) / rhythmic_dur if rhythmic_dur > 0 else 0
    raw_non_rhythmic_ps = (vocals_a['raw'] + other_a['raw']) / non_rhythmic_dur if non_rhythmic_dur > 0 else 0
    raw_ratio = raw_rhythmic_ps / raw_non_rhythmic_ps if raw_non_rhythmic_ps > 0 else float('inf')

    print(f'\n  GATING EFFECTIVENESS')
    print(f'    Raw event ratio (before gating):   {raw_ratio:.2f}x')
    if non_rhythmic_gated_ps > 0:
        print(f'    Gated event ratio (after gating):  {ratio:.2f}x')
        print(f'    Improvement:                       {ratio / raw_ratio:.2f}x')
    else:
        print(f'    Gated event ratio (after gating):  inf')

    # Per-track details for interesting cases
    print(f'\n  TRACK DETAILS')
    print(f'    {"Track":<40s}  {"D.lock":>6s} {"V.lock":>6s} {"D.gate%":>7s} {"V.gate%":>7s} {"D.bpm":>6s}')
    print(f'    {"-"*40}  {"-"*6} {"-"*6} {"-"*7} {"-"*7} {"-"*6}')
    for r in all_results:
        d = r['drums']
        v = r['vocals']
        d_gate = f'{d.gating_ratio*100:.0f}%'
        v_gate = f'{v.gating_ratio*100:.0f}%'
        d_lock = 'YES' if d.locked else 'no'
        v_lock = 'YES' if v.locked else 'no'
        d_bpm = f'{d.bpm:.0f}' if d.bpm > 0 else '-'
        name = d.track[:40]
        print(f'    {name:<40s}  {d_lock:>6s} {v_lock:>6s} {d_gate:>7s} {v_gate:>7s} {d_bpm:>6s}')

    print(f'{"=" * 78}')


def main():
    parser = argparse.ArgumentParser(description='Evaluate rhythm pipeline on MUSDB18')
    parser.add_argument('--root', default=None, help='MUSDB18 root directory')
    parser.add_argument('--wav', action='store_true', help='Use WAV (HQ) format')
    parser.add_argument('--adaptive', action='store_true', help='Use adaptive bands')
    parser.add_argument('--limit', type=int, default=None, help='Limit to N tracks')
    args = parser.parse_args()

    kwargs = {}
    if args.root:
        kwargs['root'] = args.root
        if args.wav:
            kwargs['is_wav'] = True
    else:
        kwargs['download'] = True

    mus = musdb.DB(**kwargs)
    tracks = list(mus)
    if args.limit:
        tracks = tracks[:args.limit]

    print(f'Evaluating {len(tracks)} tracks '
          f'({"adaptive" if args.adaptive else "static"} bands)...')

    all_results = []
    for i, track in enumerate(tracks):
        print(f'  [{i+1}/{len(tracks)}] {track.name}...', end='', flush=True)
        results = evaluate_track(track, adaptive=args.adaptive)
        d, v = results['drums'], results['vocals']
        print(f'  D:{d.gated_events}/{d.raw_events} V:{v.gated_events}/{v.raw_events}'
              f'  lock:{"Y" if d.locked else "n"}/{"Y" if v.locked else "n"}')
        all_results.append(results)

    print_report(all_results)


if __name__ == '__main__':
    main()
