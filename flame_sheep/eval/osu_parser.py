"""Parse osu beatmap files and derive beat/downbeat timestamps.

osu .osu format (v14): INI-like sections, the ones we care about are
[General] (audio filename + lead-in) and [TimingPoints]. Each timing
point is `offset_ms, beat_length_ms, meter, sample_set, sample_idx,
volume, uninherited, kiai`.

Uninherited timing points (column 7 = 1, equivalently beat_length > 0)
encode the actual tempo. Inherited timing points (beat_length < 0)
are slider-velocity multipliers for the gameplay engine and have no
musical-timing meaning — we ignore them.

Beat derivation: for each uninherited segment, emit beats at
`offset, offset + beat_length, offset + 2*beat_length, ...` up to the
next uninherited point (or audio end for the last segment). Downbeats
are every meter-th beat counted within the segment (so the segment's
first beat is a downbeat, and downbeats reset on each tempo change —
matches how mappers think about measures across tempo shifts).

osu times are "game time"; the audio file's t=0 corresponds to game
time = AudioLeadIn. Convert to audio time by subtracting lead-in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class TimingPoint:
    offset_ms: float
    beat_length_ms: float  # >0: uninherited (tempo). <0: inherited (sv).
    meter: int             # numerator of time signature
    uninherited: bool


@dataclass
class OsuMap:
    """Parsed osu beatmap. Use `.beat_times(audio_duration_sec)` to get
    the actual beat array; the parser by itself doesn't know the audio
    length."""
    audio_filename: str
    audio_lead_in_ms: int
    timing_points: list[TimingPoint] = field(default_factory=list)

    @property
    def uninherited_points(self) -> list[TimingPoint]:
        return [tp for tp in self.timing_points if tp.uninherited]

    def beat_times(self, audio_duration_sec: float
                    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (beat_times_sec, downbeat_times_sec).

        Both arrays are in audio time (lead-in subtracted) and sorted
        ascending. audio_duration_sec is required because the last
        uninherited segment has no terminator in the file.
        """
        ups = self.uninherited_points
        if not ups:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        audio_duration_ms = audio_duration_sec * 1000.0

        beats_ms: list[float] = []
        downbeats_ms: list[float] = []
        for i, tp in enumerate(ups):
            start_ms = tp.offset_ms
            end_ms = (ups[i + 1].offset_ms if i + 1 < len(ups)
                      else audio_duration_ms)
            if tp.beat_length_ms <= 0:
                # Malformed — uninherited but non-positive beat length.
                continue
            t = start_ms
            beat_idx = 0
            # Guard against absurdly small beat_length producing infinite loops.
            if tp.beat_length_ms < 1.0:
                continue
            while t < end_ms:
                beats_ms.append(t)
                if beat_idx % tp.meter == 0:
                    downbeats_ms.append(t)
                t += tp.beat_length_ms
                beat_idx += 1

        beats_sec = (np.asarray(beats_ms, dtype=np.float64) - self.audio_lead_in_ms) / 1000.0
        downbeats_sec = (np.asarray(downbeats_ms, dtype=np.float64)
                         - self.audio_lead_in_ms) / 1000.0
        # Clip away anything that fell into pre-audio negative time.
        beats_sec = beats_sec[beats_sec >= 0]
        downbeats_sec = downbeats_sec[downbeats_sec >= 0]
        return beats_sec, downbeats_sec


def parse_osu_file(path: Path) -> OsuMap:
    """Parse an .osu file. Raises ValueError if the file lacks an
    AudioFilename or contains no parseable timing points."""
    # osu files are UTF-8 (sometimes with BOM); be forgiving on line endings.
    text = Path(path).read_text(encoding='utf-8-sig', errors='replace')
    lines = text.splitlines()

    section: str | None = None
    audio_filename: str | None = None
    audio_lead_in_ms: int = 0
    timing_points: list[TimingPoint] = []

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith('//'):
            continue
        if line.startswith('[') and line.endswith(']'):
            section = line[1:-1]
            continue

        if section == 'General':
            # "Key: value" — split on first ':' only.
            if ':' not in line:
                continue
            k, _, v = line.partition(':')
            k = k.strip()
            v = v.strip()
            if k == 'AudioFilename':
                audio_filename = v
            elif k == 'AudioLeadIn':
                try:
                    audio_lead_in_ms = int(v)
                except ValueError:
                    audio_lead_in_ms = 0

        elif section == 'TimingPoints':
            # Comma-separated values; older maps may omit trailing columns,
            # so pad/parse defensively.
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 2:
                continue
            try:
                offset_ms = float(parts[0])
                beat_length_ms = float(parts[1])
            except ValueError:
                continue
            meter = 4
            uninherited = beat_length_ms > 0
            if len(parts) >= 3:
                try:
                    meter = int(parts[2])
                    if meter <= 0:
                        meter = 4
                except ValueError:
                    pass
            if len(parts) >= 7:
                try:
                    uninherited = parts[6].strip() == '1'
                except ValueError:
                    pass
            timing_points.append(TimingPoint(
                offset_ms=offset_ms,
                beat_length_ms=beat_length_ms,
                meter=meter,
                uninherited=uninherited,
            ))

    if audio_filename is None:
        raise ValueError(f'{path}: no AudioFilename in [General]')
    if not any(tp.uninherited for tp in timing_points):
        raise ValueError(f'{path}: no uninherited timing points')

    return OsuMap(
        audio_filename=audio_filename,
        audio_lead_in_ms=audio_lead_in_ms,
        timing_points=timing_points,
    )
