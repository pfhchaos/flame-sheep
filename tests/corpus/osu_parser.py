"""Parse osu! beatmap files (.osu format).

Extracts timing points (BPM, meter) and hit objects (beat positions)
from the plaintext .osu format. Only reads the fields we need for
beat accuracy evaluation.

Reference: https://osu.ppy.sh/wiki/en/Client/File_formats/osu_(file_format)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TimingPoint:
    """A timing point from the [TimingPoints] section."""
    time_ms: float
    beat_length: float   # ms per beat (uninherited) or velocity multiplier (inherited)
    meter: int           # beats per measure
    uninherited: bool    # True = defines BPM, False = inherited (slider velocity)

    @property
    def bpm(self) -> float:
        """BPM for uninherited timing points. Returns 0 for inherited."""
        if not self.uninherited or self.beat_length <= 0:
            return 0.0
        return 60000.0 / self.beat_length


@dataclass
class HitObject:
    """A hit object from the [HitObjects] section."""
    time_ms: float
    hit_type: int        # bitmask: 1=circle, 2=slider, 8=spinner
    hit_sound: int       # bitmask: 1=normal, 2=whistle, 4=finish, 8=clap

    @property
    def is_circle(self) -> bool:
        return bool(self.hit_type & 1)

    @property
    def is_slider(self) -> bool:
        return bool(self.hit_type & 2)

    @property
    def is_spinner(self) -> bool:
        return bool(self.hit_type & 8)


@dataclass
class OsuBeatmap:
    """Parsed osu! beatmap with timing and hit object data."""
    title: str = ''
    artist: str = ''
    audio_filename: str = ''
    timing_points: list[TimingPoint] = field(default_factory=list)
    hit_objects: list[HitObject] = field(default_factory=list)

    @property
    def bpm(self) -> float:
        """BPM from the first uninherited timing point."""
        for tp in self.timing_points:
            if tp.uninherited and tp.bpm > 0:
                return tp.bpm
        return 0.0

    @property
    def onset_times(self) -> list[float]:
        """Hit object times in seconds, sorted."""
        times = [ho.time_ms / 1000.0 for ho in self.hit_objects
                 if not ho.is_spinner]  # exclude spinners (sustained, not onset)
        return sorted(times)

    @property
    def has_tempo_changes(self) -> bool:
        """True if multiple uninherited timing points with different BPM."""
        bpms = [tp.bpm for tp in self.timing_points if tp.uninherited and tp.bpm > 0]
        return len(set(round(b, 1) for b in bpms)) > 1


def parse_osu(text: str) -> OsuBeatmap:
    """Parse a .osu file from its text content."""
    beatmap = OsuBeatmap()
    section = ''

    for line in text.split('\n'):
        line = line.strip()
        if not line or line.startswith('//'):
            continue

        # Section headers
        if line.startswith('[') and line.endswith(']'):
            section = line[1:-1]
            continue

        # Metadata
        if section == 'General':
            if line.startswith('AudioFilename:'):
                beatmap.audio_filename = line.split(':', 1)[1].strip()

        elif section == 'Metadata':
            if line.startswith('Title:'):
                beatmap.title = line.split(':', 1)[1].strip()
            elif line.startswith('Artist:'):
                beatmap.artist = line.split(':', 1)[1].strip()

        elif section == 'TimingPoints':
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    time_ms = float(parts[0])
                    beat_length = float(parts[1])
                    meter = int(parts[2]) if len(parts) > 2 else 4
                    uninherited = bool(int(parts[6])) if len(parts) > 6 else True
                    beatmap.timing_points.append(TimingPoint(
                        time_ms=time_ms,
                        beat_length=beat_length,
                        meter=meter,
                        uninherited=uninherited,
                    ))
                except (ValueError, IndexError):
                    continue

        elif section == 'HitObjects':
            parts = line.split(',')
            if len(parts) >= 5:
                try:
                    time_ms = float(parts[2])
                    hit_type = int(parts[3])
                    hit_sound = int(parts[4])
                    beatmap.hit_objects.append(HitObject(
                        time_ms=time_ms,
                        hit_type=hit_type,
                        hit_sound=hit_sound,
                    ))
                except (ValueError, IndexError):
                    continue

    return beatmap


def parse_osu_file(path: Path) -> OsuBeatmap:
    """Parse a .osu file from disk."""
    return parse_osu(path.read_text(encoding='utf-8'))
