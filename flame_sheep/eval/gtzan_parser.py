"""Parse GTZAN-rhythm annotations (Marchand/Fresnel/Peeters, ISMIR 2015).

Canonical install layout (see evals/HANDOFF_GTZAN.md):
    <root>/genres/<genre>/<genre>.<NNNNN>.wav
    <root>/annotations/beats/<genre>.<NNNNN>.beats
    <root>/annotations/tempo/<genre>.<NNNNN>.bpm

Both halves (audio + annotations) are renamed at install time so a
single `<genre>.<NNNNN>` stem joins all three files; the parser does
no translation.

Beat file format: tab-separated `<time_sec>\t<bar_position>` lines.
`bar_position == 1` marks a downbeat; all rows mark beats.

Skipping rules:
  * reggae.00086 — annotation missing in the upstream tarball.
  * jazz.00054   — standard MIREX exclusion (audio quality has been
                   flagged across published evals; matches mirdata).

Both skips are handled by the collector (`iter_tracks`), not by the
parse function itself — `parse_track()` is a pure file reader.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Hand-curated skip list. Add only when the upstream annotation is
# unusable (missing, malformed) AND a published source documents why.
KNOWN_BAD_TRACKS: frozenset[str] = frozenset({
    'reggae.00086',  # annotation missing upstream
    'jazz.00054',    # standard MIREX exclusion
})


@dataclass
class GtzanTrack:
    """Parsed GTZAN annotation for one track."""
    track_id: str                   # e.g. 'blues.00000'
    audio_path: Path
    beat_times: np.ndarray          # all beats, seconds, sorted
    downbeat_times: np.ndarray      # subset where bar_position == 1
    tempo_bpm: float


def _audio_path(track_id: str, root: Path) -> Path:
    """Resolve audio file under `<root>/genres/<genre>/<track_id>.wav`.
    Genre is the prefix of track_id before the first dot."""
    genre = track_id.split('.', 1)[0]
    return root / 'genres' / genre / f'{track_id}.wav'


def parse_track(track_id: str, root: Path) -> GtzanTrack:
    """Read annotations for one track from a canonical GTZAN root.

    Raises FileNotFoundError if any required annotation file is missing,
    ValueError if a file is malformed.
    """
    audio = _audio_path(track_id, root)
    beats_file = root / 'annotations' / 'beats' / f'{track_id}.beats'
    bpm_file = root / 'annotations' / 'tempo' / f'{track_id}.bpm'

    if not beats_file.exists():
        raise FileNotFoundError(beats_file)
    if not bpm_file.exists():
        raise FileNotFoundError(bpm_file)

    # Beats: `time<TAB>bar_pos` (most genres) or just `time` per line
    # (a subset of jazz tracks ship without bar-position annotations).
    # Use loadtxt for the standard 2-column case; if it raises or we
    # see only one column, treat as times-only and set downbeats empty.
    try:
        arr = np.loadtxt(beats_file, delimiter='\t', ndmin=2)
    except ValueError as e:
        raise ValueError(f'{beats_file}: unparseable: {e}') from e
    if arr.size == 0:
        raise ValueError(f'{beats_file}: empty')
    if arr.shape[1] >= 2:
        beat_times = np.ascontiguousarray(arr[:, 0], dtype=np.float64)
        bar_pos = arr[:, 1].astype(np.int64)
        downbeat_times = beat_times[bar_pos == 1]
    else:
        # Times-only file (jazz subset). Beats yes, downbeats unknown.
        beat_times = np.ascontiguousarray(arr[:, 0], dtype=np.float64)
        downbeat_times = np.asarray([], dtype=np.float64)

    # Tempo: single scalar.
    tempo_bpm = float(np.loadtxt(bpm_file))

    return GtzanTrack(
        track_id=track_id,
        audio_path=audio,
        beat_times=beat_times,
        downbeat_times=downbeat_times,
        tempo_bpm=tempo_bpm,
    )


def iter_tracks(root: Path) -> list[GtzanTrack]:
    """Walk a canonical GTZAN root and yield every usable track.

    Usable = annotations parse AND audio file exists. Skips
    `KNOWN_BAD_TRACKS` and any track whose audio file is missing.
    """
    beats_dir = root / 'annotations' / 'beats'
    if not beats_dir.exists():
        return []
    tracks: list[GtzanTrack] = []
    for beats_file in sorted(beats_dir.glob('*.beats')):
        track_id = beats_file.stem  # 'blues.00000'
        if track_id in KNOWN_BAD_TRACKS:
            continue
        try:
            track = parse_track(track_id, root)
        except (FileNotFoundError, ValueError):
            continue
        if not track.audio_path.exists():
            continue
        tracks.append(track)
    return tracks
