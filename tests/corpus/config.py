"""Corpus configuration — load corpus.toml and resolve paths."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CORPUS_TOML = Path(__file__).parent.parent / 'corpus.toml'


@dataclass
class OsuEntry:
    """An osu! beatmap set entry from the corpus config."""
    id: int
    name: str
    tags: list[str] = field(default_factory=list)
    expected_bpm: float = 0.0


@dataclass
class YouTubeEntry:
    """A YouTube clip entry from the corpus config."""
    url: str
    name: str
    start: float | None = None
    duration: float | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class CorpusConfig:
    """Parsed corpus.toml configuration."""
    osu_cache_dir: Path
    osu_beatmaps: list[OsuEntry]
    youtube_cache_dir: Path
    youtube_clips: list[YouTubeEntry]
    musdb18_root: Path | None

    @classmethod
    def load(cls, path: Path = CORPUS_TOML) -> CorpusConfig:
        """Load corpus config from TOML file."""
        with open(path, 'rb') as f:
            data = tomllib.load(f)

        osu = data.get('sources', {}).get('osu', {})
        yt = data.get('sources', {}).get('youtube', {})
        musdb = data.get('sources', {}).get('musdb18', {})

        osu_cache = Path(osu.get('cache_dir', '~/.cache/flame-sheep/corpus/osu')).expanduser()
        yt_cache = Path(yt.get('cache_dir', '~/.cache/flame-sheep/corpus/youtube')).expanduser()
        musdb_root = Path(musdb['root']).expanduser() if 'root' in musdb else None

        osu_entries = [
            OsuEntry(
                id=bm['id'],
                name=bm.get('name', ''),
                tags=bm.get('tags', []),
                expected_bpm=bm.get('expected_bpm', 0.0),
            )
            for bm in osu.get('beatmaps', [])
        ]

        yt_entries = [
            YouTubeEntry(
                url=clip['url'],
                name=clip['name'],
                start=clip.get('start'),
                duration=clip.get('duration'),
                tags=clip.get('tags', []),
            )
            for clip in yt.get('clips', [])
        ]

        return cls(
            osu_cache_dir=osu_cache,
            osu_beatmaps=osu_entries,
            youtube_cache_dir=yt_cache,
            youtube_clips=yt_entries,
            musdb18_root=musdb_root,
        )
