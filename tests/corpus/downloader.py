"""Corpus downloader — fetch osu! beatmaps and YouTube clips.

osu! beatmaps: downloads .osz (ZIP), extracts .osu metadata + audio.
YouTube clips: downloads via yt-dlp, clips to specified range.

All audio cached to ~/.cache/flame-sheep/corpus/ (gitignored).
Only the corpus.toml config goes in the repo.
"""

from __future__ import annotations

import io
import logging
import shutil
import subprocess
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_CACHE = Path('~/.cache/flame-sheep/corpus').expanduser()


def download_osz(beatmapset_id: int, cache_dir: Path = DEFAULT_CACHE) -> Path:
    """Download an osu! beatmap set (.osz) and extract to cache.

    Returns the directory containing .osu files and audio.
    Skips download if already cached.
    """
    import requests

    dest = cache_dir / 'osu' / str(beatmapset_id)
    if dest.exists() and any(dest.glob('*.osu')):
        log.debug(f'osu! {beatmapset_id}: already cached at {dest}')
        return dest

    dest.mkdir(parents=True, exist_ok=True)

    # Try mirrors in order
    urls = [
        f'https://catboy.best/d/{beatmapset_id}n',
        f'https://api.chimu.moe/v1/download/{beatmapset_id}',
    ]

    for url in urls:
        try:
            log.info(f'osu! {beatmapset_id}: downloading from {url}')
            resp = requests.get(url, timeout=30, stream=True)
            if resp.status_code == 200:
                data = resp.content
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    z.extractall(dest)
                log.info(f'osu! {beatmapset_id}: extracted to {dest}')
                return dest
            else:
                log.debug(f'  {url}: HTTP {resp.status_code}')
        except Exception as e:
            log.debug(f'  {url}: {e}')

    raise RuntimeError(f'Failed to download osu! beatmapset {beatmapset_id}')


def download_youtube(url: str, name: str, cache_dir: Path = DEFAULT_CACHE,
                     start: float | None = None,
                     duration: float | None = None) -> Path:
    """Download a YouTube clip via yt-dlp and extract audio.

    Returns path to the downloaded audio file.
    Skips download if already cached.
    """
    dest_dir = cache_dir / 'youtube'
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f'{name}.wav'

    if dest.exists():
        log.debug(f'YouTube "{name}": already cached at {dest}')
        return dest

    if not shutil.which('yt-dlp'):
        raise RuntimeError('yt-dlp not found — install with: pip install yt-dlp')

    log.info(f'YouTube "{name}": downloading from {url}')

    # Build yt-dlp command
    cmd = [
        'yt-dlp',
        '-x',                          # extract audio
        '--audio-format', 'wav',       # convert to WAV
        '-o', str(dest_dir / f'{name}.%(ext)s'),
    ]

    # Add time range if specified
    if start is not None or duration is not None:
        sections = ''
        s = start or 0
        if duration is not None:
            sections = f'*{s}-{s + duration}'
        else:
            sections = f'*{s}-inf'
        cmd.extend(['--download-sections', sections])
        cmd.extend(['--force-keyframes-at-cuts'])

    cmd.append(url)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        log.error(f'yt-dlp failed: {result.stderr[:500]}')
        raise RuntimeError(f'yt-dlp failed for "{name}"')

    # yt-dlp might produce slightly different filename
    candidates = list(dest_dir.glob(f'{name}.*'))
    if not candidates:
        raise RuntimeError(f'No output file found for "{name}"')

    actual = candidates[0]
    if actual != dest:
        actual.rename(dest)

    log.info(f'YouTube "{name}": saved to {dest}')
    return dest


def find_osu_files(beatmap_dir: Path) -> list[Path]:
    """Find all .osu files in a beatmap directory."""
    return sorted(beatmap_dir.glob('*.osu'))


def find_audio_file(beatmap_dir: Path, audio_filename: str) -> Path | None:
    """Find the audio file in a beatmap directory."""
    audio = beatmap_dir / audio_filename
    if audio.exists():
        return audio
    # Try case-insensitive match
    for f in beatmap_dir.iterdir():
        if f.name.lower() == audio_filename.lower():
            return f
    return None
