#!/usr/bin/env python3
"""Scrape Electric Sheep reference images and store resized in DB.

Downloads the official rendered PNG for each sheep, resizes to a target
resolution, and stores as a BLOB in the esheep database. These serve as
ground truth for comparing our renders against what voters actually saw.

Usage:
    python tools/scrape_esheep_images.py --size 512
    python tools/scrape_esheep_images.py --size 512 --generation 247 --delay 1.0
"""

from __future__ import annotations

import argparse
import io
import logging
import sqlite3
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

BASE_URL = 'https://electricsheep.com/archives'
DB_PATH = Path.home() / '.local' / 'share' / 'flame-sheep' / 'esheep.db'
USER_AGENT = 'flame-sheep-research/1.0 (fractal flame renderer; polite scraper)'
DEFAULT_DELAY = 1.0
DEFAULT_SIZE = 512


def _ensure_image_column(db: sqlite3.Connection) -> None:
    cols = {r[1] for r in db.execute('PRAGMA table_info(sheep)').fetchall()}
    if 'reference_image' not in cols:
        db.execute('ALTER TABLE sheep ADD COLUMN reference_image BLOB')
        db.commit()
        log.info('Added reference_image column')


def _fetch_image(generation: int, sheep_id: int, delay: float) -> bytes | None:
    time.sleep(delay)
    url = f'{BASE_URL}/generation-{generation}/{sheep_id}/electricsheep.{generation}.{sheep_id}.png'
    req = Request(url, headers={'User-Agent': USER_AGENT})
    try:
        with urlopen(req, timeout=30) as resp:
            return resp.read()
    except HTTPError as e:
        if e.code == 404:
            log.debug('404: gen %d sheep %d', generation, sheep_id)
            return None
        log.warning('HTTP %d: gen %d sheep %d', e.code, generation, sheep_id)
        return None
    except (URLError, TimeoutError) as e:
        log.warning('Fetch failed: gen %d sheep %d — %s', generation, sheep_id, e)
        return None


def _resize_to_square(image_data: bytes, size: int) -> bytes:
    """Resize image to size×size, preserving aspect ratio with black padding."""
    img = Image.open(io.BytesIO(image_data))
    # Resize preserving aspect ratio
    img.thumbnail((size, size), Image.LANCZOS)
    # Paste onto black square
    square = Image.new('RGB', (size, size), (0, 0, 0))
    offset_x = (size - img.width) // 2
    offset_y = (size - img.height) // 2
    square.paste(img, (offset_x, offset_y))
    buf = io.BytesIO()
    square.save(buf, format='PNG', optimize=True)
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(description='Scrape Electric Sheep reference images')
    parser.add_argument('--delay', type=float, default=DEFAULT_DELAY,
                        help='Seconds between requests (default: %(default)s)')
    parser.add_argument('--size', type=int, default=DEFAULT_SIZE,
                        help='Target image size (default: %(default)s)')
    parser.add_argument('--generation', type=int, default=None,
                        help='Only scrape a specific generation')
    parser.add_argument('--db', type=str, default=str(DB_PATH))
    parser.add_argument('--limit', type=int, default=None,
                        help='Max images to fetch')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )

    db = sqlite3.connect(args.db)
    _ensure_image_column(db)

    # Find sheep without reference images
    query = '''SELECT generation, sheep_id FROM sheep
               WHERE genome_xml IS NOT NULL
               AND reference_image IS NULL'''
    params: list = []
    if args.generation is not None:
        query += ' AND generation = ?'
        params.append(args.generation)
    query += ' ORDER BY rating DESC'
    if args.limit:
        query += f' LIMIT {args.limit}'

    rows = db.execute(query, params).fetchall()
    log.info('Found %d sheep without reference images', len(rows))

    fetched = 0
    failed = 0
    for generation, sheep_id in rows:
        image_data = _fetch_image(generation, sheep_id, args.delay)
        if image_data is None:
            failed += 1
            # Store empty blob so we don't retry 404s
            db.execute('UPDATE sheep SET reference_image=? WHERE generation=? AND sheep_id=?',
                       (b'', generation, sheep_id))
            db.commit()
            continue

        resized = _resize_to_square(image_data, args.size)
        db.execute('UPDATE sheep SET reference_image=? WHERE generation=? AND sheep_id=?',
                   (resized, generation, sheep_id))
        db.commit()
        fetched += 1

        if fetched % 25 == 0:
            elapsed_est = fetched * args.delay
            log.info('Fetched %d/%d images (%d failed)', fetched, len(rows), failed)

    log.info('Done: %d fetched, %d failed, %d total', fetched, failed, len(rows))
    db.close()


if __name__ == '__main__':
    main()
