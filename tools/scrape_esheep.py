#!/usr/bin/env python3
"""Scrape Electric Sheep genome archives.

Downloads sheep IDs, ratings, and genome XML from electricsheep.com/archives.
Stores everything in a local SQLite database for analysis and re-rendering.

Be polite: 2-second delay between requests by default.

Usage:
    python tools/scrape_esheep.py [--delay 2.0] [--generation 247]
    python tools/scrape_esheep.py --all
"""

from __future__ import annotations

import argparse
import logging
import re
import sqlite3
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

log = logging.getLogger(__name__)

BASE_URL = 'https://electricsheep.com/archives'

GENERATIONS = [247, 245, 244, 243, 242, 198, 191, 169, 165]

DB_PATH = Path.home() / '.local' / 'share' / 'flame-sheep' / 'esheep.db'

USER_AGENT = 'flame-sheep-research/1.0 (fractal flame renderer; polite scraper)'

DEFAULT_DELAY = 2.0


def _fetch(url: str, delay: float) -> str | None:
    """Fetch a URL with polite delay and error handling."""
    time.sleep(delay)
    req = Request(url, headers={'User-Agent': USER_AGENT})
    try:
        with urlopen(req, timeout=30) as resp:
            return resp.read().decode('utf-8', errors='replace')
    except HTTPError as e:
        if e.code == 404:
            log.warning('404: %s', url)
            return None
        log.error('HTTP %d: %s', e.code, url)
        return None
    except (URLError, TimeoutError) as e:
        log.error('Fetch failed: %s — %s', url, e)
        return None


def _ensure_db(db: sqlite3.Connection) -> None:
    db.executescript('''
        CREATE TABLE IF NOT EXISTS sheep (
            generation  INTEGER NOT NULL,
            sheep_id    INTEGER NOT NULL,
            rating      INTEGER,
            genome_xml  TEXT,
            scraped_at  TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (generation, sheep_id)
        );
        CREATE INDEX IF NOT EXISTS idx_sheep_rating
            ON sheep(rating DESC);
    ''')


def _parse_index_page(html: str) -> list[tuple[int, int]]:
    """Parse a best/page index and return [(sheep_id, rating), ...]."""
    results = []
    # HTML structure per entry:
    #   <a href="../../sheep/52506/view.html"><img ...></a><br>
    #   <font size="-2">192</font>
    for match in re.finditer(
        r'/sheep/(\d+)/view\.html"[^>]*>.*?<font[^>]*>\s*(\d+)\s*</font>',
        html, re.DOTALL
    ):
        sheep_id = int(match.group(1))
        rating = int(match.group(2))
        results.append((sheep_id, rating))

    if not results:
        # Fallback: pair up IDs and ratings found separately
        ids = re.findall(r'/sheep/(\d+)/view\.html', html)
        ratings = re.findall(r'<font[^>]*>\s*(\d+)\s*</font>', html)
        if len(ids) == len(ratings):
            results = [(int(i), int(r)) for i, r in zip(ids, ratings)]

    return results


def _find_next_page(html: str) -> str | None:
    """Find the 'next page' link if present."""
    m = re.search(r'href="([^"]*)"[^>]*>\s*next\s*page', html, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def _fetch_genome(generation: int, sheep_id: int, delay: float) -> str | None:
    """Fetch the genome XML for a specific sheep."""
    import html as html_mod

    url = f'{BASE_URL}/generation-{generation}/sheep/{sheep_id}/genome.html'
    page = _fetch(url, delay)
    if page is None:
        return None

    # The XML is typically HTML-encoded (&#60; for <, &#62; for >)
    # with <br> tags as line separators
    decoded = html_mod.unescape(page)

    # Strip <br> and <br/> tags that were used as line breaks in the display
    decoded = re.sub(r'<br\s*/?>', '\n', decoded)

    # Try to find <flame ...>...</flame> in the decoded content
    m = re.search(r'(<flame\b.*?</flame>)', decoded, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Some older genomes might use self-closing <flame .../> without </flame>
    m = re.search(r'(<flame\b[^>]*>.*?(?:</flame>|<xform[^/]*/>(?:\s|$)+))',
                  decoded, re.DOTALL)
    if m:
        xml = m.group(1).strip()
        if not xml.endswith('</flame>'):
            xml += '\n</flame>'
        return xml

    log.warning('No genome XML found for gen %d sheep %d', generation, sheep_id)
    return None


def scrape_generation(db: sqlite3.Connection, generation: int,
                      delay: float) -> int:
    """Scrape all sheep from a generation. Returns count scraped."""
    log.info('Scraping generation %d...', generation)

    # First, collect all sheep IDs and ratings from index pages
    all_sheep: list[tuple[int, int]] = []
    page_url = f'{BASE_URL}/generation-{generation}/best/page/index.html'

    while page_url:
        html = _fetch(page_url, delay)
        if html is None:
            break

        entries = _parse_index_page(html)
        if not entries:
            log.warning('No entries found on page: %s', page_url)
            break

        all_sheep.extend(entries)
        log.info('  page: %d entries (total so far: %d)', len(entries),
                 len(all_sheep))

        next_link = _find_next_page(html)
        if next_link:
            # Resolve relative URL
            if next_link.startswith('http'):
                page_url = next_link
            else:
                # Relative to current page
                base = page_url.rsplit('/', 1)[0]
                page_url = f'{base}/{next_link}'
        else:
            page_url = None

    log.info('Found %d sheep in generation %d', len(all_sheep), generation)

    # Now fetch genomes for sheep we don't already have
    scraped = 0
    for sheep_id, rating in all_sheep:
        # Check if we already have this one
        existing = db.execute(
            'SELECT genome_xml FROM sheep WHERE generation=? AND sheep_id=?',
            (generation, sheep_id)
        ).fetchone()

        if existing and existing[0]:
            continue  # Already have genome

        # Insert/update the rating even if we can't get the genome yet
        genome_xml = _fetch_genome(generation, sheep_id, delay)

        db.execute(
            '''INSERT INTO sheep (generation, sheep_id, rating, genome_xml)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(generation, sheep_id)
               DO UPDATE SET rating=excluded.rating,
                             genome_xml=COALESCE(excluded.genome_xml, genome_xml),
                             scraped_at=datetime('now')''',
            (generation, sheep_id, rating, genome_xml),
        )
        db.commit()
        scraped += 1

        if scraped % 25 == 0:
            log.info('  %d/%d genomes fetched', scraped, len(all_sheep))

    log.info('Generation %d: %d new genomes scraped', generation, scraped)
    return scraped


def main():
    parser = argparse.ArgumentParser(description='Scrape Electric Sheep archives')
    parser.add_argument('--delay', type=float, default=DEFAULT_DELAY,
                        help='Seconds between requests (default: %(default)s)')
    parser.add_argument('--generation', type=int,
                        help='Scrape a specific generation')
    parser.add_argument('--all', action='store_true',
                        help='Scrape all known generations')
    parser.add_argument('--db', type=str, default=str(DB_PATH),
                        help='Database path (default: %(default)s)')
    parser.add_argument('--index-only', action='store_true',
                        help='Only scrape index pages (IDs + ratings), skip genomes')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(db_path))
    _ensure_db(db)

    if args.generation:
        generations = [args.generation]
    elif args.all:
        generations = GENERATIONS
    else:
        parser.error('Specify --generation N or --all')

    total = 0
    for gen in generations:
        try:
            n = scrape_generation(db, gen, args.delay)
            total += n
        except Exception:
            log.exception('Failed on generation %d', gen)

    log.info('Done: %d total genomes scraped', total)
    db.close()


if __name__ == '__main__':
    main()
