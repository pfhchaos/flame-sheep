"""Add palette metrics columns to esheep.db and fill them.

ALTER TABLE sheep ADD palette_mean_step, palette_max_step
  (idempotent — silently ignores duplicate-column error)

Then iterates every row, parses palette (inline or indexed via flam3 library), computes
discontinuity metrics, UPDATEs row. Single pass, ~1 minute for 18K rows.

After this lands the trainer can filter ES via:
    SELECT generation, sheep_id FROM sheep WHERE palette_mean_step <= 0.049
"""

from __future__ import annotations

import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from analyze_es_palette_structure import (
    ESHEEP_DB, PALETTE_LIBRARY, load_palette_library, _parse_inline_palette,
)


def discontinuity(palette: np.ndarray) -> tuple[float, float]:
    diffs = np.linalg.norm(np.diff(palette, axis=0), axis=1)
    return float(diffs.mean()), float(diffs.max())


def ensure_columns(con: sqlite3.Connection) -> None:
    cols = {row[1] for row in con.execute("PRAGMA table_info(sheep)")}
    if 'palette_mean_step' not in cols:
        con.execute("ALTER TABLE sheep ADD COLUMN palette_mean_step REAL")
        print('  added palette_mean_step')
    if 'palette_max_step' not in cols:
        con.execute("ALTER TABLE sheep ADD COLUMN palette_max_step REAL")
        print('  added palette_max_step')
    con.commit()


def main():
    print(f'--- Opening {ESHEEP_DB} ---')
    con = sqlite3.connect(ESHEEP_DB)
    ensure_columns(con)

    print('--- Loading palette library ---')
    library = load_palette_library(PALETTE_LIBRARY)
    print(f'  {len(library)} library palettes')

    print('--- Computing + writing metrics per row ---')
    cur = con.execute(
        "SELECT generation, sheep_id, genome_xml FROM sheep "
        "WHERE genome_xml IS NOT NULL"
    )
    rows = cur.fetchall()
    print(f'  {len(rows)} rows')

    updates = []
    n_skipped = 0
    for gen, sid, xml in rows:
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            n_skipped += 1
            continue
        if '<color index=' in xml:
            pal = _parse_inline_palette(root)
        elif root.get('palette') is not None:
            pidx = int(root.get('palette'))
            if pidx not in library:
                n_skipped += 1
                continue
            pal = library[pidx]
        else:
            n_skipped += 1
            continue
        ms, mx = discontinuity(pal)
        updates.append((ms, mx, gen, sid))

    print(f'  computed: {len(updates)}  skipped: {n_skipped}')
    print('--- Writing UPDATEs ---')
    con.executemany(
        "UPDATE sheep SET palette_mean_step = ?, palette_max_step = ? "
        "WHERE generation = ? AND sheep_id = ?",
        updates,
    )
    con.commit()
    con.close()

    print('--- Verification ---')
    con = sqlite3.connect(ESHEEP_DB)
    rows = con.execute(
        "SELECT COUNT(*), AVG(palette_mean_step), MIN(palette_mean_step), "
        "MAX(palette_mean_step) FROM sheep WHERE palette_mean_step IS NOT NULL"
    ).fetchone()
    print(f'  filled: {rows[0]}  avg: {rows[1]:.4f}  '
          f'min: {rows[2]:.4f}  max: {rows[3]:.4f}')
    for thr in [0.049, 0.085, 0.126]:
        n = con.execute(
            "SELECT COUNT(*) FROM sheep WHERE palette_mean_step <= ?", (thr,)
        ).fetchone()[0]
        print(f'  palette_mean_step ≤ {thr}: {n}')
    con.close()


if __name__ == '__main__':
    main()
