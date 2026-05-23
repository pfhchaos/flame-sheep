#!/usr/bin/env python3
"""Render the top N and bottom N genomes by CNN score for visual inspection.

Outputs to two subdirectories with rank-prefixed filenames so the file
manager sorts them visually (rank 01 = highest scored in top/, lowest
scored in bottom/).

Usage:
    python tools/render_top_bottom.py
    python tools/render_top_bottom.py --n 30 --output ~/renders/inspection
"""
from __future__ import annotations

import argparse
import io
import shutil
import sqlite3
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flame_sheep.storage import _db_path


def render_one(conn, genome_id: int, output_dir: Path, prefix: str) -> bool:
    """Render a single genome. Files prefixed for visual sorting."""
    row = conn.execute(
        '''SELECT render_static FROM genome_blobs WHERE genome_id = ?''',
        (genome_id,)
    ).fetchone()
    if row is None or row[0] is None:
        return False
    img = Image.open(io.BytesIO(row[0]))
    img.save(output_dir / f'{prefix}.png')
    return True


def main():
    parser = argparse.ArgumentParser(description='Render top/bottom N by CNN score')
    parser.add_argument('--n', type=int, default=20,
                        help='How many top and bottom genomes to render')
    parser.add_argument('--output', type=Path,
                        default=Path.home() / 'renders' / 'inspection',
                        help='Base output directory')
    args = parser.parse_args()

    out_top = args.output / 'top'
    out_bot = args.output / 'bottom'

    # Clean previous renders to avoid stale files
    for d in (out_top, out_bot):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(_db_path()))

    # Top N by score
    top_rows = conn.execute('''
        SELECT id, cnn_score FROM genomes
         WHERE COALESCE(archived, 0) = 0 AND cnn_score IS NOT NULL
         ORDER BY cnn_score DESC LIMIT ?
    ''', (args.n,)).fetchall()

    # Bottom N by score
    bot_rows = conn.execute('''
        SELECT id, cnn_score FROM genomes
         WHERE COALESCE(archived, 0) = 0 AND cnn_score IS NOT NULL
         ORDER BY cnn_score ASC LIMIT ?
    ''', (args.n,)).fetchall()

    print(f'Rendering top {len(top_rows)} → {out_top}')
    for rank, (gid, score) in enumerate(top_rows, 1):
        # rank=01 is highest score; rank=20 is N-th highest
        prefix = f'rank_{rank:02d}_score_{score:.3f}_id_{gid}'
        ok = render_one(conn, gid, out_top, prefix)
        print(f'  {prefix}: {"ok" if ok else "MISSING RENDER"}')

    print(f'Rendering bottom {len(bot_rows)} → {out_bot}')
    for rank, (gid, score) in enumerate(bot_rows, 1):
        # rank=01 is lowest score; rank=20 is N-th lowest
        prefix = f'rank_{rank:02d}_score_{score:.3f}_id_{gid}'
        ok = render_one(conn, gid, out_bot, prefix)
        print(f'  {prefix}: {"ok" if ok else "MISSING RENDER"}')

    conn.close()
    print()
    print(f'Done. Inspect: {args.output}')


if __name__ == '__main__':
    main()
