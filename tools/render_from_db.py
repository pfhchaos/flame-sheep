#!/usr/bin/env python3
"""Render fractal images from stored histogram blobs in the DB.

Shows what the CNN scorer sees. Outputs PNG images from the 4-channel
domain-native representation (H/S/L/A) or the raw tonemapped render.

Usage:
    python tools/render_from_db.py 42                    # single genome
    python tools/render_from_db.py 42 103 285             # multiple genomes
    python tools/render_from_db.py --archived             # all archived genomes
    python tools/render_from_db.py 42 --output ~/renders/ # custom output dir
    python tools/render_from_db.py 42 --channels          # save per-channel images
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def render_genome(genome_id: int, conn, output_dir: Path,
                  show_channels: bool = False) -> bool:
    """Render a genome from its DB blobs. Returns True if successful."""
    row = conn.execute(
        '''SELECT render_static, render_swept,
                  hist_static, hist_swept, hist_first_hit
           FROM genomes WHERE id = ?''',
        (genome_id,)
    ).fetchone()

    if row is None:
        print(f'  #{genome_id}: not found')
        return False

    render_static, render_swept, hist_static, hist_swept, hist_first_hit = row

    # Save the raw PNG render if available
    if render_static:
        img = Image.open(io.BytesIO(render_static))
        img.save(output_dir / f'genome_{genome_id}_render.png')

    if render_swept:
        img = Image.open(io.BytesIO(render_swept))
        img.save(output_dir / f'genome_{genome_id}_swept.png')

    # Build domain-native channels from histograms
    if hist_static and hist_swept:
        from flame_sheep.scoring_channels import (
            unpack_static_histogram, unpack_histogram, normalize_channels,
        )

        hits, colors = unpack_static_histogram(hist_static)
        swept = unpack_histogram(hist_swept)
        first_hit = unpack_histogram(hist_first_hit, dtype=np.uint8) if hist_first_hit else None

        channels = normalize_channels(hits, colors, swept, first_hit,
                                      output_size=hits.shape[0])
        # channels is (4, H, W) float32 [0, 1]

        # Composite: map H/S/L/A to a visible RGB image
        # L as brightness, H as hue (mapped through a simple colormap), S as saturation
        H, S, L, A = channels[0], channels[1], channels[2], channels[3]

        # Simple HSL-ish composite: use L for structure, tint by H
        rgb = np.zeros((channels.shape[1], channels.shape[2], 3), dtype=np.float32)
        # Map H (0-1) to RGB via simple hue wheel
        rgb[:, :, 0] = np.clip(np.abs(H * 6 - 3) - 1, 0, 1) * L
        rgb[:, :, 1] = np.clip(2 - np.abs(H * 6 - 2), 0, 1) * L
        rgb[:, :, 2] = np.clip(2 - np.abs(H * 6 - 4), 0, 1) * L

        composite = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(composite).save(output_dir / f'genome_{genome_id}_domain.png')

        if show_channels:
            names = ['H_palette', 'S_swept', 'L_structure', 'A_emergence']
            for i, name in enumerate(names):
                ch = (channels[i] * 255).astype(np.uint8)
                Image.fromarray(ch).save(
                    output_dir / f'genome_{genome_id}_{name}.png')

        print(f'  #{genome_id}: saved (render + domain + {"channels" if show_channels else "composite"})')
    elif render_static:
        print(f'  #{genome_id}: saved (render only, no histograms)')
    else:
        print(f'  #{genome_id}: no render data')
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description='Render fractals from DB histogram blobs')
    parser.add_argument('ids', nargs='*', type=int, help='Genome IDs to render')
    parser.add_argument('--archived', action='store_true',
                        help='Render all archived genomes')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: ~/renders/)')
    parser.add_argument('--channels', action='store_true',
                        help='Save individual channel images (H/S/L/A)')
    parser.add_argument('--by-reason', action='store_true',
                        help='Sort archived genomes into subdirs by archive_reason')
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else Path.home() / 'renders'
    output_dir.mkdir(parents=True, exist_ok=True)

    import sqlite3
    from flame_sheep.storage import _db_path
    conn = sqlite3.connect(str(_db_path()))

    if args.archived:
        rows = conn.execute(
            'SELECT id, archive_reason FROM genomes WHERE archived = 1 ORDER BY archive_reason, id'
        ).fetchall()
        if args.by_reason:
            # Group into subdirectories by reason
            by_reason: dict[str, list[int]] = {}
            for gid, reason in rows:
                key = reason or 'unknown'
                by_reason.setdefault(key, []).append(gid)
            print(f'Rendering {len(rows)} archived genomes by reason to {output_dir}')
            rendered = 0
            for reason, gids in sorted(by_reason.items()):
                reason_dir = output_dir / reason
                reason_dir.mkdir(parents=True, exist_ok=True)
                print(f'\n  {reason}: {len(gids)} genomes')
                for gid in gids:
                    if render_genome(gid, conn, reason_dir, show_channels=args.channels):
                        rendered += 1
        else:
            ids = [r[0] for r in rows]
            print(f'Rendering {len(ids)} archived genomes to {output_dir}')
            rendered = 0
            for gid in ids:
                if render_genome(gid, conn, output_dir, show_channels=args.channels):
                    rendered += 1
    elif args.ids:
        ids = args.ids
        print(f'Rendering {len(ids)} genomes to {output_dir}')
        rendered = 0
        for gid in ids:
            if render_genome(gid, conn, output_dir, show_channels=args.channels):
                rendered += 1
    else:
        parser.print_help()
        return

    conn.close()
    print(f'\nDone: {rendered} rendered to {output_dir}')


if __name__ == '__main__':
    main()
