#!/usr/bin/env python3
"""Sort rendered genome images using imv keybindings.

Opens all renders in imv with custom keybindings:
  Right / l  → trash (confirm archive)
  Left  / h  → unfair (restore from archive)
  Space      → skip to next
  q          → quit

Moves images into trash/ or unfair/ subdirectories via imv's exec command.
After sorting, use --apply to update the DB.

Usage:
    python tools/sort_renders.py ~/renders/archived/
    python tools/sort_renders.py ~/renders/archived/ --apply
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def extract_genome_id(filename: str) -> int | None:
    """Extract genome ID from filename like genome_42_render.png."""
    parts = filename.split('_')
    if len(parts) >= 2 and parts[0] == 'genome':
        try:
            return int(parts[1])
        except ValueError:
            pass
    return None


def sort_interactive(render_dir: Path):
    """Sort renders using imv with custom keybindings."""
    trash_dir = render_dir / 'trash'
    unfair_dir = render_dir / 'unfair'
    trash_dir.mkdir(exist_ok=True)
    unfair_dir.mkdir(exist_ok=True)

    # Find all render images (the main RGB render, not channels)
    renders = sorted(render_dir.glob('genome_*_render.png'))
    if not renders:
        print('No render images found')
        return

    print(f'{len(renders)} images to sort')
    print('Controls inside imv:')
    print('  Right / l  → trash (confirm archive)')
    print('  Left  / h  → unfair (restore)')
    print('  Space      → skip')
    print('  q          → quit')

    # Shell script that moves all files for a genome to a target dir
    # $1 = current file path, $2 = target dir
    move_script = render_dir / '_sort_move.sh'
    move_script.write_text(f'''#!/bin/sh
FILE="$1"
DEST="$2"
# Extract genome prefix (genome_42)
BASE=$(basename "$FILE")
PREFIX=$(echo "$BASE" | sed 's/\\(_render\\|_swept\\|_domain\\|_H_palette\\|_S_swept\\|_L_structure\\|_A_emergence\\).png//')
DIR=$(dirname "$FILE")
# Move all related files
for f in "$DIR/${{PREFIX}}"_*.png; do
    [ -f "$f" ] && mv "$f" "$DEST/"
done
''')
    move_script.chmod(0o755)

    # Build imv command with keybindings
    cmd = [
        'imv-wayland',
        '-c', f'bind l exec {move_script} "$imv_current_file" "{trash_dir}"; next',
        '-c', f'bind <Right> exec {move_script} "$imv_current_file" "{trash_dir}"; next',
        '-c', f'bind h exec {move_script} "$imv_current_file" "{unfair_dir}"; next',
        '-c', f'bind <Left> exec {move_script} "$imv_current_file" "{unfair_dir}"; next',
        '-c', 'bind <space> next',
    ] + [str(r) for r in renders]

    subprocess.run(cmd)

    # Clean up
    move_script.unlink(missing_ok=True)

    # Report
    n_trash = len(list(trash_dir.glob('*_render.png')))
    n_unfair = len(list(unfair_dir.glob('*_render.png')))
    n_skipped = len(renders) - n_trash - n_unfair
    print(f'\nResults:')
    print(f'  trash/:  {n_trash} confirmed archives')
    print(f'  unfair/: {n_unfair} to restore')
    print(f'  skipped: {n_skipped}')

    if n_unfair > 0:
        print(f'\nRun with --apply to restore unfair genomes in DB')


def apply_sorts(render_dir: Path):
    """Update DB based on sorted folders."""
    import sqlite3
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from flame_sheep.storage import _db_path

    conn = sqlite3.connect(str(_db_path()))

    unfair_dir = render_dir / 'unfair'
    restored = 0
    if unfair_dir.exists():
        for f in unfair_dir.glob('genome_*_render.png'):
            gid = extract_genome_id(f.name)
            if gid is not None:
                conn.execute('UPDATE genomes SET archived = 0, pruner_checked = 1 WHERE id = ?', (gid,))
                restored += 1

    conn.commit()
    print(f'Restored {restored} genomes from unfair/')

    archived = conn.execute('SELECT COUNT(*) FROM genomes WHERE archived = 1').fetchone()[0]
    active = conn.execute('SELECT COUNT(*) FROM genomes WHERE COALESCE(archived, 0) = 0').fetchone()[0]
    print(f'DB state: {archived} archived, {active} active')
    conn.close()


def main():
    parser = argparse.ArgumentParser(description='Sort rendered genome images')
    parser.add_argument('dir', type=str, help='Directory with rendered images')
    parser.add_argument('--apply', action='store_true',
                        help='Apply sorts to DB (restore unfair/ genomes)')
    args = parser.parse_args()

    render_dir = Path(args.dir)
    if not render_dir.exists():
        print(f'Directory not found: {render_dir}')
        return

    if args.apply:
        apply_sorts(render_dir)
    else:
        sort_interactive(render_dir)


if __name__ == '__main__':
    main()
