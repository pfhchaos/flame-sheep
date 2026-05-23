#!/usr/bin/env python3
"""Direct upvote / downvote of genomes by ID.

Upvoting automatically triggers the thumbs-up breeder (~7 jittered
children) per Library.rate's existing behavior. Downvoting does not
breed.

Usage:
    python tools/vote.py up 2456                  # single upvote
    python tools/vote.py up 2456 127 357          # multi-upvote
    python tools/vote.py down 768 22 1853         # multi-downvote
    python tools/vote.py up 2456 --dry-run        # show what would happen
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flame_sheep.storage import Library


def main():
    parser = argparse.ArgumentParser(description='Upvote or downvote genomes by ID')
    parser.add_argument('direction', choices=['up', 'down'],
                        help='up = +1 (triggers breeding), down = -1')
    parser.add_argument('ids', type=int, nargs='+',
                        help='One or more genome IDs')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would happen without recording votes')
    args = parser.parse_args()

    rating = +1 if args.direction == 'up' else -1
    direction_word = 'upvote' if rating > 0 else 'downvote'

    lib = Library()
    try:
        # Sanity check: confirm all IDs exist as active genomes
        for gid in args.ids:
            row = lib.conn.execute(
                'SELECT COALESCE(archived, 0), cnn_score FROM genomes WHERE id = ?',
                (gid,),
            ).fetchone()
            if row is None:
                print(f'  #{gid}: NOT FOUND in DB (skipping)')
                continue
            archived, score = row
            arch_str = ' [archived]' if archived else ''
            score_str = f'score={score:.3f}' if score is not None else 'unscored'
            print(f'  #{gid}: {direction_word} ({score_str}{arch_str})')
            if args.dry_run:
                continue
            lib.rate('genome', gid, rating)

        if args.dry_run:
            print(f'\nDry run — no votes recorded. Re-run without --dry-run to apply.')
        else:
            print(f'\nRecorded {len(args.ids)} {direction_word}(s).')
            if rating > 0:
                print('Thumbs-up breeder will have queued ~7 children per upvote '
                      '(check logs for [thumbsup-breed] entries).')
    finally:
        lib.close()


if __name__ == '__main__':
    main()
