#!/usr/bin/env python3
"""One-off: breed children for direct genome upvotes recorded in the
current generation, where the breeder didn't fire at insert time.

Background: the thumbs-up breeder was added to Library.rate() during
this session, but a wallpaper process that was running before the
patch landed would have inserted gen-1 ratings without triggering
the breed. Worse — the same stale wallpaper also missed the
generation-stamping change, so its inserts default to generation=0
even though they're conceptually gen 1.

Two-step recovery:
  1. --retag-since DATE: re-stamp ratings created since DATE from
     gen 0 to current_generation. Use the wallpaper-restart date,
     since everything before that was pre-generational gen 0.
  2. Then walk current-gen direct upvotes on active genomes and call
     breed_thumbsup_children for each.

Not idempotent — running twice will breed another 7 children per
upvoted genome. Intended as a one-off catch-up tool.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flame_sheep.storage import Library


def main():
    parser = argparse.ArgumentParser(
        description='Retroactively breed children for gen-N direct upvotes.')
    parser.add_argument('--generation', type=int, default=None,
                        help='Generation to scan (default: current).')
    parser.add_argument('--n-children', type=int, default=7,
                        help='Children to breed per upvoted genome (default 7, '
                             'matches Library.rate default).')
    parser.add_argument('--retag-since', type=str, default=None,
                        help='Re-stamp ratings created since this date (e.g. '
                             '"2026-05-20") from generation=0 to '
                             'current_generation, before breeding. Use this '
                             'if a stale wallpaper recorded recent upvotes '
                             'with default generation=0.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be done without doing it.')
    args = parser.parse_args()

    lib = Library()
    gen = args.generation if args.generation is not None else lib.get_current_generation()

    # Step 1 (optional): retag mistagged ratings
    if args.retag_since:
        rows = lib.conn.execute(
            '''SELECT id, created_at, target_id, rating, source
                 FROM ratings
                WHERE generation = 0
                  AND created_at >= ?
                  AND target_type = 'genome'
                ORDER BY created_at''',
            (args.retag_since,)
        ).fetchall()
        print(f'Retag: {len(rows)} ratings created since {args.retag_since} '
              f'are currently tagged generation=0; would retag to '
              f'generation={gen}.')
        for r in rows:
            print(f'  id={r[0]} {r[1]} target={r[2]} rating={r[3]:+d} '
                  f'source={r[4]!r}')
        if not args.dry_run:
            lib.conn.execute(
                '''UPDATE ratings SET generation = ?
                    WHERE generation = 0
                      AND created_at >= ?
                      AND target_type = 'genome' ''',
                (gen, args.retag_since)
            )
            lib.conn.commit()
            print(f'  -> updated {len(rows)} rows to generation={gen}')
        print()

    rows = lib.conn.execute(
        '''SELECT DISTINCT r.target_id
             FROM ratings r
             JOIN genomes g ON g.id = r.target_id
            WHERE r.target_type = 'genome'
              AND r.rating > 0
              AND r.generation = ?
              AND r.source = 'direct'
              AND COALESCE(g.archived, 0) = 0
            ORDER BY r.target_id''',
        (gen,)
    ).fetchall()
    parent_ids = [r[0] for r in rows]

    print(f'Generation {gen}: {len(parent_ids)} direct upvotes on active '
          f'genomes to breed from.')
    if args.dry_run:
        print('--dry-run: would breed', args.n_children,
              'children for each. Parents:', parent_ids)
        lib.close()
        return

    total_children = 0
    for pid in parent_ids:
        children = lib.breed_thumbsup_children(pid, n_children=args.n_children)
        print(f'  genome #{pid} -> {len(children)} children {children}')
        total_children += len(children)

    print(f'\nDone. Bred {total_children} new genomes from {len(parent_ids)} '
          f'parents. They land without renders — gpu_render_worker will '
          f'pick them up when next online.')
    lib.close()


if __name__ == '__main__':
    main()
