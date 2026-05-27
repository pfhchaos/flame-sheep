"""Ingest blind_model_eval votes into pairwise_ratings as training data.

Blind eval votes are random-sample preference judgments — broader
distribution coverage than compare-mode's active-learning picks, which
makes them strictly more valuable training signal. This script reads the
JSON output of blind_model_eval and appends each vote as a row in
pairwise_ratings with source='blind_eval' for provenance.

Idempotent on (winner_id, loser_id) pairs: re-ingesting the same JSON
won't create duplicates. Tagged with the current generation from
metadata.

Usage:
    python tools/ingest_blind_eval.py /tmp/blind_eval/blind_eval_results.json
    python tools/ingest_blind_eval.py FILE --dry-run    # preview only
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flame_sheep.storage import _db_path


def main():
    parser = argparse.ArgumentParser(description='Ingest blind eval votes')
    parser.add_argument('results_json', type=Path,
                        help='Path to blind_eval_results.json')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be inserted without writing')
    parser.add_argument('--db', type=Path, default=None,
                        help='DB path (default: library.db)')
    parser.add_argument('--source', type=str, default='blind_eval',
                        help='source label for pairwise_ratings (default: blind_eval)')
    args = parser.parse_args()

    if not args.results_json.exists():
        sys.exit(f'no such file: {args.results_json}')

    data = json.loads(args.results_json.read_text())
    pairs = data.get('pairs', [])
    if not pairs:
        sys.exit('no votes in results JSON')

    db_path = str(args.db) if args.db else str(_db_path())
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Current generation for stamping new rows
    row = conn.execute(
        "SELECT value FROM metadata WHERE key='current_generation'"
    ).fetchone()
    current_gen = int(row[0]) if row else 0

    # Map each vote to (winner_id, loser_id)
    to_insert: list[tuple[int, int]] = []
    for p in pairs:
        user = p.get('user')
        if user not in ('a', 'b'):
            continue  # skipped or invalid
        if user == 'a':
            winner, loser = p['gid_a'], p['gid_b']
        else:
            winner, loser = p['gid_b'], p['gid_a']
        to_insert.append((winner, loser))

    # Deduplicate against existing pairwise_ratings (winner_id, loser_id) and
    # also the inverse — the inverse with same orientation would be a real
    # contradiction (rare with random sampling, but possible) so we don't
    # filter inverses; we let SQLite store them and the training-time
    # pair-weighting handles conflicting signals.
    existing = set(
        conn.execute(
            'SELECT winner_id, loser_id FROM pairwise_ratings'
        ).fetchall()
    )
    new = [(w, l) for (w, l) in to_insert if (w, l) not in existing]

    print(f'Source JSON: {args.results_json}')
    print(f'  votes: {len(pairs)}  recorded (a/b): {len(to_insert)}  '
          f'new: {len(new)}  duplicates: {len(to_insert) - len(new)}')
    print(f'  current_generation: {current_gen}')
    print(f'  source label: {args.source!r}')

    if args.dry_run:
        print('--dry-run: not writing')
        if new[:10]:
            print('  first up to 10 new pairs (winner, loser):')
            for w, l in new[:10]:
                print(f'    {w} > {l}')
        return

    if not new:
        print('nothing to ingest')
        return

    conn.executemany(
        'INSERT INTO pairwise_ratings (winner_id, loser_id, source, generation) '
        'VALUES (?, ?, ?, ?)',
        [(w, l, args.source, current_gen) for w, l in new],
    )
    conn.commit()
    print(f'inserted {len(new)} pairwise_ratings rows')
    conn.close()


if __name__ == '__main__':
    main()
