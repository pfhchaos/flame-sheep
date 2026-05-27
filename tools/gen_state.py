#!/usr/bin/env python3
"""Inspect / control the generation-advance latch state.

Usage:
    python tools/gen_state.py                    # show
    python tools/gen_state.py --show             # show (default)
    python tools/gen_state.py --reset            # force back to idle
    python tools/gen_state.py --start-advance    # kickstart from idle:
                                                 # sets target=current+1,
                                                 # state=awaiting_rescore
    python tools/gen_state.py --force-advance STATE
                                                 # CAS from current to STATE

States:
    idle | awaiting_rescore | awaiting_breed
    awaiting_score_new | awaiting_prune

The latch is a metadata-table state machine that orchestrates the post-
train pipeline: score_worker handles rescore transitions, pruner_worker
handles breed + prune. See docs/generational_architecture.md and
flame_sheep/gen_advance.py.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flame_sheep.storage import Library
from flame_sheep.gen_advance import (
    ALL_STATES, STATE_IDLE,
    STATE_AWAITING_RESCORE, STATE_AWAITING_BREED,
    STATE_AWAITING_SCORE_NEW, STATE_AWAITING_PRUNE,
    rescore_complete, all_active_fully_scored,
    compute_breed_count, measure_disagreement,
    TARGET_POP_SIZE,
)
from flame_sheep.genome.scoring.cnn_scorer import _default_weights_path


def _weights_hash() -> str | None:
    p = _default_weights_path()
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _stale_score_count(lib: Library, weights_hash: str) -> int:
    return lib.conn.execute(
        'SELECT COUNT(*) FROM genomes g '
        'JOIN genome_blobs b ON b.genome_id = g.id '
        'WHERE COALESCE(g.archived, 0) = 0 '
        '  AND b.render_static IS NOT NULL '
        '  AND (g.cnn_weights_hash IS NULL OR g.cnn_weights_hash != ?)',
        (weights_hash,)
    ).fetchone()[0]


def _active_pop(lib: Library) -> int:
    return lib.conn.execute(
        'SELECT COUNT(*) FROM genomes WHERE COALESCE(archived, 0) = 0'
    ).fetchone()[0]


def show(lib: Library) -> None:
    state = lib.get_gen_advance_state()
    target = lib.get_gen_advance_target()
    current = lib.get_current_generation()
    pop = _active_pop(lib)
    wh = _weights_hash()

    print(f'gen_advance_state  = {state}')
    print(f'gen_advance_target = {target}')
    print(f'current_generation = {current}')
    print(f'active_pop         = {pop} (target {TARGET_POP_SIZE})')
    print(f'deployed_weights   = {wh or "(missing)"}')

    if state == STATE_AWAITING_RESCORE and wh:
        stale = _stale_score_count(lib, wh)
        done = stale == 0
        print(f'stale_score_count  = {stale} (rendered, '
              f'{"complete" if done else "in-progress"})')
    elif state == STATE_AWAITING_SCORE_NEW and wh:
        # Stricter check — includes unrendered bred genomes
        stale_all = lib.conn.execute(
            'SELECT COUNT(*) FROM genomes g '
            'WHERE COALESCE(g.archived, 0) = 0 '
            '  AND (g.cnn_weights_hash IS NULL OR g.cnn_weights_hash != ?)',
            (wh,)
        ).fetchone()[0]
        stale_rendered = _stale_score_count(lib, wh)
        unrendered = stale_all - stale_rendered
        done = stale_all == 0
        print(f'stale_all_active   = {stale_all} '
              f'(of which {unrendered} unrendered, awaiting render_worker; '
              f'{stale_rendered} rendered, awaiting score_worker) '
              f'({"complete" if done else "in-progress"})')
    elif state == STATE_AWAITING_BREED:
        disagreement = measure_disagreement(lib.conn)
        n = compute_breed_count(disagreement, pop)
        print(f'planned_breed_count = {n} (disagreement={disagreement:.3f})')
    elif state == STATE_AWAITING_PRUNE:
        excess = pop - TARGET_POP_SIZE
        print(f'pop_excess_over_target = {excess}')


def reset(lib: Library) -> None:
    state = lib.get_gen_advance_state()
    if state == STATE_IDLE:
        print('already idle')
        return
    print(f'resetting latch: {state} → idle (gen unchanged)')
    lib.clear_gen_advance()


def start_advance(lib: Library) -> None:
    """Kickstart the pipeline from idle. Equivalent to the trainer's
    post-train hook — sets target=current+1 and flips state to
    awaiting_rescore. Use this when a training run completed without
    auto-latching (legacy trainer, manual deploy, etc.)."""
    state = lib.get_gen_advance_state()
    if state != STATE_IDLE:
        print(f'cannot start: state is {state!r}, not idle. '
              f'reset first or use --force-advance', file=sys.stderr)
        sys.exit(1)
    target = lib.get_current_generation() + 1
    if not lib.cas_gen_advance_state(STATE_IDLE, STATE_AWAITING_RESCORE,
                                      target=target):
        print('CAS lost — state changed under us, try again', file=sys.stderr)
        sys.exit(1)
    print(f'started gen advance: idle → awaiting_rescore (target gen {target})')


def force_advance(lib: Library, new_state: str) -> None:
    if new_state not in ALL_STATES:
        print(f'unknown state: {new_state}. valid: {ALL_STATES}',
              file=sys.stderr)
        sys.exit(2)
    current = lib.get_gen_advance_state()
    if current == new_state:
        print(f'already in {new_state}, no-op')
        return
    if not lib.cas_gen_advance_state(current, new_state):
        # Lost the race — re-read and report
        actual = lib.get_gen_advance_state()
        print(f'CAS failed: expected {current!r}, was {actual!r}',
              file=sys.stderr)
        sys.exit(1)
    print(f'forced advance: {current} → {new_state}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    g = parser.add_mutually_exclusive_group()
    g.add_argument('--show', action='store_true',
                   help='Print current latch state and pipeline diagnostics (default)')
    g.add_argument('--reset', action='store_true',
                   help='Force latch back to idle (does NOT change current_generation)')
    g.add_argument('--start-advance', action='store_true',
                   help='Kickstart pipeline from idle (target=current_gen+1)')
    g.add_argument('--force-advance', metavar='STATE',
                   help='CAS-flip to the given state from whatever current is')
    args = parser.parse_args()

    lib = Library()
    try:
        if args.reset:
            reset(lib)
        elif args.start_advance:
            start_advance(lib)
        elif args.force_advance:
            force_advance(lib, args.force_advance)
        else:
            show(lib)
    finally:
        lib.close()


if __name__ == '__main__':
    main()
