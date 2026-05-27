"""Blind validation: random pairs of genomes, user votes, tally per-model agreement.

Pre-renders 50 random pairs from active genomes (no GPU needed — uses
existing render_static blobs in the DB). For each pair, pre-computes both
models' scores. Then prompts the user interactively for blind votes.
After all votes, reports which model agrees with the user more often.

The validation gates the "deploy curriculum model" decision. Random sampling
(not adjacent-in-score or top-shifted) avoids the convenience-sampling bias
of looking at extremes where models always disagree most.

Usage:
    python tools/blind_model_eval.py \\
        --model-a ~/datasets/esheep-cnn/cnn_stage3_LSHA_personal.npz \\
        --model-b PATH/TO/DEPLOYED.npz \\
        --n 50

If --model-b is omitted, uses whatever's at the deployed path (per
flame_sheep.cnn_scorer.DEPLOYED_WEIGHTS_PATH).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flame_sheep.genome.scoring.cnn_scorer import load_cnn_weights_file
from flame_sheep.storage import Library, _db_path
from wallpaper_ml.vk_compute import VkCompute
from wallpaper_ml import build_cnn_scorer
from train_cnn_vk import MODEL_CONFIGS
from finetune_cnn_vk import DbImageStore


def score_genomes(weights_path: Path, genome_ids: list[int], db_path: str,
                  image_size: int = 256, batch_size: int = 8) -> dict[int, float]:
    """Run a model over all genome_ids; return {gid: score}."""
    gpu = VkCompute()
    LAYERS = list(MODEL_CONFIGS['25k'])
    model = build_cnn_scorer(gpu, LAYERS, batch_size=batch_size,
                             image_size=image_size, mlp_head=True)
    weights, _ = load_cnn_weights_file(str(weights_path))
    model.load_weights(weights)

    from flame_sheep.genome.scoring.scoring_channels import load_normalization_sidecar
    from flame_sheep.storage import NORMALIZATION_VERSION
    # Use Library's normalization (matches what genomes were rendered against)
    lib = Library()
    normalization = lib.get_normalization(NORMALIZATION_VERSION)
    lib.close()

    store = DbImageStore(db_path, image_size, channels='domain', cache_gb=2.0,
                         normalization=normalization, n_channels=4)
    input_buf = gpu.create_buffer(batch_size * 4 * image_size * image_size * 4)

    scores: dict[int, float] = {}
    for i in range(0, len(genome_ids), batch_size):
        batch_ids = genome_ids[i:i + batch_size]
        B = len(batch_ids)
        if B == 0:
            continue
        batch = np.zeros((batch_size, 4, image_size, image_size),
                         dtype=np.float32)
        valid = []
        for j, gid in enumerate(batch_ids):
            img = store.get(gid)
            if img is not None:
                batch[j] = img
                valid.append(gid)
        if not valid:
            continue
        gpu.upload(input_buf, batch)
        out = model.forward(input_buf, batch_size, (4, image_size, image_size))
        out_scores = gpu.download(out, np.float32, batch_size)
        for j, gid in enumerate(batch_ids):
            if gid in valid:
                scores[gid] = float(out_scores[j])
    store.close()
    return scores


def render_pair(conn, gid_a: int, gid_b: int, out_a: Path, out_b: Path) -> bool:
    """Extract render_static from DB for both genomes, save as PNGs.

    Returns True if both renders existed and were written.
    """
    rows = conn.execute(
        'SELECT genome_id, render_static FROM genome_blobs '
        'WHERE genome_id IN (?, ?) AND render_static IS NOT NULL',
        (gid_a, gid_b),
    ).fetchall()
    blobs = {gid: blob for gid, blob in rows}
    if gid_a not in blobs or gid_b not in blobs:
        return False
    out_a.write_bytes(blobs[gid_a])
    out_b.write_bytes(blobs[gid_b])
    return True


def main():
    parser = argparse.ArgumentParser(description='Blind A/B model evaluation')
    parser.add_argument('--model-a', type=Path, required=True,
                        help='First model weights (.npz)')
    parser.add_argument('--model-b', type=Path, required=True,
                        help='Second model weights (.npz)')
    parser.add_argument('--label-a', type=str, default='A',
                        help='Short label for model A (default: "A")')
    parser.add_argument('--label-b', type=str, default='B',
                        help='Short label for model B (default: "B")')
    parser.add_argument('--n', type=int, default=50,
                        help='Number of pairs to evaluate (default: 50)')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('/tmp/blind_eval'))
    parser.add_argument('--db', type=Path, default=None,
                        help='DB path (default: library.db)')
    parser.add_argument('--seed', type=int, default=None,
                        help='RNG seed for pair sampling (default: random)')
    parser.add_argument('--live', action='store_true',
                        help='Drive the running wallpaper via control pipe '
                             '(show <gid>) instead of saving PNGs. Requires '
                             'wallpaper running; falls back to PNG mode if '
                             'pipe is missing.')
    parser.add_argument('--pipe', type=Path,
                        default=Path(os.environ.get(
                            'FLAME_SHEEP_CTL',
                            os.path.expanduser('~/.local/share/flame-sheep/ctl'))),
                        help='Control pipe path (only used with --live)')
    args = parser.parse_args()

    db_path = str(args.db) if args.db else str(_db_path())
    print(f'DB: {db_path}')
    print(f'Model A ({args.label_a}): {args.model_a}')
    print(f'Model B ({args.label_b}): {args.model_b}')

    # Get list of active genomes with renders
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        'SELECT b.genome_id FROM genome_blobs b '
        'JOIN genomes g ON g.id = b.genome_id '
        'WHERE b.render_static IS NOT NULL '
        '  AND g.archived = 0'
    ).fetchall()
    active_ids = [r[0] for r in rows]
    print(f'Active genomes with renders: {len(active_ids)}')

    if len(active_ids) < 2 * args.n:
        print(f'WARNING: only {len(active_ids)} genomes available, '
              f'requested {args.n} pairs')

    # Random pair sampling
    rng = np.random.default_rng(args.seed)
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(pairs) < args.n:
        gid_a, gid_b = rng.choice(active_ids, 2, replace=False)
        key = (min(int(gid_a), int(gid_b)), max(int(gid_a), int(gid_b)))
        if key in seen:
            continue
        seen.add(key)
        # Randomize a/b ordering so display position doesn't bias
        if rng.random() < 0.5:
            pairs.append((int(gid_a), int(gid_b)))
        else:
            pairs.append((int(gid_b), int(gid_a)))

    # Live mode: pipe must exist. Fall back to PNG mode if requested but
    # pipe missing (with explicit warning).
    live_mode = args.live and args.pipe.exists()
    if args.live and not args.pipe.exists():
        print(f'WARNING: --live requested but pipe missing at {args.pipe}')
        print('  falling back to PNG mode')

    # Always create output dir — used for results JSON in both modes,
    # and PNG files in PNG mode. Skipping this in live mode caused the
    # incremental save + final-write to crash with FileNotFoundError.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if live_mode:
        # In live mode, we don't need to pre-render — pairs are existence-
        # filtered against the DB (both genomes must have renders so the
        # wallpaper can display them).
        valid_pairs: list[tuple[int, tuple[int, int]]] = []
        for n, (gid_a, gid_b) in enumerate(pairs, start=1):
            rows = conn.execute(
                'SELECT genome_id FROM genome_blobs '
                'WHERE genome_id IN (?, ?) AND render_static IS NOT NULL',
                (gid_a, gid_b),
            ).fetchall()
            if len(rows) == 2:
                valid_pairs.append((n, (gid_a, gid_b)))
        print(f'{len(valid_pairs)} valid pairs ready for live display')
    else:
        # PNG mode: extract render_static blobs to disk for image viewer use
        valid_pairs = []
        for n, (gid_a, gid_b) in enumerate(pairs, start=1):
            out_a = args.output_dir / f'pair_{n:02d}_a.png'
            out_b = args.output_dir / f'pair_{n:02d}_b.png'
            if render_pair(conn, gid_a, gid_b, out_a, out_b):
                valid_pairs.append((n, (gid_a, gid_b)))
            else:
                out_a.unlink(missing_ok=True)
                out_b.unlink(missing_ok=True)
        print(f'Rendered {len(valid_pairs)} valid pairs to {args.output_dir}/')
    conn.close()

    # Pre-compute scores under both models
    all_gids = sorted({gid for _, (a, b) in valid_pairs for gid in (a, b)})
    print(f'Scoring {len(all_gids)} unique genomes under both models...')
    print(f'  ({args.label_a})...')
    scores_a = score_genomes(args.model_a, all_gids, db_path)
    print(f'  ({args.label_b})...')
    scores_b = score_genomes(args.model_b, all_gids, db_path)

    # Interactive voting loop
    print()
    print('=' * 60)
    print('BLIND VOTE')
    print('=' * 60)
    if live_mode:
        print(f'Live mode: wallpaper will display each genome via {args.pipe}')
        print('For each pair: shows A first. Type:')
        print('  s = swap to other side')
        print('  a = vote A is better')
        print('  b = vote B is better')
        print('  k = skip this pair')
        print('  q = quit (saves partial results)')
    else:
        print(f'Open {args.output_dir}/ in an image viewer.')
        print('For each pair, view pair_NN_a.png and pair_NN_b.png.')
        print('Type "a", "b", or "skip" (or "quit" to bail early).')
    print()

    def send_show(gid: int) -> None:
        with open(args.pipe, 'w') as f:
            f.write(f'show {gid}\n')

    def send_unshow() -> None:
        try:
            with open(args.pipe, 'w') as f:
                f.write('unshow\n')
        except OSError:
            pass

    # Incremental save: write the running results JSON after each vote so a
    # crash or quit doesn't destroy the work already done.
    log_path = args.output_dir / 'blind_eval_results.json'
    pair_lookup_early = dict(valid_pairs)

    def save_partial(votes_so_far: dict[int, str]) -> None:
        out = {
            'model_a': str(args.model_a), 'label_a': args.label_a,
            'model_b': str(args.model_b), 'label_b': args.label_b,
            'pairs': [
                {
                    'n': n,
                    'gid_a': pair_lookup_early[n][0],
                    'gid_b': pair_lookup_early[n][1],
                    'user': user_choice,
                    'score_a_for_a': scores_a.get(pair_lookup_early[n][0]),
                    'score_a_for_b': scores_a.get(pair_lookup_early[n][1]),
                    'score_b_for_a': scores_b.get(pair_lookup_early[n][0]),
                    'score_b_for_b': scores_b.get(pair_lookup_early[n][1]),
                }
                for n, user_choice in votes_so_far.items()
            ],
        }
        log_path.write_text(json.dumps(out, indent=2))

    votes: dict[int, str] = {}  # {pair_n: 'a' | 'b'}
    try:
        for n, (gid_a, gid_b) in valid_pairs:
            print(f'  pair {n:02d}/{len(valid_pairs)}  (a={gid_a}  b={gid_b})')
            if live_mode:
                # Start showing A; user can swap with 's'
                current = 'a'
                send_show(gid_a)
                print(f'    → showing A (#{gid_a})')
            while True:
                choice = input('    > ').strip().lower()
                if live_mode and choice in ('s', 'swap'):
                    current = 'b' if current == 'a' else 'a'
                    new_gid = gid_b if current == 'b' else gid_a
                    send_show(new_gid)
                    print(f'    → showing {current.upper()} (#{new_gid})')
                    continue
                if choice in ('a', 'b'):
                    break
                if choice in ('k', 'skip'):
                    choice = 'skip'
                    break
                if choice in ('q', 'quit'):
                    choice = 'quit'
                    break
                if live_mode:
                    print('    type a / b / s / k / q')
                else:
                    print('    type a / b / skip / quit')
            if choice == 'quit':
                break
            if choice == 'skip':
                continue
            votes[n] = choice
            save_partial(votes)  # persist after every vote — crash-safe
    finally:
        if live_mode:
            send_unshow()

    # Tally agreement
    print()
    print('=' * 60)
    print('RESULTS')
    print('=' * 60)
    print(f'Total votes: {len(votes)}')
    if not votes:
        print('No votes recorded.')
        return

    a_correct = 0  # times model A's preference matched user
    b_correct = 0
    both_agree = 0  # both models picked same side
    a_only = 0      # only A matched user
    b_only = 0      # only B matched user
    neither = 0
    pair_lookup = dict(valid_pairs)

    for n, user_choice in votes.items():
        gid_a, gid_b = pair_lookup[n]
        s_a_for_a = scores_a.get(gid_a, 0.0)
        s_a_for_b = scores_a.get(gid_b, 0.0)
        s_b_for_a = scores_b.get(gid_a, 0.0)
        s_b_for_b = scores_b.get(gid_b, 0.0)

        a_picks = 'a' if s_a_for_a > s_a_for_b else 'b'
        b_picks = 'a' if s_b_for_a > s_b_for_b else 'b'

        a_matches = (a_picks == user_choice)
        b_matches = (b_picks == user_choice)
        a_correct += a_matches
        b_correct += b_matches
        if a_matches and b_matches:
            both_agree += 1
        elif a_matches:
            a_only += 1
        elif b_matches:
            b_only += 1
        else:
            neither += 1

    n = len(votes)
    print(f'  {args.label_a} agreement: {a_correct}/{n} ({a_correct/n:.1%})')
    print(f'  {args.label_b} agreement: {b_correct}/{n} ({b_correct/n:.1%})')
    print()
    print(f'  Both models matched user:    {both_agree}/{n} ({both_agree/n:.1%})')
    print(f'  Only {args.label_a} matched: {a_only}/{n} ({a_only/n:.1%})')
    print(f'  Only {args.label_b} matched: {b_only}/{n} ({b_only/n:.1%})')
    print(f'  Neither matched:             {neither}/{n} ({neither/n:.1%})')

    # Save raw data for re-analysis
    out = {
        'model_a': str(args.model_a), 'label_a': args.label_a,
        'model_b': str(args.model_b), 'label_b': args.label_b,
        'pairs': [
            {
                'n': n,
                'gid_a': pair_lookup[n][0],
                'gid_b': pair_lookup[n][1],
                'user': user_choice,
                'score_a_for_a': scores_a.get(pair_lookup[n][0]),
                'score_a_for_b': scores_a.get(pair_lookup[n][1]),
                'score_b_for_a': scores_b.get(pair_lookup[n][0]),
                'score_b_for_b': scores_b.get(pair_lookup[n][1]),
            }
            for n, user_choice in votes.items()
        ],
    }
    log_path = args.output_dir / 'blind_eval_results.json'
    log_path.write_text(json.dumps(out, indent=2))
    print(f'\nResults saved to {log_path}')


if __name__ == '__main__':
    main()
