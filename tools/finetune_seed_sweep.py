#!/usr/bin/env python3
"""Single-epoch finetune seed sweep — diagnose why epoch-1 val_acc starts low.

Hypothesis test for the 35-40% epoch-1 floor. A model fresh from Kaiming
init evaluated on balanced pairwise comparisons should sit at ~50% by
definition (random scores, random ranking). If we consistently see 35-40%,
something is biasing the initial predictions in the WRONG direction.

Two RNGs control the variance:

- --init-seeds: Kaiming weight init (--seed in finetune_cnn_vk.py).
  Tests "is the model init systematically miscalibrated?"
- --val-seeds: train/val split (--val-seed). Tests "is one specific val
  split unusually hard, or do all splits show the same floor?"

Run with both for the full matrix; the result table tells you which axis
the variance lives on.

Usage:
    # Init variance only (1 val split, 8 model inits)
    python tools/finetune_seed_sweep.py --init-seeds 1 2 3 4 5 6 7 8

    # Val variance only (1 model init, 5 val splits)
    python tools/finetune_seed_sweep.py --init-seeds 42 --val-seeds 1 2 3 4 5

    # Both — 4×4 matrix (16 runs, ~50 minutes)
    python tools/finetune_seed_sweep.py \\
        --init-seeds 1 2 3 4 --val-seeds 1 2 3 4
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

EPOCH_RE = re.compile(
    r'Epoch\s+1/\s*\d+\s+loss=(?P<loss>[\d.]+)\s+val_acc=(?P<val>[\d.]+)')


def run_one(init_seed: int, val_seed: int, args) -> tuple[float, float] | None:
    """Run a single 1-epoch finetune. Returns (loss, val_acc) on success."""
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp_out:
        out_path = tmp_out.name
    log_path = out_path + '.log'

    cmd = [
        sys.executable, str(Path(__file__).parent / 'finetune_cnn_vk.py'),
        '--from-scratch',
        '--channels', 'domain',
        '--mlp-head',
        '--model-size', args.model_size,
        '--data-mode', args.data_mode,
        '--lr', str(args.lr),
        '--epochs', '1',
        '--seed', str(init_seed),
        '--val-seed', str(val_seed),
        '--output', out_path,
    ]
    with open(log_path, 'w') as logf:
        subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, check=False)

    Path(out_path).unlink(missing_ok=True)
    log_text = Path(log_path).read_text()
    Path(log_path).unlink(missing_ok=True)

    m = EPOCH_RE.search(log_text)
    if not m:
        # Print last few lines of log to help diagnose
        tail = '\n'.join(log_text.splitlines()[-5:])
        print(f'  init={init_seed} val={val_seed}: FAILED to parse epoch 1. Tail:\n{tail}')
        return None
    return float(m.group('loss')), float(m.group('val'))


def main():
    parser = argparse.ArgumentParser(
        description='Single-epoch finetune seed sweep')
    parser.add_argument('--init-seeds', type=int, nargs='+', default=[42],
                        help='Model init seeds (default: [42])')
    parser.add_argument('--val-seeds', type=int, nargs='+', default=[42],
                        help='Val-split seeds (default: [42])')
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--model-size', default='25k')
    parser.add_argument('--data-mode', default='mixed',
                        choices=['mixed', 'pairwise', 'thumbs'])
    args = parser.parse_args()

    n_runs = len(args.init_seeds) * len(args.val_seeds)
    print(f'Sweep: {len(args.init_seeds)} init seeds × {len(args.val_seeds)} val seeds '
          f'= {n_runs} runs (~3min each)')

    results: dict[tuple[int, int], tuple[float, float]] = {}
    for vi in args.val_seeds:
        for ii in args.init_seeds:
            res = run_one(ii, vi, args)
            if res is None:
                continue
            results[(ii, vi)] = res
            print(f'  init={ii:3d}  val={vi:3d}  loss={res[0]:.4f}  val_acc={res[1]:.4f}')

    if not results:
        print('No successful runs.')
        sys.exit(1)

    # Summary table — rows are init seeds, cols are val seeds
    print()
    print('=' * 70)
    print('VAL_ACC TABLE  (rows = init seed, cols = val seed)')
    print('=' * 70)
    header = '          ' + '  '.join(f'val={v:3d}' for v in args.val_seeds)
    print(header)
    for ii in args.init_seeds:
        cells = []
        for vi in args.val_seeds:
            r = results.get((ii, vi))
            cells.append(f'  {r[1]:.4f}' if r else '   N/A ')
        print(f'init={ii:3d} ' + '  '.join(cells))

    # Per-axis stats
    all_vals = [v[1] for v in results.values()]
    import statistics
    print()
    print(f'Overall: n={len(all_vals)}  mean={statistics.mean(all_vals):.4f}  '
          f'stdev={statistics.stdev(all_vals) if len(all_vals)>1 else 0:.4f}  '
          f'range=[{min(all_vals):.4f}, {max(all_vals):.4f}]')

    if len(args.init_seeds) > 1 and len(args.val_seeds) > 1:
        # Variance by axis
        by_init = [[results[(ii, vi)][1] for vi in args.val_seeds
                    if (ii, vi) in results] for ii in args.init_seeds]
        by_val = [[results[(ii, vi)][1] for ii in args.init_seeds
                   if (ii, vi) in results] for vi in args.val_seeds]
        init_means = [statistics.mean(b) for b in by_init if b]
        val_means = [statistics.mean(b) for b in by_val if b]
        print(f'Init-seed spread (mean over val splits): '
              f'range=[{min(init_means):.4f}, {max(init_means):.4f}]')
        print(f'Val-seed spread (mean over init seeds):  '
              f'range=[{min(val_means):.4f}, {max(val_means):.4f}]')

    print()
    if statistics.mean(all_vals) < 0.48:
        print('Mean below random — systematic bias in initial predictions.')
        print('Test next: bias init at 0.0 instead of 0.1, OR pair-direction audit.')
    elif statistics.mean(all_vals) > 0.52:
        print('Mean above random — random init has some signal, learning is real.')
    else:
        print('Mean near random (~0.5) — model is genuinely uninitialized, '
              'no init pathology.')


if __name__ == '__main__':
    main()
