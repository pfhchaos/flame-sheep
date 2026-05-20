#!/usr/bin/env python3
"""Print the retraining recommendation for the current generation.

Reads judgment counts from the Library DB and applies the heuristic
documented in Library.retrain_recommendation:
  - Below 10K total judgments: fixed threshold of 2000 new this gen
  - Above 10K total: 1.5x of previous generation's count
  - Each thumb and each compare counts as one judgment

Just logs the recommendation — doesn't trigger anything. Run after a
compare/rating session to see if it's time to retrain. The actual
train + breed + advance_generation cycle is still manual.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flame_sheep.storage import Library


def main():
    parser = argparse.ArgumentParser(
        description='Check whether enough new-generation judgments have '
                    'accumulated to warrant retraining.')
    parser.add_argument('--fixed-threshold', type=int, default=2000,
                        help='Fixed judgment count threshold for early '
                             'generations (default: 2000)')
    parser.add_argument('--relative-threshold', type=float, default=1.5,
                        help='Multiplier of previous-gen count, used after '
                             'cutover (default: 1.5)')
    parser.add_argument('--cutover-total', type=int, default=10_000,
                        help='Total-judgments threshold to switch from '
                             'fixed to relative mode (default: 10000)')
    args = parser.parse_args()

    lib = Library()
    rec = lib.retrain_recommendation(
        fixed_threshold=args.fixed_threshold,
        relative_threshold=args.relative_threshold,
        cutover_total=args.cutover_total,
    )
    print(rec['message'])
    print()
    for key in ('current_gen', 'judgments_this_gen', 'thumbs_this_gen',
                'pairwise_this_gen', 'judgments_prev_gen', 'all_gen_total',
                'threshold', 'threshold_mode', 'should_retrain'):
        print(f'  {key}: {rec[key]}')
    lib.close()
    sys.exit(0 if rec['should_retrain'] else 1)


if __name__ == '__main__':
    main()
