"""Dummy batch worker used to exercise the scheduler end-to-end
without involving real GPU work. Pretends each "unit of work" takes
~100ms; logs every completed unit so we can see when the scheduler
let it run vs paused it.

Usage (the scheduler spawns this automatically in the demo path):
    python -m flame_sheep.scheduler._dummy_worker --pause-flag /path/to/flag
"""
from __future__ import annotations

import argparse
import time

from .pause_flag import PauseFlagReader
from .policy import BatchState


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pause-flag', required=True)
    parser.add_argument('--unit-ms', type=int, default=100,
                        help='simulated cost per unit of work')
    parser.add_argument('--slow-delay-ms', type=int, default=500,
                        help='extra sleep between units when SLOW')
    parser.add_argument('--pause-poll-ms', type=int, default=100)
    parser.add_argument('--max-units', type=int, default=0,
                        help='exit after N units (0 = forever)')
    args = parser.parse_args()

    reader = PauseFlagReader(args.pause_flag)
    completed = 0
    t0 = time.perf_counter()

    try:
        while True:
            state = reader.state
            if state is BatchState.PAUSE:
                time.sleep(args.pause_poll_ms / 1000.0)
                continue
            if state is BatchState.SLOW:
                time.sleep(args.slow_delay_ms / 1000.0)
            # Simulate a unit of work
            time.sleep(args.unit_ms / 1000.0)
            completed += 1
            elapsed = time.perf_counter() - t0
            print(f'[dummy] unit={completed} t={elapsed:.2f}s '
                   f'state={state.value}', flush=True)
            if args.max_units and completed >= args.max_units:
                break
    finally:
        reader.close()


if __name__ == '__main__':
    main()
