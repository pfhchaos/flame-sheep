"""flame_sheep.scheduler CLI — for now just `--watch` to stream GPU
load to stdout so we can verify the perf binding works against a
real workload.

Usage:
    python -m flame_sheep.scheduler --watch
    python -m flame_sheep.scheduler --watch --interval-ms 100
"""
from __future__ import annotations

import argparse
import time

from .gpu_load import GpuLoadSampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true',
                        help='stream GPU busy fraction every interval')
    parser.add_argument('--interval-ms', type=int, default=200)
    parser.add_argument('--seconds', type=float, default=0.0,
                        help='stop after N seconds (0 = forever)')
    args = parser.parse_args()

    if not args.watch:
        parser.error('only --watch is implemented; pass --watch')

    sampler = GpuLoadSampler()
    if not sampler.available:
        raise SystemExit(f'sampler unavailable: {sampler.unavailable_reason}')

    t0 = time.perf_counter()
    interval = args.interval_ms / 1000.0
    next_t = t0 + interval
    try:
        while True:
            now = time.perf_counter()
            sleep_for = next_t - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            next_t += interval
            result = sampler.sample()
            if result is None:
                continue
            busy, window = result
            bar_n = int(busy * 40)
            bar = '#' * min(bar_n, 40) + ' ' * max(0, 40 - bar_n)
            print(f'{time.perf_counter() - t0:7.2f}s  '
                  f'{busy*100:5.1f}%  |{bar}|', flush=True)
            if args.seconds and (time.perf_counter() - t0) > args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        sampler.close()


if __name__ == '__main__':
    main()
