"""flame_sheep.scheduler CLI.

Subcommands:
    watch    — stream GPU busy fraction (validates the perf binding)
    demo     — spawn a dummy batch worker + run the policy loop;
                shows the scheduler actually yielding to load

Examples:
    python -m flame_sheep.scheduler watch --seconds 5
    python -m flame_sheep.scheduler demo --seconds 30
"""
from __future__ import annotations

import argparse
import sys
import time

from .batch_worker import BatchWorker
from .gpu_load import GpuLoadSampler
from .policy import BatchState, Policy


def _cmd_watch(args):
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
            busy, _ = result
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


def _cmd_demo(args):
    """E2E: spawn dummy worker, tick policy, log state transitions.
    Useful for validating the scheduler architecture before wiring
    real consumers."""
    sampler = GpuLoadSampler()
    if not sampler.available:
        print(f'[scheduler] sampler unavailable: '
              f'{sampler.unavailable_reason}', file=sys.stderr)
        print('[scheduler] policy will stay RUN — no GPU-load gating',
              file=sys.stderr)
    policy = Policy(sampler,
                     sample_interval_s=args.sample_interval_ms / 1000.0,
                     pause_threshold=args.pause_threshold,
                     slow_threshold=args.slow_threshold)

    worker_argv = [sys.executable, '-m',
                    'flame_sheep.scheduler._dummy_worker']
    with BatchWorker('dummy', worker_argv) as worker:
        worker.start()
        last_state = None
        t0 = time.perf_counter()
        try:
            while True:
                state = policy.tick()
                worker.flag.set(state)
                if state is not last_state:
                    elapsed = time.perf_counter() - t0
                    print(f'[scheduler] t={elapsed:6.2f}s  '
                          f'state {(last_state.value if last_state else "init")} '
                          f'→ {state.value}  '
                          f'(rolling_max={policy.rolling_max*100:.1f}%)',
                          flush=True)
                    last_state = state
                if worker.poll() is not None:
                    break
                if args.seconds and (time.perf_counter() - t0) > args.seconds:
                    break
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            sampler.close()


def main():
    parser = argparse.ArgumentParser(prog='flame_sheep.scheduler')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_watch = sub.add_parser('watch',
                              help='stream GPU busy fraction')
    p_watch.add_argument('--interval-ms', type=int, default=200)
    p_watch.add_argument('--seconds', type=float, default=0.0,
                          help='stop after N seconds (0 = forever)')
    p_watch.set_defaults(fn=_cmd_watch)

    p_demo = sub.add_parser('demo',
                             help='spawn dummy worker + run policy loop')
    p_demo.add_argument('--seconds', type=float, default=30.0)
    p_demo.add_argument('--sample-interval-ms', type=int, default=250)
    p_demo.add_argument('--pause-threshold', type=float, default=0.85)
    p_demo.add_argument('--slow-threshold', type=float, default=0.55)
    p_demo.set_defaults(fn=_cmd_demo)

    args = parser.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
