#!/usr/bin/env python3
"""Validate that Mesa Xe's DRM scheduler actually honors
VK_KHR_global_priority for cross-process arbitration.

Two phases:

  Phase 1: baseline.  Run ChaosGame.frame() in a tight loop at each
    priority alone, measure throughput. All four should report
    similar throughput — there's no contention, so priority shouldn't
    matter when the process has the GPU to itself.

  Phase 2: contention. Spawn TWO subprocesses concurrently — one at
    HIGH priority, one at LOW. Both hammer the GPU. After warmup,
    compare each one's throughput against its alone-baseline.

  If priority works:
    HIGH/HIGH_alone ratio: ~1.0   (HIGH keeps the GPU)
    LOW/LOW_alone   ratio: ~0.0   (LOW is starved)

  If priority is ignored:
    Both ratios ~0.5 (50/50 split)

  Anything in between = partial / soft arbitration.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import vulkan as vk

PRIORITY_LOOKUP = {
    'LOW':      vk.VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR,
    'MEDIUM':   vk.VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR,
    'HIGH':     vk.VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR,
    'REALTIME': vk.VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR,
}


def run_worker(priority_name: str, seconds: float, label: str) -> None:
    """Spawned as child via subprocess; runs chaos game at the given
    priority for `seconds`, prints per-window throughput to stdout."""
    from viz_authoring.vk.context import VkContext
    from flame_sheep.rendering.vk.chaos_game import ChaosGame
    from viz_authoring.vk.wallpaper_demo import _genome_from_catalog

    prio = PRIORITY_LOOKUP[priority_name]
    ctx = VkContext(instance_extensions=[],
                     pipeline_cache_path=None,
                     queue_priority=prio)
    ctx.select_device()

    cg = ChaosGame(ctx, 3713, 1278, n_walkers=65536)
    kwargs, _, zoom, rotation, center = _genome_from_catalog(2505)
    cg.set_genome(**kwargs)
    cg.reset_walkers(seed=0)

    # Warmup — compile pipeline, settle into steady state
    for _ in range(5):
        cg.frame(iterations=300, zoom=zoom, rotation=rotation,
                  center=center, decay=0.3)

    t_start = time.perf_counter()
    samples = 0
    last_report = t_start
    samples_at_last_report = 0
    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= seconds:
            break
        cg.frame(iterations=300, zoom=zoom, rotation=rotation,
                  center=center, decay=0.3)
        samples += 1
        now = time.perf_counter()
        if now - last_report >= 1.0:
            window = now - last_report
            window_samples = samples - samples_at_last_report
            print(f'[{label}] t={elapsed:5.1f}s '
                  f'rate={window_samples/window:.2f}fps '
                  f'avg_frame={1000*window/window_samples:.2f}ms',
                  flush=True)
            last_report = now
            samples_at_last_report = samples

    total = time.perf_counter() - t_start
    print(f'[{label}] FINAL: {samples} frames in {total:.2f}s = '
          f'{samples/total:.2f}fps', flush=True)
    cg.cleanup()
    ctx.cleanup()


def parse_final(stdout: str) -> float | None:
    """Pull the FINAL fps from a worker's stdout."""
    for line in stdout.splitlines():
        if 'FINAL:' in line and 'fps' in line:
            try:
                return float(line.split('=')[-1].split('fps')[0].strip())
            except ValueError:
                continue
    return None


def spawn_worker(priority_name: str, seconds: float, label: str
                  ) -> subprocess.Popen:
    venv_py = Path(__file__).parents[2] / '.venv/bin/python'
    return subprocess.Popen(
        [str(venv_py), __file__, '--child',
         '--priority', priority_name,
         '--seconds', str(seconds),
         '--label', label],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ,
              'PYTHONPATH': ':'.join([
                  str(Path(__file__).parents[2] / 'viz_authoring/src'),
                  str(Path(__file__).parents[2] / 'flame_sheep_audio/src'),
                  str(Path(__file__).parents[2]),
              ])},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', action='store_true',
                        help='internal: run as the GPU worker')
    parser.add_argument('--priority', default='MEDIUM',
                        choices=list(PRIORITY_LOOKUP))
    parser.add_argument('--seconds', type=float, default=10.0)
    parser.add_argument('--label', default='worker')
    args = parser.parse_args()

    if args.child:
        run_worker(args.priority, args.seconds, args.label)
        return

    # Parent: orchestrate the experiment.
    print('=== Phase 1: baseline (each priority alone, 6s) ===')
    baselines = {}
    for name in ('LOW', 'HIGH'):
        print(f'--- {name} alone ---')
        p = spawn_worker(name, 6.0, name)
        out, _ = p.communicate()
        print(out)
        baselines[name] = parse_final(out)
        if baselines[name] is None:
            print(f'!! could not parse final fps for {name}')
            return

    print()
    print('=== Phase 2: contention (HIGH + LOW concurrently, 10s) ===')
    p_high = spawn_worker('HIGH', 10.0, 'HIGH')
    p_low = spawn_worker('LOW', 10.0, 'LOW')
    out_high, _ = p_high.communicate()
    out_low, _ = p_low.communicate()
    print('--- HIGH stdout ---')
    print(out_high)
    print('--- LOW stdout ---')
    print(out_low)

    contended = {
        'HIGH': parse_final(out_high),
        'LOW':  parse_final(out_low),
    }
    print()
    print('=== Verdict ===')
    if None in contended.values():
        print('!! could not parse contended final fps; check worker output')
        return
    for name in ('HIGH', 'LOW'):
        base = baselines[name]
        cont = contended[name]
        ratio = cont / base if base > 0 else 0.0
        print(f'  {name}: baseline={base:.2f}fps  contended={cont:.2f}fps  '
              f'kept {ratio*100:.1f}% of throughput')

    h_ratio = contended['HIGH'] / baselines['HIGH']
    l_ratio = contended['LOW']  / baselines['LOW']
    print()
    if h_ratio > 0.85 and l_ratio < 0.30:
        print('  → priority HONORED. HIGH dominates, LOW starved. '
              'Cross-process priority works.')
    elif 0.30 <= h_ratio <= 0.70 and 0.30 <= l_ratio <= 0.70:
        print('  → priority IGNORED. ~50/50 split regardless. '
              'Driver doesn\'t arbitrate by priority.')
    else:
        print('  → priority PARTIALLY effective. Soft arbitration; '
              'meaningful but not absolute.')


if __name__ == '__main__':
    main()
