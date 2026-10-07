#!/usr/bin/env python3
"""Measure wall-clock time for ChaosGame.frame() at production canvas
size on a real catalog genome.

Used as before/after comparator for the spec-const productionization.
The wallpaper's per-frame log shows chaos_game time on real workloads
but bundles in upload + tonemap. This isolates just the compute pass
so we can attribute speedup unambiguously.

Headline metric: per-frame ms across N warm-up + M timed iterations.
Run it before + after a perf change and diff.
"""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='Bench ChaosGame.frame() at wallpaper-scale canvas')
    parser.add_argument('--canvas-w', type=int, default=3713,
                        help='Match the user\'s actual multi-monitor canvas')
    parser.add_argument('--canvas-h', type=int, default=1278)
    parser.add_argument('--genome-id', type=int, default=2505,
                        help='Catalog genome to render')
    parser.add_argument('--iterations', type=int, default=150,
                        help='Chaos game iters per frame (matches wallpaper)')
    parser.add_argument('--n-walkers', type=int, default=65536)
    parser.add_argument('--warmup-frames', type=int, default=10)
    parser.add_argument('--timed-frames', type=int, default=60)
    args = parser.parse_args()

    from viz_authoring.vk.context import VkContext
    from flame_sheep.rendering.vk.chaos_game import ChaosGame
    from viz_authoring.vk.wallpaper_demo import _genome_from_catalog

    ctx = VkContext(instance_extensions=[], pipeline_cache_path=None)
    ctx.select_device()
    print(f'device: {ctx.device_name}')

    cg = ChaosGame(ctx, args.canvas_w, args.canvas_h,
                    n_walkers=args.n_walkers)
    print(f'canvas: {args.canvas_w}x{args.canvas_h}, '
          f'{args.n_walkers} walkers, {args.iterations} iters/frame')

    kwargs, _palette, zoom, rotation, center = _genome_from_catalog(
        args.genome_id)
    cg.set_genome(**kwargs)
    cg.reset_walkers(seed=0)
    print(f'genome {args.genome_id}: '
          f'{kwargs["n_transforms"]} transforms, '
          f'final_xform={kwargs["has_final_xform"]}')

    # Warm up
    for _ in range(args.warmup_frames):
        cg.frame(iterations=args.iterations, zoom=zoom,
                  rotation=rotation, center=center, decay=0.3)

    # Timed
    samples = []
    for _ in range(args.timed_frames):
        t0 = time.perf_counter()
        cg.frame(iterations=args.iterations, zoom=zoom,
                  rotation=rotation, center=center, decay=0.3)
        samples.append((time.perf_counter() - t0) * 1000.0)

    cg.cleanup()
    ctx.cleanup()

    samples.sort()
    median = samples[len(samples) // 2]
    p10 = samples[len(samples) // 10]
    p90 = samples[9 * len(samples) // 10]
    mean = statistics.mean(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0

    print()
    print(f'frames timed: {len(samples)}')
    print(f'  mean:   {mean:.2f}ms  (≈ {1000/mean:.1f} fps)')
    print(f'  median: {median:.2f}ms')
    print(f'  p10:    {p10:.2f}ms')
    print(f'  p90:    {p90:.2f}ms')
    print(f'  stdev:  {stdev:.2f}ms')


if __name__ == '__main__':
    main()
