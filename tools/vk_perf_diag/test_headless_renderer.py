#!/usr/bin/env python3
"""Smoke test for HeadlessVkRenderer: load a catalog genome, render
both histogram and PNG, write PNG to /tmp.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / 'viz_authoring/src'))
sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np

from viz_authoring.vk.context import VkContext
from viz_authoring.vk.headless import HeadlessVkRenderer
from viz_authoring.vk.wallpaper_demo import _genome_from_catalog


def main():
    ctx = VkContext(instance_extensions=[], pipeline_cache_path=None)
    ctx.select_device()
    print(f'device: {ctx.device_name}')

    renderer = HeadlessVkRenderer(ctx, 512, 512, n_walkers=14720)
    print('renderer created')

    kwargs, palette, _zoom, _rot, _center = _genome_from_catalog(2505)
    print(f'genome 2505 loaded: {kwargs["n_transforms"]} transforms')

    renderer.set_genome(**kwargs)
    renderer.set_palette(palette)
    renderer.reset_walkers(seed=0)

    # Burn-in then accumulate a fresh render
    renderer.clear_histogram()
    renderer.dispatch_chaos_game(iterations=50)
    renderer.clear_histogram()
    renderer.dispatch_chaos_game(iterations=400)

    hits, color_acc = renderer.download_histogram()
    print(f'histogram: shape={hits.shape} max={hits.max()} '
          f'mean={hits.mean():.2f} nonzero={(hits>0).sum()}')

    transform_hits = renderer.download_transform_hits()
    print(f'transform_hits: shape={transform_hits.shape} '
          f'max={transform_hits.max()}')

    t0 = time.perf_counter()
    png_bytes = renderer.snapshot_png(gamma=6.0)
    dt = time.perf_counter() - t0
    print(f'snapshot_png: {len(png_bytes)} bytes in {dt*1000:.1f}ms')

    out = Path('/tmp/headless_test.png')
    out.write_bytes(png_bytes)
    print(f'wrote {out}')

    renderer.cleanup()
    ctx.cleanup()


if __name__ == '__main__':
    main()
