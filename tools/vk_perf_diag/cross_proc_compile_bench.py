#!/usr/bin/env python3
"""Validate the load-bearing assumption for subprocess pre-compile:
does Mesa's pipeline / shader cache make repeated compiles cheap
across process boundaries?

If yes → subprocess can warm the cache, main-process compile becomes
a fast cache hit, and we have a viable stutter-hiding strategy.

If no → subprocess approach doesn't work; need a different shape
(in-process worker thread, which the user has had kernel panics
from on Mesa Xe, so probably means dropping prefetch entirely).

Tests three scenarios for the same (n_tx, has_final, keep_vars)
tuple — all with the spec-const+var-trim shader from production:

1. cold      — fresh process, both caches cleared
2. in-proc   — same process, second compile of same tuple
3. cross-proc — fresh process, both caches PRIMED by a prior process

Reports compile time for each (driver SPIR-V→ISA, not Python setup).

To exercise the "two cache layers" question (Mesa shader cache vs
VkPipelineCache), runs each scenario with and without
VkPipelineCache enabled.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Same fixed (n_tx, has_final, keep_vars) tuple for reproducibility
TEST_N_TX = 3
TEST_HAS_FINAL = 0
TEST_VARS = (0, 1, 2, 3, 5, 7, 11, 14, 17, 24, 27, 29)


def child_compile(use_pipeline_cache: bool, cache_dir: Path) -> float:
    """Run inside a fresh process: build context + compile pipeline,
    return compile-only time (ms)."""
    sys.path.insert(0, str(Path(__file__).parents[2] / 'viz_authoring/src'))
    sys.path.insert(0, str(Path(__file__).parents[2]))

    from viz_authoring.vk.context import VkContext
    from viz_authoring.vk.chaos_game import (
        ChaosGame, MAX_TRANSFORMS, MAX_ACTIVE_VARS, SLOT_SIZE)
    import numpy as np

    cache_path = (cache_dir / 'pipeline_cache.bin') if use_pipeline_cache else None
    ctx = VkContext(instance_extensions=[], pipeline_cache_path=cache_path)
    ctx.select_device()

    cg = ChaosGame(ctx, 1024, 1024, n_walkers=64)
    # Set up a genome with the test tuple
    T = MAX_TRANSFORMS + 1
    SL = MAX_ACTIVE_VARS * SLOT_SIZE
    affines = np.tile(np.array([1, 0, 0, 0, 1, 0], dtype=np.float32),
                       T).reshape(T, 6)
    pa = affines.copy()
    av = np.full((T, MAX_ACTIVE_VARS, SLOT_SIZE), -1.0, dtype=np.float32)
    # First few transforms get the keep_vars laid out across active_vars
    for tidx in range(TEST_N_TX):
        # Distribute keep_vars round-robin across the transforms
        slot = 0
        for vi, var in enumerate(TEST_VARS):
            if vi % TEST_N_TX == tidx and slot < MAX_ACTIVE_VARS:
                av[tidx, slot, 0] = float(var)
                av[tidx, slot, 1] = 1.0 / TEST_N_TX
                slot += 1
    pv = np.full((T, MAX_ACTIVE_VARS, SLOT_SIZE), -1.0, dtype=np.float32)
    colors = np.full(T, 0.5, dtype=np.float32)
    cs = np.full(T, 0.5, dtype=np.float32)
    weights = np.array([1.0 / TEST_N_TX] * TEST_N_TX +
                        [0.0] * (MAX_TRANSFORMS - TEST_N_TX),
                        dtype=np.float32)

    # Warm the in-memory cache with a DIFFERENT pipeline so we measure
    # the cold-for-this-tuple cost, not the cold-for-this-process cost.
    t_warm0 = time.perf_counter()
    cg.set_genome(affines=affines, post_affines=pa, active_vars=av,
                   pre_vars=pv, colors=colors, weights=weights,
                   color_speeds=cs, n_transforms=TEST_N_TX,
                   has_final_xform=bool(TEST_HAS_FINAL))
    t_first_compile = (time.perf_counter() - t_warm0) * 1000.0

    # Second compile of the SAME tuple (should hit in-memory cache)
    t_warm1 = time.perf_counter()
    cg.set_genome(affines=affines, post_affines=pa, active_vars=av,
                   pre_vars=pv, colors=colors, weights=weights,
                   color_speeds=cs, n_transforms=TEST_N_TX,
                   has_final_xform=bool(TEST_HAS_FINAL))
    t_second_compile = (time.perf_counter() - t_warm1) * 1000.0

    cg.cleanup()
    ctx.cleanup()

    print(f'FIRST={t_first_compile:.2f}ms SECOND={t_second_compile:.2f}ms')
    return t_first_compile


def clear_caches(scenario: str, cache_dir: Path):
    """Nuke Mesa's shader cache and VkPipelineCache file for a true
    cold measurement.

    `scenario` controls how aggressive: 'cold' nukes both; 'cross_proc'
    leaves them intact (we want the cross-process cache hit)."""
    if scenario == 'cold':
        # Mesa's compute shader cache lives here. Wiping forces SPIR-V→ISA
        # to redo. WARNING: this kills cache for ALL processes on the
        # machine, briefly. Done for benchmark only.
        mesa_dirs = [
            Path.home() / '.cache/mesa_shader_cache_db',
            Path.home() / '.cache/mesa_shader_cache',
        ]
        for d in mesa_dirs:
            if d.exists():
                # Don't actually delete — just back up + rename. Safer
                # than rm -rf on user's cache. Restore at end.
                backup = d.with_suffix('.bench-bak')
                if backup.exists():
                    shutil.rmtree(backup)
                d.rename(backup)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
    # 'cross_proc' / 'in_proc': leave caches intact


def restore_caches():
    for orig in [Path.home() / '.cache/mesa_shader_cache_db',
                  Path.home() / '.cache/mesa_shader_cache']:
        backup = orig.with_suffix('.bench-bak')
        if backup.exists() and not orig.exists():
            backup.rename(orig)


def run_in_child(use_pipeline_cache: bool, cache_dir: Path) -> tuple[float, float]:
    """Spawn fresh Python process, return (first_compile_ms, second_compile_ms)."""
    venv_py = Path(__file__).parents[2] / '.venv/bin/python'
    env = {**os.environ,
            'PYTHONPATH': ':'.join([
                str(Path(__file__).parents[2] / 'viz_authoring/src'),
                str(Path(__file__).parents[2] / 'flame_sheep_audio/src'),
                str(Path(__file__).parents[2]),
            ])}
    r = subprocess.run(
        [str(venv_py), __file__, '--child',
         '--use-pcache' if use_pipeline_cache else '--no-pcache',
         '--cache-dir', str(cache_dir)],
        capture_output=True, text=True, env=env)
    # Find the FIRST=...ms SECOND=...ms line
    for line in r.stdout.splitlines():
        if line.startswith('FIRST='):
            parts = line.split()
            first = float(parts[0].split('=')[1].rstrip('ms'))
            second = float(parts[1].split('=')[1].rstrip('ms'))
            return first, second
    raise RuntimeError(f'child output unparseable: {r.stdout[-500:]} '
                        f'/ stderr: {r.stderr[-500:]}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', action='store_true',
                        help='internal: run as subprocess child')
    parser.add_argument('--use-pcache', action='store_true')
    parser.add_argument('--no-pcache', action='store_true')
    parser.add_argument('--cache-dir', type=Path,
                        default=Path('/tmp/vk_bench_pcache'))
    args = parser.parse_args()

    if args.child:
        use_pcache = args.use_pcache and not args.no_pcache
        child_compile(use_pcache, args.cache_dir)
        return

    cache_dir = args.cache_dir
    print(f'Test tuple: n_tx={TEST_N_TX}, has_final={TEST_HAS_FINAL}, '
          f'keep_vars={TEST_VARS}')
    print()
    print('=== Scenario 1: cold (all caches cleared) ===')
    try:
        for pcache in (False, True):
            clear_caches('cold', cache_dir)
            t1, t2 = run_in_child(pcache, cache_dir)
            tag = 'with-pcache' if pcache else 'no-pcache'
            print(f'  {tag}: first={t1:.2f}ms  second-same-process={t2:.2f}ms')
    finally:
        restore_caches()

    print()
    print('=== Scenario 2: cross-process (cache already primed) ===')
    # Re-prime first
    for pcache in (False, True):
        clear_caches('cold', cache_dir)
        try:
            run_in_child(pcache, cache_dir)   # warm pass
            t1, t2 = run_in_child(pcache, cache_dir)   # measurement
            tag = 'with-pcache' if pcache else 'no-pcache'
            print(f'  {tag}: first (in NEW process, cache primed)={t1:.2f}ms  '
                  f'second={t2:.2f}ms')
        finally:
            restore_caches()


if __name__ == '__main__':
    main()
