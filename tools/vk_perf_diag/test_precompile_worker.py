#!/usr/bin/env python3
"""Standalone smoke test for flame_sheep.scheduler.precompile_worker.

Spawns the worker, feeds it 5 fake (n_tx, has_final, var_set) tuples,
times each. After the worker exits, spawns it AGAIN and feeds the
same tuples — the second run should hit Mesa's cache and compile
near-instantly. That validates the precompile-warms-main-render
hypothesis end-to-end on the real production canvas + shader.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from flame_sheep.scheduler.pause_flag import PauseFlag
from flame_sheep.scheduler.policy import BatchState

TUPLES = [
    {'n_transforms': 2, 'has_final_xform': False,
     'vars': [0, 1, 7, 13]},
    {'n_transforms': 3, 'has_final_xform': False,
     'vars': [0, 1, 2, 3, 7, 11, 14, 17]},
    {'n_transforms': 4, 'has_final_xform': True,
     'vars': [0, 1, 2, 5, 9, 12, 18, 22, 27, 33, 41]},
    {'n_transforms': 5, 'has_final_xform': False,
     'vars': [0, 2, 4, 6, 8, 10, 14, 18, 24, 30, 36, 42, 48]},
    {'n_transforms': 6, 'has_final_xform': True,
     'vars': [0, 1, 3, 5, 7, 11, 14, 17, 24, 27, 29, 32, 38, 44]},
]


def run_pass(label: str, canvas_w: int, canvas_h: int):
    print(f'\n=== {label} ===')
    with PauseFlag('precompile-test') as flag:
        flag.set(BatchState.RUN)
        argv = [
            sys.executable, '-m', 'flame_sheep.scheduler.precompile_worker',
            '--pause-flag', str(flag.path),
            '--canvas-w', str(canvas_w),
            '--canvas-h', str(canvas_h),
        ]
        p = subprocess.Popen(argv, stdin=subprocess.PIPE, text=True)
        t0 = time.perf_counter()
        for t in TUPLES:
            p.stdin.write(json.dumps(t) + '\n')
            p.stdin.flush()
        p.stdin.close()  # signal EOF → worker exits cleanly
        rc = p.wait()
        elapsed = time.perf_counter() - t0
        print(f'[wrapper] worker exited rc={rc} after {elapsed:.2f}s '
              f'({len(TUPLES)} tuples)')


def main():
    canvas_w = 3713   # matches user's actual wallpaper canvas
    canvas_h = 1278

    # Pass 1: cold or partial cache (depending on what's lying around
    # in ~/.cache/mesa_shader_cache_db). This pass exists to warm it.
    run_pass('Pass 1 (cold or partial)', canvas_w, canvas_h)

    # Pass 2: cache should be hot — compile times should drop sharply
    run_pass('Pass 2 (cache warm)', canvas_w, canvas_h)


if __name__ == '__main__':
    main()
