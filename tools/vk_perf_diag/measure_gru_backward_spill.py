#!/usr/bin/env python3
"""Measure Mesa-Xe Vulkan ISA stats for gru_seq_backward.comp.

The hypothesis under test: the single-dispatch BPTT backward shader is
register-spilling (similar to the chaos shader pre-trim), and that
spillage is part of why beat-RNN training + wallpaper wedges the GuC
scheduler. If confirmed, the structural fix is the same shape as the
chaos shader fix — break it into smaller per-shader bodies, here by
chunking timesteps across multiple dispatches.

T, H, and I are all push constants (runtime), so the inner loops don't
unroll. Per-shader register pressure is structural — it doesn't depend
on T. So one compile is enough to characterize the spill regime.

For reference, the chaos shader's spill threshold (see
spill_threshold.csv) was N=75 (first spills) → N=127 (1929:3326 spills,
SIMD8, 61M cycles, the unusable end of the range that motivated trim).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DUMP_OUT = HERE / 'gru_backward_isa.dump'


VK_RUNNER = r"""
import sys
from pathlib import Path
sys.path.insert(0, '{project}/wallpaper_ml/src')

from wallpaper_ml.vk_compute import VkCompute

gpu = VkCompute()

# Bindings 0..9 — sizes are nominal; pipeline compile doesn't care about
# real shape, only that the descriptor types and count match. Use a few
# KB each, more than enough for the compiler to lay out the load/store
# accesses.
NUM = 10
bufs = [gpu.create_buffer(64 * 1024, usage='storage') for _ in range(NUM)]

shader = Path('{project}/wallpaper_ml/src/wallpaper_ml/shaders/rnn/gru_seq_backward.comp')
p = gpu.create_pipeline(str(shader), buffers=bufs, push_constant_size=16)
# No dispatch needed — pipeline create already triggered Mesa compile.
"""


STAT_RE = re.compile(
    r'^SIMD(?P<simd>\d+) shader:\s+(?P<instr>\d+) instructions\.\s+'
    r'(?P<loops>\d+) loops?\.\s+(?P<cycles>\d+) cycles\.\s+'
    r'(?P<spills>\d+):(?P<fills>\d+) spills:fills,\s+'
    r'(?P<sends>\d+) sends',
    re.MULTILINE)


def main():
    env = {**os.environ,
           'MESA_SHADER_CACHE_DISABLE': '1',
           'INTEL_DEBUG': 'cs'}
    print('compiling gru_seq_backward.comp with INTEL_DEBUG=cs ...')
    result = subprocess.run(
        [sys.executable, '-c', VK_RUNNER.format(project=PROJECT)],
        env=env, capture_output=True, text=True)
    DUMP_OUT.write_text(result.stderr)
    if result.returncode != 0:
        print(f'WARN: subprocess exit {result.returncode}')
        print(f'stdout: {result.stdout[-500:]}')
        print(f'stderr tail: {result.stderr[-500:]}')

    stats = [m.groupdict() for m in STAT_RE.finditer(result.stderr)]
    if not stats:
        print('no Mesa stat lines parsed; see', DUMP_OUT)
        return 1

    print()
    print(f'  {"SIMD":>5}  {"inst":>6}  {"cycles":>11}  '
          f'{"spills":>6}  {"fills":>6}  {"sends":>5}  {"loops":>5}')
    print('  ' + '-' * 60)
    for s in stats:
        print(f'  SIMD{s["simd"]:>2}  '
              f'{int(s["instr"]):>6}  '
              f'{int(s["cycles"]):>11}  '
              f'{int(s["spills"]):>6}  '
              f'{int(s["fills"]):>6}  '
              f'{int(s["sends"]):>5}  '
              f'{int(s["loops"]):>5}')

    print()
    print('reference (chaos shader trim_threshold.csv):')
    print('  N= 50 (under threshold)   SIMD8  5357 inst   0:0  spills   5.4M cycles')
    print('  N= 75 (knee)              SIMD8 11914 inst 323:429 spills 14.4M cycles')
    print('  N=127 (full untrimmed)    SIMD8 43377 inst 1929:3326 spills 61.6M cycles')
    print()
    print(f'dump: {DUMP_OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
