#!/usr/bin/env python3
"""Measure where Mesa-Xe Vulkan starts spilling registers on the REAL
flame_chaos.comp shader as we cap how many variations are kept in the
big switch dispatcher.

Approach: take the real flame_chaos.comp + variations.glsl (the actual
production shader the wallpaper compiles), apply a source-transform
that strips switch cases at var_idx >= N (they fall through to
`default: return p;`), and measure the resulting Mesa stats.

This is more faithful to the real workload than a synthetic generator
would be — the wrapping shader (pre-variations, post-affines, final-
xform, color blend, atomic plot) is identical to production. The only
variable is how many switch cases live in the compiled code.

The first run showed a synthetic version stayed at SIMD16 + 0 spills
even at N=100, while the REAL shader at N=127 compiles to SIMD8 with
5255 spill/fill ops. So the synthetic missed structural pressure
present in the real shader; this version goes straight to the source.

Output:
  - stdout table: N → SIMD, instr, cycles, spills, fills, sends
  - tools/vk_perf_diag/spill_threshold.csv
  - tools/vk_perf_diag/spill_threshold_N{NN}.dump per-N raw ISA
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
CSV_OUT = HERE / 'spill_threshold.csv'

# Coarser at low N, finer near the suspected knee. The full shader at
# N=127 is SIMD8 + 5255 spills, so we expect the transition somewhere
# between 30 and 100.
N_VALUES = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 105, 127]


# ---------------------------------------------------------------------------
# Subprocess runner — generates the trim transform inline (closure
# captures don't survive pickling for subprocess), drops it into the
# Vulkan compile path, captures Mesa's INTEL_DEBUG=cs stderr.
# ---------------------------------------------------------------------------

VK_RUNNER = r"""
import os, re, struct, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, '{project}/viz_authoring/src')
sys.path.insert(0, '{project}')

from viz_authoring.vk.context import VkContext
from viz_authoring.vk.chaos_game import _symmetry_inject, MAX_TRANSFORMS, MAX_ACTIVE_VARS, SLOT_SIZE
from viz_authoring.vk.pipeline import ComputePipeline

N = {n}

def trim_then_inject(src: str) -> str:
    # First apply the symmetry-group injection that chaos_game's
    # default transform would have run. Then strip switch cases at
    # var_idx >= N from apply_single_variation. The trimmed cases fall
    # through to `default: return p;` so the shader still compiles.
    src = _symmetry_inject(src)
    def drop(m):
        idx = int(m.group(1))
        return '' if idx >= N else m.group(0)
    # Match the indent-prefixed "case NN: return var_xxx(...);" lines.
    pattern = r'^[ \t]+case\s+(\d+):\s+return var_\w+\([^)]*\);\s*\n'
    return re.sub(pattern, drop, src, flags=re.MULTILINE)

# Build a ChaosGame-style pipeline against the real flame_chaos.comp.
ctx = VkContext(instance_extensions=[], pipeline_cache_path=None)
ctx.select_device()

n_pixels = 256 * 256
hist = ctx.create_buffer(n_pixels * 2 * 4, 'storage')
walkers = ctx.create_buffer(64 * 3 * 4, 'storage')
T = MAX_TRANSFORMS + 1
SL = MAX_ACTIVE_VARS * SLOT_SIZE
affines = ctx.create_buffer(T * 6 * 4, 'storage')
avars = ctx.create_buffer(T * SL * 4, 'storage')
colors = ctx.create_buffer(T * 4, 'storage')
weights = ctx.create_buffer(MAX_TRANSFORMS * 4, 'storage')
cspeeds = ctx.create_buffer(T * 4, 'storage')
thits = ctx.create_buffer(n_pixels * MAX_TRANSFORMS * 4, 'storage')
postaff = ctx.create_buffer(T * 6 * 4, 'storage')
prevars = ctx.create_buffer(T * SL * 4, 'storage')

ctx.upload(walkers, np.random.default_rng(0).uniform(-1, 1, (64, 3)).astype(np.float32))
ctx.upload(weights, np.array([1.0]+[0]*(MAX_TRANSFORMS-1), dtype=np.float32))

shader = Path('{project}/viz_authoring/src/viz_authoring/vk/shaders/flame_chaos.comp')
p = ComputePipeline(
    ctx, shader,
    buffers=[hist, walkers, affines, avars, colors, weights, cspeeds, thits, postaff, prevars],
    push_constant_size=64,
    source_transform=trim_then_inject,
)
push = struct.pack('i 4x 2f 2f 2f 4i I 2I', 1, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 256, 256, 30, 0, 0, 0, n_pixels)
ctx.dispatch_compute(p, groups_x=1, push_constants=push)
p.cleanup(); ctx.cleanup()
"""


def compile_and_dump(n: int) -> str:
    env = {**os.environ,
           'MESA_SHADER_CACHE_DISABLE': '1',
           'INTEL_DEBUG': 'cs'}
    result = subprocess.run(
        [sys.executable, '-c', VK_RUNNER.format(project=PROJECT, n=n)],
        env=env, capture_output=True, text=True)
    dump_path = HERE / f'spill_threshold_N{n:03d}.dump'
    dump_path.write_text(result.stderr)
    if result.returncode != 0:
        print(f'  WARN n={n}: exit {result.returncode}')
        print(f'    stdout tail: {result.stdout[-400:]}')
    return result.stderr


# Mesa emits one stat line per (shader, simd-variant). Capture all.
STAT_RE = re.compile(
    r'^SIMD(?P<simd>\d+) shader:\s+(?P<instr>\d+) instructions\.\s+'
    r'(?P<loops>\d+) loops?\.\s+(?P<cycles>\d+) cycles\.\s+'
    r'(?P<spills>\d+):(?P<fills>\d+) spills:fills,\s+'
    r'(?P<sends>\d+) sends',
    re.MULTILINE)


def parse_stats(text: str) -> list[dict]:
    return [m.groupdict() for m in STAT_RE.finditer(text)]


def main():
    print(f'{"N":>4}  {"SIMD":>5}  {"inst":>6}  {"cycles":>11}  '
          f'{"spills":>6}  {"fills":>6}  {"sends":>5}  {"loops":>5}')
    print('-' * 70)

    rows = []
    for n in N_VALUES:
        dump = compile_and_dump(n)
        stats = parse_stats(dump)
        # The chaos compute shader is the only compute pipeline we
        # dispatched, so all stat lines belong to it (one per SIMD
        # variant). Show each.
        if not stats:
            print(f'{n:>4}  no stats parsed; see dump')
            continue
        for s in stats:
            print(f'{n:>4}  '
                  f'SIMD{s["simd"]:>2}  '
                  f'{int(s["instr"]):>6}  '
                  f'{int(s["cycles"]):>11}  '
                  f'{int(s["spills"]):>6}  '
                  f'{int(s["fills"]):>6}  '
                  f'{int(s["sends"]):>5}  '
                  f'{int(s["loops"]):>5}')
            rows.append(dict(n=n, **{k: int(v) for k, v in s.items()}))

    if rows:
        with open(CSV_OUT, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=[
                'n', 'simd', 'instr', 'loops', 'cycles',
                'spills', 'fills', 'sends'])
            w.writeheader()
            w.writerows(rows)
        print(f'\ncsv: {CSV_OUT}')

        # Locate the knee: first N where any variant has spills > 0.
        spilled = [r for r in rows if r['spills'] > 0]
        if spilled:
            first = min(spilled, key=lambda r: (r['n'], -r['simd']))
            print(f'\nfirst N with spills: {first["n"]} '
                  f'(SIMD{first["simd"]}: '
                  f'{first["spills"]}:{first["fills"]})')
        else:
            print('\nno spills observed in any variant')

        # SIMD-drop knee: first N where SIMD8 appears alongside SIMD16.
        simds_per_n = {}
        for r in rows:
            simds_per_n.setdefault(r['n'], set()).add(r['simd'])
        simd8_n = [n for n, simds in simds_per_n.items() if 8 in simds]
        if simd8_n:
            print(f'first N where Mesa emits SIMD8: {min(simd8_n)} '
                  f'(register pressure forced narrower variant)')


if __name__ == '__main__':
    main()
