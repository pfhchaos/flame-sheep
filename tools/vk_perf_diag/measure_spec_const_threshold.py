#!/usr/bin/env python3
"""Re-run the spill-threshold sweep on the SPEC-CONST variant of
flame_chaos.comp and compare to the plain push-constant version.

Same approach as measure_spill_threshold.py — source-transform strips
switch cases at var_idx ≥ N, capture Mesa stats per N. The only
difference: the shader is flame_chaos_specconst.comp with four
specialization constants (SC_n_transforms, SC_has_final_xform,
SC_width, SC_height) bound at pipeline-creation time.

Specialization values chosen to match the prior measurement's
implicit settings (n_transforms=1, has_final_xform=0, 256×256 canvas)
so the per-N rows are apples-to-apples. The hypothesis is that the
spec-const'd shader stays in SIMD16 to a higher N because the driver
can eliminate the pick_transform loop's runtime branching and the
final-xform code block entirely.
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
CSV_OUT = HERE / 'spec_const_threshold.csv'

# Same N grid as the unfixed measurement for a clean diff.
N_VALUES = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 105, 127]


VK_RUNNER = r"""
import os, re, struct, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, '{project}/viz_authoring/src')
sys.path.insert(0, '{project}')

from viz_authoring.vk.context import VkContext
from flame_sheep.rendering.vk.chaos_game import _symmetry_inject, MAX_TRANSFORMS, MAX_ACTIVE_VARS, SLOT_SIZE
from viz_authoring.vk.pipeline import ComputePipeline

N = {n}

def trim_then_inject(src: str) -> str:
    src = _symmetry_inject(src)
    def drop(m):
        idx = int(m.group(1))
        return '' if idx >= N else m.group(0)
    pattern = r'^[ \t]+case\s+(\d+):\s+return var_\w+\([^)]*\);\s*\n'
    return re.sub(pattern, drop, src, flags=re.MULTILINE)

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

shader = Path('{project}/flame_sheep/rendering/vk/shaders/flame_chaos_specconst.comp')
# Push constant layout shrinks: u_zoom + cos/sin + center + iter +
# seed + offset + stride = 2f+1f+1f+2f+1i+1u+1u+1u = 44 bytes
p = ComputePipeline(
    ctx, shader,
    buffers=[hist, walkers, affines, avars, colors, weights, cspeeds, thits, postaff, prevars],
    push_constant_size=48,
    source_transform=trim_then_inject,
    specialization={{
        0: 1,    # SC_n_transforms
        1: 0,    # SC_has_final_xform
        2: 256,  # SC_width
        3: 256,  # SC_height
    }},
)
push = struct.pack('2f 2f 2f i I 2I',
                    1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 30, 0, 0, n_pixels)
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
    dump_path = HERE / f'spec_const_threshold_N{n:03d}.dump'
    dump_path.write_text(result.stderr)
    if result.returncode != 0:
        print(f'  WARN n={n}: exit {result.returncode}')
        print(f'    stdout tail: {result.stdout[-300:]}')
    return result.stderr


STAT_RE = re.compile(
    r'^SIMD(?P<simd>\d+) shader:\s+(?P<instr>\d+) instructions\.\s+'
    r'(?P<loops>\d+) loops?\.\s+(?P<cycles>\d+) cycles\.\s+'
    r'(?P<spills>\d+):(?P<fills>\d+) spills:fills,\s+'
    r'(?P<sends>\d+) sends',
    re.MULTILINE)


def parse_stats(text):
    return [m.groupdict() for m in STAT_RE.finditer(text)]


def main():
    print(f'{"N":>4}  {"SIMD":>5}  {"inst":>6}  {"cycles":>11}  '
          f'{"spills":>6}  {"fills":>6}  {"sends":>5}  {"loops":>5}')
    print('-' * 70)
    rows = []
    for n in N_VALUES:
        dump = compile_and_dump(n)
        stats = parse_stats(dump)
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

        spilled = [r for r in rows if r['spills'] > 0]
        if spilled:
            first = min(spilled, key=lambda r: (r['n'], -r['simd']))
            print(f'first N with spills: {first["n"]} (SIMD{first["simd"]})')
        simds = {}
        for r in rows:
            simds.setdefault(r['n'], set()).add(r['simd'])
        simd8 = [n for n, s in simds.items() if 8 in s]
        if simd8:
            print(f'first N where Mesa emits SIMD8: {min(simd8)}')


if __name__ == '__main__':
    main()
