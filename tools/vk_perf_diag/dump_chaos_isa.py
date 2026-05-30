#!/usr/bin/env python3
"""Compile + dispatch the trimmed chaos shader on both backends, capturing
Mesa's Intel Gen ISA dump for side-by-side analysis.

Outputs:
  tools/vk_perf_diag/dump_vk.txt   — ANV (Vulkan)
  tools/vk_perf_diag/dump_gl.txt   — iris (OpenGL)
  tools/vk_perf_diag/dump_vk.spv   — raw SPIR-V (for spirv-dis manual inspection)

Why this exists: profiling shows the Vulkan chaos game is ~2× slower
than the GL chaos game on Mesa-Xe for the same shader. The leading
hypothesis is divergent atomic-add codegen — the GL driver may emit a
hardware-optimized atomic sequence that Mesa's ANV path doesn't apply.
This script dumps the ISA from both backends on a structurally-identical
shader so we can read the relevant fragments by hand.

Usage:
  python tools/vk_perf_diag/dump_chaos_isa.py

The script sets MESA_SHADER_CACHE_DISABLE=1 + INTEL_DEBUG=cs internally
— Mesa's cache otherwise skips the recompile and silently produces an
empty dump.
"""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VK_SRC = HERE / 'trimmed_chaos_vk.comp'
GL_SRC = HERE / 'trimmed_chaos_gl.comp'
VK_DUMP = HERE / 'dump_vk.txt'
GL_DUMP = HERE / 'dump_gl.txt'
SPIRV_RAW = HERE / 'dump_vk.spv'
SPIRV_DIS = HERE / 'dump_vk.spvdis'


# ---------------------------------------------------------------------------
# Vulkan path
# ---------------------------------------------------------------------------

VK_RUNNER = r'''
import struct, numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '{project}/viz_authoring/src')
from viz_authoring.vk.context import VkContext
from viz_authoring.vk.pipeline import ComputePipeline

ctx = VkContext(instance_extensions=[], pipeline_cache_path=None)
ctx.select_device()
N_WALKERS = 64
hist = ctx.create_buffer(64 * 64 * 4, 'storage')
walkers = ctx.create_buffer(N_WALKERS * 3 * 4, 'storage')
rng = np.random.default_rng(0)
ctx.upload(walkers, rng.uniform(-1, 1, (N_WALKERS, 3)).astype(np.float32))
ctx.zero_buffer(hist)
p = ComputePipeline(ctx, Path('{vk_src}'),
                     buffers=[hist, walkers], push_constant_size=16)
push = struct.pack('3iI', 64, 64, 30, 12345)
ctx.dispatch_compute(p, groups_x=N_WALKERS // 64, push_constants=push)
p.cleanup(); ctx.cleanup()
'''


def dump_vulkan():
    print('--- Vulkan / ANV ---')
    env = {**os.environ,
           'MESA_SHADER_CACHE_DISABLE': '1',
           'INTEL_DEBUG': 'cs'}
    runner = VK_RUNNER.format(project=HERE.parents[1], vk_src=VK_SRC)
    result = subprocess.run(
        [sys.executable, '-c', runner],
        env=env, capture_output=True, text=True)
    # The dump comes out on stderr from Mesa
    VK_DUMP.write_text(result.stderr)
    print(f'  wrote {VK_DUMP}: {VK_DUMP.stat().st_size} bytes')
    if result.returncode != 0:
        print(f'  WARNING: exit code {result.returncode}')
        print(f'  stdout: {result.stdout[:500]}')

    # Also dump the SPIR-V via compile_shader + spirv-dis for cross-check.
    sys.path.insert(0, str(HERE.parents[1] / 'viz_authoring' / 'src'))
    from viz_authoring.vk.shader import compile_shader  # noqa
    spv = compile_shader(VK_SRC)
    SPIRV_RAW.write_bytes(spv)
    if shutil.which('spirv-dis'):
        dis = subprocess.run(['spirv-dis', str(SPIRV_RAW)],
                              capture_output=True, text=True)
        SPIRV_DIS.write_text(dis.stdout)
        print(f'  wrote {SPIRV_DIS}: {SPIRV_DIS.stat().st_size} bytes')


# ---------------------------------------------------------------------------
# GL path — use moderngl with a standalone EGL context
# ---------------------------------------------------------------------------

GL_RUNNER = r'''
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import moderngl
import numpy as np
ctx = moderngl.create_context(standalone=True, backend='egl')
src = open('{gl_src}').read()
shader = ctx.compute_shader(src)
N_WALKERS = 64
hist = ctx.buffer(reserve=64 * 64 * 4)
walkers = ctx.buffer(np.random.default_rng(0).uniform(-1, 1, (N_WALKERS, 3))
                      .astype(np.float32).tobytes())
hist.bind_to_storage_buffer(0)
walkers.bind_to_storage_buffer(1)
shader['u_width'] = 64
shader['u_height'] = 64
shader['u_iterations'] = 30
shader['u_rng_seed'] = 12345
shader.run(group_x=N_WALKERS // 64)
ctx.finish()
'''


def dump_gl():
    print('--- OpenGL / iris ---')
    env = {**os.environ,
           'MESA_SHADER_CACHE_DISABLE': '1',
           'INTEL_DEBUG': 'cs',
           'MESA_GLSL_CACHE_DISABLE': 'true'}
    runner = GL_RUNNER.format(gl_src=GL_SRC)
    result = subprocess.run(
        [sys.executable, '-c', runner],
        env=env, capture_output=True, text=True)
    GL_DUMP.write_text(result.stderr)
    print(f'  wrote {GL_DUMP}: {GL_DUMP.stat().st_size} bytes')
    if result.returncode != 0:
        print(f'  WARNING: exit code {result.returncode}')
        print(f'  stdout: {result.stdout[:500]}')


VK_FULL_DUMP = HERE / 'dump_vk_full.txt'
GL_FULL_DUMP = HERE / 'dump_gl_full.txt'

VK_FULL_RUNNER = r'''
import sys
sys.path.insert(0, '{project}/viz_authoring/src')
sys.path.insert(0, '{project}')
import struct, numpy as np
from viz_authoring.vk.context import VkContext
from viz_authoring.vk.chaos_game import ChaosGame

# Building ChaosGame triggers compile of the full flame_chaos.comp
# (with symmetry-group injection + variations.glsl include resolution).
ctx = VkContext(instance_extensions=[], pipeline_cache_path=None)
ctx.select_device()
cg = ChaosGame(ctx, 256, 256, n_walkers=64)
# Run a frame so the driver actually dispatches the pipeline.
cg.frame(iterations=30)
cg.cleanup(); ctx.cleanup()
'''


def dump_vulkan_full():
    print('--- Vulkan / ANV — FULL flame_chaos.comp ---')
    env = {**os.environ,
           'MESA_SHADER_CACHE_DISABLE': '1',
           'INTEL_DEBUG': 'cs'}
    runner = VK_FULL_RUNNER.format(project=HERE.parents[1])
    result = subprocess.run(
        [sys.executable, '-c', runner],
        env=env, capture_output=True, text=True)
    VK_FULL_DUMP.write_text(result.stderr)
    print(f'  wrote {VK_FULL_DUMP}: {VK_FULL_DUMP.stat().st_size} bytes')


GL_FULL_RUNNER = r'''
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import sys
sys.path.insert(0, '{project}')
import numpy as np
import moderngl
from flame_sheep.rendering import FlameRenderer, GpuContext
from flame_sheep_audio import N_BINS

ctx = moderngl.create_context(standalone=True, backend='egl')
renderer = FlameRenderer(GpuContext(ctx, 256, 256))
# Force a chaos-game dispatch so the driver actually compiles the shader.
from flame_sheep.genome import Genome
g = Genome.random(np.random.default_rng(42))
renderer.upload_genome(g)
renderer.upload_audio(np.zeros(N_BINS, dtype=np.float32))
renderer.reset_walkers()
renderer.clear_histogram()
renderer.dispatch_chaos_game(iterations=30)
ctx.finish()
'''


def dump_gl_full():
    print('--- OpenGL / iris — FULL flame.comp ---')
    env = {**os.environ,
           'MESA_SHADER_CACHE_DISABLE': '1',
           'INTEL_DEBUG': 'cs',
           'MESA_GLSL_CACHE_DISABLE': 'true'}
    runner = GL_FULL_RUNNER.format(project=HERE.parents[1])
    result = subprocess.run(
        [sys.executable, '-c', runner],
        env=env, capture_output=True, text=True)
    GL_FULL_DUMP.write_text(result.stderr)
    print(f'  wrote {GL_FULL_DUMP}: {GL_FULL_DUMP.stat().st_size} bytes')
    if result.returncode != 0:
        print(f'  WARN: exit={result.returncode}; stdout={result.stdout[:300]}')


if __name__ == '__main__':
    dump_vulkan()
    dump_gl()
    dump_vulkan_full()
    dump_gl_full()

    print()
    print('Next steps for the IR diff:')
    print(f'  spirv-dis output  : {SPIRV_DIS}')
    print(f'  Vulkan Gen ISA    : {VK_DUMP}')
    print(f'  OpenGL Gen ISA    : {GL_DUMP}')
    print(f'  diff side-by-side : diff -y {VK_DUMP} {GL_DUMP} | less -R')
    print()
    print('Focus area: search the ISA dumps for "ugm" (Unified Global '
          'Memory) messages with "atomic" in the descriptor — those are '
          'the hardware atomic-add operations.')
