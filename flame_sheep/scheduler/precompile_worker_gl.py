"""GL precompile worker — subprocess that warms Mesa's GL shader
cache for FlameRenderer chaos programs so the wallpaper's first
render of each new variation set doesn't pay a ~600ms compile
stutter.

Backend-twin of precompile_worker.py (Vk). Same protocol, same
scheduler integration, swaps ChaosGame for FlameRenderer. The
backend-agnostic scheduler infrastructure (PauseFlag, Policy,
PrecompileDriver, near_tuples) doesn't change.

How it works
------------
1. Spawned by the wallpaper with a pause-flag path on argv.
2. Opens its OWN headless moderngl context + GpuContext + FlameRenderer.
3. Reads tuple specs from stdin, one JSON object per line:
       {"n_transforms": 3, "has_final_xform": false, "vars": [0,1,3,7]}
4. For each tuple, calls
   FlameRenderer._get_chaos_shader_for_keep_vars(keep_vars) which
   compiles + caches. Mesa's automatic GL shader cache (under
   ~/.cache/mesa_shader_cache) persists the compile result; when
   the main wallpaper process compiles the same key later it's a
   ~5ms cache hit instead of 600ms.
5. Writes a warm marker file (shared with the wallpaper via
   flame_sheep.rendering.vk.pipeline_warm) so FlameRenderer.is_shader_warm()
   on the wallpaper side returns True for compiled keys.
6. Between compiles, checks the pause flag and yields per the
   wallpaper-signal scheduler.

Process isolation: own context, own everything. The kernel-panic
risk the user repeatedly flagged is THREADS sharing a GL context,
not subprocess EGL contexts. Per the configurability-discipline
memory, this is the proven-reliable pattern.

Canvas dimensions aren't argv args (unlike the Vk worker) because
GL FlameRenderer doesn't use spec constants tied to canvas size —
the trim is purely source-level. A nominal small canvas is used
for the headless context's buffer allocation, just to satisfy
FlameRenderer's init contract.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from .pause_flag import PauseFlagReader
from .policy import BatchState

log = logging.getLogger('flame_sheep.scheduler.precompile_worker_gl')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pause-flag', required=True,
                        help='path to scheduler pause flag mmap')
    parser.add_argument('--slow-delay-ms', type=int, default=500)
    parser.add_argument('--pause-poll-ms', type=int, default=200)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='[precompile-gl %(levelname)s] %(message)s')

    # Defer GL imports until after argv parsing so --help doesn't
    # drag in a full moderngl context.
    import moderngl
    from viz_authoring.gpu_context import GpuContext
    from flame_sheep.rendering import FlameRenderer

    reader = PauseFlagReader(args.pause_flag)

    # Headless standalone GL context. Canvas dimensions are nominal —
    # FlameRenderer allocates buffers sized for them but we never
    # dispatch chaos, so the buffer contents don't matter. Smaller is
    # cheaper at init.
    ctx = moderngl.create_standalone_context()
    log.info(f'GL renderer: {ctx.info["GL_RENDERER"]}')
    gpu = GpuContext(ctx, canvas_w=256, canvas_h=256, ppmm=1.0)
    renderer = FlameRenderer(gpu)
    log.info('FlameRenderer initialized; ready for tuples on stdin')

    compiled = 0
    skipped = 0
    # Keys compiled this run, for dedup. We deliberately do NOT retain the
    # compiled programs (see the release in the loop below), so we can't
    # dedup against renderer._chaos_shader_cache — track keys in a cheap
    # set of frozensets instead (bytes, not ~0.9MB/key of GL program).
    seen: set[frozenset] = set()
    t_start = time.perf_counter()

    sys.stdout.write('READY\n')
    sys.stdout.flush()

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                spec = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning(f'bad json: {e}: {line[:200]}')
                sys.stdout.write('DONE 0 err\n')
                sys.stdout.flush()
                continue

            # n_transforms / has_final_xform are part of the Vk
            # tuple protocol (spec constants there) but don't change
            # the GL shader — only vars matters. Accept them in the
            # JSON for protocol compatibility, then ignore.
            keep_vars = frozenset(int(v) for v in spec.get('vars', []))

            while True:
                state = reader.state
                if state is BatchState.PAUSE:
                    time.sleep(args.pause_poll_ms / 1000.0)
                    continue
                if state is BatchState.SLOW:
                    time.sleep(args.slow_delay_ms / 1000.0)
                break

            if keep_vars in seen:
                skipped += 1
                sys.stdout.write('DONE 0 dup\n')
                sys.stdout.flush()
                continue

            t0 = time.perf_counter()
            prog = renderer._get_chaos_shader_for_keep_vars(keep_vars)
            dt = (time.perf_counter() - t0) * 1000.0
            # Our ONLY deliverable is that compile's side effects: Mesa's
            # on-disk shader cache is now warm and the warm marker is
            # written. The in-memory GL program never renders in this
            # process, so keeping it just accumulates Mesa-compiled code
            # (~0.9MB/key; 14.9k keys -> ~8GB of swapped-out heap observed
            # on a long run). Drop it immediately.
            renderer._chaos_shader_cache.pop(keep_vars, None)
            prog.release()
            seen.add(keep_vars)
            compiled += 1
            cache_status = 'hit' if dt < 20.0 else 'cold'
            log.info(
                f'compiled keep_vars=({len(keep_vars)} vars) in '
                f'{dt:.1f}ms [{cache_status}] '
                f'[total: {compiled} compiled, {skipped} skipped]')
            sys.stdout.write(f'DONE {dt:.2f} {cache_status}\n')
            sys.stdout.flush()
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        elapsed = time.perf_counter() - t_start
        log.info(f'exiting after {elapsed:.2f}s: compiled={compiled} '
                  f'skipped={skipped}')
        reader.close()


if __name__ == '__main__':
    main()
