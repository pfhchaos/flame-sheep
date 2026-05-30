"""Precompile worker — subprocess that warms Mesa's shader cache
for chaos-game pipelines so the wallpaper's first render of each
genome doesn't pay a ~200-400ms compile stutter.

How it works
------------
1. Spawned by the scheduler with the wallpaper's canvas dims on argv
   (SC_width / SC_height are spec consts, so each pipeline is
   canvas-specific).
2. Opens its OWN VkContext + ChaosGame.
3. Reads tuple specs from stdin, one JSON object per line:
       {"n_transforms": 3, "has_final_xform": false, "vars": [0,1,3,7]}
4. For each tuple, calls ChaosGame._get_chaos_pipe(...) which compiles
   and caches the pipeline. Mesa's automatic shader cache (under
   ~/.cache/mesa_shader_cache_db) persists the SPIR-V→ISA result;
   when the main wallpaper process compiles the same tuple later
   it's a 3.5ms cache hit instead of a 200-400ms cold compile (see
   tools/vk_perf_diag/cross_proc_compile_bench.py).
5. Between compiles, checks the pause flag and either continues
   (RUN), sleeps briefly (SLOW), or sleeps longer (PAUSE).
6. Exits when stdin closes (EOF) — parent's signal to clean up.

Process isolation: own context, own command pool, own everything.
Mesa-Xe + multiprocessing.fork has crashed sway in the past;
subprocess.Popen avoids that entirely.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from .pause_flag import PauseFlagReader
from .policy import BatchState

log = logging.getLogger('flame_sheep.scheduler.precompile_worker')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pause-flag', required=True,
                        help='path to scheduler pause flag mmap')
    parser.add_argument('--canvas-w', type=int, required=True)
    parser.add_argument('--canvas-h', type=int, required=True)
    parser.add_argument('--slow-delay-ms', type=int, default=500)
    parser.add_argument('--pause-poll-ms', type=int, default=200)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='[precompile %(levelname)s] %(message)s')

    # Imports are deferred until after argv parsing so --help doesn't
    # drag in a full Vulkan context.
    from viz_authoring.vk.context import VkContext
    from viz_authoring.vk.chaos_game import ChaosGame

    reader = PauseFlagReader(args.pause_flag)

    # Use the same pipeline-cache path the wallpaper does, so the
    # VkPipelineCache (in addition to Mesa's shader cache) is shared.
    # Default ctor path is ~/.local/share/flame-sheep/vulkan/pipeline_cache.bin.
    ctx = VkContext(instance_extensions=[])
    ctx.select_device()
    log.info(f'device: {ctx.device_name}')
    log.info(f'canvas: {args.canvas_w}x{args.canvas_h}')

    # ChaosGame allocates the full per-canvas buffer set. We need
    # exactly the same shape ChaosGame uses, because the pipeline
    # descriptor layout has to match what set_genome would produce
    # at render time (the cache key includes that implicitly via
    # buffer count + canvas dims). n_walkers is irrelevant to the
    # cached compile — pass the default.
    cg = ChaosGame(ctx, args.canvas_w, args.canvas_h)
    log.info('chaos game initialized; ready for tuples on stdin')

    compiled = 0
    skipped = 0
    t_start = time.perf_counter()

    # Tell the parent we're ready for the first tuple. The driver
    # thread reads READY and writes the first tuple in response.
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
                # Stay in lockstep — emit DONE so the driver moves on.
                sys.stdout.write('DONE 0 err\n')
                sys.stdout.flush()
                continue

            n_tx = int(spec['n_transforms'])
            has_final = 1 if spec.get('has_final_xform') else 0
            keep_vars = frozenset(int(v) for v in spec.get('vars', []))

            # Cooperative yield — block here while scheduler says PAUSE.
            while True:
                state = reader.state
                if state is BatchState.PAUSE:
                    time.sleep(args.pause_poll_ms / 1000.0)
                    continue
                if state is BatchState.SLOW:
                    time.sleep(args.slow_delay_ms / 1000.0)
                break

            key = (n_tx, has_final, keep_vars)
            if key in cg._chaos_pipes:
                skipped += 1
                # Ack so driver sends the next tuple.
                sys.stdout.write('DONE 0 dup\n')
                sys.stdout.flush()
                continue

            t0 = time.perf_counter()
            cg._get_chaos_pipe(n_tx, has_final, keep_vars)
            dt = (time.perf_counter() - t0) * 1000.0
            compiled += 1
            # Cache-hit heuristic: <20ms = Mesa shader cache hit;
            # higher = cold compile. Driver uses this to track win.
            cache_status = 'hit' if dt < 20.0 else 'cold'
            log.info(
                f'compiled tuple ({n_tx}, hf={has_final}, '
                f'{len(keep_vars)} vars) in {dt:.1f}ms [{cache_status}] '
                f'[total: {compiled} compiled, {skipped} skipped]')
            sys.stdout.write(f'DONE {dt:.2f} {cache_status}\n')
            sys.stdout.flush()
    except (KeyboardInterrupt, BrokenPipeError):
        # BrokenPipeError = parent closed stdout reader at shutdown
        # while we were writing a DONE ack. Expected; exit cleanly.
        pass
    finally:
        elapsed = time.perf_counter() - t_start
        log.info(f'exiting after {elapsed:.2f}s: compiled={compiled} '
                  f'skipped={skipped}')
        cg.cleanup()
        ctx.cleanup()
        reader.close()


if __name__ == '__main__':
    main()
