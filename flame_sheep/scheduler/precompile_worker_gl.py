"""GL precompile worker — subprocess that warms Mesa's GL shader
cache for FlameRenderer chaos programs so the wallpaper's first
render of each new variation set doesn't pay a ~600ms compile
stutter.

Backend-twin of precompile_worker.py (Vk). It swaps ChaosGame for
FlameRenderer; the backend-agnostic scheduler infrastructure
(PauseFlag, Policy, near_tuples) doesn't change.

How it works (step 2 of the shader-warming redesign)
----------------------------------------------------
Step 1 kept the ``queue.PriorityQueue`` + dedup in ``PrecompileDriver``
and streamed one compile-at-a-time to the worker. **Step 2 moves the
scheduling INTO the worker** — the worker now owns the queue, the
``(priority, counter, tuple)`` tie-break, the ``_seen`` + ``is_warm``
dedup, and the pop-highest-priority → compile → release loop. The
driver is reduced to a thin IPC shim that serializes requests to our
stdin.

Two threads run here:
  * a GL-FREE **stdin reader thread** (``_stdin_loop`` → ``dispatch_line``)
    that parses each request line and mutates the ``CompileQueue``. It
    never touches GL and never touches the Library (so SQLite stays
    single-threaded on the main thread). This is the thread that lets a
    late high-priority enqueue (the -100 reactive cold-miss) overtake
    queued work: it is received even while the main thread is mid-compile,
    so it is popped next — the same "re-prioritization reflects in the
    next compile" guarantee step 1 got from the one-in-flight handshake,
    now achieved naturally by the in-worker queue.
  * the **main compile loop** (owns the GL context + Library): pops the
    highest-priority task, and either compiles it (disk-cache warm +
    marker write) or, for a flood task, computes ``near_tuples`` with
    its own Library and SELF-SCHEDULES the resulting tuples back into the
    queue at LOW. The flood round-trip to the driver (step 1's
    FLOOD/FLOOD_DONE back-channel) is gone — the computation is now a
    task on the worker's own queue.

Request protocol (driver → worker stdin, one JSON object per line):
  * ``{"enqueue": {"priority": P, "tuples": [[n, hf, [v..]], ...]}}``
  * ``{"flood":   {"seed_ids": [..], "hops": 2, "fan_per_node": 5,
                    "priority": P}}``
  * ``{"plan":    {"req": R, "arrival_id": G, "loop_id": L,
                    "loop_step": S}}`` — step 3. Routed to a DEDICATED planner
    thread (not the compile queue), which computes the 3 next-move targets from
    G and writes a ``{"plan": {...}}`` response to stdout. A separate thread so
    the PLAN can never sit behind an in-flight cold compile (resolved decision
    #3). See ``scheduler.plan_protocol`` for the wire shapes.
  * ``{"n_transforms": .., "has_final_xform": .., "vars": [..]}`` — a
    single bare compile spec, accepted for back-compat / manual piping.

Policy throttling is UNCHANGED by this step. The Policy decision lives in
the driver (it owns the render-side signal sampler) and is published to
the shared ``PauseFlag`` mmap; the worker continues to read that flag
between compiles and PAUSE/SLOW accordingly. Moving the queue into the
worker is orthogonal to throttling — the throttle was always the pause
flag, never the stdin handshake.

Process isolation: own GL context, own Library, own everything. The
kernel-panic risk the user repeatedly flagged is THREADS sharing a GL
context, not subprocess EGL contexts. The reader thread added here is
GL-free and Library-free, so neither invariant is violated.

Canvas dimensions aren't argv args (unlike the Vk worker) because GL
FlameRenderer doesn't use spec constants tied to canvas size — the trim
is purely source-level. A nominal small canvas is used for the headless
context's buffer allocation, just to satisfy FlameRenderer's init
contract.
"""
from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import threading
import time

from .pause_flag import PauseFlagReader
from .policy import BatchState
from .precompile_warm import near_tuples, tuple_from_wire

log = logging.getLogger('flame_sheep.scheduler.precompile_worker_gl')

# Priority the self-scheduled flood tuples are enqueued at — LOW, the
# cushion for walks that wander past the loop members. Matches the
# priority the step-1 driver re-enqueued streamed flood tuples at.
_FLOOD_RESULT_PRIORITY = 10

# Internal queue payload sentinel distinguishing a flood-compute task
# from a compile task (a bare (n, hf, keep_vars) tuple). A compile
# tuple's first field is an int count, never this string, so the two
# never collide — and the monotonic counter in the queue key means two
# payloads are never actually compared (see CompileQueue).
_FLOOD_TASK = '__flood__'


def compute_flood(lib, flood_spec: dict) -> set[tuple[int, int, frozenset[int]]]:
    """Compute the 2-hop graph flood for a loop-change request.

    This is the DB+math work that used to run SYNCHRONOUSLY on the render
    thread (~31 load_genome + nearest_transitions queries). It runs in
    the worker PROCESS, on the main thread that owns the Library (so the
    single SQLite connection stays thread-local). GPU-free and pure —
    no GL context is touched — so it is directly unit-testable with a
    FakeLibrary.

    Step 2: the result is SELF-SCHEDULED into the worker's own queue by
    the caller (main loop) — there is no FLOOD/FLOOD_DONE stream back to
    the driver anymore. On empty seeds, returns an empty set. On error,
    logs and returns an empty set (the task is simply dropped).
    """
    try:
        seed_ids = [int(s) for s in flood_spec.get('seed_ids', [])]
        if not seed_ids:
            return set()
        hops = int(flood_spec.get('hops', 2))
        fan_per_node = int(flood_spec.get('fan_per_node', 5))
        return near_tuples(lib, seed_ids, hops=hops, fan_per_node=fan_per_node)
    except Exception:
        log.exception('flood computation failed')
        return set()


class CompileQueue:
    """The compile scheduler, moved out of ``PrecompileDriver`` (step 2).

    Owns the ``queue.PriorityQueue``, the ``(priority, counter, payload)``
    tie-break, and the two dedup layers the driver used to own:
      * ``_seen`` — a set of full ``(n, has_final, keep_vars)`` tuples, so
        a tuple already queued this run is not re-queued;
      * ``pipeline_warm.is_warm`` — a tuple whose Mesa disk cache was
        already populated in any prior session is skipped (recompiling it
        would just re-touch the same disk cache for no benefit).
    Both are exactly the checks ``PrecompileDriver.enqueue`` did in step 1,
    relocated verbatim so dedup/priority behavior is equivalent.

    GL-free: ``enqueue`` / ``enqueue_flood`` / ``get`` are pure data ops,
    safe to call from the stdin reader thread. The counter is monotonic
    under ``_lock``, so it is always unique and the queue never compares
    two payloads (which would TypeError on frozenset/str/tuple mixes).
    """

    def __init__(self):
        self._q: queue.PriorityQueue = queue.PriorityQueue()
        # tuple -> best (lowest) priority queued for it this run. PRIORITY-AWARE
        # dedup: a plain set would drop a later re-request outright, silently
        # swallowing the reactive -100 "about to show this" warm whenever its
        # target was already queued at LOW(10) by the loop-change flood -> the
        # candidate stays buried behind the flood and the render thread cold-
        # compiles it. Keyed on the best priority instead, an UPGRADE re-queues.
        self._seen: dict[tuple[int, int, frozenset[int]], int] = {}
        self._counter = 0
        self._lock = threading.Lock()

    def enqueue(self, tuples, priority: int = 0) -> int:
        """Dedup + push compile tasks at ``priority`` (lower = served first).
        Returns the number actually enqueued (after the priority-aware _seen +
        is_warm skipping).

        A tuple already queued is re-pushed ONLY if ``priority`` is strictly
        better (lower) than the best it was queued at — the imminent-warm
        preemption. ``queue.PriorityQueue`` has no decrease-key, so the upgrade
        is a second push; the stale worse-priority entry pops later and is
        skipped by the main loop's compile-time ``seen_keepvars`` dedup."""
        from flame_sheep.rendering.vk import pipeline_warm
        added = 0
        with self._lock:
            for t in tuples:
                prev = self._seen.get(t)
                if prev is not None and prev <= priority:
                    continue                      # same-or-worse -> already covered
                self._seen[t] = priority          # new, or an upgrade
                n_tx, has_final, keep_vars = t
                # Warm markers are keyed (0, False, keep_vars): for the GL chaos
                # shader only keep_vars matters, so that is how both renderer
                # (_get_chaos_shader_for_keep_vars / is_shader_warm) and the
                # worker WRITE them. Check with the SAME key — checking with the
                # real (n_tx, has_final) means a real genome tuple (n_tx != 0)
                # never matches its own marker and the disk-warm skip is inert.
                if pipeline_warm.is_warm(0, False, keep_vars):
                    continue
                self._q.put((priority, self._counter, t))
                self._counter += 1
                added += 1
        return added

    def enqueue_flood(self, flood_spec: dict) -> None:
        """Schedule a flood-COMPUTE task (not a compile) at the request's
        priority. The main loop runs ``compute_flood`` when it pops this,
        then self-schedules the results at LOW. Deliberately NOT deduped:
        a flood is a one-shot computation, and its product tuples are
        deduped when they re-enter via ``enqueue``."""
        priority = int(flood_spec.get('priority', 0))
        with self._lock:
            self._q.put((priority, self._counter, (_FLOOD_TASK, flood_spec)))
            self._counter += 1

    def get(self, timeout: float):
        """Pop the highest-priority task, or raise ``queue.Empty`` after
        ``timeout`` seconds."""
        return self._q.get(timeout=timeout)


def dispatch_line(line: str, cq: CompileQueue, plan_queue=None) -> str:
    """GL-FREE stdin routing: classify one request line and apply it to
    the compile queue (or, for a PLAN request, the planner queue). Returns a
    short tag naming the branch taken, for logging and for the mutation-provable
    routing unit test:

        'flood'    — a {"flood": {...}} request → cq.enqueue_flood
        'enqueue'  — a {"enqueue": {...}} compile batch → cq.enqueue
        'plan'     — a {"plan": {...}} request → plan_queue.put (step 3). This
                     is the PREEMPT path: PLAN never enters the compile queue
                     (cq), so it can never sit behind a 50–350ms cold compile —
                     a separate planner thread drains plan_queue concurrently
                     with the main thread's compiles (resolved decision #3).
        'empty'    — blank line (ignored)
        'bad-json' — unparseable (ignored, logged)
        'unknown'  — parsed but no recognized key / not an object
                     (ignored, logged — any other control line lands here)

    Factored out of ``main()`` precisely so routing is testable without a
    GL context (step 1 had this inline in the stdin loop and notated it as
    a GPU-gated skip; step 2 closes that gap)."""
    line = line.strip()
    if not line:
        return 'empty'
    try:
        spec = json.loads(line)
    except json.JSONDecodeError as e:
        log.warning(f'bad json: {e}: {line[:200]}')
        return 'bad-json'
    if not isinstance(spec, dict):
        log.warning(f'unexpected request (not an object): {line[:200]}')
        return 'unknown'
    if 'flood' in spec:
        cq.enqueue_flood(spec['flood'])
        return 'flood'
    if 'enqueue' in spec:
        req = spec['enqueue']
        priority = int(req.get('priority', 0))
        tuples = [tuple_from_wire(x) for x in req.get('tuples', [])]
        cq.enqueue(tuples, priority)
        return 'enqueue'
    if 'plan' in spec:
        if plan_queue is not None:
            plan_queue.put(spec['plan'])
            return 'plan'
        log.warning('plan request but no planner queue; ignoring')
        return 'unknown'
    log.warning(f'unknown request, ignoring: {line[:200]}')
    return 'unknown'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pause-flag', required=True,
                        help='path to scheduler pause flag mmap')
    parser.add_argument('--slow-delay-ms', type=int, default=500)
    parser.add_argument('--pause-poll-ms', type=int, default=200)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    # Log to flame-sheep.log (the rotating file the wallpaper uses) so worker
    # activity is watchable alongside the render thread instead of stranded on
    # the inherited stderr/tty. NOTE: two processes then share one
    # RotatingFileHandler path — appends interleave fine, but rotation (at 5MB)
    # is racy across processes. Acceptable for short verification runs; the
    # proper fix (QueueHandler, or a separate worker logfile) is deferred.
    from ..logging_config import setup_logging
    setup_logging(level='DEBUG' if args.verbose else 'INFO')

    # Defer GL imports until after argv parsing so --help doesn't
    # drag in a full moderngl context.
    import moderngl
    from viz_authoring.gpu_context import GpuContext
    from flame_sheep.rendering import FlameRenderer
    from flame_sheep.storage.library import Library

    reader = PauseFlagReader(args.pause_flag)

    # Headless standalone GL context. Canvas dimensions are nominal —
    # FlameRenderer allocates buffers sized for them but we never
    # dispatch chaos, so the buffer contents don't matter. Smaller is
    # cheaper at init.
    ctx = moderngl.create_standalone_context()
    log.info(f'GL renderer: {ctx.info["GL_RENDERER"]}')
    gpu = GpuContext(ctx, canvas_w=256, canvas_h=256, ppmm=1.0)
    renderer = FlameRenderer(gpu)
    # Own Library (own SQLite connection). Touched ONLY on this main
    # thread (compute_flood runs here), so the connection stays
    # thread-local; the stdin reader thread never touches it.
    lib = Library()
    log.info('FlameRenderer + Library initialized; ready for requests on stdin')

    cq = CompileQueue()
    # PLAN request lane (step 3). GL-free + separate from the compile queue so a
    # PLAN never waits behind an in-flight cold compile on the main thread — the
    # dedicated planner thread below drains it concurrently (resolved decision
    # #3: PLAN preempts compiles).
    plan_queue: queue.Queue = queue.Queue()
    stop = threading.Event()

    def _stdin_loop():
        """GL-free: read request lines, route them into the compile queue or the
        planner queue. EOF (driver closed stdin → shutdown) stops the loop."""
        try:
            for line in sys.stdin:
                if stop.is_set():
                    break
                dispatch_line(line, cq, plan_queue)
        except (KeyboardInterrupt, BrokenPipeError):
            pass
        finally:
            stop.set()

    reader_t = threading.Thread(target=_stdin_loop,
                                name='precompile-stdin', daemon=True)
    reader_t.start()

    # Planner thread (step 3). DB-ONLY (no GL), its own Library (own SQLite
    # connection, touched only here) + its own RNG. It answers PLAN requests
    # concurrently with the main thread's compiles and writes the response to
    # stdout, which the driver's reader thread parses. GL-free, so running the
    # planner on a thread violates neither invariant (no GL-on-thread; the
    # SQLite connection stays thread-local to this thread).
    def _planner_loop():
        from .planner import WorkerPlanner
        from .plan_protocol import parse_plan_request, make_plan_response
        import numpy as _np
        from flame_sheep.storage.library import Library as _Library
        try:
            planner_lib = _Library()
        except Exception:
            log.exception('[planner] Library init failed; planner disabled')
            return
        planner = WorkerPlanner(planner_lib, rng=_np.random.default_rng())
        log.info('[planner] ready (DB-only, own Library + RNG)')
        try:
            while not stop.is_set():
                try:
                    spec = plan_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                try:
                    req = parse_plan_request(spec)
                    t0 = time.perf_counter()
                    moves = planner.plan(req['arrival_id'], req['loop_id'],
                                         req['loop_step'])
                    resp = make_plan_response(
                        req['req'], req['arrival_id'],
                        moves['follow'], moves['break'], moves['walk'])
                    sys.stdout.write(json.dumps(resp) + '\n')
                    sys.stdout.flush()
                    # Pre-warm feed (step 4+): the render now has its targets;
                    # enqueue their shaders so the worker warms WHERE THE RENDER
                    # WILL GO. MED priority (0) — above the background flood (10),
                    # below the render's imminent -100; the priority-aware dedup
                    # upgrades the tuple if the render later -100's it on arrival.
                    warm = moves.get('warm')
                    if warm:
                        cq.enqueue(warm, priority=0)
                    dt = (time.perf_counter() - t0) * 1000.0
                    log.debug('[planner] answered req=%s arrival=#%s in %.1fms '
                              '(+%d warm)', req['req'], req['arrival_id'], dt,
                              len(warm) if warm else 0)
                except Exception:
                    log.exception('[planner] request failed: %r', spec)
        finally:
            try:
                planner_lib.close()
            except Exception:
                pass

    planner_t = threading.Thread(target=_planner_loop,
                                 name='precompile-planner', daemon=True)
    planner_t.start()

    compiled = 0
    skipped = 0
    # Keys compiled this run, for the compile-time dedup layer (keyed on
    # keep_vars, since for GL only keep_vars affects the shader). We do
    # NOT retain the compiled programs (see the release below), so we
    # can't dedup against renderer._chaos_shader_cache — track a cheap
    # set of frozensets instead (bytes, not ~0.9MB/key of GL program).
    seen_keepvars: set[frozenset] = set()
    t_start = time.perf_counter()

    try:
        while not stop.is_set():
            try:
                priority, _seq, payload = cq.get(timeout=0.5)
            except queue.Empty:
                continue

            # Flood-compute task: run near_tuples with our Library (on
            # this main thread) and self-schedule the result at LOW.
            if isinstance(payload, tuple) and len(payload) == 2 \
                    and payload[0] == _FLOOD_TASK:
                result = compute_flood(lib, payload[1])
                if result:
                    n_added = cq.enqueue(result, priority=_FLOOD_RESULT_PRIORITY)
                    log.debug(f'flood self-scheduled {n_added} tuple(s) at LOW')
                continue

            # Compile task: payload is (n_transforms, has_final, keep_vars).
            keep_vars = payload[2]

            # Policy enforcement (unchanged): read the shared pause flag
            # the driver's policy loop publishes, and PAUSE/SLOW between
            # compiles. Flood tasks are not gated (cheap DB, not a compile).
            while True:
                state = reader.state
                if state is BatchState.PAUSE:
                    time.sleep(args.pause_poll_ms / 1000.0)
                    continue
                if state is BatchState.SLOW:
                    time.sleep(args.slow_delay_ms / 1000.0)
                break

            if keep_vars in seen_keepvars:
                skipped += 1
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
            seen_keepvars.add(keep_vars)
            compiled += 1
            cache_status = 'hit' if dt < 20.0 else 'cold'
            log.info(
                f'compiled keep_vars=({len(keep_vars)} vars) in '
                f'{dt:.1f}ms [{cache_status}] '
                f'[total: {compiled} compiled, {skipped} skipped]')
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        stop.set()
        elapsed = time.perf_counter() - t_start
        log.info(f'exiting after {elapsed:.2f}s: compiled={compiled} '
                  f'skipped={skipped}')
        reader.close()
        try:
            lib.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
