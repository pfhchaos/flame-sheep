"""Thin IPC shim to the precompile worker subprocess (step 2).

Step 1 had this class OWN the scheduling: a ``queue.PriorityQueue``, the
``(priority, counter, tuple)`` tie-break, the ``_seen`` + ``is_warm``
dedup, and a one-compile-at-a-time send/ack handshake with the worker.
**Step 2 moves all of that INTO the worker** (``precompile_worker_gl``:
``CompileQueue`` + the compile loop). What's left here is a shim:

  * ``enqueue(tuples, priority)``   → serialize an enqueue request to the
                                       worker's stdin;
  * ``request_flood(seed_ids, ...)`` → serialize a flood request;
  * ``start`` / ``stop``            → subprocess lifecycle;
  * ``_policy_loop``                → the Policy throttle, UNCHANGED.

The worker queues / dedups / schedules internally. There is no send
thread (no driver-side queue to drain) and no recv thread (the
READY/DONE and FLOOD/FLOOD_DONE stdout protocol is gone — the worker logs
its own stats and self-schedules the flood). The only driver → worker
channel is stdin; writes are serialized by ``_stdin_lock`` because
several render-side threads call ``enqueue`` concurrently (render loop,
compare-prewarm thread, axis gate).

Why writing straight to stdin doesn't block the render thread: the
worker's stdin READER thread does only JSON-parse + queue.put (never GL,
never DB, never a blocking compile — those are on the worker's main
thread), so it drains the pipe continuously and the OS pipe buffer never
fills. The late-arriving high-priority enqueue (the -100 reactive
cold-miss) is therefore received even while the worker is mid-compile and
is popped next — the same re-prioritization guarantee step 1 got from the
one-in-flight handshake.

Policy throttling (unchanged): the Policy decision stays HERE because it
owns the render-side signal sampler; ``_policy_loop`` publishes the
BatchState to the shared ``PauseFlag`` mmap, and the worker reads that
flag between compiles. Moving the queue into the worker is orthogonal to
throttling — the throttle was always the pause flag, never the handshake.

Usage:
    driver = PrecompileDriver(canvas_w=0, canvas_h=0, policy=policy,
                              worker_module='...precompile_worker_gl')
    driver.start()
    driver.enqueue(near_term, priority=0)
    driver.enqueue([next_genome_tuple], priority=-100)  # imminent
    driver.request_flood(seed_ids, hops=2, fan_per_node=5)
    ...
    driver.stop()
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from typing import Iterable

from .batch_worker import _child_preexec
from .pause_flag import PauseFlag
from .policy import Policy

log = logging.getLogger(__name__)


class PrecompileDriver:
    """Manages one precompile worker subprocess + the Policy throttle.

    The compile queue / dedup / scheduling live in the worker (step 2);
    this object only serializes requests to the worker's stdin and ticks
    the Policy into the shared pause flag.
    """

    def __init__(self, canvas_w: int, canvas_h: int, policy: Policy,
                 worker_module: str = 'flame_sheep.scheduler.precompile_worker'):
        """worker_module: dotted path to the subprocess entry point.
        Default is the Vk worker. Pass 'flame_sheep.scheduler.
        precompile_worker_gl' for the GL backend. The request protocol
        (enqueue/flood JSON on stdin) is the same for both — only the
        actual compile target differs."""
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.policy = policy
        self._worker_module = worker_module
        self._flag = PauseFlag('precompile')
        self._proc: subprocess.Popen | None = None
        self._policy_t: threading.Thread | None = None
        self._stop = threading.Event()
        # Serializes concurrent writes to the worker's stdin — several
        # render-side threads call enqueue (render loop, compare-prewarm,
        # axis gate), and interleaved file.write()s would corrupt the
        # newline-framed JSON.
        self._stdin_lock = threading.Lock()

    def start(self) -> None:
        argv = [sys.executable, '-u',
                 '-m', self._worker_module,
                 '--pause-flag', str(self._flag.path)]
        # Vk worker needs canvas dims (SC_width/SC_height spec
        # constants); GL worker doesn't. Pass when supported, omit
        # otherwise — gl worker would reject the unknown flag.
        if self._worker_module.endswith('precompile_worker'):
            argv += ['--canvas-w', str(self.canvas_w),
                     '--canvas-h', str(self.canvas_h)]
        log.info(f'[precompile] spawning: {" ".join(argv)}')
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=None,        # inherit; worker logs via logging→stderr.
                                # No stdout protocol anymore, so nothing to
                                # drain and no recv thread.
            stderr=None,        # inherit; worker logs flow to parent stderr
            text=True,
            preexec_fn=_child_preexec,
            bufsize=1,          # line-buffered (works with text=True)
        )
        self._policy_t = threading.Thread(
            target=self._policy_loop, name='precompile-policy',
            daemon=True)
        self._policy_t.start()

    def drain_state_distribution(self) -> dict:
        """Pass-through to the policy — see Policy.drain_state_distribution."""
        return self.policy.drain_state_distribution()

    def _write(self, obj: dict) -> None:
        """Serialize one request object to the worker's stdin as a single
        newline-framed JSON line. No-op (logged) if the worker isn't up or
        the pipe broke."""
        line = json.dumps(obj)
        with self._stdin_lock:
            if self._proc is None or self._proc.stdin is None:
                return
            try:
                self._proc.stdin.write(line + '\n')
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                log.warning(f'[precompile] send failed: {e}')

    def enqueue(self, tuples: Iterable[tuple[int, int, frozenset[int]]],
                 priority: int = 0) -> int:
        """Serialize an enqueue request to the worker; it queues + dedups
        internally. Returns the number of tuples REQUESTED (post-dedup
        counting now happens in the worker and isn't reported back — the
        value is used only for caller-side logging). Empty input is a
        no-op returning 0."""
        tuples = list(tuples)
        if not tuples:
            return 0
        from .precompile_warm import tuple_to_wire
        self._write({'enqueue': {
            'priority': priority,
            'tuples': [tuple_to_wire(t) for t in tuples],
        }})
        return len(tuples)

    def request_flood(self, seed_ids: Iterable[int], hops: int = 2,
                      fan_per_node: int = 5, priority: int = 0) -> None:
        """Ask the worker to compute the 2-hop graph flood for these seed
        genome ids. The worker runs near_tuples with ITS Library (DB work
        in the worker process, not on the render thread), then
        self-schedules the resulting cache keys into its own queue at LOW.
        No-op on empty seed_ids."""
        seed_ids = [int(s) for s in seed_ids]
        if not seed_ids:
            return
        self._write({'flood': {
            'seed_ids': seed_ids,
            'hops': hops,
            'fan_per_node': fan_per_node,
            'priority': priority,
        }})

    def _policy_loop(self):
        """Tick the policy at its natural cadence; mirror state into the
        pause flag. UNCHANGED from step 1 — the worker reads this flag
        between compiles."""
        interval = self.policy.sample_interval_s
        while not self._stop.is_set():
            state = self.policy.tick()
            self._flag.set(state)
            time.sleep(interval)

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.stdin.close()  # EOF → worker's stdin loop ends
            except OSError:
                pass
            try:
                self._proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
        self._flag.close()
        log.info('[precompile] stopped (worker logs its own compile stats)')
