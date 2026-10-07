"""Drives the precompile_worker subprocess with a priority queue.

Owns the queue of (priority, tuple) pairs. Runs two daemon threads:
  - send_thread: pops highest-priority tuple, writes to worker stdin
  - recv_thread: reads worker stdout for READY / DONE acks; releases
                 the send_thread to pick next tuple

Why split into two threads: keeps each thread's I/O blocking simple
(write to a pipe, read from a pipe). The shared state between them
is one Event ('worker_ready') and the priority queue itself, both
already thread-safe.

Usage:
    driver = PrecompileDriver(canvas_w=3713, canvas_h=1278,
                                policy=policy)
    driver.start()
    driver.enqueue(tuples_set, priority=10)   # catalog warm
    driver.enqueue(near_term, priority=0)     # anticipated
    driver.enqueue([next_genome_tuple], priority=-10)  # imminent
    ...
    driver.stop()
"""
from __future__ import annotations

import logging
import queue
import subprocess
import sys
import threading
import time
from typing import Iterable

from .batch_worker import BatchWorker, _child_preexec
from .pause_flag import PauseFlag
from .policy import Policy

log = logging.getLogger(__name__)


class PrecompileDriver:
    """Manages one precompile worker subprocess + its input queue."""

    def __init__(self, canvas_w: int, canvas_h: int, policy: Policy,
                 worker_module: str = 'flame_sheep.scheduler.precompile_worker'):
        """worker_module: dotted path to the subprocess entry point.
        Default is the Vk worker. Pass 'flame_sheep.scheduler.
        precompile_worker_gl' for the GL backend. The protocol
        (READY/DONE tags on stdout, tuple JSON on stdin) is the same
        for both — only the actual compile target differs."""
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.policy = policy
        self._worker_module = worker_module
        # Counter breaks priority ties FIFO — heapq is stable-on-tuple,
        # so this prevents comparison of the dict/frozenset payloads
        # which would TypeError.
        self._counter = 0
        self._counter_lock = threading.Lock()
        self._q: queue.PriorityQueue = queue.PriorityQueue()
        # Set of cache keys already enqueued so we don't double-add.
        self._seen: set = set()
        self._seen_lock = threading.Lock()
        self._flag = PauseFlag('precompile')
        self._proc: subprocess.Popen | None = None
        self._send_t: threading.Thread | None = None
        self._recv_t: threading.Thread | None = None
        self._policy_t: threading.Thread | None = None
        self._stop = threading.Event()
        self._worker_ready = threading.Event()
        # Stats
        self.stats_compiled = 0
        self.stats_skipped = 0
        self.stats_cold = 0
        self.stats_hit = 0

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
            stdout=subprocess.PIPE,
            stderr=None,        # inherit; worker logs flow to parent stderr
            text=True,
            preexec_fn=_child_preexec,
            bufsize=1,          # line-buffered (works with text=True)
        )
        self._send_t = threading.Thread(
            target=self._send_loop, name='precompile-send', daemon=True)
        self._recv_t = threading.Thread(
            target=self._recv_loop, name='precompile-recv', daemon=True)
        self._policy_t = threading.Thread(
            target=self._policy_loop, name='precompile-policy',
            daemon=True)
        self._send_t.start()
        self._recv_t.start()
        self._policy_t.start()

    def drain_state_distribution(self) -> dict:
        """Pass-through to the policy — see Policy.drain_state_distribution."""
        return self.policy.drain_state_distribution()

    def enqueue(self, tuples: Iterable[tuple[int, int, frozenset[int]]],
                 priority: int = 0) -> int:
        """Add tuples to the queue at given priority. Lower priority
        number = served first (matches heapq semantics).
        Returns the number actually enqueued (dedup'd against seen
        AND against the disk warm-marker convention — tuples whose
        Mesa shader cache has already been populated in any prior
        session are silently skipped, since re-compiling them via
        the worker would just dutifully re-touch the same disk cache
        for no real benefit).
        """
        from flame_sheep.rendering.vk import pipeline_warm
        added = 0
        skipped_warm = 0
        with self._seen_lock:
            for t in tuples:
                if t in self._seen:
                    continue
                self._seen.add(t)
                n_tx, has_final, keep_vars = t
                if pipeline_warm.is_warm(n_tx, bool(has_final), keep_vars):
                    skipped_warm += 1
                    continue
                with self._counter_lock:
                    n = self._counter
                    self._counter += 1
                self._q.put((priority, n, t))
                added += 1
        if skipped_warm:
            log.debug(
                f'[precompile] enqueue: added={added} '
                f'skipped_warm={skipped_warm} (already in Mesa cache)')
        return added

    def _send_loop(self):
        """Pull from queue, write to worker stdin. Wait for the
        worker to signal READY between sends (one-at-a-time, so
        re-prioritization always reflects in next compile)."""
        from .precompile_warm import tuple_to_json_line
        # First send waits for the worker's initial READY.
        self._worker_ready.wait()
        while not self._stop.is_set():
            try:
                priority, _seq, t = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                line = tuple_to_json_line(t)
                self._worker_ready.clear()  # consumed; worker won't be
                                              # ready again until next ack
                self._proc.stdin.write(line + '\n')
                self._proc.stdin.flush()
                # Block until worker acknowledges (DONE) — keeps the
                # one-tuple-in-flight invariant.
                if not self._worker_ready.wait(timeout=120.0):
                    log.warning('[precompile] worker stalled >120s '
                                'waiting for DONE')
            except (BrokenPipeError, OSError) as e:
                log.warning(f'[precompile] send failed: {e}')
                break

    def _recv_loop(self):
        """Read worker stdout. Each line is either:
            READY            — initial handshake
            DONE <ms> <tag>  — last compile finished
        We treat both as "worker is ready for next tuple."
        """
        proc = self._proc
        while not self._stop.is_set():
            line = proc.stdout.readline()
            if not line:
                break  # worker exited / closed stdout
            line = line.strip()
            if line == 'READY':
                self._worker_ready.set()
                continue
            if line.startswith('DONE'):
                parts = line.split()
                tag = parts[2] if len(parts) >= 3 else ''
                if tag == 'hit':
                    self.stats_hit += 1; self.stats_compiled += 1
                elif tag == 'cold':
                    self.stats_cold += 1; self.stats_compiled += 1
                elif tag == 'dup':
                    self.stats_skipped += 1
                self._worker_ready.set()

    def _policy_loop(self):
        """Tick the policy at the policy's natural cadence; mirror
        state into the pause flag."""
        interval = self.policy.sample_interval_s
        while not self._stop.is_set():
            state = self.policy.tick()
            self._flag.set(state)
            time.sleep(interval)

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.stdin.close()
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
        log.info(f'[precompile] stopped. '
                 f'compiled={self.stats_compiled} '
                 f'(cold={self.stats_cold}, hit={self.stats_hit}, '
                 f'skipped={self.stats_skipped})')
