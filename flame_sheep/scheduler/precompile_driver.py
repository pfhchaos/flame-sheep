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
        # Accumulates flood tuples the worker streams back (FLOOD lines)
        # until its FLOOD_DONE, then they're enqueued at LOW. Touched
        # only by recv_thread, so no lock needed.
        self._flood_buf: list[tuple[int, int, frozenset[int]]] = []
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

    def request_flood(self, seed_ids: Iterable[int], hops: int = 2,
                      fan_per_node: int = 5, priority: int = 0) -> None:
        """Ask the worker to compute the 2-hop graph flood for these
        seed genome ids. The worker runs near_tuples with ITS Library
        (DB work in the worker process, NOT on the render thread) and
        streams the resulting cache keys back; recv_thread enqueues them
        at LOW. The request itself rides the same stdin pipe as compiles,
        so it goes through the priority queue (default MED, so it's
        computed right after the loop-member tuples the render thread
        enqueues alongside it).

        Step-1 interim: the driver still owns the queue; step 2 moves it
        into the worker.
        """
        seed_ids = [int(s) for s in seed_ids]
        if not seed_ids:
            return
        with self._counter_lock:
            n = self._counter
            self._counter += 1
        self._q.put((priority, n, {
            'type': 'flood',
            'seed_ids': seed_ids,
            'hops': hops,
            'fan_per_node': fan_per_node,
        }))

    def _send_loop(self):
        """Pull from queue, write to worker stdin. Wait for the
        worker to signal READY between sends (one-at-a-time, so
        re-prioritization always reflects in next compile)."""
        import json as _json
        from .precompile_warm import tuple_to_json_line
        # First send waits for the worker's initial READY.
        self._worker_ready.wait()
        while not self._stop.is_set():
            try:
                priority, _seq, payload = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if isinstance(payload, dict) and payload.get('type') == 'flood':
                    # Flood-compute request: worker runs near_tuples and
                    # streams the keys back (recv_loop enqueues them).
                    line = _json.dumps({'flood': {
                        'seed_ids': payload['seed_ids'],
                        'hops': payload['hops'],
                        'fan_per_node': payload['fan_per_node'],
                    }})
                else:
                    line = tuple_to_json_line(payload)
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
            if line.startswith('FLOOD_DONE'):
                # Worker finished a flood computation. Enqueue the keys it
                # streamed at LOW (same priority the render thread used to
                # enqueue the inline-computed flood). Dedup (_seen +
                # is_warm) applies via enqueue(), as before.
                tuples = self._flood_buf
                self._flood_buf = []
                if tuples:
                    self.enqueue(tuples, priority=10)
                self._worker_ready.set()
                continue
            if line.startswith('FLOOD '):
                from .precompile_warm import json_line_to_tuple
                try:
                    self._flood_buf.append(json_line_to_tuple(line[len('FLOOD '):]))
                except Exception as e:
                    log.warning(f'[precompile] bad FLOOD line: {e}')
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
