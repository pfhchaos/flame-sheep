"""Regression: the three background GA workers must use a spawn
multiprocessing context, never the Linux-default fork.

Background (DECISIONS.md 2026-10-10): the workers are started in
`runtime/wallpaper.py` AFTER the GL context, renderer, and genome
Library are built. With the default `fork` start method on Linux +
py3.13, each child inherited a copy-on-write snapshot of the parent
render process's entire address space:

  - the live GL context + GPU buffers (nvtop showed all three workers
    holding an identical ~595 MiB of inherited VRAM they never use), and
  - every open fd, including the Library's open SQLite connection
    (`Library.conn`) — the documented SQLite-across-fork corruption
    footgun.

The author intended these to be fully isolated processes (that is WHY
they are processes, not threads — no shared-state/thread safety). `fork`
silently broke that isolation. `spawn` restores it: the child starts
from a fresh interpreter, re-imports only what it needs, and inherits no
GL context, no GPU memory, and no parent fds. The stop Event and the
Process must come from the SAME spawn context or the Event won't ride
the spawn Process cleanly.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from flame_sheep.genome.pruner import BackgroundPruner
from flame_sheep.genome.score_worker import BackgroundCpuScorer
from flame_sheep.transitions.worker import BackgroundTransitionScorer

WORKER_CLASSES = [
    BackgroundCpuScorer,
    BackgroundPruner,
    BackgroundTransitionScorer,
]


@pytest.mark.parametrize('cls', WORKER_CLASSES, ids=lambda c: c.__name__)
def test_worker_uses_spawn_context(cls):
    """The context a worker builds its Process + Event from must be
    spawn. This is the core lock on the fix — if it ever reverts to
    fork (or an implicit default), every forked child re-inherits the
    parent's GL context, ~595 MiB of VRAM, and the open SQLite fd."""
    w = cls(db_path=':memory:')
    assert w._ctx.get_start_method() == 'spawn', (
        f'{cls.__name__} must use a spawn context, '
        f'got {w._ctx.get_start_method()!r}'
    )
    # The Process factory bound to this context must be the spawn one.
    assert w._ctx.Process._start_method == 'spawn'


def _wait_for_stop(stop_event, done_path: str) -> None:
    """Trivial worker target: block until the stop Event fires, then
    touch a sentinel file. Picklable (module-level) so it rides a spawn
    Process. Exercises exactly the 'does a spawn-context Event ride a
    spawn-context Process' wiring without starting the heavy real
    worker main (which would load the CNN / hit the real DB)."""
    stop_event.wait(timeout=30.0)
    Path(done_path).write_text('stopped')


@pytest.mark.parametrize('cls', WORKER_CLASSES, ids=lambda c: c.__name__)
def test_stop_event_rides_spawn_process(cls):
    """A default-context Event does NOT ride a spawn Process cleanly;
    both must come from the same spawn context. Build a Process from the
    worker's own `_ctx` driven by the worker's own `_stop` Event and
    confirm setting the Event actually releases the child."""
    w = cls(db_path=':memory:')
    with tempfile.TemporaryDirectory(prefix='fs-spawn-') as d:
        sentinel = os.path.join(d, 'done')
        w._stop.clear()
        p = w._ctx.Process(
            target=_wait_for_stop, args=(w._stop, sentinel), daemon=True)
        p.start()
        assert p._start_method == 'spawn'
        assert p.is_alive()
        w._stop.set()
        p.join(timeout=15.0)
        assert not p.is_alive(), 'child did not observe the spawn-context Event'
        assert p.exitcode == 0
        assert Path(sentinel).read_text() == 'stopped'


def _report_open_fd_targets(q) -> None:
    """Worker target: report the resolved targets of this process's open
    file descriptors. Under fork the child would inherit the parent's
    open fds (including the SQLite DB connection); under spawn it starts
    clean. Module-level so it's picklable under spawn."""
    fd_dir = '/proc/self/fd'
    targets = []
    for name in os.listdir(fd_dir):
        try:
            targets.append(os.readlink(os.path.join(fd_dir, name)))
        except OSError:
            pass
    q.put(targets)


@pytest.mark.parametrize('cls', WORKER_CLASSES, ids=lambda c: c.__name__)
def test_spawn_child_does_not_inherit_parent_db_fd(cls):
    """The SQLite-across-fork footgun, locked directly to the REAL
    worker's own `_ctx` — the attribute the fix controls — not a fresh
    ad hoc `get_context('spawn')` (that version was tautological: it
    never touched `cls` at all, so it stayed green even after a worker
    reverted to fork).

    The parent holds an open connection to a DB file, standing in for
    `Library.conn` at the moment wallpaper.py starts the workers. A
    child spawned from `cls`'s own `_ctx` must NOT inherit that fd.
    If `cls.__init__` is ever reverted to build `_ctx` from 'fork'
    instead of 'spawn', this must go RED: a forked child inherits the
    parent's entire fd table, including this one, at fork time."""
    w = cls(db_path=':memory:')
    with tempfile.TemporaryDirectory(prefix='fs-fdtest-') as d:
        import sqlite3
        db_path = os.path.join(d, 'genomes.db')
        conn = sqlite3.connect(db_path)
        conn.execute('CREATE TABLE t (x)')
        conn.commit()
        try:
            q = w._ctx.Queue()
            p = w._ctx.Process(target=_report_open_fd_targets, args=(q,))
            p.start()
            child_fd_targets = q.get(timeout=15.0)
            p.join(timeout=15.0)
        finally:
            conn.close()
        resolved_db = os.path.realpath(db_path)
        inherited = [t for t in child_fd_targets
                     if os.path.realpath(t) == resolved_db]
        assert not inherited, (
            f'{cls.__name__}: child spawned from worker._ctx inherited '
            f'parent DB fd(s): {inherited} — this is the SQLite-across-'
            f'fork corruption footgun the spawn switch exists to prevent'
        )
