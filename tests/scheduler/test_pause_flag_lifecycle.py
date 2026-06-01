"""Lifecycle property tests for PauseFlag / PauseFlagReader.

Before 2026-05-31, PauseFlag.__init__ created+truncated the file
and PauseFlag.close() unlinked it — meaning workers HAD to be
spawned after their corresponding writer, couldn't survive a writer
restart, and were blocked by stale PAUSE if the writer was slow to
start. The user flagged this as a wart during the GL precompile
worker integration.

The new contract:
- Reader can be created BEFORE writer; defaults to RUN (fail-open).
- Writer close() does NOT unlink; survives writer restart.
- Writer re-init does NOT truncate; preserves existing state byte.
- Reader's mmap stays valid through writer init/close cycles.

These properties make the flag file a long-lived shared resource
rather than a writer-owned ephemeral file, matching the design
intent of decoupled lifecycles across the scheduler graph.
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pytest


@pytest.fixture
def isolated_runtime(monkeypatch):
    """Each test gets a fresh XDG_RUNTIME_DIR so flag files don't
    bleed between tests."""
    tmp = tempfile.mkdtemp(prefix='flame-sheep-test-')
    monkeypatch.setenv('XDG_RUNTIME_DIR', tmp)
    yield tmp
    # Best-effort cleanup; tmpfs goes away on reboot anyway.


def test_reader_before_writer_defaults_run(isolated_runtime):
    """Worker can open the flag before the writer ever exists.
    Defaults to RUN (fail-open) so the worker isn't blocked by
    stale-PAUSE state from a writer that's slow to start or
    never appears."""
    from flame_sheep.scheduler.pause_flag import PauseFlagReader, _runtime_dir
    from flame_sheep.scheduler.policy import BatchState

    p = _runtime_dir() / 'pre_writer.flag'
    r = PauseFlagReader(p)
    assert r.state is BatchState.RUN
    r.close()


def test_writer_updates_pre_existing_reader(isolated_runtime):
    """Writer that comes up after the reader can still drive its
    state — they end up mmaping the same persistent file."""
    from flame_sheep.scheduler.pause_flag import (
        PauseFlag, PauseFlagReader, _runtime_dir)
    from flame_sheep.scheduler.policy import BatchState

    p = _runtime_dir() / 'late_writer.flag'
    r = PauseFlagReader(p)
    assert r.state is BatchState.RUN

    w = PauseFlag('late_writer')
    w.set(BatchState.PAUSE)
    time.sleep(0.01)  # mmap propagation is essentially immediate; nominal slack
    assert r.state is BatchState.PAUSE

    w.set(BatchState.SLOW)
    time.sleep(0.01)
    assert r.state is BatchState.SLOW

    w.close()
    r.close()


def test_writer_close_does_not_unlink(isolated_runtime):
    """File survives writer close. Future writers/readers see the
    same byte. Lets workers persist across writer restart cycles."""
    from flame_sheep.scheduler.pause_flag import PauseFlag

    w = PauseFlag('persist_test')
    w.set_path = w.path  # capture before close
    w.close()
    assert w.path.exists(), 'file should persist after writer close'


def test_reader_survives_writer_close(isolated_runtime):
    """Reader's mmap stays valid when the writer goes away."""
    from flame_sheep.scheduler.pause_flag import (
        PauseFlag, PauseFlagReader, _runtime_dir)
    from flame_sheep.scheduler.policy import BatchState

    w = PauseFlag('survival_test')
    w.set(BatchState.PAUSE)
    time.sleep(0.01)

    r = PauseFlagReader(_runtime_dir() / 'survival_test.flag')
    assert r.state is BatchState.PAUSE

    w.close()
    # Reader still works after writer is gone.
    assert r.state is BatchState.PAUSE
    r.close()


def test_writer_reinit_preserves_state(isolated_runtime):
    """Re-creating the writer does NOT truncate the file's content.
    Critical: pre-existing readers' mmaps would break if the file
    got truncated (their view would read past-EOF as zero = RUN,
    independent of what subsequent writes do)."""
    from flame_sheep.scheduler.pause_flag import (
        PauseFlag, PauseFlagReader, _runtime_dir)
    from flame_sheep.scheduler.policy import BatchState

    r = PauseFlagReader(_runtime_dir() / 'reinit_test.flag')

    w1 = PauseFlag('reinit_test')
    w1.set(BatchState.PAUSE)
    time.sleep(0.01)
    assert r.state is BatchState.PAUSE
    w1.close()

    # Writer re-init: must preserve state, must not corrupt reader's mmap.
    w2 = PauseFlag('reinit_test')
    assert r.state is BatchState.PAUSE, (
        'writer re-init must not truncate; pre-existing readers depend on it')

    w2.set(BatchState.SLOW)
    time.sleep(0.01)
    assert r.state is BatchState.SLOW

    w2.close()
    r.close()
