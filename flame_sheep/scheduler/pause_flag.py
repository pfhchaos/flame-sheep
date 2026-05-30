"""Single-byte shared-memory pause flag for batch workers.

Pattern: scheduler creates the flag, hands its name to a worker
subprocess. Both mmap the same file. Scheduler writes a state byte;
worker reads it between units of work (e.g. after each compile)
and either continues, slows, or pauses accordingly.

Why a file + mmap rather than e.g. a Unix pipe or shared array
through multiprocessing:
  - Worker is a subprocess.Popen, not a multiprocessing.Process.
    Mesa-Xe + multiprocessing.fork crashes sway (known); subprocess
    is the safer pattern. Multiprocessing's primitives don't apply.
  - mmap is the simplest cross-process byte handoff with no IPC
    round-trip — worker reads in O(ns).
  - Persistent file (in /run/user/$UID) survives across worker
    crashes, lets the scheduler re-spawn and re-attach.

The byte values match scheduler.policy.BatchState (RUN=0, SLOW=1,
PAUSE=2). Workers should treat unknown values as PAUSE — safer
than RUN since over-pausing only wastes some throughput, but
under-pausing can blow the wallpaper's frame budget.
"""
from __future__ import annotations

import mmap
import os
import struct
from pathlib import Path

from .policy import BatchState


# Byte encoding for BatchState. Keep contiguous so unknown-value
# check is `>= MAX`.
_STATE_TO_BYTE = {
    BatchState.RUN:   0,
    BatchState.SLOW:  1,
    BatchState.PAUSE: 2,
}
_BYTE_TO_STATE = {v: k for k, v in _STATE_TO_BYTE.items()}


def _runtime_dir() -> Path:
    """XDG_RUNTIME_DIR / flame-sheep / scheduler — tmpfs-backed on
    Linux, auto-cleaned on session end."""
    base = os.environ.get('XDG_RUNTIME_DIR',
                           f'/run/user/{os.getuid()}')
    d = Path(base) / 'flame-sheep' / 'scheduler'
    d.mkdir(parents=True, exist_ok=True)
    return d


class PauseFlag:
    """Writer side. Scheduler creates one of these per managed worker;
    passes `path` to the worker so the worker can open the read side.

    The flag lives for the lifetime of the PauseFlag instance — close()
    unmaps and deletes the file. Re-creating with the same name resets
    the flag (which is fine).
    """

    def __init__(self, name: str):
        self.path = _runtime_dir() / f'{name}.flag'
        # Truncate to 1 byte and initialize to PAUSE (workers start
        # paused; scheduler ticks the policy and flips to RUN once
        # there's data to act on).
        with open(self.path, 'wb') as f:
            f.write(bytes([_STATE_TO_BYTE[BatchState.PAUSE]]))
        # mmap for fast writes from the policy loop.
        self._fd = os.open(self.path, os.O_RDWR)
        self._mm = mmap.mmap(self._fd, 1, mmap.MAP_SHARED,
                              mmap.PROT_READ | mmap.PROT_WRITE)

    def set(self, state: BatchState) -> None:
        b = _STATE_TO_BYTE[state]
        # Single-byte write to mmap is atomic on x86; no torn reads.
        self._mm[0] = b

    @property
    def current(self) -> BatchState:
        return _BYTE_TO_STATE.get(self._mm[0], BatchState.PAUSE)

    def close(self):
        try:
            self._mm.close()
        except Exception:
            pass
        try:
            os.close(self._fd)
        except OSError:
            pass
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class PauseFlagReader:
    """Reader side. Worker subprocess opens the flag by path and
    polls `state` between units of work.

    Open is cheap (single mmap); polling is just an mmap byte read,
    no syscall.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fd = os.open(self.path, os.O_RDONLY)
        self._mm = mmap.mmap(self._fd, 1, mmap.MAP_SHARED,
                              mmap.PROT_READ)

    @property
    def state(self) -> BatchState:
        return _BYTE_TO_STATE.get(self._mm[0], BatchState.PAUSE)

    def close(self):
        try:
            self._mm.close()
        except Exception:
            pass
        try:
            os.close(self._fd)
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
