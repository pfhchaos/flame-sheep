"""Small helpers for subprocess lifecycle hygiene.

set_pdeathsig — kernel-kill children when their parent dies, so
crash / kill -9 doesn't leave orphans (we saw a precompile_worker
running for hours after flame-sheep died).

acquire_single_instance_lock — pidfile + flock so a second
instance of e.g. the audio daemon refuses to start instead of
fighting the first for the input device.
"""
from __future__ import annotations

import ctypes
import fcntl
import logging
import os
import signal
from pathlib import Path

log = logging.getLogger(__name__)

_PR_SET_PDEATHSIG = 1


def set_pdeathsig(sig: int = signal.SIGTERM) -> None:
    """prctl(PR_SET_PDEATHSIG, sig) — kernel sends `sig` to THIS
    process when its parent dies. Best-effort; silently no-ops if
    prctl isn't available (non-Linux, sandbox, etc.).

    Caveat: fires when the spawning THREAD exits, not the whole
    parent process. Our workers are all spawned from the main thread
    so this matches process lifetime in practice, but watch out if
    you ever spawn from a daemon thread."""
    try:
        libc = ctypes.CDLL('libc.so.6', use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, sig, 0, 0, 0)
    except (OSError, AttributeError):
        pass


# Hold acquired lock fds at module scope so they outlive the caller's
# function frame — fd close releases the kernel lock. Keyed by name so
# the same process can hold multiple non-overlapping locks.
_held_locks: dict[str, int] = {}


def _runtime_dir() -> Path:
    """$XDG_RUNTIME_DIR/flame-sheep, falling back to /tmp/flame-sheep-$UID.
    tmpfs preferred — pidfile must not survive a reboot."""
    xdg = os.environ.get('XDG_RUNTIME_DIR')
    base = Path(xdg) if xdg else Path(f'/tmp/flame-sheep-{os.getuid()}')
    d = base / 'flame-sheep'
    d.mkdir(parents=True, exist_ok=True)
    return d


def acquire_single_instance_lock(name: str) -> bool:
    """Take an exclusive non-blocking flock on $RUNTIME_DIR/<name>.pid.

    Returns True if we got the lock — caller can proceed. Writes our
    pid into the file for diagnostic purposes. The lock fd is kept
    open at module scope; the kernel releases the lock when the
    process exits (or when close() is called).

    Returns False if another process already holds the lock — caller
    should exit cleanly (we're a no-op duplicate, not an error). The
    holder's pid is logged so the user can find it.

    Stale-pidfile case is handled implicitly: if the previous holder
    died, the kernel released the lock automatically, so a fresh
    LOCK_NB attempt succeeds. The pid in the file may be stale until
    we overwrite it."""
    path = _runtime_dir() / f'{name}.pid'
    # Open with O_RDWR | O_CREAT (no O_TRUNC — we want to preserve the
    # existing pid if the lock attempt fails, so we can report it).
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        # Already locked. Read whatever pid is in the file (best-effort).
        try:
            holder = os.read(fd, 32).decode('ascii', errors='replace').strip()
        except OSError:
            holder = '?'
        os.close(fd)
        log.info(f'[single-instance:{name}] already running '
                  f'(pid={holder}); exiting')
        return False
    # We got the lock — overwrite with our pid, keep fd open.
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    os.write(fd, f'{os.getpid()}\n'.encode('ascii'))
    _held_locks[name] = fd
    log.info(f'[single-instance:{name}] acquired '
              f'(pid={os.getpid()}, file={path})')
    return True
