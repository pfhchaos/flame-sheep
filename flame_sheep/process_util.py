"""Small helpers for subprocess lifecycle hygiene.

Currently just set_pdeathsig — used by every flame-sheep worker
(precompile, render, anything else we add later) to ensure the
kernel kills them when the parent flame-sheep process dies.

Before this, we'd see orphaned precompile_workers running for hours
after the wallpaper died — they don't notice their parent is gone
until they try to write to a closed pipe, which can be a long time
if the queue is empty.
"""
from __future__ import annotations

import ctypes
import signal

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
