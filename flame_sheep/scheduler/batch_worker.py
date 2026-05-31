"""BatchWorker: subprocess wrapper that owns a PauseFlag.

Scheduler creates a BatchWorker, starts it with a command + args.
The worker process gets the flag path on argv (or env) and reads
state between work units. Scheduler ticks its policy and flips the
flag.

Lifecycle:
    bw = BatchWorker('precompile', ['python', '-m',
                                       'flame_sheep.scheduler.precompile_worker',
                                       '--queue', queue_path])
    bw.start()           # launches subprocess at nice +19 + ionice idle
    while running:
        policy.tick()
        bw.flag.set(policy.state)
        if bw.poll() is not None:
            break       # worker exited (queue drained, error, etc.)
    bw.stop()           # SIGTERM, then SIGKILL if needed

The worker subprocess's argv is augmented with --pause-flag <path>
automatically so individual worker scripts don't need to know
about the flag plumbing.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from typing import Sequence

from .pause_flag import PauseFlag
from .policy import BatchState

log = logging.getLogger(__name__)

# nice / ionice values for the BATCH tier. Highest niceness (lowest
# priority) for CPU; "idle" class for I/O. The kernel scheduler then
# starves the worker when anything else wants the CPU or disk —
# which is what we want.
BATCH_NICE = 19
BATCH_IONICE_CLASS = 3  # IOPRIO_CLASS_IDLE


def _ionice_to_idle():
    """ionice(2) syscall — set the current process's I/O priority
    class to IDLE. Called from the child preexec hook so it applies
    to the worker, not the scheduler.

    Done via ctypes rather than shelling to ionice(1) — same logic
    as the perf binding (no shell-out for things that are just
    syscalls)."""
    import ctypes
    libc = ctypes.CDLL('libc.so.6', use_errno=True)
    SYS_ioprio_set = 251  # x86_64
    IOPRIO_WHO_PROCESS = 1
    # ioprio_set(which, who, ioprio) — ioprio packs class in upper
    # bits, data in lower (data ignored for IDLE class).
    ioprio = BATCH_IONICE_CLASS << 13
    libc.syscall(SYS_ioprio_set, IOPRIO_WHO_PROCESS, 0, ioprio)


def _set_pdeathsig():
    from ..process_util import set_pdeathsig
    set_pdeathsig()


def _child_preexec():
    """Run in the child after fork, before exec.
    Apply nice + ionice so the worker starts deprioritized; also
    install PR_SET_PDEATHSIG so the child dies with its parent."""
    os.nice(BATCH_NICE)
    try:
        _ionice_to_idle()
    except OSError:
        # ionice failure is non-fatal; nice alone still helps.
        pass
    try:
        _set_pdeathsig()
    except (OSError, AttributeError):
        # prctl unavailable / call failed — fall back to manual
        # cleanup via atexit. Worth logging but not fatal.
        pass


class BatchWorker:
    """Manages one batch subprocess + its pause flag.

    `name` is a short identifier used to name the flag file and to
    distinguish workers in logs.
    `argv` is the command to run (must be a list — passed straight
    to subprocess.Popen, no shell). The path to the pause flag is
    appended automatically as `--pause-flag <path>` unless `argv`
    already contains `--pause-flag`.
    """

    def __init__(self, name: str, argv: Sequence[str]):
        self.name = name
        self.flag = PauseFlag(name)
        full_argv = list(argv)
        if '--pause-flag' not in full_argv:
            full_argv += ['--pause-flag', str(self.flag.path)]
        self._argv = full_argv
        self._proc: subprocess.Popen | None = None

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            raise RuntimeError(f'worker {self.name!r} already running')
        log.info(f'[scheduler] starting batch worker {self.name!r}: '
                  f'{" ".join(self._argv)}')
        self._proc = subprocess.Popen(
            self._argv,
            preexec_fn=_child_preexec,
            # Inherit stdout/stderr so the worker's logs flow into
            # the parent's terminal / log. If we want capture later
            # we can switch to PIPE.
        )

    def poll(self) -> int | None:
        """None if still running, exit code otherwise."""
        if self._proc is None:
            return None
        return self._proc.poll()

    def stop(self, timeout_s: float = 5.0) -> int | None:
        """SIGTERM, wait `timeout_s`, then SIGKILL if still alive.
        Returns exit code, or None if the worker was never started."""
        if self._proc is None:
            return None
        if self._proc.poll() is not None:
            return self._proc.returncode

        log.info(f'[scheduler] stopping worker {self.name!r} '
                  f'(pid={self._proc.pid})')
        self._proc.terminate()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                return self._proc.returncode
            time.sleep(0.05)
        log.warning(f'[scheduler] worker {self.name!r} did not exit '
                     f'on SIGTERM; sending SIGKILL')
        self._proc.kill()
        self._proc.wait()
        return self._proc.returncode

    def close(self):
        try:
            self.stop()
        finally:
            self.flag.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
