"""System-load-aware pacing for background workers.

Workers (genome scorer, transition scorer, pruner, GPU renderer) do
their work while the system isn't busy. Each iteration consults the
1-minute load average; if the box is busy, the worker sleeps for a
beat and rechecks. Keeps the wallpaper at 60fps even with batch CPU
work running in the background.

Defaults are the values all four workers had been running with (6.0
threshold, 10s check interval) — the constants were duplicated across
workers but turned out to be the same number. The principled
per-worker variance is the IDLE_CHECK_INTERVAL (how often to poll the
DB for new work), which stays in each worker.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as MpEvent

# 1-minute load average above which workers should pause and recheck
# later. Set to roughly 1.5× the number of cores on a typical desktop:
# normal interactive load sits well below; a sustained spike pushes
# above. Workers yield to keep the wallpaper smooth.
LOAD_THRESHOLD = 6.0

# Seconds to wait before rechecking after a loaded-state hit. Long
# enough that brief CPU spikes (a compile, a tab switch) don't make
# the worker thrash; short enough that workers resume promptly once
# the spike passes.
LOAD_CHECK_INTERVAL = 10.0


def is_loaded(threshold: float = LOAD_THRESHOLD) -> bool:
    """True if the 1-minute load average exceeds `threshold`."""
    return os.getloadavg()[0] > threshold


def sleep_if_loaded(stop_event: 'MpEvent',
                    threshold: float = LOAD_THRESHOLD,
                    check_interval: float = LOAD_CHECK_INTERVAL) -> bool:
    """Pace one iteration of a load-aware worker loop.

    If the system is loaded, sleep `check_interval` seconds
    (interruptible by `stop_event`) and return True — caller should
    `continue` and try again. Otherwise return False immediately —
    caller proceeds to do work.

    Pattern at the top of a worker's main loop:

        while not stop_event.is_set():
            if sleep_if_loaded(stop_event):
                continue
            ...do one unit of work...
    """
    if is_loaded(threshold):
        stop_event.wait(check_interval)
        return True
    return False
