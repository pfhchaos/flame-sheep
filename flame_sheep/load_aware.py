"""Shim re-exporting from the viz_authoring package.

System-load-aware pacing for background workers — extracted into the
standalone viz_authoring package in Stage 9 of `docs/reorg_plan.md`.
This shim keeps `from flame_sheep.load_aware import sleep_if_loaded`
working for any in-flame consumer that wasn't updated.
"""
from viz_authoring.load_aware import (  # noqa: F401
    LOAD_THRESHOLD,
    LOAD_CHECK_INTERVAL,
    is_loaded,
    sleep_if_loaded,
)
