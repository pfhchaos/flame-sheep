"""Flame-specific wrapper around viz_authoring's worker subprocess
bootstrap. Adds flame-specific schema-init after the generic setup.

viz_authoring's `init_worker_subprocess` is intentionally
schema-agnostic — it doesn't know about flame_sheep's genome /
loops / ratings tables. This wrapper bridges that gap by running
`flame_sheep.storage._ensure_schema` after the generic init.

Workers should import from here, not from viz_authoring directly:

    from flame_sheep.worker_bootstrap import init_worker_subprocess
    log, conn = init_worker_subprocess('my_worker_name', db_path)
    # conn already has flame_sheep schema migrations applied
"""
from __future__ import annotations

import logging
import sqlite3

from viz_authoring.worker_bootstrap import (
    init_worker_subprocess as _generic_init,
)


def init_worker_subprocess(logger_name: str,
                            db_path: str,
                            busy_timeout_ms: int = 30_000,
                            silence_pil: bool = False,
                            ) -> tuple[logging.Logger, sqlite3.Connection]:
    """flame-sheep wrapper around viz_authoring.init_worker_subprocess
    that additionally runs flame_sheep's `_ensure_schema` on the
    returned connection. See the underlying function's docstring for
    full bootstrap details.

    Also installs PR_SET_PDEATHSIG so the kernel kills this worker
    when its parent (the wallpaper) dies. Covers crash, kill -9, and
    SIGSEGV restart-loop death modes that an explicit shutdown chain
    can't reach. The 5-day-old orphan wallpaper-child processes we
    cleaned up on 2026-06-09 are the failure mode this prevents:
    they inherited fd 17 (Wayland connection) from their parent and
    kept it alive long after the parent was gone.
    """
    # Install pdeathsig FIRST — before any work that might fail and
    # leave us orphaned. Even if the bootstrap below raises, the
    # signal handler is already armed.
    from .process_util import set_pdeathsig
    set_pdeathsig()

    log, conn = _generic_init(
        logger_name, db_path,
        busy_timeout_ms=busy_timeout_ms,
        silence_pil=silence_pil,
    )
    from .storage import _ensure_schema
    _ensure_schema(conn)
    return log, conn
