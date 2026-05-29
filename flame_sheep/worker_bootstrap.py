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
    full bootstrap details."""
    log, conn = _generic_init(
        logger_name, db_path,
        busy_timeout_ms=busy_timeout_ms,
        silence_pil=silence_pil,
    )
    from .storage import _ensure_schema
    _ensure_schema(conn)
    return log, conn
