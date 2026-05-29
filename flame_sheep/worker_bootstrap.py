"""Standard subprocess bootstrap for the background workers (genome
scorer, transition scorer, pruner, GPU renderer).

Each worker spawns via `multiprocessing.Process(target=_X_main)`. On
entry to that function the subprocess inherits the parent's logging
handlers via fork — they need clearing and reconfiguring so the
subprocess's output goes to stdout cleanly. The process also wants to
be niced (background priority — we yield to interactive work) and
typically opens its own SQLite connection with a long busy_timeout so
contention with the parent's writes doesn't fail loud.

That setup pattern was duplicated across all four workers with subtle
copy-paste variation. Centralized here so future workers don't have
to remember the four steps and the variations are explicit (silence_pil
and busy_timeout are the parameters that actually differed).

Companion to `flame_sheep.load_aware` (which handles the per-iteration
pacing). Together they cover the boilerplate slice of the worker
lifecycle; the worker-specific work lives in each worker module.
"""
from __future__ import annotations

import logging
import os
import sqlite3


_DEFAULT_LOG_FORMAT = '%(asctime)s %(name)s %(levelname)s %(message)s'
_DEFAULT_DATEFMT = '%H:%M:%S'
_DEFAULT_BUSY_TIMEOUT_MS = 30_000  # generous default; renderer overrides to 5_000


def init_worker_subprocess(logger_name: str,
                            db_path: str,
                            busy_timeout_ms: int = _DEFAULT_BUSY_TIMEOUT_MS,
                            silence_pil: bool = False,
                            ) -> tuple[logging.Logger, sqlite3.Connection]:
    """Bootstrap a background-worker subprocess.

    Steps:
      1. Clear inherited logging handlers, reconfigure for stdout
      2. Optionally silence PIL's INFO logs (workers that touch images)
      3. Nice the process to background priority (level 19)
      4. Open SQLite connection with `busy_timeout_ms` PRAGMA
      5. Run schema migrations via `_ensure_schema`

    Returns:
        (log, conn) — the worker-named logger and an open SQLite
        connection. Caller owns `conn` (responsible for `conn.close()`
        in its own try/finally, typically as part of the worker's
        graceful shutdown).
    """
    # Subprocess fork inherits parent's handlers; clear so the
    # subprocess's basicConfig actually takes effect.
    for name in ('flame_sheep', 'flame_sheep_audio', None):
        logging.getLogger(name).handlers.clear()
    logging.basicConfig(level=logging.INFO,
                        format=_DEFAULT_LOG_FORMAT,
                        datefmt=_DEFAULT_DATEFMT)
    if silence_pil:
        logging.getLogger('PIL').setLevel(logging.WARNING)
    log = logging.getLogger(logger_name)

    # Lowest user-priority. Workers run only when interactive work
    # isn't holding the CPU.
    try:
        os.nice(19)
    except OSError:
        pass  # already at lowest priority or not permitted

    conn = sqlite3.connect(db_path)
    conn.execute(f'PRAGMA busy_timeout={busy_timeout_ms}')
    from .storage import _ensure_schema
    _ensure_schema(conn)

    return log, conn
