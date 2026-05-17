"""Background genome pruner — archives genomes that fail stability checks.

Continuously scans genomes that haven't been checked, runs
check_stability() on each, and archives failures. Follows the
same load-aware daemon pattern as transition_worker and cpu_score_worker.

Phase 1: stability pruning (flying dots, pulsars)
Phase 2 (future): distance-based thinning of near-duplicates
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import sqlite3

log = logging.getLogger(__name__)


def _pruner_main(db_path: str, stop_event: multiprocessing.synchronize.Event) -> None:
    """Entry point for the pruner subprocess."""
    for name in ('flame_sheep', 'flame_sheep_audio', None):
        logging.getLogger(name).handlers.clear()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S')
    log = logging.getLogger('flame_sheep.pruner_worker')

    try:
        os.nice(19)
    except OSError:
        pass

    from .storage import _ensure_schema, _genome_from_json

    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA busy_timeout=30000')
    _ensure_schema(conn)

    log.info('pruner worker started')

    try:
        while not stop_event.is_set():
            if os.getloadavg()[0] > BackgroundPruner.LOAD_THRESHOLD:
                stop_event.wait(BackgroundPruner.LOAD_CHECK_INTERVAL)
                continue

            # Two pruning strategies:
            # 1. GPU-rendered genomes: check img_coverage from DB (trustworthy)
            # 2. Unrendered genomes: fall back to CPU check_stability()

            row = conn.execute(
                '''SELECT id, params, img_coverage, render_static
                   FROM genomes
                   WHERE COALESCE(archived, 0) = 0
                     AND COALESCE(pruner_checked, 0) = 0
                   LIMIT 1'''
            ).fetchone()

            if row is None:
                stop_event.wait(BackgroundPruner.IDLE_CHECK_INTERVAL)
                continue

            gid, params_json, img_coverage, has_render = row[0], row[1], row[2], row[3]
            try:
                should_archive = False
                reason = None

                if has_render is not None and img_coverage is not None:
                    # GPU render exists — trust the coverage metric
                    if img_coverage < 0.02:
                        should_archive = True
                        reason = 'low_coverage'
                else:
                    # No GPU render yet — use CPU stability check
                    genome = _genome_from_json(params_json)
                    if not genome.check_stability():
                        should_archive = True
                        reason = 'stability'

                if should_archive:
                    conn.execute(
                        'UPDATE genomes SET archived=1, pruner_checked=1, archive_reason=? WHERE id=?',
                        (reason, gid))
                    log.info('archived genome #%d (%s)', gid, reason)
                else:
                    conn.execute(
                        'UPDATE genomes SET pruner_checked=1 WHERE id=?',
                        (gid,))
                conn.commit()

            except sqlite3.OperationalError as e:
                if 'locked' in str(e):
                    log.debug(f'DB locked checking genome #{gid}, will retry')
                    stop_event.wait(2.0)
                else:
                    log.exception(f'failed to check genome #{gid}')
                    conn.execute(
                        'UPDATE genomes SET pruner_checked=1 WHERE id=?',
                        (gid,))
                    conn.commit()
            except Exception:
                log.exception(f'failed to check genome #{gid}')
                conn.execute(
                    'UPDATE genomes SET pruner_checked=1 WHERE id=?',
                    (gid,))
                conn.commit()

    finally:
        conn.close()


class BackgroundPruner:
    """Load-aware background process that archives unstable genomes."""

    LOAD_THRESHOLD = 6.0
    LOAD_CHECK_INTERVAL = 10.0
    IDLE_CHECK_INTERVAL = 60.0

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._process: multiprocessing.Process | None = None
        self._stop = multiprocessing.Event()

    def start(self) -> None:
        self._stop.clear()
        self._process = multiprocessing.Process(
            target=_pruner_main,
            args=(self._db_path, self._stop),
            daemon=True, name='genome-pruner')
        self._process.start()
        log.info('pruner worker started (pid=%d)', self._process.pid)

    def stop(self) -> None:
        self._stop.set()
        if self._process is not None:
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.kill()
            self._process = None
