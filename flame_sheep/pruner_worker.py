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

    # Periodic retrain-recommendation check. Runs on a slow cadence
    # (every 5 min) inside the pruner's loop — same DB connection, no
    # extra thread, no GPU involvement. Just queries judgment counts
    # against current_generation and logs the recommendation. The
    # actual train+breed cycle is still manual; this is just the
    # nudge that says "you have enough new data, consider retraining."
    import time as _time
    from .storage import Library
    last_retrain_check = 0.0
    RETRAIN_CHECK_INTERVAL = 300.0  # seconds
    _lib_for_retrain = None  # lazy

    try:
        while not stop_event.is_set():
            # Periodic retrain heuristic — independent of the pruner's
            # genome-iteration cadence, throttled by wall time so it
            # doesn't spam the log.
            now = _time.monotonic()
            if now - last_retrain_check >= RETRAIN_CHECK_INTERVAL:
                last_retrain_check = now
                try:
                    if _lib_for_retrain is None:
                        _lib_for_retrain = Library()
                    rec = _lib_for_retrain.retrain_recommendation()
                    level = log.warning if rec['should_retrain'] else log.info
                    level('[retrain] %s', rec['message'])
                except Exception:
                    log.exception('[retrain] recommendation check failed')

            # Generation-advance latch: this worker owns the breed and
            # prune steps. Both can be heavy; do at most one per iteration
            # and let the next loop tick re-check state.
            try:
                from .gen_advance import (
                    bulk_breed, rank_prune, compute_breed_count,
                    measure_disagreement,
                    STATE_AWAITING_BREED, STATE_AWAITING_SCORE_NEW,
                    STATE_AWAITING_PRUNE, STATE_IDLE,
                )
                from .storage import Library
                if _lib_for_retrain is None:
                    _lib_for_retrain = Library()
                gen_state = _lib_for_retrain.get_gen_advance_state()

                if gen_state == STATE_AWAITING_BREED:
                    target = _lib_for_retrain.get_gen_advance_target()
                    disagreement = measure_disagreement(conn)
                    pop = conn.execute(
                        'SELECT COUNT(*) FROM genomes WHERE COALESCE(archived,0)=0'
                    ).fetchone()[0]
                    n = compute_breed_count(disagreement, pop)
                    log.info('[gen-advance] bulk-breed start: target gen %s, '
                             'pop=%d, disagreement=%.3f, count=%d',
                             target, pop, disagreement, n)
                    result = bulk_breed(_lib_for_retrain, n)
                    log.info('[gen-advance] bulk-breed done: produced %d/%d '
                             'children. Flipping → awaiting_score_new',
                             len(result['ids']), n)
                    _lib_for_retrain.cas_gen_advance_state(
                        STATE_AWAITING_BREED, STATE_AWAITING_SCORE_NEW)
                    continue

                if gen_state == STATE_AWAITING_PRUNE:
                    target = _lib_for_retrain.get_gen_advance_target()
                    current_gen = _lib_for_retrain.get_current_generation()
                    result = rank_prune(_lib_for_retrain, current_gen)
                    new_gen = _lib_for_retrain.advance_generation()
                    _lib_for_retrain.clear_gen_advance()
                    log.warning('[gen-advance] COMPLETE: now at generation %d '
                                '(target was %s, archived %d, pop %d → %d)',
                                new_gen, target, result['archived'],
                                result['pop_before'], result['pop_after'])
                    continue
            except Exception:
                log.exception('[gen-advance] step failed')

            if os.getloadavg()[0] > BackgroundPruner.LOAD_THRESHOLD:
                stop_event.wait(BackgroundPruner.LOAD_CHECK_INTERVAL)
                continue

            row = conn.execute(
                '''SELECT id, params, img_coverage
                   FROM genomes
                   WHERE COALESCE(archived, 0) = 0
                     AND COALESCE(pruner_checked, 0) = 0
                   LIMIT 1'''
            ).fetchone()

            if row is None:
                stop_event.wait(BackgroundPruner.IDLE_CHECK_INTERVAL)
                continue

            gid, params_json, img_coverage = row
            try:
                reason = None

                # Check 1: GPU coverage (only for rendered genomes)
                if img_coverage is not None and img_coverage < 0.02:
                    reason = 'low_coverage'
                    log.info('archived genome #%d (img_coverage=%.4f)', gid, img_coverage)

                # Check 2: CPU viability (bbox + pulsar, NOT cell coverage)
                if reason is None:
                    genome = _genome_from_json(params_json)
                    if not genome.check_stability():
                        reason = 'stability'
                        log.info('archived genome #%d (failed stability)', gid)

                if reason:
                    conn.execute(
                        'UPDATE genomes SET archived=1, pruner_checked=1, archive_reason=? WHERE id=?',
                        (reason, gid))
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
        if _lib_for_retrain is not None:
            try:
                _lib_for_retrain.close()
            except Exception:
                pass


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
