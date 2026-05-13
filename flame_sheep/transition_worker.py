"""Background transition scorer — computes pairwise distances between genomes.

Two phases:
  1. Backfill: compute variation_signature for genomes that lack one
  2. Score: compute transition distances within signature bins

Runs as a daemon subprocess. Load-aware (nice 19, pauses on high load).
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import sqlite3

log = logging.getLogger(__name__)


def _transition_main(db_path: str, stop_event: multiprocessing.synchronize.Event) -> None:
    """Entry point for the transition score subprocess."""
    for name in ('flame_sheep', 'flame_sheep_audio', None):
        logging.getLogger(name).handlers.clear()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S')
    log = logging.getLogger('flame_sheep.transition_worker')

    try:
        os.nice(19)
    except OSError:
        pass

    from .storage import _ensure_schema, _genome_from_json
    from .transition import (
        variation_signature, compute_transition_distance,
        signature_distance, CACHE_THRESHOLD, MAX_SIGNATURE_DISTANCE,
    )

    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA busy_timeout=5000')
    _ensure_schema(conn)

    try:
        while not stop_event.is_set():
            if os.getloadavg()[0] > BackgroundTransitionScorer.LOAD_THRESHOLD:
                stop_event.wait(BackgroundTransitionScorer.LOAD_CHECK_INTERVAL)
                continue

            # Phase 1: backfill variation_signature
            row = conn.execute(
                '''SELECT id, params FROM genomes
                   WHERE variation_signature IS NULL
                   LIMIT 1''',
            ).fetchone()

            if row is not None:
                gid, params_json = row
                try:
                    genome = _genome_from_json(params_json)
                    sig = variation_signature(genome)
                    n_xforms = len(genome.transforms)
                    conn.execute(
                        'UPDATE genomes SET variation_signature=?, n_transforms=? WHERE id=?',
                        (sig, n_xforms, gid),
                    )
                    conn.commit()
                    log.debug(f'backfill genome #{gid} sig={sig}')
                except Exception:
                    log.exception(f'failed to backfill genome #{gid}')
                    # Mark with empty signature so we don't retry forever
                    conn.execute(
                        "UPDATE genomes SET variation_signature='' WHERE id=?",
                        (gid,),
                    )
                    conn.commit()
                continue

            # Phase 2: compute transition distances
            row = conn.execute(
                '''SELECT g.id, g.params, g.variation_signature
                   FROM genomes g
                   WHERE g.variation_signature IS NOT NULL
                     AND g.variation_signature != ''
                     AND g.id NOT IN (
                         SELECT DISTINCT genome_a FROM genome_transitions
                     )
                   LIMIT 1''',
            ).fetchone()

            if row is None:
                stop_event.wait(BackgroundTransitionScorer.IDLE_CHECK_INTERVAL)
                continue

            gid_a, params_a, sig_a = row
            try:
                genome_a = _genome_from_json(params_a)

                # Find candidate genomes in same signature bin
                candidates = conn.execute(
                    '''SELECT id, params, variation_signature
                       FROM genomes
                       WHERE id != ?
                         AND variation_signature IS NOT NULL
                         AND variation_signature != ''
                    ''',
                    (gid_a,),
                ).fetchall()

                n_computed = 0
                n_cached = 0
                for gid_b, params_b, sig_b in candidates:
                    # Pre-filter by signature distance
                    if signature_distance(sig_a, sig_b) > MAX_SIGNATURE_DISTANCE:
                        continue

                    genome_b = _genome_from_json(params_b)
                    dist = compute_transition_distance(
                        genome_a, genome_b, sig_a=sig_a, sig_b=sig_b)
                    n_computed += 1

                    if dist < CACHE_THRESHOLD:
                        conn.execute(
                            'INSERT OR REPLACE INTO genome_transitions VALUES (?, ?, ?)',
                            (gid_a, gid_b, dist),
                        )
                        conn.execute(
                            'INSERT OR REPLACE INTO genome_transitions VALUES (?, ?, ?)',
                            (gid_b, gid_a, dist),
                        )
                        n_cached += 1

                # Insert a self-transition (distance=0) to mark this genome as processed
                conn.execute(
                    'INSERT OR REPLACE INTO genome_transitions VALUES (?, ?, ?)',
                    (gid_a, gid_a, 0.0),
                )
                conn.commit()

                log.debug(f'genome #{gid_a}: {n_computed} compared, {n_cached} cached')

            except Exception:
                log.exception(f'failed to compute transitions for genome #{gid_a}')
                # Mark as processed to avoid infinite retry
                conn.execute(
                    'INSERT OR REPLACE INTO genome_transitions VALUES (?, ?, ?)',
                    (gid_a, gid_a, 0.0),
                )
                conn.commit()

    finally:
        conn.close()


class BackgroundTransitionScorer:
    """Load-aware background process that computes genome transition distances."""

    LOAD_THRESHOLD = 6.0
    LOAD_CHECK_INTERVAL = 10.0
    IDLE_CHECK_INTERVAL = 30.0

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._process: multiprocessing.Process | None = None
        self._stop = multiprocessing.Event()

    def start(self) -> None:
        self._stop.clear()
        self._process = multiprocessing.Process(
            target=_transition_main,
            args=(self._db_path, self._stop),
            daemon=True, name='transition-scorer')
        self._process.start()
        log.info('transition worker started (pid=%d)', self._process.pid)

    def stop(self) -> None:
        self._stop.set()
        if self._process is not None:
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.kill()
            self._process = None
