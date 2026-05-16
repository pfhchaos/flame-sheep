"""Background CPU scorer — computes metrics from rendered images.

Picks up genomes that have been rendered (render_version current) but
not yet scored (score_version outdated). Reads PNG blobs from the DB,
runs image-based and experimental metrics, stores numeric scores.

Runs as a daemon subprocess. Load-aware (nice 19, pauses on high load).
"""

from __future__ import annotations

import io
import logging
import multiprocessing
import os
import sqlite3
import zlib

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

SCORE_VERSION = 8  # v8: + CNN aesthetic score


def _score_genome(render_static: bytes, render_swept: bytes | None,
                  hist_static: bytes | None = None,
                  hist_transform: bytes | None = None,
                  hist_swept: bytes | None = None,
                  hist_first_hit: bytes | None = None,
                  render_size: int = 512,
                  cnn_model=None,
                  ) -> dict[str, float]:
    """Compute all metrics from stored blobs."""
    from .image_scorer import score_from_image
    from .experimental_metrics import score_all

    scores: dict[str, float] = {}

    # Load static image
    static_img = np.array(Image.open(io.BytesIO(render_static)))

    # Image-based scoring (existing)
    scores.update(score_from_image(static_img))

    # Load swept image if available
    swept_img = None
    if render_swept:
        swept_img = np.array(Image.open(io.BytesIO(render_swept)))

    # Experimental metrics
    scores.update(score_all(static_img, swept_img))

    # CNN aesthetic score — prefer domain-native histograms over PNG
    if cnn_model is not None:
        try:
            import torch

            if hist_static is not None and hist_swept is not None:
                from .cnn_scorer import _prepare_input_domain
                tensor = _prepare_input_domain(hist_static, hist_swept, hist_first_hit)
            elif render_swept is not None:
                from .cnn_scorer import _prepare_input
                tensor = _prepare_input(render_static, render_swept)
            else:
                tensor = None

            if tensor is not None:
                with torch.no_grad():
                    scores['cnn_score'] = cnn_model(tensor).item()
        except Exception:
            pass  # non-fatal: other scores still valid

    # Histogram-based scoring (if blobs stored)
    if hist_static is not None:
        from .genome import _score_from_histogram, _score_symmetry

        raw = zlib.decompress(hist_static)
        n_pixels = render_size * render_size
        hit_grid = np.frombuffer(raw, dtype=np.uint32, count=n_pixels
                                 ).reshape(render_size, render_size).astype(np.float64)
        color_grid_raw = np.frombuffer(raw, dtype=np.uint32, offset=n_pixels * 4,
                                       count=n_pixels
                                       ).reshape(render_size, render_size)

        with np.errstate(divide='ignore', invalid='ignore'):
            color_grid = np.where(
                hit_grid > 0,
                color_grid_raw.astype(np.float64) / (hit_grid * 1_000_000.0),
                0.0,
            )

        hist_scores = _score_from_histogram(hit_grid, color_grid)
        sym_scores = _score_symmetry(hit_grid)
        # Prefix to avoid collision with image-based scores
        for k, v in hist_scores.items():
            scores.setdefault(k, v)
        scores.update(sym_scores)

        # Cluster scoring
        from .cluster_scorer import score_from_clusters
        cl_scores = score_from_clusters(hit_grid, color_grid)
        scores.update(cl_scores)

    # Transform-based clustering (if stored)
    if hist_static is not None and hist_transform is not None:
        from .cluster_scorer import score_from_transform_hits
        import struct as _struct

        tf_raw = zlib.decompress(hist_transform)
        # v4+ blobs have an 8-byte dimension header from pack_histogram().
        # Detect by checking if first 8 bytes decode to plausible dimensions.
        try:
            hdr_h, hdr_w = _struct.unpack('II', tf_raw[:8])
            if hdr_h == render_size and hdr_w == render_size:
                tf_raw = tf_raw[8:]
        except _struct.error:
            pass
        n_total = len(tf_raw) // 4
        n_transforms = n_total // (render_size * render_size)
        transform_hits = np.frombuffer(tf_raw, dtype=np.uint32
                                       ).reshape(render_size, render_size, n_transforms)
        tf_scores = score_from_transform_hits(hit_grid, transform_hits)
        scores.update(tf_scores)

    scores.setdefault('detail_sensitivity', 0.0)
    return scores


def _refresh_loop_fitness(conn: sqlite3.Connection, log: logging.Logger) -> None:
    """Recompute fitness for all loops after a scoring pass completes."""
    from .storage import Library
    try:
        lib = Library.__new__(Library)
        lib.conn = conn
        loop_ids = [r[0] for r in conn.execute('SELECT id FROM loops').fetchall()]
        for lid in loop_ids:
            lib.update_loop_fitness(lid)
        if loop_ids:
            log.info('Refreshed fitness for %d loops after scoring pass', len(loop_ids))
    except Exception:
        log.exception('Failed to refresh loop fitness')


def _rescore_cnn(conn, row, cnn_model, weights_hash, log,
                 store_detail: bool = False):
    """Re-score a single genome with CNN only (heuristics unchanged)."""
    import torch

    gid = row[0]
    render_static, render_swept = row[1], row[2]
    hist_static, hist_swept, hist_first_hit = row[3], row[4], row[5]

    try:
        if hist_static is not None and hist_swept is not None:
            from .cnn_scorer import _prepare_input_domain
            tensor = _prepare_input_domain(hist_static, hist_swept, hist_first_hit)
        elif render_swept is not None:
            from .cnn_scorer import _prepare_input
            tensor = _prepare_input(render_static, render_swept)
        else:
            conn.execute(
                'UPDATE genomes SET cnn_weights_hash=? WHERE id=?',
                (weights_hash, gid))
            conn.commit()
            return

        with torch.no_grad():
            score = cnn_model(tensor).item()

        if store_detail:
            import json
            existing = conn.execute(
                'SELECT cnn_scores_detail FROM genomes WHERE id=?', (gid,)
            ).fetchone()
            detail = json.loads(existing[0]) if existing and existing[0] else {}
            detail[weights_hash] = score
            conn.execute(
                'UPDATE genomes SET cnn_score=?, cnn_weights_hash=?, cnn_scores_detail=? WHERE id=?',
                (score, weights_hash, json.dumps(detail), gid))
        else:
            conn.execute(
                'UPDATE genomes SET cnn_score=?, cnn_weights_hash=? WHERE id=?',
                (score, weights_hash, gid))
        conn.commit()
        log.debug(f'CNN rescore genome #{gid}: {score:.3f}')
    except Exception:
        log.exception(f'CNN rescore failed for genome #{gid}')
        conn.execute(
            'UPDATE genomes SET cnn_weights_hash=? WHERE id=?',
            (weights_hash, gid))
        conn.commit()


def _score_main(db_path: str, stop_event: multiprocessing.synchronize.Event) -> None:
    """Entry point for the CPU score subprocess."""
    # Clear inherited handlers from fork (parent's setup_logging adds to named loggers)
    for name in ('flame_sheep', 'flame_sheep_audio', None):
        logger = logging.getLogger(name)
        logger.handlers.clear()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger('PIL').setLevel(logging.WARNING)
    log = logging.getLogger('flame_sheep.cpu_score_worker')

    try:
        os.nice(19)
    except OSError:
        pass

    from .gpu_render_worker import RENDER_VERSION
    from .storage import _ensure_schema

    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA busy_timeout=5000')
    _ensure_schema(conn)

    # Load CNN model once (if weights available) + compute weights hash
    cnn_model = None
    cnn_weights_hash = None
    try:
        import hashlib
        from .cnn_scorer import load_model, _default_weights_path
        weights_path = _default_weights_path()
        if weights_path.exists():
            cnn_model = load_model()
            cnn_weights_hash = hashlib.sha256(weights_path.read_bytes()).hexdigest()[:16]
            log.info('CNN scorer loaded from %s (hash=%s)', weights_path, cnn_weights_hash)
        else:
            log.warning('CNN weights not found at %s — CNN scoring disabled',
                        weights_path)
    except Exception:
        log.exception('Failed to load CNN scorer — CNN scoring disabled')

    cnn_weights_mtime = weights_path.stat().st_mtime if (cnn_model and weights_path.exists()) else 0

    # Config for multi-model detail tracking
    from .config import cfg
    store_cnn_detail = getattr(getattr(cfg, 'scoring', None), 'store_cnn_detail', False)

    was_scoring = False  # track when we transition from scoring → idle

    try:
        while not stop_event.is_set():
            if os.getloadavg()[0] > BackgroundCpuScorer.LOAD_THRESHOLD:
                stop_event.wait(BackgroundCpuScorer.LOAD_CHECK_INTERVAL)
                continue

            row = conn.execute(
                '''SELECT id, render_static, render_swept,
                          hist_static, hist_transform,
                          hist_swept, hist_first_hit
                   FROM genomes
                   WHERE render_version >= ?
                     AND (score_version IS NULL OR score_version < ?)
                     AND render_static IS NOT NULL
                   LIMIT 1''',
                (RENDER_VERSION, SCORE_VERSION),
            ).fetchone()

            if row is None:
                # CNN-only re-score: find genomes with stale weights hash
                cnn_row = None
                if cnn_model is not None and cnn_weights_hash:
                    cnn_row = conn.execute(
                        '''SELECT id, render_static, render_swept,
                                  hist_static, hist_swept, hist_first_hit
                           FROM genomes
                           WHERE render_version >= ?
                             AND render_static IS NOT NULL
                             AND (cnn_weights_hash IS NULL OR cnn_weights_hash != ?)
                           LIMIT 1''',
                        (RENDER_VERSION, cnn_weights_hash),
                    ).fetchone()
                    if cnn_row is not None:
                        _rescore_cnn(conn, cnn_row, cnn_model, cnn_weights_hash, log,
                                     store_detail=store_cnn_detail)
                        was_scoring = True
                        continue

                # Both passes idle — refresh loop fitness once
                if was_scoring:
                    _refresh_loop_fitness(conn, log)
                    was_scoring = False

                # Hot-reload weights if file changed
                if weights_path and weights_path.exists():
                    try:
                        current_mtime = weights_path.stat().st_mtime
                        if current_mtime != cnn_weights_mtime:
                            cnn_model = load_model()
                            cnn_weights_hash = hashlib.sha256(
                                weights_path.read_bytes()).hexdigest()[:16]
                            cnn_weights_mtime = current_mtime
                            log.info('CNN weights reloaded (hash=%s)', cnn_weights_hash)
                            was_scoring = True  # trigger rescore pass
                            continue
                    except Exception:
                        log.exception('Failed to reload CNN weights, keeping previous')

                stop_event.wait(BackgroundCpuScorer.IDLE_CHECK_INTERVAL)
                continue

            was_scoring = True

            gid = row[0]
            render_static = row[1]
            render_swept = row[2]
            hist_static = row[3]
            hist_transform = row[4]
            hist_swept = row[5]
            hist_first_hit = row[6]

            try:
                scores = _score_genome(render_static, render_swept,
                                       hist_static, hist_transform,
                                       hist_swept=hist_swept,
                                       hist_first_hit=hist_first_hit,
                                       cnn_model=cnn_model)

                # Build SET clause dynamically from available scores
                score_cols = [
                    'coverage', 'entropy', 'color_entropy', 'balance',
                    'complexity', 'edge_sharpness', 'contour_coherence',
                    'symmetry_max', 'rotational', 'reflective', 'radial',
                    'periodic', 'fractal_dim', 'self_similarity',
                    'detail_sensitivity',
                    'img_coverage', 'img_structural_edges',
                    'img_euclidean_edges', 'img_color_edges',
                    'img_color_regions', 'img_color_coherence',
                    'img_color_variety',
                    'cl_coverage', 'cl_edge_sharpness', 'cl_symmetry_best',
                    'cl_cluster_count', 'cl_dominance', 'cl_balance',
                    'tf_coverage', 'tf_n_clusters', 'tf_avg_purity',
                    'tf_symmetry_best', 'tf_balance', 'tf_separation',
                    # Experimental
                    'sw_rotational', 'sw_coverage',
                    'freq_high_ratio', 'freq_peak_scale',
                    'lacunarity', 'radial_slope', 'radial_r2',
                    'compactness', 'bbox_aspect', 'filament_count',
                    'contrast_ratio', 'angular_uniformity',
                    # CNN
                    'cnn_score',
                ]

                set_parts = []
                values = []
                for col in score_cols:
                    if col in scores:
                        set_parts.append(f'{col}=?')
                        values.append(scores[col])

                set_parts.append('score_version=?')
                values.append(SCORE_VERSION)
                if 'cnn_score' in scores and cnn_weights_hash:
                    set_parts.append('cnn_weights_hash=?')
                    values.append(cnn_weights_hash)
                    # Multi-model detail tracking
                    if store_cnn_detail:
                        import json
                        existing = conn.execute(
                            'SELECT cnn_scores_detail FROM genomes WHERE id=?', (gid,)
                        ).fetchone()
                        detail = json.loads(existing[0]) if existing and existing[0] else {}
                        detail[cnn_weights_hash] = scores['cnn_score']
                        set_parts.append('cnn_scores_detail=?')
                        values.append(json.dumps(detail))
                values.append(gid)

                sql = f'UPDATE genomes SET {", ".join(set_parts)} WHERE id=?'
                conn.execute(sql, values)
                conn.commit()

                log.debug(f'genome #{gid}  '
                          f'img_cov={scores.get("img_coverage", 0):.3f}  '
                          f'filaments={scores.get("filament_count", 0):.0f}  '
                          f'sw_rot={scores.get("sw_rotational", 0):.3f}')

            except Exception:
                log.exception(f'failed to score genome #{gid}')
                conn.execute(
                    'UPDATE genomes SET score_version=? WHERE id=?',
                    (SCORE_VERSION, gid),
                )
                conn.commit()

    finally:
        conn.close()


class BackgroundCpuScorer:
    """Load-aware background process that scores rendered genomes."""

    LOAD_THRESHOLD = 6.0
    LOAD_CHECK_INTERVAL = 10.0
    IDLE_CHECK_INTERVAL = 10.0

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._process: multiprocessing.Process | None = None
        self._stop = multiprocessing.Event()

    def start(self) -> None:
        self._stop.clear()
        self._process = multiprocessing.Process(
            target=_score_main,
            args=(self._db_path, self._stop),
            daemon=True, name='cpu-scorer')
        self._process.start()
        log.info('CPU score worker started (pid=%d)', self._process.pid)

    def stop(self) -> None:
        self._stop.set()
        if self._process is not None:
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.kill()
            self._process = None
