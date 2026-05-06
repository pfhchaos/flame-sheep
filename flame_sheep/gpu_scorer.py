"""Standalone GPU genome scorer.

Renders genomes headlessly on the GPU and scores them using the full
histogram, color, and per-transform hit data. Much higher quality than
the CPU background scorer (128x128 / 50K iterations) — uses the same
FlameRenderer pipeline as the live wallpaper.

Run standalone:
    python -m flame_sheep.gpu_scorer [--size 512] [--frames 60] [--all]

Or import and call score_genome_gpu() directly.
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SCORE_VERSION = 5  # v5: GPU render + transform-based clustering

COLOR_SCALE = 1_000_000.0
DEFAULT_SIZE = 512
DEFAULT_FRAMES = 60


def _create_context():
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    import moderngl
    ctx = moderngl.create_context(standalone=True, backend='egl')
    log.info('GL context: %s', ctx.info['GL_RENDERER'])
    return ctx


def score_genome_gpu(genome, renderer, n_frames: int = DEFAULT_FRAMES,
                     ) -> dict[str, float]:
    """Render a genome on GPU and compute all scores.

    Args:
        genome: Genome object.
        renderer: FlameRenderer instance (already created with desired size).
        n_frames: Number of frames to accumulate.

    Returns:
        Dict of all score columns.
    """
    from .genome import _score_from_histogram, _score_symmetry
    from .cluster_scorer import score_from_clusters, score_from_transform_hits
    from .renderer import N_ITERS
    from flame_sheep_audio import N_BINS

    renderer.upload_genome(genome)
    renderer.reset_walkers()
    renderer.upload_audio(np.zeros(N_BINS, dtype=np.float32))
    renderer.clear_transform_hits()

    for _ in range(n_frames):
        renderer.clear_histogram()
        renderer.dispatch_chaos_game(iterations=N_ITERS)
        renderer.ctx.memory_barrier()

    # Read back raw data
    hit_counts, color_accs = renderer.histogram_data()
    transform_hits = renderer.transform_hits_data()

    hit_grid = hit_counts.astype(np.float64)

    # Recover average color index per pixel (undo COLOR_SCALE packing)
    with np.errstate(divide='ignore', invalid='ignore'):
        color_grid = np.where(
            hit_counts > 0,
            color_accs.astype(np.float64) / (hit_grid * COLOR_SCALE),
            0.0,
        )

    # Core histogram + symmetry scores
    scores = _score_from_histogram(hit_grid, color_grid)
    scores.update(_score_symmetry(hit_grid))

    # Cluster-based scoring (c-value)
    cl_scores = score_from_clusters(hit_grid, color_grid)
    scores.update(cl_scores)

    # Transform-based clustering
    n_transforms = len(genome.transforms)
    tf_scores = score_from_transform_hits(
        hit_grid, transform_hits[:, :, :n_transforms])
    scores.update(tf_scores)

    # detail_sensitivity: not computed here (needs half-iteration snapshot)
    scores.setdefault('detail_sensitivity', 0.0)

    return scores


def _db_path() -> Path:
    return Path.home() / '.local' / 'share' / 'flame-sheep' / 'library.db'


def _update_genome(conn: sqlite3.Connection, gid: int,
                   scores: dict[str, float]) -> None:
    conn.execute(
        '''UPDATE genomes
           SET coverage=?, entropy=?, color_entropy=?,
               balance=?, complexity=?,
               edge_sharpness=?, contour_coherence=?,
               symmetry_max=?, rotational=?, reflective=?,
               radial=?, periodic=?, fractal_dim=?,
               self_similarity=?, detail_sensitivity=?,
               centroid_x=?, centroid_y=?,
               cl_coverage=?, cl_edge_sharpness=?,
               cl_symmetry_best=?, cl_cluster_count=?,
               cl_dominance=?, cl_balance=?,
               cluster_detail=?,
               tf_coverage=?, tf_n_clusters=?, tf_avg_purity=?,
               tf_symmetry_best=?, tf_balance=?, tf_separation=?,
               score_version=?
           WHERE id=?''',
        (scores['coverage'], scores['entropy'],
         scores['color_entropy'], scores['balance'],
         scores['complexity'],
         scores.get('edge_sharpness', 0.0),
         scores.get('contour_coherence', 0.0),
         scores['symmetry_max'], scores['rotational'],
         scores['reflective'], scores['radial'],
         scores['periodic'], scores['fractal_dim'],
         scores['self_similarity'], scores['detail_sensitivity'],
         scores.get('centroid_offset_x'),
         scores.get('centroid_offset_y'),
         scores.get('cl_coverage'),
         scores.get('cl_edge_sharpness'),
         scores.get('cl_symmetry_best'),
         scores.get('cl_cluster_count'),
         scores.get('cl_dominance'),
         scores.get('cl_balance'),
         scores.get('cluster_detail'),
         scores.get('tf_coverage'),
         scores.get('tf_n_clusters'),
         scores.get('tf_avg_purity'),
         scores.get('tf_symmetry_best'),
         scores.get('tf_balance'),
         scores.get('tf_separation'),
         SCORE_VERSION,
         gid),
    )


def main():
    parser = argparse.ArgumentParser(description='GPU genome scorer')
    parser.add_argument('--size', type=int, default=DEFAULT_SIZE,
                        help='Render resolution (default: %(default)s)')
    parser.add_argument('--frames', type=int, default=DEFAULT_FRAMES,
                        help='Frames to accumulate (default: %(default)s)')
    parser.add_argument('--all', action='store_true',
                        help='Re-score all genomes, not just outdated ones')
    parser.add_argument('--limit', type=int, default=0,
                        help='Max genomes to score (0 = unlimited)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )

    from .storage import _genome_from_json, _ensure_schema
    from .renderer import FlameRenderer

    db = _db_path()
    conn = sqlite3.connect(str(db))
    _ensure_schema(conn)

    if args.all:
        query = 'SELECT id, params FROM genomes ORDER BY id'
        params = ()
    else:
        query = ('SELECT id, params FROM genomes '
                 'WHERE score_version IS NULL OR score_version < ? '
                 'ORDER BY id')
        params = (SCORE_VERSION,)

    rows = conn.execute(query, params).fetchall()
    total = len(rows)
    if args.limit > 0:
        rows = rows[:args.limit]

    if not rows:
        log.info('No genomes to score')
        return

    log.info('Scoring %d / %d genomes at %dx%d, %d frames',
             len(rows), total, args.size, args.size, args.frames)

    ctx = _create_context()
    renderer = FlameRenderer(ctx, args.size, args.size)

    scored = 0
    t0 = time.monotonic()

    for gid, params_json in rows:
        try:
            genome = _genome_from_json(params_json)
            scores = score_genome_gpu(genome, renderer, n_frames=args.frames)
            _update_genome(conn, gid, scores)
            conn.commit()
            scored += 1

            if scored % 50 == 0:
                elapsed = time.monotonic() - t0
                rate = scored / elapsed
                remaining = (len(rows) - scored) / rate
                log.info('  %d/%d scored (%.1f/s, ~%.0fs remaining)',
                         scored, len(rows), rate, remaining)

        except Exception:
            log.exception('Failed to score genome #%d', gid)
            conn.execute(
                'UPDATE genomes SET score_version=? WHERE id=?',
                (SCORE_VERSION, gid),
            )
            conn.commit()

    elapsed = time.monotonic() - t0
    log.info('Done: %d genomes in %.1fs (%.1f/s)', scored, elapsed,
             scored / elapsed if elapsed > 0 else 0)

    ctx.release()
    conn.close()


if __name__ == '__main__':
    main()
