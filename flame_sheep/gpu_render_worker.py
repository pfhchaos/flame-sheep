"""Background GPU renderer — renders genomes headlessly for scoring.

Runs as a daemon subprocess during wallpaper operation. Creates its own
EGL context, renders static + swept images for each unscored genome,
stores PNGs (and optionally raw histograms) in the DB.

The CPU score worker picks up rendered genomes and computes metrics
from the stored images.
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import os
import sqlite3
import time
import zlib

import numpy as np

log = logging.getLogger(__name__)

RENDER_VERSION = 1  # bump to re-render all genomes

COLOR_SCALE = 1_000_000.0


def _render_main(db_path: str, stop_event: multiprocessing.synchronize.Event) -> None:
    """Entry point for the GPU render subprocess."""
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S')
    log = logging.getLogger('flame_sheep.gpu_render_worker')

    try:
        os.nice(19)
    except OSError:
        pass

    from .config import cfg

    render_size = getattr(getattr(cfg, 'scoring', None), 'render_size', 512)
    render_sleep = getattr(getattr(cfg, 'scoring', None), 'render_sleep', 2.0)
    store_histograms = getattr(getattr(cfg, 'scoring', None), 'store_histograms', False)

    # Create standalone GPU context
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    import moderngl
    try:
        ctx = moderngl.create_context(standalone=True, backend='egl')
    except Exception as e:
        log.error(f'Failed to create EGL context: {e}')
        return

    from .renderer import FlameRenderer, N_ITERS
    from .storage import _genome_from_json, _ensure_schema
    from flame_sheep_audio import N_BINS
    from .genome import _score_from_histogram

    renderer = FlameRenderer(ctx, render_size, render_size, scoring=True)

    # Fixed rainbow palette for scoring renders (genome doesn't own a palette)
    _hues = np.linspace(0, 1, 256, endpoint=False)
    _rainbow = np.zeros((256, 3), dtype=np.float32)
    for i, h in enumerate(_hues):
        # HSV to RGB, S=1, V=1
        c = 1.0
        x = c * (1 - abs((h * 6) % 2 - 1))
        hi = int(h * 6) % 6
        if hi == 0:   _rainbow[i] = [c, x, 0]
        elif hi == 1: _rainbow[i] = [x, c, 0]
        elif hi == 2: _rainbow[i] = [0, c, x]
        elif hi == 3: _rainbow[i] = [0, x, c]
        elif hi == 4: _rainbow[i] = [x, 0, c]
        else:         _rainbow[i] = [c, 0, x]
    _gray_palette = np.tile(
        np.linspace(0, 1, 256, dtype=np.float32), (3, 1)).T.copy()
    log.info(f'GPU render worker started ({render_size}x{render_size}, '
             f'sleep={render_sleep}s, histograms={store_histograms})')

    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA busy_timeout=5000')
    _ensure_schema(conn)

    n_frames = 60
    swept_steps = 36

    try:
        while not stop_event.is_set():
            # Load-aware: pause when system is busy
            if os.getloadavg()[0] > BackgroundGpuRenderer.LOAD_THRESHOLD:
                stop_event.wait(BackgroundGpuRenderer.LOAD_CHECK_INTERVAL)
                continue

            # Find next genome to render
            row = conn.execute(
                'SELECT id, params FROM genomes '
                'WHERE render_version IS NULL OR render_version < ? '
                'LIMIT 1',
                (RENDER_VERSION,),
            ).fetchone()

            if row is None:
                stop_event.wait(BackgroundGpuRenderer.IDLE_CHECK_INTERVAL)
                continue

            gid, params_json = row
            try:
                genome = _genome_from_json(params_json)

                # --- Static render ---
                renderer.upload_genome(genome)
                renderer.upload_palette(_rainbow)
                renderer.reset_walkers()
                renderer.upload_audio(np.zeros(N_BINS, dtype=np.float32))
                if store_histograms:
                    renderer.clear_transform_hits()

                for _ in range(n_frames):
                    renderer.clear_histogram()
                    renderer.dispatch_chaos_game(iterations=N_ITERS)
                    ctx.memory_barrier()

                render_static = renderer.snapshot_png()

                # Read histogram data for centroid (and optionally store)
                hit_counts, color_accs = renderer.histogram_data()
                hist_static_blob = None
                hist_transform_blob = None
                if store_histograms:
                    hist_static_blob = zlib.compress(
                        hit_counts.tobytes() + color_accs.tobytes())
                    transform_hits = renderer.transform_hits_data()
                    hist_transform_blob = zlib.compress(transform_hits.tobytes())

                # --- Swept render ---
                renderer.upload_genome(genome)
                renderer.reset_walkers()
                renderer.upload_audio(np.zeros(N_BINS, dtype=np.float32))
                renderer.clear_histogram()

                base_rotation = genome.rotation
                for i in range(swept_steps):
                    angle = base_rotation + (2.0 * math.pi * i / swept_steps)
                    renderer.set_rotation(angle)
                    renderer.dispatch_chaos_game(iterations=N_ITERS)
                    ctx.memory_barrier()
                renderer.set_rotation(base_rotation)

                # Swept centroid from histogram (before snapshot)
                swept_hits, swept_colors = renderer.histogram_data()

                # Grayscale palette for swept snapshot (color is meaningless)
                renderer.upload_palette(_gray_palette)
                render_swept = renderer.snapshot_png()

                hist_swept_blob = None
                if store_histograms:
                    hist_swept_blob = zlib.compress(swept_hits.tobytes())

                # Compute swept centroid + balance
                swept_hit_grid = swept_hits.astype(np.float64)
                with np.errstate(divide='ignore', invalid='ignore'):
                    swept_color_grid = np.where(
                        swept_hits > 0,
                        swept_colors.astype(np.float64) / (swept_hit_grid * COLOR_SCALE),
                        0.0,
                    )
                swept_scores = _score_from_histogram(swept_hit_grid, swept_color_grid)

                # --- Store results ---
                conn.execute(
                    '''UPDATE genomes
                       SET render_static=?, render_swept=?,
                           hist_static=?, hist_swept=?, hist_transform=?,
                           centroid_x=?, centroid_y=?, balance=?,
                           render_version=?
                       WHERE id=?''',
                    (render_static, render_swept,
                     hist_static_blob, hist_swept_blob, hist_transform_blob,
                     swept_scores.get('centroid_offset_x'),
                     swept_scores.get('centroid_offset_y'),
                     swept_scores.get('balance'),
                     RENDER_VERSION,
                     gid),
                )
                conn.commit()

                log.debug(f'genome #{gid}  '
                          f'static={len(render_static)//1024}KB  '
                          f'swept={len(render_swept)//1024}KB  '
                          f'centroid=({swept_scores.get("centroid_offset_x", 0):.3f}, '
                          f'{swept_scores.get("centroid_offset_y", 0):.3f})')

            except Exception:
                log.exception(f'failed to render genome #{gid}')
                conn.execute(
                    'UPDATE genomes SET render_version=? WHERE id=?',
                    (RENDER_VERSION, gid),
                )
                conn.commit()

            # Yield GPU time to wallpaper
            stop_event.wait(render_sleep)

    finally:
        conn.close()
        ctx.release()


class BackgroundGpuRenderer:
    """Load-aware background process that renders unscored genomes."""

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
            target=_render_main,
            args=(self._db_path, self._stop),
            daemon=True, name='gpu-render')
        self._process.start()
        log.info('GPU render worker started (pid=%d)', self._process.pid)

    def stop(self) -> None:
        self._stop.set()
        if self._process is not None:
            self._process.join(timeout=10.0)
            if self._process.is_alive():
                self._process.kill()
            self._process = None
