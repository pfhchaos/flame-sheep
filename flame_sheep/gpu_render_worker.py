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

RENDER_VERSION = 10  # v10: first-hit = single-frame snapshots at iter=N (match live wallpaper semantics)

COLOR_SCALE = 1_000_000.0


def _render_main(db_path: str, stop_event: multiprocessing.synchronize.Event) -> None:
    """Entry point for the GPU render subprocess."""
    for name in ('flame_sheep', 'flame_sheep_audio', None):
        logging.getLogger(name).handlers.clear()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger('PIL').setLevel(logging.WARNING)
    log = logging.getLogger('flame_sheep.gpu_render_worker')

    try:
        os.nice(19)
    except OSError:
        pass

    # Let wallpaper start up before competing for GPU
    time.sleep(10)

    from .config import cfg

    scoring_cfg = getattr(cfg, 'scoring', None)
    render_size = getattr(scoring_cfg, 'render_size', 512)
    render_sleep = getattr(scoring_cfg, 'render_sleep', 2.0)

    # Walker count: match the walker-per-pixel density of an assumed live render
    # so the CNN sees attractor convergence matching what it scores at runtime.
    live_w = getattr(scoring_cfg, 'assumed_live_render_w', 1080)
    live_h = getattr(scoring_cfg, 'assumed_live_render_h', 1080)
    live_walkers = getattr(scoring_cfg, 'live_walkers', 65536)
    walker_density = live_walkers / (live_w * live_h)
    scoring_walkers = max(64, int(walker_density * render_size * render_size))
    # Round to multiple of 64 (workgroup size)
    scoring_walkers = (scoring_walkers // 64) * 64

    # Create standalone GPU context
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    import moderngl
    try:
        ctx = moderngl.create_context(standalone=True, backend='egl')
    except Exception as e:
        log.error(f'Failed to create EGL context: {e}')
        return

    from .renderer import FlameRenderer, N_ITERS, LIVE_ITER_MIN, LIVE_ITER_MAX
    from .storage import _genome_from_json, _ensure_schema
    from flame_sheep_audio import N_BINS
    from .genome import _score_from_histogram

    renderer = FlameRenderer(ctx, render_size, render_size,
                              scoring=True, n_walkers=scoring_walkers)

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
    from .scoring_channels import pack_histogram, pack_static_histogram
    log.info(f'GPU render worker started ({render_size}x{render_size}, '
             f'{scoring_walkers} walkers, sleep={render_sleep}s)')

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

                # --- Static render with first-hit snapshots ---
                renderer.upload_genome(genome)
                renderer.upload_palette(_rainbow)
                renderer.reset_walkers()
                renderer.upload_audio(np.zeros(N_BINS, dtype=np.float32))
                renderer.clear_transform_hits()

                # First-hit iteration response: for each pixel, record the
                # smallest iter count at which it appears in a single live frame.
                # This matches what the live wallpaper actually does: each
                # frame is a fresh dispatch at iter=N (driven by audio energy)
                # into a cleared histogram. So we snapshot single-frame renders
                # at 16 iter counts spanning the live range.
                #
                # Encoding: 0 = trunk (appears at LIVE_ITER_MIN, denset),
                #           240 = wispy (only appears at LIVE_ITER_MAX),
                #           255 = sentinel for "never appears at any energy".
                # The 15-step gap between 240 and 255 separates "barely visible
                # at peak energy" from "never visible".
                n_snapshots = 16
                snap_iters = np.linspace(LIVE_ITER_MIN, LIVE_ITER_MAX,
                                         n_snapshots, dtype=int)

                # Burn-in: get walkers to converged positions before the
                # first snapshot, otherwise burn-in trace pollutes snap 0.
                # Live wallpaper walkers persist across frames so they're
                # always converged in steady state — match that.
                renderer.clear_histogram()
                renderer.dispatch_chaos_game(iterations=50)
                ctx.memory_barrier()

                first_hit = None
                for snap_idx, target in enumerate(snap_iters):
                    renderer.clear_histogram()
                    renderer.dispatch_chaos_game(iterations=int(target))
                    ctx.memory_barrier()
                    hits, _ = renderer.histogram_data()
                    if first_hit is None:
                        first_hit = np.full(hits.shape, 255, dtype=np.uint8)
                    mapped = snap_idx * 240 // (n_snapshots - 1)
                    first_hit[(hits > 0) & (first_hit == 255)] = mapped

                # hist_static is the single live frame at peak energy
                # (LIVE_ITER_MAX). The last snapshot above already rendered
                # exactly that, but transform_hits accumulated across the loop —
                # do a fresh dispatch to get clean transform_hits too.
                renderer.clear_histogram()
                renderer.clear_transform_hits()
                renderer.dispatch_chaos_game(iterations=LIVE_ITER_MAX)
                ctx.memory_barrier()

                render_static = renderer.snapshot_png()

                # Read histogram data
                hit_counts, color_accs = renderer.histogram_data()
                hist_static_blob = pack_static_histogram(hit_counts, color_accs)
                transform_hits = renderer.transform_hits_data()
                hist_transform_blob = pack_histogram(transform_hits)
                hist_first_hit_blob = pack_histogram(first_hit) if first_hit is not None else None

                # --- Swept render ---
                renderer.upload_genome(genome)
                renderer.reset_walkers()
                renderer.upload_audio(np.zeros(N_BINS, dtype=np.float32))
                renderer.clear_histogram()

                base_rotation = genome.rotation
                for i in range(swept_steps):
                    angle = base_rotation + (2.0 * math.pi * i / swept_steps)
                    rotated = genome.rotated(angle - base_rotation)
                    renderer.upload_genome(rotated)
                    # Fix viewport to base rotation — only affines change
                    renderer.set_rotation(base_rotation)
                    renderer.dispatch_chaos_game(iterations=N_ITERS)
                    ctx.memory_barrier()

                # Swept centroid from histogram (before snapshot)
                swept_hits, swept_colors = renderer.histogram_data()

                # Grayscale palette for swept snapshot (color is meaningless)
                renderer.upload_palette(_gray_palette)
                render_swept = renderer.snapshot_png()

                hist_swept_blob = pack_histogram(swept_hits)

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
                # Scalar metadata stays on the genomes table.
                conn.execute(
                    '''UPDATE genomes
                       SET centroid_x=?, centroid_y=?, balance=?,
                           render_version=?
                       WHERE id=?''',
                    (swept_scores.get('centroid_offset_x'),
                     swept_scores.get('centroid_offset_y'),
                     swept_scores.get('balance'),
                     RENDER_VERSION,
                     gid),
                )
                # Per-channel normalization stats for the trainer. Compute
                # by re-running normalize_channels on this render's
                # histograms; trivial CPU cost relative to the render.
                from .scoring_channels import normalize_channels
                try:
                    channels = normalize_channels(hit_counts, color_accs,
                                                  swept_hits, first_hit)
                    chan_means = channels.mean(axis=(1, 2)).astype(np.float64)
                    chan_stds = channels.std(axis=(1, 2)).astype(np.float64)
                except Exception as _e:
                    log.warning(f'channel stats failed for gid {gid}: {_e}')
                    chan_means = (None, None, None, None)
                    chan_stds = (None, None, None, None)

                # Blobs + stats go to the sibling table — keep them off the wide row.
                conn.execute(
                    '''INSERT OR REPLACE INTO genome_blobs
                       (genome_id, render_static, render_swept,
                        hist_static, hist_swept, hist_transform, hist_first_hit,
                        mean_h, mean_s, mean_l, mean_a,
                        std_h,  std_s,  std_l,  std_a)
                       VALUES (?, ?, ?, ?, ?, ?, ?,
                               ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (gid, render_static, render_swept,
                     hist_static_blob, hist_swept_blob, hist_transform_blob,
                     hist_first_hit_blob,
                     float(chan_means[0]) if chan_means[0] is not None else None,
                     float(chan_means[1]) if chan_means[1] is not None else None,
                     float(chan_means[2]) if chan_means[2] is not None else None,
                     float(chan_means[3]) if chan_means[3] is not None else None,
                     float(chan_stds[0]) if chan_stds[0] is not None else None,
                     float(chan_stds[1]) if chan_stds[1] is not None else None,
                     float(chan_stds[2]) if chan_stds[2] is not None else None,
                     float(chan_stds[3]) if chan_stds[3] is not None else None),
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


def main():
    """CLI entry point: render all unscored genomes.

    Can also run standalone for bulk rendering. The wallpaper startup
    now launches this as a background worker automatically.
    """
    import argparse

    parser = argparse.ArgumentParser(description='GPU genome renderer')
    parser.add_argument('--all', action='store_true',
                        help='Re-render all genomes')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger('PIL').setLevel(logging.WARNING)

    from .storage import _db_path
    stop = multiprocessing.Event()
    db = str(_db_path())

    if args.all:
        conn = sqlite3.connect(db)
        conn.execute('UPDATE genomes SET render_version = 0')
        conn.commit()
        conn.close()

    _render_main(db, stop)


if __name__ == '__main__':
    main()
