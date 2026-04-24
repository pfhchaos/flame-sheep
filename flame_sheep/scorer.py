"""Background symmetry scorer — load-aware CPU scoring thread.

Scores genomes' symmetry metrics using a pure CPU chaos game,
independent of the render pipeline. Produces clean, reproducible
scores from raw genome structure with no visual effects baked in.

Runs at lowest CPU priority (nice 19) and pauses when system
load is high (VM gaming, emerge builds, etc.).
"""

import logging
import os
import sqlite3
import threading

import numpy as np

log = logging.getLogger(__name__)


class BackgroundScorer:
    """Load-aware background thread that scores unscored genomes."""

    LOAD_THRESHOLD = 6.0      # pause when 1-min load avg exceeds this
    LOAD_CHECK_INTERVAL = 10.0  # seconds between load checks when busy
    IDLE_CHECK_INTERVAL = 30.0  # seconds between checks when all scored
    GRID_SIZE = 128           # histogram resolution for symmetry detection
    N_ITERATIONS = 50_000     # chaos game iterations per genome

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self):
        self._thread = threading.Thread(
            target=self._run, daemon=True, name='bg-scorer')
        self._thread.start()
        log.info('Background scorer started')

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run(self):
        try:
            os.nice(19)
        except OSError:
            pass  # nice may fail in some environments

        # Own DB connection (SQLite requires per-thread)
        conn = sqlite3.connect(self._db_path)

        try:
            while not self._stop.is_set():
                if self._system_busy():
                    self._stop.wait(self.LOAD_CHECK_INTERVAL)
                    continue

                row = conn.execute(
                    'SELECT id, params FROM genomes '
                    'WHERE symmetry_max IS NULL OR self_similarity IS NULL '
                    'LIMIT 1'
                ).fetchone()

                if row is None:
                    self._stop.wait(self.IDLE_CHECK_INTERVAL)
                    continue

                gid, params_json = row
                try:
                    scores = self._score_genome(params_json)
                    conn.execute(
                        '''UPDATE genomes
                           SET symmetry_max=?, rotational=?, reflective=?,
                               radial=?, periodic=?, fractal_dim=?,
                               self_similarity=?
                           WHERE id=?''',
                        (scores['symmetry_max'], scores['rotational'],
                         scores['reflective'], scores['radial'],
                         scores['periodic'], scores['fractal_dim'],
                         scores['self_similarity'], gid),
                    )
                    conn.commit()
                    log.debug(f'[scorer] genome #{gid}  '
                              f'sym={scores["symmetry_max"]:.3f}  '
                              f'self_sim={scores["self_similarity"]:.3f}  '
                              f'fdim={scores["fractal_dim"]:.3f}')
                except Exception:
                    log.exception(f'[scorer] failed to score genome #{gid}')
                    # Mark as scored with zeros to avoid retrying broken genomes
                    conn.execute(
                        '''UPDATE genomes
                           SET symmetry_max=0, rotational=0, reflective=0,
                               radial=0, periodic=0, fractal_dim=0,
                               self_similarity=0
                           WHERE id=?''',
                        (gid,),
                    )
                    conn.commit()
        finally:
            conn.close()

    def _system_busy(self) -> bool:
        load1 = os.getloadavg()[0]
        return load1 > self.LOAD_THRESHOLD

    def _score_genome(self, params_json: str) -> dict[str, float]:
        from .storage import _genome_from_json
        from .genome import _score_from_histogram, _score_symmetry

        genome = _genome_from_json(params_json)

        # Run chaos game — pure IFS, no morph/zoom/effects
        grid_size = self.GRID_SIZE
        fuse = 20
        bound = 4.0
        rng = np.random.default_rng()

        hit_grid = np.zeros((grid_size, grid_size), dtype=np.float64)
        color_grid = np.zeros((grid_size, grid_size), dtype=np.float64)

        weights = np.array([tr.weight for tr in genome.transforms],
                           dtype=np.float64)
        weights /= weights.sum()
        cumw = np.cumsum(weights)

        x, y, c = 0.0, 0.0, 0.5

        for i in range(fuse + self.N_ITERATIONS):
            r = rng.random()
            tidx = min(int(np.searchsorted(cumw, r)),
                       len(genome.transforms) - 1)
            tr = genome.transforms[tidx]
            a, b, cc, d, e, f = tr.affine
            nx = a * x + b * y + cc
            ny = d * x + e * y + f

            # Apply dominant variation (same as score_cpu)
            best_var = int(np.argmax(tr.variations))
            w = float(tr.variations[best_var])
            if w > 0.0:
                from .genome import _apply_variation_cpu
                nx, ny = _apply_variation_cpu(best_var, nx, ny, w)

            x, y = nx, ny
            c = (c + tr.color) * 0.5

            if not (np.isfinite(x) and np.isfinite(y)):
                break

            if i >= fuse and abs(x) < bound and abs(y) < bound:
                gx = int((x + bound) / (2 * bound) * grid_size)
                gy = int((y + bound) / (2 * bound) * grid_size)
                gx = max(0, min(grid_size - 1, gx))
                gy = max(0, min(grid_size - 1, gy))
                hit_grid[gy, gx] += 1.0
                color_grid[gy, gx] += c

        scores = _score_from_histogram(hit_grid, color_grid)
        scores.update(_score_symmetry(hit_grid))
        return scores
