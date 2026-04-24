"""Background symmetry scorer — load-aware CPU scoring process.

Scores genomes' symmetry metrics using a pure CPU chaos game,
independent of the render pipeline. Produces clean, reproducible
scores from raw genome structure with no visual effects baked in.

Runs as a separate process (not thread) to avoid GIL contention
with the render loop. Lowest CPU priority (nice 19), pauses when
system load is high (VM gaming, emerge builds, etc.).
"""

import logging
import multiprocessing
import os
import sqlite3

import numpy as np

log = logging.getLogger(__name__)


def _scorer_main(db_path: str, stop_event):
    """Entry point for the scorer subprocess."""
    # Reconfigure logging in the child process
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S')
    log = logging.getLogger('flame_sheep.scorer')

    try:
        os.nice(19)
    except OSError:
        pass

    conn = sqlite3.connect(db_path)

    try:
        while not stop_event.is_set():
            if os.getloadavg()[0] > BackgroundScorer.LOAD_THRESHOLD:
                stop_event.wait(BackgroundScorer.LOAD_CHECK_INTERVAL)
                continue

            row = conn.execute(
                'SELECT id, params FROM genomes '
                'WHERE symmetry_max IS NULL OR self_similarity IS NULL '
                'LIMIT 1'
            ).fetchone()

            if row is None:
                stop_event.wait(BackgroundScorer.IDLE_CHECK_INTERVAL)
                continue

            gid, params_json = row
            try:
                scores = _score_genome(params_json)
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
                log.debug(f'genome #{gid}  '
                          f'sym={scores["symmetry_max"]:.3f}  '
                          f'self_sim={scores["self_similarity"]:.3f}  '
                          f'fdim={scores["fractal_dim"]:.3f}')
            except Exception:
                log.exception(f'failed to score genome #{gid}')
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


def _score_genome(params_json: str) -> dict[str, float]:
    """Run CPU chaos game and compute all symmetry metrics."""
    from flame_sheep.storage import _genome_from_json
    from flame_sheep.genome import _score_from_histogram, _score_symmetry
    from flame_sheep.variations import apply_variations_cpu

    genome = _genome_from_json(params_json)

    grid_size = BackgroundScorer.GRID_SIZE
    n_iter = BackgroundScorer.N_ITERATIONS
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

    for i in range(fuse + n_iter):
        r = rng.random()
        tidx = min(int(np.searchsorted(cumw, r)),
                   len(genome.transforms) - 1)
        tr = genome.transforms[tidx]
        a, b, cc, d, e, f = tr.affine
        nx = a * x + b * y + cc
        ny = d * x + e * y + f

        nx, ny = apply_variations_cpu(tr.variations, nx, ny)

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


class BackgroundScorer:
    """Load-aware background process that scores unscored genomes."""

    LOAD_THRESHOLD = 6.0
    LOAD_CHECK_INTERVAL = 10.0
    IDLE_CHECK_INTERVAL = 30.0
    GRID_SIZE = 128
    N_ITERATIONS = 50_000

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._process: multiprocessing.Process | None = None
        self._stop = multiprocessing.Event()

    def start(self):
        self._stop.clear()
        self._process = multiprocessing.Process(
            target=_scorer_main,
            args=(self._db_path, self._stop),
            daemon=True, name='bg-scorer')
        self._process.start()
        log.info('Background scorer started (pid=%d)', self._process.pid)

    def stop(self):
        self._stop.set()
        if self._process is not None:
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.kill()
            self._process = None
