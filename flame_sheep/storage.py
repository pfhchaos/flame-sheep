"""
Persistent storage for genomes, loops, and ratings.

SQLite database at ~/.local/share/flame-sheep/library.db

Tables:
  genomes    — individual genomes with aesthetic scores and serialized parameters
  loops      — ordered sequences of genomes that form coherent visual cycles
  loop_items — join table: (loop_id, position, genome_id, motion_field)
  ratings    — user like/dislike history for genomes and loops

Motion fields:
  Each transition between adjacent genomes in a loop has a 3x3 vector field
  (18 floats: 9 cells × 2 components) describing the apparent visual motion.
  Stored on loop_items as a blob — characterizes the transition FROM this
  genome TO the next one in the loop.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from .genome import Genome, Transform, NUM_VARIATIONS

_DEFAULT_DIR = Path.home() / '.local' / 'share' / 'flame-sheep'
_DB_NAME = 'library.db'

# Motion field resolution: 3x3 grid of 2D vectors
MOTION_GRID = 3


def _db_path(data_dir: Path | None = None) -> Path:
    d = data_dir or _DEFAULT_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d / _DB_NAME


def _connect(data_dir: Path | None = None) -> sqlite3.Connection:
    path = _db_path(data_dir)
    conn = sqlite3.connect(str(path))
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA foreign_keys=ON')
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS genomes (
            id          INTEGER PRIMARY KEY,
            created_at  TEXT NOT NULL DEFAULT (datetime('now')),
            params      TEXT NOT NULL,          -- JSON: transforms, palette, zoom, rotation, center
            coverage      REAL,
            entropy       REAL,
            color_entropy REAL,
            balance       REAL,
            complexity    REAL,
            centroid_x    REAL,
            centroid_y    REAL,
            score_version INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS loops (
            id              INTEGER PRIMARY KEY,
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            name            TEXT,
            fitness         REAL DEFAULT 0.0,   -- composite fitness (auto + user)
            mean_coherence  REAL,               -- average motion coherence
            min_coherence   REAL,               -- worst transition coherence
            diversity       REAL,               -- visual diversity across the loop
            palette_flow    REAL,               -- smoothness of color transitions
            smoothness      REAL,               -- evenness of step distances (1=even, 0=snappy)
            parent_a        INTEGER REFERENCES loops(id),  -- breeding parent (nullable)
            parent_b        INTEGER REFERENCES loops(id)   -- breeding parent (nullable)
        );

        CREATE TABLE IF NOT EXISTS loop_items (
            loop_id     INTEGER NOT NULL REFERENCES loops(id) ON DELETE CASCADE,
            position    INTEGER NOT NULL,        -- 0-based order in the loop
            genome_id   INTEGER NOT NULL REFERENCES genomes(id),
            motion_field BLOB,                   -- 18 float32s: 3x3 grid of 2D vectors (to next)
            PRIMARY KEY (loop_id, position)
        );

        CREATE TABLE IF NOT EXISTS palettes (
            id          INTEGER PRIMARY KEY,
            created_at  TEXT NOT NULL DEFAULT (datetime('now')),
            data        BLOB NOT NULL,             -- 256*3 float32 = 3072 bytes
            mean_rgb    BLOB,                      -- 3 float32s, for quick distance estimates
            contrast    REAL,                      -- luminance range
            saturation  REAL,                      -- average saturation
            harmony     REAL,                      -- color relationship quality
            smoothness  REAL,                      -- gradient smoothness
            fitness     REAL                       -- composite palette quality
        );

        CREATE TABLE IF NOT EXISTS ratings (
            id          INTEGER PRIMARY KEY,
            created_at  TEXT NOT NULL DEFAULT (datetime('now')),
            target_type TEXT NOT NULL,            -- 'genome' or 'loop'
            target_id   INTEGER NOT NULL,
            rating      INTEGER NOT NULL          -- +1 = like, -1 = dislike
        );

        CREATE INDEX IF NOT EXISTS idx_loop_items_genome ON loop_items(genome_id);
        CREATE INDEX IF NOT EXISTS idx_ratings_target ON ratings(target_type, target_id);
    ''')

    # Add source column to ratings if it doesn't exist
    rating_cols = {r[1] for r in conn.execute('PRAGMA table_info(ratings)').fetchall()}
    if 'source' not in rating_cols:
        conn.execute("ALTER TABLE ratings ADD COLUMN source TEXT DEFAULT 'loop'")

    # Add symmetry columns if they don't exist (no migration system yet)
    existing = {r[1] for r in conn.execute('PRAGMA table_info(genomes)').fetchall()}
    for col in ('symmetry_max', 'rotational', 'reflective', 'radial', 'periodic',
                 'fractal_dim', 'self_similarity', 'detail_sensitivity',
                 'centroid_x', 'centroid_y',
                 'edge_sharpness', 'contour_coherence',
                 'img_coverage', 'img_structural_edges', 'img_euclidean_edges',
                 'img_color_edges', 'img_color_regions',
                 'img_color_coherence', 'img_color_variety',
                 'cl_coverage', 'cl_edge_sharpness', 'cl_symmetry_best',
                 'cl_cluster_count', 'cl_dominance', 'cl_balance',
                 'cluster_detail',
                 'tf_coverage', 'tf_n_clusters', 'tf_avg_purity',
                 'tf_symmetry_best', 'tf_balance', 'tf_separation',
                 # Experimental metrics (throw at wall)
                 'sw_rotational', 'sw_coverage',
                 'freq_high_ratio', 'freq_peak_scale',
                 'lacunarity',
                 'radial_slope', 'radial_r2',
                 'compactness',
                 'bbox_aspect',
                 'filament_count',
                 'contrast_ratio',
                 'angular_uniformity',
                 'cnn_score'):
        if col not in existing:
            conn.execute(f'ALTER TABLE genomes ADD COLUMN {col} REAL')
    for col in ('render_static', 'render_swept',
                'hist_static', 'hist_swept', 'hist_transform', 'hist_first_hit'):
        if col not in existing:
            conn.execute(f'ALTER TABLE genomes ADD COLUMN {col} BLOB')
    if 'cnn_weights_hash' not in existing:
        conn.execute('ALTER TABLE genomes ADD COLUMN cnn_weights_hash TEXT')
    for col in ('score_version', 'render_version'):
        if col not in existing:
            conn.execute(f'ALTER TABLE genomes ADD COLUMN {col} INTEGER DEFAULT 0')

    # Transition scoring columns
    if 'variation_signature' not in existing:
        conn.execute('ALTER TABLE genomes ADD COLUMN variation_signature TEXT')
    if 'n_transforms' not in existing:
        conn.execute('ALTER TABLE genomes ADD COLUMN n_transforms INTEGER')

    # Add loop_type column if it doesn't exist
    loop_cols = {r[1] for r in conn.execute('PRAGMA table_info(loops)').fetchall()}
    if 'loop_type' not in loop_cols:
        conn.execute("ALTER TABLE loops ADD COLUMN loop_type TEXT DEFAULT 'cyclic'")

    if 'framing_version' not in existing:
        conn.execute('ALTER TABLE genomes ADD COLUMN framing_version INTEGER DEFAULT 0')

    # Transition cache table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS genome_transitions (
            genome_a    INTEGER NOT NULL,
            genome_b    INTEGER NOT NULL,
            distance    REAL NOT NULL,
            PRIMARY KEY (genome_a, genome_b)
        )
    ''')
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_transitions_a
        ON genome_transitions(genome_a, distance)
    ''')

    # Pairwise comparison ratings
    conn.execute('''
        CREATE TABLE IF NOT EXISTS pairwise_ratings (
            id          INTEGER PRIMARY KEY,
            created_at  TEXT NOT NULL DEFAULT (datetime('now')),
            winner_id   INTEGER NOT NULL,
            loser_id    INTEGER NOT NULL,
            source      TEXT DEFAULT 'compare'
        )
    ''')

    conn.commit()

    # Variation reindex migration: move parameterized waves/popcorn/rings/fan
    # from indices 15/17/21/22 to 70/71/72/73.  The originals now read from
    # the affine at render time.
    _migrate_variation_reindex(conn)


def _migrate_variation_reindex(conn: sqlite3.Connection) -> None:
    """One-time migration: shift parameterized variation weights to new indices.

    Old layout: idx 15=waves(param), 17=popcorn(param), 21=rings(param), 22=fan(param)
    New layout: idx 15=waves(affine), 17=popcorn(affine), 21=rings(affine), 22=fan(affine)
                idx 70=waves_param, 71=popcorn_param, 72=rings_param, 73=fan_param
    """
    # Check if migration already done — look for a marker in the metadata
    existing_tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if 'metadata' not in existing_tables:
        conn.execute('CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)')
        conn.commit()

    row = conn.execute(
        "SELECT value FROM metadata WHERE key='variation_reindex_v1'").fetchone()
    if row is not None:
        return  # already migrated

    # Count genomes to migrate
    rows = conn.execute('SELECT id, params FROM genomes').fetchall()
    if not rows:
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES ('variation_reindex_v1', 'done')")
        conn.commit()
        return

    # Index mapping: old → new
    moves = {15: 70, 17: 71, 21: 72, 22: 73}
    migrated = 0

    for genome_id, params_json in rows:
        data = json.loads(params_json)
        changed = False
        for td in data['transforms']:
            variations = td['variations']
            # Pad to new size if needed
            while len(variations) < 75:
                variations.append(0.0)
            # Move weights from old indices to new
            for old_idx, new_idx in moves.items():
                if variations[old_idx] > 1e-6:
                    variations[new_idx] = variations[old_idx]
                    variations[old_idx] = 0.0
                    changed = True
            td['variations'] = variations
        if changed:
            migrated += 1
            new_json = json.dumps(data, separators=(',', ':'))
            conn.execute('UPDATE genomes SET params=?, score_version=0 WHERE id=?',
                         (new_json, genome_id))

    conn.execute(
        "INSERT INTO metadata (key, value) VALUES ('variation_reindex_v1', 'done')")
    conn.commit()

    if migrated:
        import sys
        print(f'[storage] Migrated {migrated}/{len(rows)} genomes '
              f'(variation reindex v1)', file=sys.stderr)


# ------------------------------------------------------------------
# Genome serialization
# ------------------------------------------------------------------

def _transform_to_dict(tr: Transform) -> dict:
    """Serialize a Transform to a JSON-compatible dict."""
    d = {
        'affine': tr.affine.tolist(),
        'variations': tr.variations.tolist(),
        'color': tr.color,
        'weight': tr.weight,
        'var_params': tr.var_params,
    }
    if tr.post_affine is not None:
        d['post_affine'] = tr.post_affine.tolist()
    if tr.pre_variations is not None:
        d['pre_variations'] = tr.pre_variations.tolist()
    return d


def _transform_from_dict(td: dict) -> Transform:
    """Deserialize a Transform from a JSON dict."""
    tr = Transform()
    tr.affine = np.array(td['affine'], dtype=np.float32)
    # Backwards compat: old genomes have shorter variation arrays
    raw_vars = np.array(td['variations'], dtype=np.float32)
    if len(raw_vars) < NUM_VARIATIONS:
        tr.variations = np.zeros(NUM_VARIATIONS, dtype=np.float32)
        tr.variations[:len(raw_vars)] = raw_vars
    else:
        tr.variations = raw_vars
    tr.color = td['color']
    tr.weight = td['weight']
    tr.var_params = td.get('var_params', {})
    if 'post_affine' in td:
        tr.post_affine = np.array(td['post_affine'], dtype=np.float32)
    if 'pre_variations' in td:
        raw_pre = np.array(td['pre_variations'], dtype=np.float32)
        if len(raw_pre) < NUM_VARIATIONS:
            tr.pre_variations = np.zeros(NUM_VARIATIONS, dtype=np.float32)
            tr.pre_variations[:len(raw_pre)] = raw_pre
        else:
            tr.pre_variations = raw_pre
    return tr


def _genome_to_json(g: Genome) -> str:
    """Serialize a Genome to a JSON string."""
    data = {
        'transforms': [_transform_to_dict(tr) for tr in g.transforms],
        'palette': g.palette.tolist(),
        'zoom': g.zoom,
        'rotation': g.rotation,
        'center': g.center.tolist(),
    }
    if g.final_xform is not None:
        data['final_xform'] = _transform_to_dict(g.final_xform)
    return json.dumps(data, separators=(',', ':'))


def _genome_from_json(s: str) -> Genome:
    """Deserialize a Genome from a JSON string."""
    data = json.loads(s)
    g = Genome()
    g.transforms = [_transform_from_dict(td) for td in data['transforms']]
    if 'final_xform' in data:
        g.final_xform = _transform_from_dict(data['final_xform'])
    g.palette = np.array(data['palette'], dtype=np.float32)
    g.zoom = data['zoom']
    g.rotation = data['rotation']
    g.center = np.array(data['center'], dtype=np.float32)
    return g


# ------------------------------------------------------------------
# Motion field computation
# ------------------------------------------------------------------

def compute_motion_field(genome_a: Genome, genome_b: Genome,
                         n_test: int = 5000, fuse: int = 20,
                         bound: float = 4.0) -> np.ndarray:
    """
    Compute a 3x3 grid of 2D motion vectors characterizing the visual
    transition from genome_a to genome_b.

    For each cell in the 3x3 grid, runs a chaos game for both genomes,
    computes the density centroid of points landing in that cell, and
    returns the displacement vector (centroid_b - centroid_a).

    Returns shape (3, 3, 2) float32.
    """
    grid = MOTION_GRID
    centroids_a = _grid_centroids(genome_a, grid, n_test, fuse, bound)
    centroids_b = _grid_centroids(genome_b, grid, n_test, fuse, bound)
    return (centroids_b - centroids_a).astype(np.float32)


def _grid_centroids(genome: Genome, grid: int, n_test: int,
                    fuse: int, bound: float) -> np.ndarray:
    """
    Run chaos game and compute density-weighted centroid for each cell
    in a grid×grid partition of the viewport.

    Returns shape (grid, grid, 2) — centroid (x, y) per cell.
    """
    from .genome import _apply_variation_cpu

    rng = np.random.default_rng()
    weights = np.array([tr.weight for tr in genome.transforms], dtype=np.float64)
    weights /= weights.sum()
    cumw = np.cumsum(weights)

    # Accumulators: sum of positions and count per cell
    cell_sum = np.zeros((grid, grid, 2), dtype=np.float64)
    cell_count = np.zeros((grid, grid), dtype=np.float64)

    x, y = 0.0, 0.0
    cell_size = (2 * bound) / grid

    for i in range(fuse + n_test):
        r = rng.random()
        tidx = min(int(np.searchsorted(cumw, r)), len(genome.transforms) - 1)
        tr = genome.transforms[tidx]
        a, b, c, d, e, f = tr.affine
        nx = a * x + b * y + c
        ny = d * x + e * y + f

        best_var = int(np.argmax(tr.variations))
        w = float(tr.variations[best_var])
        if w > 0.0:
            nx, ny = _apply_variation_cpu(best_var, nx, ny, w, tr.affine)

        x, y = nx, ny

        if not (np.isfinite(x) and np.isfinite(y)):
            x, y = 0.0, 0.0
            continue

        if i >= fuse and abs(x) < bound and abs(y) < bound:
            gx = min(int((x + bound) / cell_size), grid - 1)
            gy = min(int((y + bound) / cell_size), grid - 1)
            cell_sum[gy, gx, 0] += x
            cell_sum[gy, gx, 1] += y
            cell_count[gy, gx] += 1

    # Compute centroids; cells with no hits get center of cell
    centroids = np.zeros((grid, grid, 2), dtype=np.float64)
    for gy in range(grid):
        for gx in range(grid):
            if cell_count[gy, gx] > 0:
                centroids[gy, gx] = cell_sum[gy, gx] / cell_count[gy, gx]
            else:
                centroids[gy, gx, 0] = -bound + (gx + 0.5) * cell_size
                centroids[gy, gx, 1] = -bound + (gy + 0.5) * cell_size
    return centroids


def motion_field_coherence(field_a: np.ndarray, field_b: np.ndarray) -> float:
    """
    How well two motion fields flow in the same direction.

    Returns a value in [-1, 1]:
      +1 = identical direction everywhere
       0 = unrelated
      -1 = exactly opposite

    Uses cosine similarity averaged over the 3x3 grid, weighted by
    magnitude (cells with more motion matter more).
    """
    a = field_a.reshape(-1, 2)
    b = field_b.reshape(-1, 2)
    mag_a = np.linalg.norm(a, axis=1)
    mag_b = np.linalg.norm(b, axis=1)
    weights = mag_a * mag_b
    total_weight = weights.sum()
    if total_weight < 1e-10:
        return 0.0
    dots = np.sum(a * b, axis=1)
    return float(np.sum(dots) / (total_weight + 1e-10) * (weights.sum() / (weights.sum() + 1e-10)))


# ------------------------------------------------------------------
# Palette fitness
# ------------------------------------------------------------------

def score_palette(palette: np.ndarray) -> dict[str, float]:
    """
    Compute quality metrics for a palette.

    palette: shape (256, 3) float32, RGB values in [0, 1].

    Returns dict with keys:
      contrast   -- luminance range (0=flat gray, 1=full black-to-white)
      saturation -- average color saturation (0=grayscale, 1=vivid)
      harmony    -- how well colors form coherent groups (0=noise, 1=clean scheme)
      smoothness -- gradient continuity (0=banded, 1=smooth transitions)
      fitness    -- weighted composite
    """
    p = palette.astype(np.float64)

    # -- contrast: range of perceived luminance --
    luminance = 0.299 * p[:, 0] + 0.587 * p[:, 1] + 0.114 * p[:, 2]
    contrast = float(luminance.max() - luminance.min())

    # -- saturation: average HSV saturation --
    maxc = p.max(axis=1)
    minc = p.min(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        sat = np.where(maxc > 1e-6, (maxc - minc) / maxc, 0.0)
    saturation = float(sat.mean())

    # -- harmony: cluster colors with simple histogram, reward 2-5 dominant hues --
    # Convert to hue (0-360)
    delta = maxc - minc
    hue = np.zeros(256)
    mask = delta > 1e-6
    r, g, b = p[:, 0], p[:, 1], p[:, 2]
    # Red is max
    m = mask & (maxc == r)
    hue[m] = (60 * ((g[m] - b[m]) / delta[m]) % 360)
    # Green is max
    m = mask & (maxc == g)
    hue[m] = 60 * ((b[m] - r[m]) / delta[m]) + 120
    # Blue is max
    m = mask & (maxc == b)
    hue[m] = 60 * ((r[m] - g[m]) / delta[m]) + 240
    hue = hue % 360

    # Hue histogram — count dominant hue groups
    hue_bins = np.histogram(hue[mask], bins=12, range=(0, 360))[0]
    if hue_bins.sum() > 0:
        hue_probs = hue_bins / hue_bins.sum()
        n_dominant = int(np.sum(hue_probs > 0.1))  # bins with >10% of mass
        # 2-5 dominant hues is ideal
        if n_dominant < 2:
            harmony = 0.3  # too monotone
        elif n_dominant <= 5:
            harmony = 1.0 - abs(n_dominant - 3) * 0.15  # peak at 3
        else:
            harmony = max(0.0, 1.0 - (n_dominant - 5) * 0.2)  # too many
    else:
        harmony = 0.3  # all achromatic

    # -- smoothness: average step size between adjacent entries --
    diffs = np.sqrt(np.sum(np.diff(p, axis=0) ** 2, axis=1))
    if len(diffs) > 0:
        mean_diff = diffs.mean()
        if mean_diff > 1e-6:
            # Low coefficient of variation = smooth, high = banded
            cv = float(diffs.std() / mean_diff)
            smoothness = max(0.0, 1.0 - cv / 2.0)
        else:
            smoothness = 1.0  # all same color, technically smooth
    else:
        smoothness = 0.0

    fitness = (
        contrast * 0.25
        + saturation * 0.25
        + harmony * 0.25
        + smoothness * 0.25
    )

    return dict(
        contrast=contrast,
        saturation=saturation,
        harmony=harmony,
        smoothness=smoothness,
        fitness=fitness,
    )


def motion_field_to_blob(field: np.ndarray) -> bytes:
    return field.astype(np.float32).tobytes()


def motion_field_from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).reshape(MOTION_GRID, MOTION_GRID, 2)


# ------------------------------------------------------------------
# Loop-level fitness
# ------------------------------------------------------------------

def score_loop(genomes: list[Genome],
               motion_fields: list[np.ndarray],
               structure: str = 'cyclic') -> dict[str, float]:
    """
    Compute automated fitness metrics for a loop.

    Parameters
    ----------
    genomes : list[Genome]
        The genomes in loop order.
    motion_fields : list[np.ndarray]
        Motion field for each transition (len == len(genomes), wrapping).
    structure : str
        'cyclic', 'palindrome', or 'rondo'.

    Returns
    -------
    dict with keys:
      mean_coherence -- average motion coherence between consecutive transitions
      min_coherence  -- worst-case coherence (one bad jerk tanks this)
      diversity      -- visual variety across genomes in the loop
      palette_flow   -- smoothness of color transitions
      fitness        -- weighted composite of the above
    """
    n = len(genomes)

    # -- motion coherence --
    coherences = []
    for i in range(len(motion_fields)):
        mf_a = motion_fields[i]
        mf_b = motion_fields[(i + 1) % len(motion_fields)]
        coherences.append(motion_field_coherence(mf_a, mf_b))
    mean_coh = float(np.mean(coherences)) if coherences else 0.0
    min_coh = float(np.min(coherences)) if coherences else 0.0

    # -- diversity: average pairwise distance between genomes --
    # High = the loop covers interesting visual ground
    # Low = all genomes look similar (boring)
    if n >= 2:
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(genomes[i].distance(genomes[j]))
        diversity = float(np.mean(dists))
    else:
        diversity = 0.0

    # -- palette flow: how smoothly colors transition around the loop --
    # Compare average palette color between consecutive genomes.
    # Small steps = smooth flow, big jumps = jarring.
    palette_diffs = []
    for i in range(n):
        p_a = genomes[i].palette.mean(axis=0)       # average RGB
        p_b = genomes[(i + 1) % n].palette.mean(axis=0)
        palette_diffs.append(float(np.linalg.norm(p_a - p_b)))
    if palette_diffs:
        # Low variance in step size = smooth; normalize by mean to get consistency
        mean_diff = np.mean(palette_diffs)
        if mean_diff > 1e-6:
            # Consistency: 1 = perfectly even steps, 0 = erratic
            consistency = 1.0 - float(np.std(palette_diffs) / mean_diff)
            consistency = max(0.0, consistency)
            # Moderate step size is best — too small = no change, too big = jarring
            # Peak at ~0.3 palette distance per step
            step_quality = 1.0 - abs(mean_diff - 0.3) / 0.3
            step_quality = max(0.0, min(1.0, step_quality))
            palette_flow = (consistency + step_quality) / 2.0
        else:
            palette_flow = 0.0  # no color change at all
    else:
        palette_flow = 0.0

    # -- smoothness: evenness of consecutive genome distances --
    # Penalizes loops where one transition is a huge jump (the snap problem).
    # Measures coefficient of variation of step distances — 0 = all equal, high = uneven.
    step_dists = []
    for i in range(n):
        step_dists.append(genomes[i].distance(genomes[(i + 1) % n]))
    if step_dists:
        mean_step = np.mean(step_dists)
        max_step = max(step_dists)
        if mean_step > 1e-6:
            # Ratio of worst step to mean — 1.0 = perfectly even, 0 = one step dominates
            smoothness = 1.0 - (max_step - mean_step) / max_step
            smoothness = max(0.0, smoothness)
        else:
            smoothness = 1.0
    else:
        smoothness = 0.0

    # -- composite fitness (structure-dependent weights) --
    if structure == 'palindrome':
        # Palindrome: wrap-around less important (reversal is smooth),
        # bidirectional coherence matters, smoothness still king
        fitness = (
            smoothness * 0.30
            + mean_coh * 0.30   # higher — coherence in both directions
            + diversity * 0.20
            + palette_flow * 0.15
            + min_coh * 0.05    # lower — wrap transition less critical
        )
    elif structure == 'rondo':
        # Rondo: home genome quality matters, diversity of episodes matters
        # min_coherence less important (transitions to/from home vary)
        fitness = (
            smoothness * 0.25
            + mean_coh * 0.20
            + diversity * 0.30   # higher — episodes should be distinct
            + palette_flow * 0.15
            + min_coh * 0.10
        )
    else:  # cyclic
        fitness = (
            smoothness * 0.30
            + mean_coh * 0.25
            + diversity * 0.20
            + palette_flow * 0.15
            + min_coh * 0.10
        )

    return dict(
        mean_coherence=mean_coh,
        min_coherence=min_coh,
        diversity=diversity,
        palette_flow=palette_flow,
        smoothness=smoothness,
        fitness=fitness,
    )


# ------------------------------------------------------------------
# Database operations
# ------------------------------------------------------------------

class Library:
    """Interface to the genome/loop database."""

    def __init__(self, data_dir: Path | None = None):
        self.conn = _connect(data_dir)

    def close(self) -> None:
        self.conn.close()

    # -- Genomes --

    def save_genome(self, genome: Genome, scores: dict[str, float] | None = None) -> int:
        """Store a genome, return its ID."""
        from .transition import variation_signature

        params = _genome_to_json(genome)
        if scores is None:
            scores = genome.aesthetic_score()
        sig = variation_signature(genome)
        n_xforms = len(genome.transforms)
        cur = self.conn.execute(
            '''INSERT INTO genomes (params, coverage, entropy, color_entropy, balance, complexity,
                                    edge_sharpness, contour_coherence,
                                    symmetry_max, rotational, reflective, radial, periodic, fractal_dim,
                                    centroid_x, centroid_y, score_version,
                                    variation_signature, n_transforms, framing_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (params, scores['coverage'], scores['entropy'],
             scores['color_entropy'], scores['balance'], scores['complexity'],
             scores.get('edge_sharpness', 0.0), scores.get('contour_coherence', 0.0),
             scores.get('symmetry_max'), scores.get('rotational'),
             scores.get('reflective'), scores.get('radial'),
             scores.get('periodic'), scores.get('fractal_dim'),
             scores.get('centroid_offset_x'), scores.get('centroid_offset_y'),
             1, sig, n_xforms, 1),
        )
        self.conn.commit()
        return cur.lastrowid

    def load_genome(self, genome_id: int) -> Genome:
        """Load a genome by ID.

        Center and zoom are baked into params from survey_and_correct()
        at creation time.
        """
        row = self.conn.execute(
            'SELECT params FROM genomes WHERE id = ?',
            (genome_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f'No genome with id {genome_id}')
        genome = _genome_from_json(row[0])
        genome.db_id = genome_id
        return genome

    def genome_scores(self, genome_id: int) -> dict[str, float]:
        """Get stored aesthetic scores for a genome."""
        row = self.conn.execute(
            'SELECT coverage, entropy, color_entropy, balance, complexity,'
            ' edge_sharpness, contour_coherence, centroid_x, centroid_y'
            ' FROM genomes WHERE id = ?',
            (genome_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f'No genome with id {genome_id}')
        return dict(coverage=row[0], entropy=row[1], color_entropy=row[2],
                    balance=row[3], complexity=row[4],
                    edge_sharpness=row[5] or 0.0, contour_coherence=row[6] or 0.0,
                    centroid_offset_x=row[7], centroid_offset_y=row[8])

    def top_genomes(self, n: int = 20, min_coverage: float = 0.01) -> list[tuple[int, dict]]:
        """Return top N genomes ranked by CNN score (preferred) or composite."""
        rows = self.conn.execute(
            '''SELECT id, coverage, entropy, color_entropy, balance, complexity,
                      cnn_score
               FROM genomes
               WHERE coverage >= ?
               ORDER BY COALESCE(cnn_score, -999) DESC,
                        (entropy + color_entropy + balance * 0.5 + complexity) DESC
               LIMIT ?''',
            (min_coverage, n),
        ).fetchall()
        return [
            (r[0], dict(coverage=r[1], entropy=r[2], color_entropy=r[3],
                        balance=r[4], complexity=r[5], cnn_score=r[6]))
            for r in rows
        ]

    def genome_count(self) -> int:
        return self.conn.execute('SELECT COUNT(*) FROM genomes').fetchone()[0]

    def genome_loop_counts(self) -> dict[int, int]:
        """Count how many loops each genome appears in."""
        rows = self.conn.execute(
            'SELECT genome_id, COUNT(DISTINCT loop_id) FROM loop_items GROUP BY genome_id'
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def genome_transition_counts(self) -> dict[int, int]:
        """Count cached transition neighbors per genome (excluding self-edges)."""
        rows = self.conn.execute(
            '''SELECT genome_a, COUNT(*) FROM genome_transitions
               WHERE genome_a != genome_b
               GROUP BY genome_a'''
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    # -- Transitions --

    def store_transition(self, genome_a: int, genome_b: int, distance: float) -> None:
        """Cache a pairwise transition distance (both directions).
        Skips zero-distance self-transitions — no useful information."""
        if distance <= 0 or genome_a == genome_b:
            return
        self.conn.execute(
            'INSERT OR REPLACE INTO genome_transitions (genome_a, genome_b, distance) VALUES (?, ?, ?)',
            (genome_a, genome_b, distance),
        )
        self.conn.execute(
            'INSERT OR REPLACE INTO genome_transitions (genome_a, genome_b, distance) VALUES (?, ?, ?)',
            (genome_b, genome_a, distance),
        )

    def nearest_transitions(self, genome_id: int, n: int = 10) -> list[tuple[int, float]]:
        """Return the N nearest genomes by transition distance."""
        rows = self.conn.execute(
            '''SELECT genome_b, distance FROM genome_transitions
               WHERE genome_a = ?
               ORDER BY distance ASC
               LIMIT ?''',
            (genome_id, n),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    # -- Loops --

    def save_loop(self, genome_ids: list[int],
                  motion_fields: list[np.ndarray] | None = None,
                  name: str | None = None,
                  loop_type: str = 'cyclic',
                  parent_a: int | None = None,
                  parent_b: int | None = None) -> int:
        """
        Store a loop as an ordered sequence of genome IDs.

        motion_fields: list of 3x3x2 arrays, one per transition.
                       Length should equal len(genome_ids) (last wraps to first).
                       If None, motion fields are computed automatically.
        loop_type: 'cyclic', 'palindrome', or 'rondo'.
        parent_a/b: loop IDs of breeding parents (for tracking lineage).
        """
        genomes = [self.load_genome(gid) for gid in genome_ids]

        if motion_fields is None:
            motion_fields = []
            for i in range(len(genome_ids)):
                motion_fields.append(compute_motion_field(
                    genomes[i], genomes[(i + 1) % len(genomes)]
                ))

        scores = score_loop(genomes, motion_fields, structure=loop_type)

        cur = self.conn.execute(
            '''INSERT INTO loops (name, fitness, mean_coherence, min_coherence,
                                  diversity, palette_flow, smoothness, loop_type,
                                  parent_a, parent_b)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (name, scores['fitness'], scores['mean_coherence'],
             scores['min_coherence'], scores['diversity'],
             scores['palette_flow'], scores['smoothness'], loop_type,
             parent_a, parent_b),
        )
        loop_id = cur.lastrowid

        for i, gid in enumerate(genome_ids):
            mf = motion_fields[i] if i < len(motion_fields) else None
            blob = motion_field_to_blob(mf) if mf is not None else None
            self.conn.execute(
                'INSERT INTO loop_items (loop_id, position, genome_id, motion_field) VALUES (?, ?, ?, ?)',
                (loop_id, i, gid, blob),
            )
        self.conn.commit()
        return loop_id

    def loop_type(self, loop_id: int) -> str:
        """Get the structure type of a loop ('cyclic', 'palindrome', 'rondo')."""
        row = self.conn.execute(
            'SELECT loop_type FROM loops WHERE id = ?', (loop_id,)
        ).fetchone()
        return row[0] if row and row[0] else 'cyclic'

    def load_loop(self, loop_id: int) -> list[tuple[int, Genome, np.ndarray | None]]:
        """
        Load a loop: returns [(genome_id, Genome, motion_field), ...] in order.
        """
        rows = self.conn.execute(
            '''SELECT li.genome_id, li.motion_field
               FROM loop_items li
               WHERE li.loop_id = ?
               ORDER BY li.position''',
            (loop_id,),
        ).fetchall()
        if not rows:
            raise KeyError(f'No loop with id {loop_id}')
        result = []
        for gid, mf_blob in rows:
            genome = self.load_genome(gid)
            mf = motion_field_from_blob(mf_blob) if mf_blob else None
            result.append((gid, genome, mf))
        return result

    def loop_count(self) -> int:
        return self.conn.execute('SELECT COUNT(*) FROM loops').fetchone()[0]

    def loop_fitness(self, loop_id: int) -> dict[str, float]:
        """Get stored fitness scores for a loop."""
        row = self.conn.execute(
            '''SELECT fitness, mean_coherence, min_coherence, diversity, palette_flow, smoothness
               FROM loops WHERE id = ?''',
            (loop_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f'No loop with id {loop_id}')
        return dict(fitness=row[0], mean_coherence=row[1], min_coherence=row[2],
                    diversity=row[3], palette_flow=row[4], smoothness=row[5])

    def update_loop_fitness(self, loop_id: int) -> None:
        """Recompute and store loop fitness from coherence + genome quality + votes.

        Components:
          - auto: mean_coherence + diversity + palette_flow + smoothness
          - genome_quality: mean genome fitness across loop genomes
          - transition_quality: smoothness of consecutive genome transitions
          - user_bonus: clamped loop vote (-1/0/+1) * 0.5
        """
        user_bonus = self.net_rating('loop', loop_id) * 0.5
        row = self.conn.execute(
            'SELECT mean_coherence, diversity, palette_flow, smoothness FROM loops WHERE id = ?',
            (loop_id,),
        ).fetchone()
        if row is None:
            return
        auto = (row[0] or 0) + (row[1] or 0) + (row[2] or 0) + (row[3] or 0)

        # Mean genome fitness across loop
        genome_ids = self.loop_genome_ids(loop_id)
        if genome_ids:
            genome_scores = [self.genome_fitness(gid) for gid in genome_ids]
            genome_quality = sum(genome_scores) / len(genome_scores)
        else:
            genome_quality = 0.0

        # Mean transition quality between consecutive genomes in loop
        transition_quality = 0.0
        if len(genome_ids) >= 2:
            dists = []
            for i in range(len(genome_ids)):
                ga = genome_ids[i]
                gb = genome_ids[(i + 1) % len(genome_ids)]
                row_t = self.conn.execute(
                    'SELECT distance FROM genome_transitions WHERE genome_a=? AND genome_b=?',
                    (ga, gb),
                ).fetchone()
                if row_t is not None:
                    dists.append(row_t[0])
            if dists:
                mean_dist = sum(dists) / len(dists)
                transition_quality = 1.0 / (1.0 + mean_dist)

        fitness = auto + genome_quality * 0.3 + transition_quality * 0.2 + user_bonus
        self.conn.execute(
            'UPDATE loops SET fitness = ? WHERE id = ?',
            (fitness, loop_id),
        )
        self.conn.commit()

    def top_loops(self, n: int = 10) -> list[tuple[int, dict]]:
        """Return top N loops ranked by fitness."""
        rows = self.conn.execute(
            '''SELECT id, fitness, mean_coherence, min_coherence, diversity, palette_flow, smoothness
               FROM loops
               ORDER BY fitness DESC
               LIMIT ?''',
            (n,),
        ).fetchall()
        return [
            (r[0], dict(fitness=r[1], mean_coherence=r[2], min_coherence=r[3],
                        diversity=r[4], palette_flow=r[5], smoothness=r[6]))
            for r in rows
        ]

    def loop_genome_ids(self, loop_id: int) -> list[int]:
        """Get just the genome IDs for a loop, in order."""
        rows = self.conn.execute(
            'SELECT genome_id FROM loop_items WHERE loop_id = ? ORDER BY position',
            (loop_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def all_genome_ids_in_loops(self) -> list[int]:
        """Return all unique genome IDs that appear in at least one loop."""
        rows = self.conn.execute(
            'SELECT DISTINCT genome_id FROM loop_items'
        ).fetchall()
        return [r[0] for r in rows]

    def nearest_loop_transitions(self, genome_id: int,
                                  exclude_loop: int | None = None,
                                  exclude_loops: set[int] | None = None,
                                  n: int = 3) -> list[tuple[int, float, float, int]]:
        """Find nearest loops via the transition graph in a single query.

        Returns [(loop_id, fitness, transition_distance, entry_genome_id)]
        sorted by distance, deduplicated to one entry per loop.
        """
        exclude = set(exclude_loops or ())
        if exclude_loop is not None:
            exclude.add(exclude_loop)

        # Join transitions → loop_items → loops to find reachable loops.
        rows = self.conn.execute(
            '''SELECT li.loop_id, l.fitness, gt.distance, gt.genome_b
               FROM genome_transitions gt
               JOIN loop_items li ON li.genome_id = gt.genome_b
               JOIN loops l ON l.id = li.loop_id
               WHERE gt.genome_a = ?
               ORDER BY gt.distance ASC''',
            (genome_id,),
        ).fetchall()

        # Deduplicate: keep closest entry per loop, skip excluded
        seen = set()
        result = []
        for lid, fitness, dist, entry_gid in rows:
            if lid in exclude or lid in seen:
                continue
            seen.add(lid)
            result.append((lid, fitness or 0.0, dist, entry_gid))
            if len(result) >= n:
                break

        return result

    def loops_containing(self, genome_id: int) -> list[tuple[int, float]]:
        """Find loops that contain a genome. Returns [(loop_id, fitness)]."""
        rows = self.conn.execute(
            '''SELECT DISTINCT li.loop_id, l.fitness
               FROM loop_items li
               JOIN loops l ON l.id = li.loop_id
               WHERE li.genome_id = ?
               ORDER BY l.fitness DESC''',
            (genome_id,),
        ).fetchall()
        return [(r[0], r[1] or 0.0) for r in rows]

    def delete_loop(self, loop_id: int) -> None:
        """Delete a loop and its items (CASCADE)."""
        self.conn.execute('DELETE FROM loops WHERE id = ?', (loop_id,))
        self.conn.commit()

    def delete_loops(self, loop_ids: list[int]) -> int:
        """Delete multiple loops. Returns count deleted."""
        if not loop_ids:
            return 0
        placeholders = ','.join('?' * len(loop_ids))
        # Clear parent references that point to loops being deleted
        self.conn.execute(
            f'UPDATE loops SET parent_a = NULL WHERE parent_a IN ({placeholders})',
            loop_ids)
        self.conn.execute(
            f'UPDATE loops SET parent_b = NULL WHERE parent_b IN ({placeholders})',
            loop_ids)
        cur = self.conn.execute(
            f'DELETE FROM loops WHERE id IN ({placeholders})', loop_ids)
        self.conn.commit()
        return cur.rowcount

    # -- Palettes (graph nodes) --

    def save_palette(self, palette: np.ndarray) -> int:
        """Store a palette with fitness scores, return its ID. palette: shape (256, 3) float32."""
        data = palette.astype(np.float32).tobytes()
        mean_rgb = palette.mean(axis=0).astype(np.float32).tobytes()
        scores = score_palette(palette)
        cur = self.conn.execute(
            '''INSERT INTO palettes (data, mean_rgb, contrast, saturation, harmony, smoothness, fitness)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (data, mean_rgb, scores['contrast'], scores['saturation'],
             scores['harmony'], scores['smoothness'], scores['fitness']),
        )
        self.conn.commit()
        return cur.lastrowid

    def load_palette(self, palette_id: int) -> np.ndarray:
        """Load a palette by ID. Returns shape (256, 3) float32."""
        row = self.conn.execute(
            'SELECT data FROM palettes WHERE id = ?', (palette_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f'No palette with id {palette_id}')
        return np.frombuffer(row[0], dtype=np.float32).reshape(256, 3).copy()

    def palette_count(self) -> int:
        return self.conn.execute('SELECT COUNT(*) FROM palettes').fetchone()[0]

    def all_palette_ids(self) -> list[int]:
        rows = self.conn.execute('SELECT id FROM palettes').fetchall()
        return [r[0] for r in rows]

    def palette_distance(self, id_a: int, id_b: int) -> float:
        """Perceptual distance between two palettes. Cheap — uses mean RGB."""
        rows = self.conn.execute(
            'SELECT id, mean_rgb FROM palettes WHERE id IN (?, ?)',
            (id_a, id_b),
        ).fetchall()
        if len(rows) != 2:
            raise KeyError(f'Palette {id_a} or {id_b} not found')
        by_id = {r[0]: np.frombuffer(r[1], dtype=np.float32) for r in rows}
        return float(np.linalg.norm(by_id[id_a] - by_id[id_b]))

    def palette_neighbors(self, palette_id: int, min_dist: float = 0.05,
                          max_dist: float = 0.4,
                          min_fitness: float = 0.0) -> list[tuple[int, float]]:
        """
        Find palettes within a distance range, optionally filtered by fitness.
        Returns [(id, distance), ...] sorted by distance.
        """
        source = self.conn.execute(
            'SELECT mean_rgb FROM palettes WHERE id = ?', (palette_id,)
        ).fetchone()
        if source is None:
            raise KeyError(f'No palette with id {palette_id}')
        source_rgb = np.frombuffer(source[0], dtype=np.float32)

        rows = self.conn.execute(
            'SELECT id, mean_rgb FROM palettes WHERE id != ? AND COALESCE(fitness, 0) >= ?',
            (palette_id, min_fitness),
        ).fetchall()
        neighbors = []
        for pid, mean_blob in rows:
            other_rgb = np.frombuffer(mean_blob, dtype=np.float32)
            d = float(np.linalg.norm(source_rgb - other_rgb))
            if min_dist <= d <= max_dist:
                neighbors.append((pid, d))
        neighbors.sort(key=lambda x: x[1])
        return neighbors

    # -- Ratings --

    def rate(self, target_type: str, target_id: int, rating: int) -> None:
        """Record a like (+1) or dislike (-1).

        Loop votes are clamped to -1/0/+1 (replaces previous vote).
        Voting on a loop also propagates the vote to all its genomes.
        """
        if target_type == 'loop':
            # Clamp: replace any existing vote on this loop
            self.conn.execute(
                'DELETE FROM ratings WHERE target_type = ? AND target_id = ?',
                (target_type, target_id),
            )
        self.conn.execute(
            'INSERT INTO ratings (target_type, target_id, rating, source) VALUES (?, ?, ?, ?)',
            (target_type, target_id, rating, 'direct'),
        )
        # Propagate loop votes to constituent genomes
        if target_type == 'loop':
            genome_ids = self.loop_genome_ids(target_id)
            for gid in genome_ids:
                self.conn.execute(
                    'INSERT INTO ratings (target_type, target_id, rating, source) VALUES (?, ?, ?, ?)',
                    ('genome', gid, rating, 'loop'),
                )
        self.conn.commit()

    def net_rating(self, target_type: str, target_id: int) -> int:
        """Sum of ratings for a target."""
        row = self.conn.execute(
            'SELECT COALESCE(SUM(rating), 0) FROM ratings WHERE target_type = ? AND target_id = ?',
            (target_type, target_id),
        ).fetchone()
        return row[0]

    def genome_fitness(self, genome_id: int) -> float:
        """Compute genome fitness from aesthetic scores + symmetry + user votes.

        Uses CNN score as primary signal when available (trained on 13 years
        of Electric Sheep crowd ratings). Falls back to cluster-based or
        histogram-based heuristics for unscored genomes.
        """
        row = self.conn.execute(
            '''SELECT coverage, entropy, color_entropy, balance, complexity,
                      edge_sharpness, contour_coherence,
                      symmetry_max, fractal_dim, cnn_score
               FROM genomes WHERE id = ?''',
            (genome_id,),
        ).fetchone()
        if row is None:
            return 0.0

        coverage = row[0] or 0
        color_entropy = row[2] or 0
        fractal_dim = row[8] or 1.0
        cnn_score = row[9]

        if cnn_score is not None:
            # CNN score is the primary aesthetic signal.
            # Raw scores are unbounded; typical range roughly -2 to +5.
            aesthetic = cnn_score
        else:
            # Fallback: hand-crafted heuristics for genomes not yet CNN-scored.
            # Coverage sweet spot: 0.15-0.40 is ideal
            if coverage < 0.05:
                coverage_score = coverage * 4
            elif coverage < 0.15:
                coverage_score = 0.2 + (coverage - 0.05) * 4
            elif coverage <= 0.40:
                coverage_score = 0.6 + (coverage - 0.15) * 1.6
            else:
                coverage_score = max(0, 1.0 - (coverage - 0.40) * 2)

            cl_row = self.conn.execute(
                'SELECT cl_coverage, cl_edge_sharpness, cl_symmetry_best, cl_dominance'
                ' FROM genomes WHERE id = ?',
                (genome_id,),
            ).fetchone()

            if cl_row and cl_row[0] is not None:
                cl_cov = cl_row[0] or 0
                cl_edges = cl_row[1] or 0
                cl_sym = cl_row[2] or 0
                eps = 0.01
                aesthetic = cl_cov * (cl_edges + eps) * (cl_sym + eps)
            else:
                eps = 0.01
                aesthetic = (
                    max(coverage_score, eps) ** 1.0
                    * max(color_entropy, eps) ** 1.0
                )

        # User signal: net votes propagated from loop ratings
        votes = self.net_rating('genome', genome_id)
        return aesthetic + votes * 0.5
