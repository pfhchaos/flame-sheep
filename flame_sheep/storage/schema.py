"""Database schema bootstrap + migrations.

`_connect()` opens the SQLite database and runs `_ensure_schema()`,
which creates tables on a fresh DB and applies any pending migrations
on an existing one. Each migration is guarded by a metadata key so it
only runs once.

Constants `BLOB_COLS`, `STATS_COLS`, `NORMALIZATION_VERSION` live here
because they describe the data model — what columns exist, what
normalization version they're populated against. Library imports them
back when it needs to enumerate columns at query time.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

_DEFAULT_DIR = Path.home() / '.local' / 'share' / 'flame-sheep'
_DB_NAME = 'library.db'

# Blob columns that live in the separate `genome_blobs` table
BLOB_COLS = ('render_static', 'render_swept',
             'hist_static', 'hist_swept', 'hist_transform', 'hist_first_hit')

# Per-channel normalization stats columns (added in channel_stats_v1 migration)
STATS_COLS = ('mean_h', 'mean_s', 'mean_l', 'mean_a',
              'std_h',  'std_s',  'std_l',  'std_a')

# Normalization version this codebase currently produces.
# Bump when normalize_channels semantics change (gamma, sentinels, etc.).
# Weights trained against an older version must not be loaded with a
# newer normalization or scores are silently garbage.
#
# v1: per-channel mean/std over every pixel (sentinels included). Library
#     and ES distributions diverge because library is 60% sentinel-pixels
#     and ES is 28% — corpus gap is mostly sentinel coverage, not visual
#     content.
# v2: per-channel mean/std over hit pixels only (sentinel mask excluded).
#     H/L mask = static_hits > 0. S mask = swept_hits > 0. A mask =
#     first_hit < 255. Library and ES collapse to nearly the same
#     distribution at the hit-pixel level, so cross-corpus transfer
#     works without per-dataset stats. Sentinel value (1.0) standardizes
#     to the same z-score across both corpora.
# v3: same stats as v2, but standardize_channels now holds sentinels
#     STATIC at a fixed out-of-distribution value (SENTINEL_STANDARDIZED
#     in scoring_channels.py) instead of running them through (x-mean)/std.
#     Also: ES renders without first_hit data fill A with sentinel 1.0
#     pre-standardization, so the model sees "all pixels are A-sentinel"
#     consistently. Together, sentinel becomes a stable categorical
#     feature at a known location across all corpora and versions.
NORMALIZATION_VERSION = 'v3'


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
    if 'cnn_scores_detail' not in existing:
        conn.execute('ALTER TABLE genomes ADD COLUMN cnn_scores_detail TEXT')
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

    if 'archived' not in existing:
        conn.execute('ALTER TABLE genomes ADD COLUMN archived INTEGER DEFAULT 0')
    if 'pruner_checked' not in existing:
        conn.execute('ALTER TABLE genomes ADD COLUMN pruner_checked INTEGER DEFAULT 0')
    if 'archive_reason' not in existing:
        conn.execute('ALTER TABLE genomes ADD COLUMN archive_reason TEXT')

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
    _migrate_blob_separation(conn)
    _migrate_channel_stats(conn)
    _migrate_channel_stats_v2(conn)
    _migrate_generational(conn)


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
        print(f'[storage] Migrated {migrated}/{len(rows)} genomes '
              f'(variation reindex v1)', file=sys.stderr)


def _migrate_blob_separation(conn: sqlite3.Connection) -> None:
    """Move 6 BLOB columns off `genomes` into a sibling `genome_blobs` table.

    Inline blobs on a wide table force SQLite to walk multi-MB row pages
    for queries that only touch scalars (compare mode candidate building
    stalled hundreds of ms before this migration). Moving them to a
    sibling table keeps the scalar pages dense.

    One-shot — guarded by metadata key 'blob_separation_v1'.
    """
    if 'metadata' not in {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}:
        conn.execute('CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)')
        conn.commit()

    row = conn.execute(
        "SELECT value FROM metadata WHERE key='blob_separation_v1'").fetchone()
    if row is not None:
        return

    # Need foreign_keys for ON DELETE CASCADE to actually fire on row delete.
    conn.execute('PRAGMA foreign_keys = ON')

    conn.execute('''
        CREATE TABLE IF NOT EXISTS genome_blobs (
            genome_id      INTEGER PRIMARY KEY
                                    REFERENCES genomes(id) ON DELETE CASCADE,
            render_static  BLOB,
            render_swept   BLOB,
            hist_static    BLOB,
            hist_swept     BLOB,
            hist_transform BLOB,
            hist_first_hit BLOB
        )
    ''')

    # Check whether the old columns even exist (fresh DBs won't have them).
    existing = {r[1] for r in conn.execute('PRAGMA table_info(genomes)').fetchall()}
    have_old_cols = all(c in existing for c in BLOB_COLS)

    if have_old_cols:
        # Copy any non-empty blob rows.
        n_before = conn.execute(
            'SELECT COUNT(*) FROM genomes '
            'WHERE render_static IS NOT NULL OR hist_static IS NOT NULL '
            'OR render_swept IS NOT NULL OR hist_swept IS NOT NULL '
            'OR hist_transform IS NOT NULL OR hist_first_hit IS NOT NULL'
        ).fetchone()[0]
        conn.execute('''
            INSERT INTO genome_blobs
                (genome_id, render_static, render_swept,
                 hist_static, hist_swept, hist_transform, hist_first_hit)
            SELECT id, render_static, render_swept,
                   hist_static, hist_swept, hist_transform, hist_first_hit
              FROM genomes
             WHERE render_static IS NOT NULL OR hist_static IS NOT NULL
               OR render_swept IS NOT NULL OR hist_swept IS NOT NULL
               OR hist_transform IS NOT NULL OR hist_first_hit IS NOT NULL
        ''')
        n_after = conn.execute('SELECT COUNT(*) FROM genome_blobs').fetchone()[0]
        if n_after != n_before:
            raise RuntimeError(
                f'blob migration row count mismatch: {n_before} expected, {n_after} copied')
        if n_after:
            print(f'[storage] Copied {n_after} blob rows to genome_blobs '
                  f'(blob separation v1)', file=sys.stderr)

    conn.execute(
        "INSERT INTO metadata (key, value) VALUES ('blob_separation_v1', 'done')")
    conn.commit()


def _migrate_channel_stats(conn: sqlite3.Connection) -> None:
    """Add 8 per-channel stats columns to genome_blobs + backfill existing rows.

    Stats are per-genome mean/std for each of the 4 input channels
    (H, S, L, A) after normalize_channels. Trainer aggregates them into
    global normalization constants (zero-mean / unit-variance) to make
    Kaiming init's distributional assumptions hold.

    One-shot — guarded by metadata key 'channel_stats_v1'. Backfills
    inline by re-running normalize_channels on stored histograms.
    """
    row = conn.execute(
        "SELECT value FROM metadata WHERE key='channel_stats_v1'").fetchone()
    if row is not None:
        return

    existing = {r[1] for r in conn.execute(
        'PRAGMA table_info(genome_blobs)').fetchall()}
    added = 0
    for col in STATS_COLS:
        if col not in existing:
            conn.execute(f'ALTER TABLE genome_blobs ADD COLUMN {col} REAL')
            added += 1
    conn.commit()

    # Backfill: compute stats for every row with histograms.
    from ..scoring_channels import (
        unpack_static_histogram, unpack_histogram, normalize_channels,
    )

    rows = conn.execute(
        '''SELECT genome_id, hist_static, hist_swept, hist_first_hit
             FROM genome_blobs
            WHERE hist_static IS NOT NULL AND hist_swept IS NOT NULL'''
    ).fetchall()

    if rows:
        import time as _time
        t0 = _time.time()
        for gid, hs, hsw, hfh in rows:
            try:
                hits, colors = unpack_static_histogram(hs)
                swept = unpack_histogram(hsw)
                first_hit = (unpack_histogram(hfh, dtype=np.uint8)
                             if hfh is not None else None)
                channels = normalize_channels(hits, colors, swept, first_hit)
                # channels shape: (4, H, W); axes 1,2 are spatial.
                means = channels.mean(axis=(1, 2)).astype(np.float64)
                stds = channels.std(axis=(1, 2)).astype(np.float64)
                conn.execute(
                    '''UPDATE genome_blobs
                          SET mean_h=?, mean_s=?, mean_l=?, mean_a=?,
                              std_h=?,  std_s=?,  std_l=?,  std_a=?
                        WHERE genome_id=?''',
                    (float(means[0]), float(means[1]),
                     float(means[2]), float(means[3]),
                     float(stds[0]), float(stds[1]),
                     float(stds[2]), float(stds[3]),
                     gid),
                )
            except Exception as e:
                print(f'[storage] channel stats backfill skipped gid {gid}: {e}',
                      file=sys.stderr)
        elapsed = _time.time() - t0
        print(f'[storage] Backfilled channel stats for {len(rows)} genomes '
              f'in {elapsed:.1f}s (channel_stats_v1)', file=sys.stderr)

    conn.execute(
        "INSERT INTO metadata (key, value) VALUES ('channel_stats_v1', 'done')")
    conn.commit()


def _migrate_channel_stats_v2(conn: sqlite3.Connection) -> None:
    """Recompute genome_blobs stats with sentinel-aware semantics.

    v1 mean/std were taken over every pixel including sentinels (H=1.0
    for never-hit, A=1.0 for first_hit==255). Library is 60% sentinels,
    ES is 28% — so per-corpus stats diverged mostly because of the
    sentinel fraction, not the actual rendered content. The standardized
    value of "this is a sentinel" landed at different z-scores across
    corpora, breaking cross-corpus equivalence the model relies on.

    v2 uses hit-only means/stds (H/L masked by static_hits>0, S by
    swept_hits>0, A by first_hit<255). The hit-pixel distributions of
    library and ES collapse to within ~5% of each other; sentinel
    z-scores become consistent across both.

    One-shot — guarded by metadata key 'channel_stats_v2'. Overwrites
    the same 8 stats columns in-place (v1's per-genome stats become
    inaccessible once v2 runs; v1's aggregate 'normalization_v1' key
    in metadata stays for any legacy weights file that wants to verify
    its origin, but won't be used by current trainers).
    """
    row = conn.execute(
        "SELECT value FROM metadata WHERE key='channel_stats_v2'").fetchone()
    if row is not None:
        return

    from ..scoring_channels import (
        unpack_static_histogram, unpack_histogram, channel_stats_hit_only,
    )

    rows = conn.execute(
        '''SELECT genome_id, hist_static, hist_swept, hist_first_hit
             FROM genome_blobs
            WHERE hist_static IS NOT NULL AND hist_swept IS NOT NULL'''
    ).fetchall()

    if rows:
        import time as _time
        t0 = _time.time()
        for gid, hs, hsw, hfh in rows:
            try:
                hits, colors = unpack_static_histogram(hs)
                swept = unpack_histogram(hsw)
                first_hit = (unpack_histogram(hfh, dtype=np.uint8)
                             if hfh is not None else None)
                means, stds = channel_stats_hit_only(
                    hits, colors, swept, first_hit)
                conn.execute(
                    '''UPDATE genome_blobs
                          SET mean_h=?, mean_s=?, mean_l=?, mean_a=?,
                              std_h=?,  std_s=?,  std_l=?,  std_a=?
                        WHERE genome_id=?''',
                    (float(means[0]), float(means[1]),
                     float(means[2]), float(means[3]),
                     float(stds[0]), float(stds[1]),
                     float(stds[2]), float(stds[3]),
                     gid),
                )
            except Exception as e:
                print(f'[storage] v2 stats backfill skipped gid {gid}: {e}',
                      file=sys.stderr)
        elapsed = _time.time() - t0
        print(f'[storage] Recomputed channel stats v2 for {len(rows)} genomes '
              f'in {elapsed:.1f}s (channel_stats_v2)', file=sys.stderr)

    conn.execute(
        "INSERT INTO metadata (key, value) VALUES ('channel_stats_v2', 'done')")
    conn.commit()


def _migrate_generational(conn: sqlite3.Connection) -> None:
    """Add generation column to ratings + pairwise_ratings tables.

    Generational architecture (see docs/generational_architecture.md):
    each rating is tagged with the model+population generation it was
    made against. Thumbs are corpus-relative judgments — only valid as
    training signal within their own generation (synthesized into
    intra-gen pairwise pairs). Pairwise compare-mode pairs are self-
    contained and durable across generations, but stamping them with
    the rating-time generation is cheap and useful for diagnostics.

    Existing data backfills to generation 0 (= "pre-generational",
    rated against the now-archived corpus that prompted today's CNN
    rework). current_generation in metadata starts at 1, defining the
    v3 pairwise-only model's deployment as the boundary. New ratings
    from this point on get stamped with current_generation at insert.

    One-shot, guarded by metadata key 'generational_v1'.
    """
    row = conn.execute(
        "SELECT value FROM metadata WHERE key='generational_v1'").fetchone()
    if row is not None:
        return

    # Add generation column to both rating tables. Default 0 backfills
    # existing rows automatically — they predate the generational system.
    for table in ('ratings', 'pairwise_ratings'):
        cols = {r[1] for r in conn.execute(
            f'PRAGMA table_info({table})').fetchall()}
        if 'generation' not in cols:
            conn.execute(
                f'ALTER TABLE {table} ADD COLUMN generation INTEGER NOT NULL DEFAULT 0')

    # Seed current_generation = 1. New ratings collected from this point
    # forward get stamped with this value (or whatever it's been advanced
    # to by subsequent train+breed cycles).
    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES "
        "('current_generation', '1')")

    conn.execute(
        "INSERT INTO metadata (key, value) VALUES ('generational_v1', 'done')")
    conn.commit()

    n_ratings = conn.execute('SELECT COUNT(*) FROM ratings').fetchone()[0]
    n_pw = conn.execute('SELECT COUNT(*) FROM pairwise_ratings').fetchone()[0]
    print(f'[storage] Generational migration: backfilled {n_ratings} ratings + '
          f'{n_pw} pairwise to generation=0, set current_generation=1 '
          f'(generational_v1)', file=sys.stderr)
