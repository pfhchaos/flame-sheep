"""Measure how well ES palettes can be approximated by our palette representation.

The transfer question: H-channel CNN features learned from ES palettes only transfer to
personal palettes if the structural space overlaps. Personal palettes are 3-6 control
points, uniformly-spaced knots, linear RGB interp. ES palettes are 256 free RGB samples.

For each ES palette, compute approximation error under:
  - Fixed uniform knots, K segments — current representation (K=2..5 covers 3-6 pts)
  - Floating knots (optimal placement via DP) — knot-freedom upgrade
  - Extended K — control-point-count upgrade

Bundled L≈S correlation test on the standardized histograms (separate analysis for the
channel ablation: if L and S are highly correlated in ES, the S channel has no signal
to learn in pretrain and the post-finetune ablation result is explained without the
LR-too-high hypothesis).
"""

from __future__ import annotations

import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


ESHEEP_DB = Path.home() / '.local/share/flame-sheep/esheep.db'
PALETTE_LIBRARY = Path('/home/chaos/projects/flam3/flam3-palettes.xml')
ESHEEP_HIST_DIR = Path.home() / 'datasets/esheep-cnn'
OUTPUT_NPZ = Path.home() / 'datasets/es_palette_fit_errors.npz'

FIXED_K = [2, 3, 4, 5, 6, 8, 10, 12, 15]
FLOAT_K_MAX = 6  # DP up to K=6; diminishing returns past that
EPS_THRESHOLDS = [0.02, 0.05, 0.10, 0.20]
N_LS_SAMPLES = 1000


# ---------------------------------------------------------------------------
# Palette extraction
# ---------------------------------------------------------------------------

def _parse_inline_palette(root: ET.Element) -> np.ndarray:
    palette = np.zeros((256, 3), dtype=np.float32)
    for c in root.findall('color'):
        idx = int(c.get('index', '0'))
        rgb = c.get('rgb', '0 0 0').split()
        if idx < 256 and len(rgb) >= 3:
            palette[idx] = [float(rgb[0]) / 255.0,
                            float(rgb[1]) / 255.0,
                            float(rgb[2]) / 255.0]
    return palette


def load_palette_library(path: Path) -> dict[int, np.ndarray]:
    """Load flam3 standard palette library. Format: 8 hex chars per entry (00RRGGBB)."""
    root = ET.parse(path).getroot()
    library: dict[int, np.ndarray] = {}
    for pal in root.findall('palette'):
        n = int(pal.get('number'))
        hex_data = ''.join(pal.get('data', '').split())
        palette = np.zeros((256, 3), dtype=np.float32)
        for i in range(256):
            chunk = hex_data[i * 8:(i + 1) * 8]
            if len(chunk) < 8:
                break
            palette[i] = [int(chunk[2:4], 16) / 255.0,
                          int(chunk[4:6], 16) / 255.0,
                          int(chunk[6:8], 16) / 255.0]
        library[n] = palette
    return library


def extract_all_palettes(db_path: Path, library: dict[int, np.ndarray]):
    """Iterate over all ES genomes; return (N, 256, 3) array + (N,) generation labels."""
    palettes = []
    generations = []
    n_inline = 0
    n_indexed = 0
    n_skipped = 0

    con = sqlite3.connect(db_path)
    cur = con.execute(
        "SELECT generation, sheep_id, genome_xml FROM sheep WHERE genome_xml IS NOT NULL"
    )
    for gen, _sid, xml in cur:
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            n_skipped += 1
            continue

        if '<color index=' in xml:
            palette = _parse_inline_palette(root)
            n_inline += 1
        elif root.get('palette') is not None:
            pidx = int(root.get('palette'))
            if pidx not in library:
                n_skipped += 1
                continue
            palette = library[pidx].copy()
            n_indexed += 1
        else:
            n_skipped += 1
            continue

        palettes.append(palette)
        generations.append(gen)

    con.close()
    print(f'  inline: {n_inline}  indexed: {n_indexed}  skipped: {n_skipped}')
    return np.stack(palettes), np.array(generations, dtype=np.int32)


# ---------------------------------------------------------------------------
# Fit error computation
# ---------------------------------------------------------------------------

def fixed_knot_error_per_K(palette: np.ndarray, K_list: list[int]) -> dict[int, float]:
    """RMS L2 error per sample for each K (uniformly-spaced knots)."""
    out = {}
    for K in K_list:
        knot_idx = np.round(np.linspace(0, 255, K + 1)).astype(int)
        recon = np.zeros_like(palette)
        for s in range(K):
            lo, hi = knot_idx[s], knot_idx[s + 1]
            seg_len = hi - lo
            if seg_len == 0:
                recon[lo] = palette[lo]
                continue
            t = np.arange(seg_len + 1) / seg_len
            recon[lo:hi + 1] = (
                (1 - t)[:, None] * palette[lo]
                + t[:, None] * palette[hi]
            )
        sq_err = ((palette - recon) ** 2).sum(axis=1)
        out[K] = float(np.sqrt(sq_err.mean()))
    return out


def segment_err_table(palette: np.ndarray) -> np.ndarray:
    """err[i][j] = sum of squared 3D L2 distances from points i..j to line i→j.

    Vectorized: for each segment length L=1..N-1, compute all (i, i+L) at once via
    sliding windows. O(N) Python iterations, all heavy work in numpy.
    """
    N = len(palette)
    err = np.full((N, N), np.inf, dtype=np.float64)
    np.fill_diagonal(err, 0.0)
    pal = palette.astype(np.float64)

    for L in range(1, N):
        starts = pal[:N - L]    # (N-L, 3)
        ends = pal[L:]          # (N-L, 3)
        t = np.arange(L + 1) / L  # (L+1,)
        # line shape: (N-L, L+1, 3)
        line = (
            (1 - t)[None, :, None] * starts[:, None, :]
            + t[None, :, None] * ends[:, None, :]
        )
        # windows of palette points: (N-L, L+1, 3) via sliding view
        windows = sliding_window_view(pal, window_shape=L + 1, axis=0)
        # sliding_window_view returns shape (N-L, 3, L+1) — swap to (N-L, L+1, 3)
        windows = np.swapaxes(windows, 1, 2)
        diff = windows - line
        err_L = (diff ** 2).sum(axis=(1, 2))
        i_arr = np.arange(N - L)
        err[i_arr, i_arr + L] = err_L
    return err


def dp_floating_knot(err: np.ndarray, K_max: int) -> np.ndarray:
    """Return cost_for_K[k] for k=1..K_max — min sum-squared-err for k-segment cover."""
    N = err.shape[0]
    f = np.full((K_max + 1, N), np.inf, dtype=np.float64)
    f[1] = err[0]  # one segment from 0 to j
    for k in range(2, K_max + 1):
        for j in range(k, N):
            candidates = f[k - 1, k - 1:j] + err[k - 1:j, j]
            f[k, j] = candidates.min()
    return f[1:K_max + 1, N - 1]


def floating_knot_error_per_K(palette: np.ndarray, K_max: int) -> dict[int, float]:
    err_table = segment_err_table(palette)
    costs = dp_floating_knot(err_table, K_max)
    N = len(palette)
    return {k + 1: float(np.sqrt(costs[k] / N)) for k in range(K_max)}


# ---------------------------------------------------------------------------
# L≈S correlation bundle
# ---------------------------------------------------------------------------

def ls_correlation_sample(hist_dir: Path, n_samples: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    files = sorted(hist_dir.glob('*_hist.npz'))
    if len(files) > n_samples:
        files = list(rng.choice(files, size=n_samples, replace=False))

    corrs = []
    for f in files:
        try:
            d = np.load(f)
            static = d['static_hits'].astype(np.float64).ravel()
            swept = d['swept_hits'].astype(np.float64).ravel()
        except (KeyError, OSError):
            continue
        # Use log-transform to match what the standardization pipeline does.
        static = np.log1p(static)
        swept = np.log1p(swept)
        if static.std() < 1e-6 or swept.std() < 1e-6:
            continue
        c = np.corrcoef(static, swept)[0, 1]
        if np.isfinite(c):
            corrs.append(c)
    return np.array(corrs)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_error_table(label: str, errors_by_K: dict[int, np.ndarray]):
    print(f'\n=== {label} ===')
    print(f'{"K":>3} | {"p50":>7} {"p90":>7} {"p99":>7} | ' +
          ' '.join(f'≤{e:.2f}' for e in EPS_THRESHOLDS))
    for K in sorted(errors_by_K.keys()):
        e = errors_by_K[K]
        p50, p90, p99 = np.percentile(e, [50, 90, 99])
        frac = [(e <= eps).mean() for eps in EPS_THRESHOLDS]
        print(f'{K:>3} | {p50:>7.4f} {p90:>7.4f} {p99:>7.4f} | ' +
              ' '.join(f'{f:>5.1%}' for f in frac))


def main():
    print('--- Loading palette library ---')
    library = load_palette_library(PALETTE_LIBRARY)
    print(f'  loaded {len(library)} library palettes')

    print('--- Extracting ES palettes ---')
    palettes, generations = extract_all_palettes(ESHEEP_DB, library)
    print(f'  total palettes: {len(palettes)}')

    print('--- Computing fixed-knot fit errors ---')
    fixed_errors: dict[int, list[float]] = {K: [] for K in FIXED_K}
    for i, pal in enumerate(palettes):
        if i % 2000 == 0:
            print(f'    {i}/{len(palettes)}')
        per_K = fixed_knot_error_per_K(pal, FIXED_K)
        for K, e in per_K.items():
            fixed_errors[K].append(e)
    fixed_errors = {K: np.array(v) for K, v in fixed_errors.items()}

    print('--- Computing floating-knot fit errors (DP) ---')
    float_errors: dict[int, list[float]] = {k: [] for k in range(1, FLOAT_K_MAX + 1)}
    for i, pal in enumerate(palettes):
        if i % 500 == 0:
            print(f'    {i}/{len(palettes)}')
        per_K = floating_knot_error_per_K(pal, FLOAT_K_MAX)
        for K, e in per_K.items():
            float_errors[K].append(e)
    float_errors = {K: np.array(v) for K, v in float_errors.items()}

    print('--- L≈S correlation across ES histograms ---')
    ls_corrs = ls_correlation_sample(ESHEEP_HIST_DIR, N_LS_SAMPLES)
    print(f'  N samples: {len(ls_corrs)}')

    print_error_table('Fixed uniform knots (current rep envelope)', fixed_errors)
    print_error_table('Floating knots (knot-freedom upgrade)', float_errors)

    print('\n=== L≈S correlation (log-static_hits vs log-swept_hits) ===')
    if len(ls_corrs):
        for q in [10, 25, 50, 75, 90]:
            print(f'  p{q:>2}: {np.percentile(ls_corrs, q):.4f}')
        for thr in [0.7, 0.9, 0.95, 0.99]:
            print(f'  frac > {thr}: {(ls_corrs > thr).mean():.1%}')

    print(f'\nSaving raw errors to {OUTPUT_NPZ}')
    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {
        'generations': generations,
        'ls_correlations': ls_corrs,
    }
    for K, v in fixed_errors.items():
        save_kwargs[f'fixed_K{K}'] = v
    for K, v in float_errors.items():
        save_kwargs[f'float_K{K}'] = v
    np.savez_compressed(OUTPUT_NPZ, **save_kwargs)


if __name__ == '__main__':
    main()
