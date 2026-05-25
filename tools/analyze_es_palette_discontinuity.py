"""Discontinuity diagnostic for ES palettes.

Splits the corpus by per-palette discontinuity (mean adjacent-index L2 step) to test
the human-designed-LUT hypothesis. If high-discontinuity palettes dominate the
unfittable mass, smooth ES is what's actually transferable; filtering pretrain to
that subset might give cleaner H feature learning.

Reuses the saved per-palette fit errors from analyze_es_palette_structure.py.
"""

from __future__ import annotations

import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from analyze_es_palette_structure import (
    ESHEEP_DB, PALETTE_LIBRARY, OUTPUT_NPZ,
    load_palette_library, _parse_inline_palette,
)


def discontinuity(palette: np.ndarray) -> tuple[float, float]:
    """Return (mean_step, max_step) — L2 distance between adjacent palette indices."""
    diffs = np.linalg.norm(np.diff(palette, axis=0), axis=1)
    return float(diffs.mean()), float(diffs.max())


def main():
    print('--- Loading saved fit errors ---')
    data = np.load(OUTPUT_NPZ)
    generations = data['generations']
    fixed_K5 = data['fixed_K5']
    float_K5 = data['float_K5']
    float_K6 = data['float_K6']
    print(f'  {len(generations)} palettes')

    print('--- Re-extracting palettes for discontinuity computation ---')
    library = load_palette_library(PALETTE_LIBRARY)
    con = sqlite3.connect(ESHEEP_DB)
    cur = con.execute(
        "SELECT generation, sheep_id, genome_xml FROM sheep "
        "WHERE genome_xml IS NOT NULL"
    )
    mean_steps = np.zeros(len(generations), dtype=np.float32)
    max_steps = np.zeros(len(generations), dtype=np.float32)
    i = 0
    for gen, _sid, xml in cur:
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            continue
        if '<color index=' in xml:
            pal = _parse_inline_palette(root)
        elif root.get('palette') is not None:
            pidx = int(root.get('palette'))
            if pidx not in library:
                continue
            pal = library[pidx]
        else:
            continue
        m, mx = discontinuity(pal)
        mean_steps[i] = m
        max_steps[i] = mx
        i += 1
        if i % 2000 == 0:
            print(f'    {i}/{len(generations)}')
    con.close()
    mean_steps = mean_steps[:i]
    max_steps = max_steps[:i]
    print(f'  extracted {i} palettes (expected {len(generations)})')

    print('\n=== Discontinuity distribution ===')
    print(f'  mean_step  p50={np.percentile(mean_steps, 50):.4f}  '
          f'p90={np.percentile(mean_steps, 90):.4f}  '
          f'p99={np.percentile(mean_steps, 99):.4f}')
    print(f'  max_step   p50={np.percentile(max_steps, 50):.4f}  '
          f'p90={np.percentile(max_steps, 90):.4f}  '
          f'p99={np.percentile(max_steps, 99):.4f}')

    # Quartile split by mean adjacent-step
    print('\n=== Fit error by mean-step quartile ===')
    q = np.percentile(mean_steps, [25, 50, 75])
    bins = np.digitize(mean_steps, q)
    print(f'  quartile thresholds (mean_step): '
          f'q25={q[0]:.4f}  q50={q[1]:.4f}  q75={q[2]:.4f}')
    print()
    print(f'  {"quartile":>10} | {"fixed_K5 p50":>13} {"≤0.10":>8} '
          f'{"≤0.20":>8} | {"float_K6 p50":>13} {"≤0.10":>8} {"≤0.20":>8}')
    for b, label in enumerate(['Q1 smoothest', 'Q2', 'Q3', 'Q4 jaggiest']):
        mask = bins == b
        n = mask.sum()
        if n == 0:
            continue
        f5 = fixed_K5[mask]
        fl6 = float_K6[mask]
        print(f'  {label:>10} | {np.percentile(f5, 50):>13.4f} '
              f'{(f5 <= 0.10).mean():>7.1%} {(f5 <= 0.20).mean():>7.1%} | '
              f'{np.percentile(fl6, 50):>13.4f} '
              f'{(fl6 <= 0.10).mean():>7.1%} {(fl6 <= 0.20).mean():>7.1%}')

    # Correlation
    corr_fixed = np.corrcoef(mean_steps, fixed_K5)[0, 1]
    corr_float = np.corrcoef(mean_steps, float_K6)[0, 1]
    print(f'\n  Pearson(mean_step, fixed_K5 err) = {corr_fixed:.3f}')
    print(f'  Pearson(mean_step, float_K6 err) = {corr_float:.3f}')

    # What if we filtered to the smoothest 25%?
    smoothest = bins == 0
    print(f'\n=== If we filtered ES pretrain corpus to smoothest 25% ===')
    print(f'  Retained: {smoothest.sum()} palettes')
    print(f'  fixed_K5 fittable (≤0.10): {(fixed_K5[smoothest] <= 0.10).mean():.1%}')
    print(f'  fixed_K5 fittable (≤0.20): {(fixed_K5[smoothest] <= 0.20).mean():.1%}')
    print(f'  float_K6 fittable (≤0.10): {(float_K6[smoothest] <= 0.10).mean():.1%}')
    print(f'  float_K6 fittable (≤0.20): {(float_K6[smoothest] <= 0.20).mean():.1%}')

    print(f'\nSaving discontinuity to {OUTPUT_NPZ}')
    save = dict(data)
    save['mean_step'] = mean_steps
    save['max_step'] = max_steps
    np.savez_compressed(OUTPUT_NPZ, **save)


if __name__ == '__main__':
    main()
