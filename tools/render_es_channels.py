#!/usr/bin/env python3
"""Render ES histogram .npz files as channel PNGs for inspection."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flame_sheep.scoring_channels import normalize_channels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+', help='hist.npz files')
    parser.add_argument('--output', default=str(Path.home() / 'renders/es-check'))
    parser.add_argument('--sidecar', default=None,
                        help='normalization_v1.json sidecar — also render standardized channels')
    args = parser.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    mean = std = None
    if args.sidecar:
        side = json.load(open(args.sidecar))
        mean = np.array(side['mean'], dtype=np.float32)
        std = np.array(side['std'], dtype=np.float32)
        print(f'Using sidecar mean={mean}, std={std}')

    for path in args.files:
        p = Path(path)
        data = np.load(p)
        hits = data['static_hits']
        colors = data['static_colors']
        swept = data['swept_hits']
        first_hit = data.get('first_hit', None)
        ch = normalize_channels(hits, colors, swept, first_hit,
                                output_size=hits.shape[0])
        name = p.stem.replace('_hist', '')
        for i, label in enumerate(['H_palette', 'S_swept', 'L_structure', 'A_emergence']):
            img = (ch[i] * 255).astype(np.uint8)
            Image.fromarray(img).save(outdir / f'{name}_{label}.png')

        if mean is not None:
            # Standardize and renormalize to [0,1] for visualization
            std_safe = np.where(std > 1e-6, std, 1.0)
            standardized = (ch - mean[:, None, None]) / std_safe[:, None, None]
            for i, label in enumerate(['H', 'S', 'L', 'A']):
                v = standardized[i]
                # Map [-3, 3] sigma range to [0, 1] for viewing
                v_viz = np.clip((v + 3) / 6, 0, 1)
                img = (v_viz * 255).astype(np.uint8)
                Image.fromarray(img).save(outdir / f'{name}_{label}_std.png')

        print(f'  {name}')

    print(f'Done: {outdir}')


if __name__ == '__main__':
    main()
