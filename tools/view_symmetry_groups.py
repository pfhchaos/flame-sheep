#!/usr/bin/env python3
"""Visualize symmetry variations using the actual CPU chaos game.

Renders each symmetry group (wallpaper/frieze) and other variations
(sattractor, icon) as flame fractal histograms. Uses the same data
and code paths as the renderer and scorer.

Usage:
    python tools/view_symmetry_groups.py                  # all wallpaper groups
    python tools/view_symmetry_groups.py --frieze          # all frieze groups
    python tools/view_symmetry_groups.py --variation icon   # single variation
    python tools/view_symmetry_groups.py --variation sattractor --param sat_m=6
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

from flame_sheep.genome import Genome, Transform, NUM_VARIATIONS
from flame_sheep.variations import Variation, random_var_params
from flame_sheep.variations._symmetry_groups import (
    WALLPAPER_GROUPS, FRIEZE_GROUPS, WALLPAPER_NAMES, FRIEZE_NAMES,
)


def make_genome(variation_idx: int, var_params: dict = None,
                rng: np.random.Generator = None) -> Genome:
    """Create a minimal genome with one transform using a single variation."""
    if rng is None:
        rng = np.random.default_rng(42)
    g = Genome()
    t = Transform()
    t.affine = np.array([0.8, 0.1, 0.0, -0.1, 0.8, 0.0], dtype=np.float32)
    t.variations = np.zeros(NUM_VARIATIONS, dtype=np.float32)
    t.variations[variation_idx] = 1.0
    t.var_params = var_params or {}
    t.weight = 1.0
    t.color = 0.5
    g.transforms = [t]
    g.zoom = 1.0
    return g


def render_histogram(genome: Genome, grid_size: int = 256,
                     n_iter: int = 100_000) -> np.ndarray:
    """Run CPU chaos game and return log-scaled histogram."""
    from flame_sheep.variations import apply_variations_cpu
    from flame_sheep.variations._cpu import _current_var_params

    # Set var_params for CPU path (wallpaper/frieze group selection)
    _current_var_params.clear()
    for tr in genome.transforms:
        _current_var_params.update(tr.var_params)

    # Patch the module-level dict
    import flame_sheep.variations._cpu as cpu_mod
    cpu_mod._current_var_params = _current_var_params

    fuse = 20
    bound = 4.0
    rng = np.random.default_rng(42)

    hit_grid = np.zeros((grid_size, grid_size), dtype=np.float64)

    weights = np.array([tr.weight for tr in genome.transforms], dtype=np.float64)
    weights /= weights.sum()
    cumw = np.cumsum(weights)

    x, y = 0.0, 0.0
    for i in range(fuse + n_iter):
        r = rng.random()
        tidx = min(int(np.searchsorted(cumw, r)), len(genome.transforms) - 1)
        tr = genome.transforms[tidx]
        a, b, c, d, e, f = tr.affine
        nx = a * x + b * y + c
        ny = d * x + e * y + f
        nx, ny = apply_variations_cpu(tr.variations, nx, ny)
        x, y = nx, ny
        if not (np.isfinite(x) and np.isfinite(y)):
            break
        if i >= fuse and abs(x) < bound and abs(y) < bound:
            gx = int((x + bound) / (2 * bound) * grid_size)
            gy = int((y + bound) / (2 * bound) * grid_size)
            gx = max(0, min(grid_size - 1, gx))
            gy = max(0, min(grid_size - 1, gy))
            hit_grid[gy, gx] += 1.0

    return np.log1p(hit_grid)


def show_wallpaper():
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    fig.suptitle('17 Wallpaper Groups', fontsize=14)
    for i in range(17):
        row, col = divmod(i, 6)
        params = {'wallpaper_group': float(i)}
        g = make_genome(Variation.WALLPAPER, params)
        hist = render_histogram(g)
        axes[row, col].imshow(hist, cmap='inferno', origin='lower')
        axes[row, col].set_title(f'{WALLPAPER_NAMES[i]} ({i})', fontsize=9)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
    axes[2, 5].set_visible(False)
    fig.tight_layout()
    fig.savefig('/tmp/wallpaper_groups.png', dpi=150)
    print('Saved /tmp/wallpaper_groups.png')
    plt.show()


def show_frieze():
    fig, axes = plt.subplots(1, 7, figsize=(21, 3))
    fig.suptitle('7 Frieze Groups', fontsize=14)
    for i in range(7):
        params = {'frieze_group': float(i)}
        g = make_genome(Variation.FRIEZE, params)
        hist = render_histogram(g)
        axes[i].imshow(hist, cmap='inferno', origin='lower')
        axes[i].set_title(f'{FRIEZE_NAMES[i]} ({i})', fontsize=9)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    fig.tight_layout()
    fig.savefig('/tmp/frieze_groups.png', dpi=150)
    print('Saved /tmp/frieze_groups.png')
    plt.show()


def show_variation(var_name: str, params: dict = None):
    var_idx = getattr(Variation, var_name.upper(), None)
    if var_idx is None:
        print(f'Unknown variation: {var_name}')
        print(f'Available: {[k for k in dir(Variation) if not k.startswith("_")]}')
        return
    if params is None:
        rng = np.random.default_rng(42)
        params = random_var_params(var_idx, rng)
    g = make_genome(var_idx, params)
    hist = render_histogram(g)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(hist, cmap='inferno', origin='lower')
    ax.set_title(f'{var_name} {params}', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    plt.show()


def show_matrix_wallpaper():
    """Show each wallpaper group's transforms as geometric operations on an L-shape."""
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    # Teardrop centered near origin — fully asymmetric, no hidden overlaps
    t = np.linspace(0, 2*np.pi, 30)
    r = 0.04 * (1 + np.cos(t))  # cardioid-ish, small
    L = np.column_stack([r * np.cos(t) + 0.02, r * np.sin(t)])

    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    fig.suptitle('17 Wallpaper Groups — Transform Elements', fontsize=14)

    for i, (name, elements) in enumerate(zip(WALLPAPER_NAMES, WALLPAPER_GROUPS)):
        row, col = divmod(i, 6)
        ax = axes[row, col]
        colors = plt.cm.Set3(np.linspace(0, 1, max(len(elements), 1)))
        patches = []
        for j, (a, b, c, d, e, f) in enumerate(elements):
            pts = np.column_stack([
                a * L[:, 0] + b * L[:, 1] + c,
                d * L[:, 0] + e * L[:, 1] + f,
            ])
            patches.append(Polygon(pts, closed=True))
        coll = PatchCollection(patches, alpha=0.5, edgecolors='black', linewidths=0.5)
        coll.set_facecolor(colors[:len(patches)])
        ax.add_collection(coll)
        ax.axhline(0, color='gray', linewidth=0.3)
        ax.axvline(0, color='gray', linewidth=0.3)
        ax.set_title(f'{name} ({len(elements)})', fontsize=8)
        ax.set_aspect('equal')
        ax.autoscale()
        ax.set_xticks([])
        ax.set_yticks([])
    axes[2, 5].set_visible(False)
    fig.tight_layout()
    fig.savefig('/tmp/wallpaper_matrix.png', dpi=150)
    print('Saved /tmp/wallpaper_matrix.png')
    plt.show()


def show_matrix_frieze():
    """Show each frieze group's transforms as geometric operations on an L-shape."""
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    L = np.array([[-0.15, -0.15], [0.15, -0.15], [0.15, -0.05],
                  [-0.05, -0.05], [-0.05, 0.15], [-0.15, 0.15]])

    fig, axes = plt.subplots(1, 7, figsize=(21, 3))
    fig.suptitle('7 Frieze Groups — Transform Elements', fontsize=14)

    for i, (name, elements) in enumerate(zip(FRIEZE_NAMES, FRIEZE_GROUPS)):
        ax = axes[i]
        colors = plt.cm.Set3(np.linspace(0, 1, max(len(elements), 1)))
        patches = []
        for j, (a, b, c, d, e, f) in enumerate(elements):
            pts = np.column_stack([
                a * L[:, 0] + b * L[:, 1] + c,
                d * L[:, 0] + e * L[:, 1] + f,
            ])
            patches.append(Polygon(pts, closed=True))
        coll = PatchCollection(patches, alpha=0.5, edgecolors='black', linewidths=0.5)
        coll.set_facecolor(colors[:len(patches)])
        ax.add_collection(coll)
        ax.axhline(0, color='gray', linewidth=0.3)
        ax.axvline(0, color='gray', linewidth=0.3)
        ax.set_title(f'{name} ({len(elements)})', fontsize=8)
        ax.set_aspect('equal')
        ax.autoscale()
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig('/tmp/frieze_matrix.png', dpi=150)
    print('Saved /tmp/frieze_matrix.png')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize symmetry variations')
    parser.add_argument('--frieze', action='store_true', help='Show frieze groups')
    parser.add_argument('--matrix', action='store_true',
                        help='Show transform elements as L-shape operations instead of fractals')
    parser.add_argument('--variation', type=str, help='Show a single variation by name')
    parser.add_argument('--param', type=str, action='append',
                        help='Set param as key=value (can repeat)')
    args = parser.parse_args()

    if args.variation:
        params = {}
        if args.param:
            for p in args.param:
                k, v = p.split('=')
                params[k] = float(v)
        show_variation(args.variation, params or None)
    elif args.matrix and args.frieze:
        show_matrix_frieze()
    elif args.matrix:
        show_matrix_wallpaper()
    elif args.frieze:
        show_frieze()
    else:
        show_wallpaper()


if __name__ == '__main__':
    main()
