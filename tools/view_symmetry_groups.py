#!/usr/bin/env python3
"""Visualize the 24 crystallographic symmetry groups.

Renders a test shape (L-shape to show orientation) through each group's
transforms. Run with: python tools/view_symmetry_groups.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# L-shape vertices (asymmetric to show rotation/reflection)
L_SHAPE = np.array([
    [0.0, 0.0], [0.3, 0.0], [0.3, 0.1], [0.1, 0.1], [0.1, 0.3], [0.0, 0.3]
])


def apply_affine(points, a, b, c, d, e, f):
    """Apply affine transform to a set of 2D points."""
    result = np.zeros_like(points)
    result[:, 0] = a * points[:, 0] + b * points[:, 1] + c
    result[:, 1] = d * points[:, 0] + e * points[:, 1] + f
    return result


# --- Wallpaper groups (SymNetG1-17) ---
# Format: (name, [(a, b, c, d, e, f), ...])
# Using stepx=0.5, stepy=0.5, sepx=0.5, sepy=0.5 as defaults

sx, sy = 0.25, 0.25  # stepx/2, stepy/2
sep = 0.25  # sepx/2, sepy/2

WALLPAPER_GROUPS = [
    ("p1 (ng1)", [
        (1, 0, -sx, 0, 1, -sy),
        (1, 0, sx, 0, 1, sy),
    ]),
    ("p2 (ng2)", [
        (1, 0, -sx, 0, 1, -sy),
        (1, 0, sx, 0, -1, sy),
    ]),
    ("pg (ng3)", [
        (1, 0, -sep-sx, 0, 1, -sep-sy),
        (-1, 0, sep+sx, 0, -1, sep+sy),
    ]),
    ("pmm (ng4)", [
        (1, 0, -sep-2-sx, 0, 1, sep-1.5-sy),
        (-1, 0, sep-sx, 0, -1, -sep-0.5-sy),
        (1, 0, -sep+sx, 0, -1, -sep+0.5+sy),
        (-1, 0, sep+2+sx, 0, 1, sep-0.5+sy),
    ]),
    ("pmg (ng5)", [
        (1, 0, -sep-2-sx, 0, 1, sep-0.5),
        (-1, 0, sep-sx, 0, -1, -sep-0.5),
        (1, 0, -sep+sx, 0, -1, -sep-0.5),
        (-1, 0, 2+sep+sx, 0, 1, sep-0.5),
    ]),
    ("pm (ng6)", [
        (1, 0, -sx, 0, 1, -sy+sep),
        (1, 0, -sx, 0, -1, -sy-sep),
        (1, 0, sx, 0, 1, sy+sep),
        (1, 0, sx, 0, -1, sy-sep),
    ]),
    ("cm (ng7)", [
        (1, 0, -sep, 0, 1, sep),
        (1, 0, sep, 0, -1, -sep),
    ]),
    ("pgg (ng8)", [
        (1, 0, -1-sep, 0, -1, -sep),
        (-1, 0, 1+sep, 0, 1, sep),
        (-1, 0, 1+sep, 0, -1, -sep),
        (1, 0, -1-sep, 0, 1, sep),
        (1, 0, -1-sep, 0, -1, sep),
        (-1, 0, 1+sep, 0, 1, -sep),
        (-1, 0, 1+sep, 0, -1, sep),
        (1, 0, -1-sep, 0, 1, -sep),
    ]),
    ("cmm (ng9)", [
        (1, 0, sep, 0, -1, -sep),
        (-1, 0, -sep, 0, 1, sep),
        (-1, 0, -sep, 0, -1, -sep),
        (1, 0, sep, 0, 1, sep),
    ]),
    ("p4 (ng10)", [
        (-1, 0, -sep, 0, -1, -sep),
        (0, 1, -sep, -1, 0, -sep),
        (0, -1, sep, 1, 0, sep),
        (1, 0, sep, 0, 1, sep),
    ]),
    ("p4m (ng11)", [
        (1, 0, sep+sx, 0, 1, sep+sy),
        (-1, 0, -sep+sx, 0, -1, -sep+sy),
        (0, 1, sep+sx, -1, 0, -sep+sy),
        (0, -1, -sep+sx, 1, 0, sep+sy),
        (-1, 0, -sep-sx, 0, 1, sep-sy),
        (0, -1, -sep-sx, -1, 0, -sep-sy),
        (0, 1, sep-sx, 1, 0, sep-sy),
        (1, 0, sep-sx, 0, -1, -sep-sy),
    ]),
    ("p4g (ng12)", [
        (1, 0, sep, 0, 1, sep),
        (-1, 0, -sep, 0, -1, -sep),
        (0, 1, sep, -1, 0, -sep),
        (0, -1, -sep, 1, 0, sep),
        (-1, 0, -sep, 0, 1, sep),
        (0, -1, -sep, -1, 0, -sep),
        (0, 1, sep, 1, 0, sep),
        (1, 0, sep, 0, -1, -sep),
    ]),
    ("p3 (ng13)", [
        (1, 0, -sx, 0, -1, -sy),
        (-0.5, -0.866, -sx, -0.866, 0.5, -sy),
        (-0.5, 0.866, -sx, 0.866, 0.5, -sy),
        (1, 0, sx, 0, -1, sy),
        (-0.5, -0.866, sx, -0.866, 0.5, sy),
        (-0.5, 0.866, sx, 0.866, 0.5, sy),
    ]),
    ("p3m1 (ng14)", [
        (1, 0, -sx, 0, 1, -sy),
        (1, 0, -sx, 0, -1, -sy),
        (-0.5, -0.866, -sx, 0.866, -0.5, -sy),
        (-0.5, 0.866, -sx, -0.866, -0.5, -sy),
        (-0.5, -0.866, -sx, -0.866, 0.5, -sy),
        (-0.5, 0.866, -sx, 0.866, 0.5, -sy),
        (1, 0, sx, 0, 1, sy),
        (1, 0, sx, 0, -1, sy),
        (-0.5, -0.866, sx, 0.866, -0.5, sy),
        (-0.5, 0.866, sx, -0.866, -0.5, sy),
        (-0.5, -0.866, sx, -0.866, 0.5, sy),
        (-0.5, 0.866, sx, 0.866, 0.5, sy),
    ]),
    ("p31m (ng15)", [
        (0, 1, -sx, -1, 0, -sy),
        (0, -1, -sx, -1, 0, -sy),
        (0.866, -0.5, -sx, 0.5, 0.866, -sy),
        (0.866, 0.5, -sx, 0.5, -0.866, -sy),
        (-0.866, -0.5, -sx, 0.5, -0.866, -sy),
        (-0.866, 0.5, -sx, 0.5, 0.866, -sy),
        (0, 1, sx, -1, 0, sy),
        (0, -1, sx, -1, 0, sy),
        (0.866, -0.5, sx, 0.5, 0.866, sy),
        (0.866, 0.5, sx, 0.5, -0.866, sy),
        (-0.866, -0.5, sx, 0.5, -0.866, sy),
        (-0.866, 0.5, sx, 0.5, 0.866, sy),
    ]),
    ("p6 (ng16)", [
        (1, 0, -sx, 0, 1, -sy),
        (0.5, -0.866, -sx, 0.866, 0.5, -sy),
        (-0.5, -0.866, -sx, 0.866, -0.5, -sy),
        (-1, 0, -sx, 0, -1, -sy),
        (-0.5, 0.866, -sx, -0.866, -0.5, -sy),
        (0.5, 0.866, -sx, -0.866, 0.5, -sy),
        (1, 0, sx, 0, 1, sy),
        (0.5, -0.866, sx, 0.866, 0.5, sy),
        (-0.5, -0.866, sx, 0.866, -0.5, sy),
        (-1, 0, sx, 0, -1, sy),
        (-0.5, 0.866, sx, -0.866, -0.5, sy),
        (0.5, 0.866, sx, -0.866, 0.5, sy),
    ]),
    ("p6m (ng17)", [
        (1, 0, -sx, 0, 1, -sy),
        (1, 0, -sx, 0, -1, -sy),
        (0.5, -0.866, -sx, 0.866, 0.5, -sy),
        (0.5, 0.866, -sx, 0.866, -0.5, -sy),
        (-0.5, -0.866, -sx, 0.866, -0.5, -sy),
        (-0.5, 0.866, -sx, 0.866, 0.5, -sy),
        (-1, 0, -sx, 0, -1, -sy),
        (-1, 0, -sx, 0, 1, -sy),
        (-0.5, 0.866, -sx, -0.866, -0.5, -sy),
        (-0.5, -0.866, -sx, -0.866, 0.5, -sy),
        (0.5, 0.866, -sx, -0.866, 0.5, -sy),
        (0.5, -0.866, -sx, -0.866, -0.5, -sy),
    ]),
]

FRIEZE_GROUPS = [
    ("p1 (bg1)", [
        (1, 0, -sx-1, 0, 1, -sy),
        (1, 0, sx, 0, 1, sy),
    ]),
    ("p11m (bg2)", [
        (1, 0, -sx-1, 0, 1, -sy),
        (1, 0, sx, 0, -1, sy+1),
    ]),
    ("p11g (bg3)", [
        (1, 0, -sx-1, 0, 1, -sy-0.5),
        (-1, 0, sx+1, 0, -1, sy+0.5),
    ]),
    ("p2 (bg4)", [
        (1, 0, -sx-1, 0, 1, -sy-0.5),
        (-1, 0, sx+1, 0, 1, sy+0.5),
    ]),
    ("p2mm (bg5)", [
        (1, 0, -sx-1, 0, 1, -sy-0.5),
        (1, 0, sx-1, 0, -1, sy+0.5),
    ]),
    ("p2mg (bg6)", [
        (1, 0, -sx-1, 0, 1, -sy-0.5),
        (-1, 0, sx+1, 0, 1, sy-0.5),
        (1, 0, -sx-1, 0, -1, -sy+0.5),
        (-1, 0, sx+1, 0, -1, sy+0.5),
    ]),
    ("p2gm (bg7)", [
        (1, 0, -sx-2, 0, 1, -sy-0.5),
        (-1, 0, sx+2, 0, 1, sy-0.5),
        (1, 0, sx, 0, -1, sy+0.5),
        (-1, 0, -sx, 0, -1, -sy+0.5),
    ]),
]


def render_group(ax, name, transforms, shape=L_SHAPE):
    """Render a symmetry group by applying all transforms to the test shape."""
    colors = plt.cm.Set3(np.linspace(0, 1, len(transforms)))
    patches = []
    patch_colors = []

    for i, (a, b, c, d, e, f) in enumerate(transforms):
        transformed = apply_affine(shape, a, b, c, d, e, f)
        patches.append(Polygon(transformed, closed=True))
        patch_colors.append(colors[i])

    collection = PatchCollection(patches, alpha=0.6, edgecolors='black',
                                  linewidths=0.5)
    collection.set_facecolor(patch_colors)
    ax.add_collection(collection)
    ax.set_title(name, fontsize=8)
    ax.set_aspect('equal')
    ax.autoscale()
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    # Wallpaper groups: 17 panels
    fig1, axes1 = plt.subplots(3, 6, figsize=(18, 9))
    fig1.suptitle('17 Wallpaper Groups (SymNetG1-17)', fontsize=14)
    for i, (name, transforms) in enumerate(WALLPAPER_GROUPS):
        row, col = divmod(i, 6)
        render_group(axes1[row, col], name, transforms)
    # Hide unused panel
    axes1[2, 5].set_visible(False)
    fig1.tight_layout()
    fig1.savefig('/tmp/wallpaper_groups.png', dpi=150)
    print('Saved /tmp/wallpaper_groups.png')

    # Frieze groups: 7 panels
    fig2, axes2 = plt.subplots(1, 7, figsize=(21, 3))
    fig2.suptitle('7 Frieze Groups (SymBandG1-7)', fontsize=14)
    for i, (name, transforms) in enumerate(FRIEZE_GROUPS):
        render_group(axes2[i], name, transforms)
    fig2.tight_layout()
    fig2.savefig('/tmp/frieze_groups.png', dpi=150)
    print('Saved /tmp/frieze_groups.png')

    plt.show()


if __name__ == '__main__':
    main()
