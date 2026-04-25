"""Crystallographic symmetry group element tables.

Single source of truth for both GPU shader generation and Python tools.
Each group is a list of affine transforms (a, b, c, d, e, f) where:
  x' = a*x + b*y + c
  y' = d*x + e*y + f

The c,f (translation) components use step=0.5 baked in.
Groups from McGregor & Watt "The Art of Graphics for the IBM PC"
pp 162-205, via JWildfire SymNetG/SymBandG implementations.
"""

# step/2 baked into translations
S = 0.25

# 17 wallpaper groups (plane symmetry)
WALLPAPER_GROUPS = [
    # 0: p1 — identity with translation
    [(1, 0, -S, 0, 1, -S),
     (1, 0, S, 0, 1, S)],

    # 1: p2 — 180° rotation
    [(1, 0, -S, 0, 1, -S),
     (-1, 0, S, 0, -1, S)],

    # 2: pm — vertical reflection
    [(1, 0, -S, 0, 1, -S),
     (1, 0, S, 0, -1, S)],

    # 3: pg — glide reflection
    [(1, 0, -S-S, 0, 1, -S-S),
     (-1, 0, S+S, 0, -1, S+S)],

    # 4: cm — centered reflection
    [(1, 0, -S, 0, 1, S),
     (1, 0, S, 0, -1, -S)],

    # 5: pmm — two perpendicular reflections
    [(1, 0, -S-2-S, 0, 1, S-1.5-S),
     (-1, 0, S-S, 0, -1, -S-0.5-S),
     (1, 0, -S+S, 0, -1, -S+0.5+S),
     (-1, 0, S+2+S, 0, 1, S-0.5+S)],

    # 6: pmg — reflection + glide
    [(1, 0, -S-2-S, 0, 1, S-0.5),
     (-1, 0, S-S, 0, -1, -S-0.5),
     (1, 0, -S+S, 0, -1, -S-0.5),
     (-1, 0, 2+S+S, 0, 1, S-0.5)],

    # 7: pgg — two glide reflections
    [(1, 0, -1-S, 0, -1, -S),
     (-1, 0, 1+S, 0, 1, S),
     (-1, 0, 1+S, 0, -1, -S),
     (1, 0, -1-S, 0, 1, S),
     (1, 0, -1-S, 0, -1, S),
     (-1, 0, 1+S, 0, 1, -S),
     (-1, 0, 1+S, 0, -1, S),
     (1, 0, -1-S, 0, 1, -S)],

    # 8: cmm — centered with two reflections
    [(1, 0, S, 0, -1, -S),
     (-1, 0, -S, 0, 1, S),
     (-1, 0, -S, 0, -1, -S),
     (1, 0, S, 0, 1, S)],

    # 9: p4 — 4-fold rotation
    [(-1, 0, -S, 0, -1, -S),
     (0, 1, -S, -1, 0, -S),
     (0, -1, S, 1, 0, S),
     (1, 0, S, 0, 1, S)],

    # 10: p4m — 4-fold rotation + reflections
    [(1, 0, S+S, 0, 1, S+S),
     (-1, 0, -S+S, 0, -1, -S+S),
     (0, 1, S+S, -1, 0, -S+S),
     (0, -1, -S+S, 1, 0, S+S),
     (-1, 0, -S-S, 0, 1, S-S),
     (0, -1, -S-S, -1, 0, -S-S),
     (0, 1, S-S, 1, 0, S-S),
     (1, 0, S-S, 0, -1, -S-S)],

    # 11: p4g — 4-fold rotation + glide
    [(1, 0, S, 0, 1, S),
     (-1, 0, -S, 0, -1, -S),
     (0, 1, S, -1, 0, -S),
     (0, -1, -S, 1, 0, S),
     (-1, 0, -S, 0, 1, S),
     (0, -1, -S, -1, 0, -S),
     (0, 1, S, 1, 0, S),
     (1, 0, S, 0, -1, -S)],

    # 12: p3 — 3-fold rotation
    [(1, 0, -S, 0, -1, -S),
     (-0.5, -0.866, -S, -0.866, 0.5, -S),
     (-0.5, 0.866, -S, 0.866, 0.5, -S),
     (1, 0, S, 0, -1, S),
     (-0.5, -0.866, S, -0.866, 0.5, S),
     (-0.5, 0.866, S, 0.866, 0.5, S)],

    # 13: p3m1 — 3-fold + reflections (type 1)
    [(1, 0, -S, 0, 1, -S),
     (1, 0, -S, 0, -1, -S),
     (-0.5, -0.866, -S, 0.866, -0.5, -S),
     (-0.5, 0.866, -S, -0.866, -0.5, -S),
     (-0.5, -0.866, -S, -0.866, 0.5, -S),
     (-0.5, 0.866, -S, 0.866, 0.5, -S),
     (1, 0, S, 0, 1, S),
     (1, 0, S, 0, -1, S),
     (-0.5, -0.866, S, 0.866, -0.5, S),
     (-0.5, 0.866, S, -0.866, -0.5, S),
     (-0.5, -0.866, S, -0.866, 0.5, S),
     (-0.5, 0.866, S, 0.866, 0.5, S)],

    # 14: p31m — 3-fold + reflections (type 2)
    [(0, 1, -S, -1, 0, -S),
     (0, -1, -S, -1, 0, -S),
     (0.866, -0.5, -S, 0.5, 0.866, -S),
     (0.866, 0.5, -S, 0.5, -0.866, -S),
     (-0.866, -0.5, -S, 0.5, -0.866, -S),
     (-0.866, 0.5, -S, 0.5, 0.866, -S),
     (0, 1, S, -1, 0, S),
     (0, -1, S, -1, 0, S),
     (0.866, -0.5, S, 0.5, 0.866, S),
     (0.866, 0.5, S, 0.5, -0.866, S),
     (-0.866, -0.5, S, 0.5, -0.866, S),
     (-0.866, 0.5, S, 0.5, 0.866, S)],

    # 15: p6 — 6-fold rotation
    [(1, 0, -S, 0, 1, -S),
     (0.5, -0.866, -S, 0.866, 0.5, -S),
     (-0.5, -0.866, -S, 0.866, -0.5, -S),
     (-1, 0, -S, 0, -1, -S),
     (-0.5, 0.866, -S, -0.866, -0.5, -S),
     (0.5, 0.866, -S, -0.866, 0.5, -S),
     (1, 0, S, 0, 1, S),
     (0.5, -0.866, S, 0.866, 0.5, S),
     (-0.5, -0.866, S, 0.866, -0.5, S),
     (-1, 0, S, 0, -1, S),
     (-0.5, 0.866, S, -0.866, -0.5, S),
     (0.5, 0.866, S, -0.866, 0.5, S)],

    # 16: p6m — 6-fold rotation + reflections
    [(1, 0, -S, 0, 1, -S),
     (1, 0, -S, 0, -1, -S),
     (0.5, -0.866, -S, 0.866, 0.5, -S),
     (0.5, 0.866, -S, 0.866, -0.5, -S),
     (-0.5, -0.866, -S, 0.866, -0.5, -S),
     (-0.5, 0.866, -S, 0.866, 0.5, -S),
     (-1, 0, -S, 0, -1, -S),
     (-1, 0, -S, 0, 1, -S),
     (-0.5, 0.866, -S, -0.866, -0.5, -S),
     (-0.5, -0.866, -S, -0.866, 0.5, -S),
     (0.5, 0.866, -S, -0.866, 0.5, -S),
     (0.5, -0.866, -S, -0.866, -0.5, -S)],
]

# 7 frieze groups (band symmetry)
FRIEZE_GROUPS = [
    # 0: p1 — translation only
    [(1, 0, -S-1, 0, 1, -S),
     (1, 0, S, 0, 1, S)],

    # 1: p11m — horizontal reflection
    [(1, 0, -S-1, 0, 1, -S),
     (1, 0, S, 0, -1, S+1)],

    # 2: p11g — glide reflection
    [(1, 0, -S-1, 0, 1, -S-0.5),
     (-1, 0, S+1, 0, -1, S+0.5)],

    # 3: p2 — 180° rotation
    [(1, 0, -S-1, 0, 1, -S-0.5),
     (-1, 0, S+1, 0, 1, S+0.5)],

    # 4: p2mm — two reflections
    [(1, 0, -S-1, 0, 1, -S-0.5),
     (1, 0, S-1, 0, -1, S+0.5)],

    # 5: p2mg — reflection + glide
    [(1, 0, -S-1, 0, 1, -S-0.5),
     (-1, 0, S+1, 0, 1, S-0.5),
     (1, 0, -S-1, 0, -1, -S+0.5),
     (-1, 0, S+1, 0, -1, S+0.5)],

    # 6: p2gm — glide + reflection
    [(1, 0, -S-2, 0, 1, -S-0.5),
     (-1, 0, S+2, 0, 1, S-0.5),
     (1, 0, S, 0, -1, S+0.5),
     (-1, 0, -S, 0, -1, -S+0.5)],
]

# Group names for display
WALLPAPER_NAMES = [
    'p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm',
    'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m',
]

FRIEZE_NAMES = [
    'p1', 'p11m', 'p11g', 'p2', 'p2mm', 'p2mg', 'p2gm',
]


def generate_glsl() -> str:
    """Generate GLSL const arrays for the symmetry group transforms.

    Returns a string to be injected into the shader source at
    the // {{SYMMETRY_GROUPS}} placeholder.
    """
    lines = ['// Auto-generated from _symmetry_groups.py — do not edit']

    # Wallpaper group element counts
    wp_counts = [len(g) for g in WALLPAPER_GROUPS]
    lines.append(f'const int WALLPAPER_COUNTS[17] = int[]({", ".join(str(c) for c in wp_counts)});')

    # Wallpaper transforms: flat array, offset by group
    lines.append('')
    lines.append('// Wallpaper group transforms (a, b, c, d, e, f per element)')
    wp_flat = []
    wp_offsets = []
    offset = 0
    for g in WALLPAPER_GROUPS:
        wp_offsets.append(offset)
        for t in g:
            wp_flat.extend(t)
        offset += len(g)
    lines.append(f'const int WALLPAPER_OFFSETS[17] = int[]({", ".join(str(o) for o in wp_offsets)});')
    floats = ', '.join(f'{v:.4f}' for v in wp_flat)
    lines.append(f'const float WALLPAPER_DATA[{len(wp_flat)}] = float[]({floats});')

    # Frieze group element counts
    fr_counts = [len(g) for g in FRIEZE_GROUPS]
    lines.append('')
    lines.append(f'const int FRIEZE_COUNTS[7] = int[]({", ".join(str(c) for c in fr_counts)});')

    fr_flat = []
    fr_offsets = []
    offset = 0
    for g in FRIEZE_GROUPS:
        fr_offsets.append(offset)
        for t in g:
            fr_flat.extend(t)
        offset += len(g)
    lines.append(f'const int FRIEZE_OFFSETS[7] = int[]({", ".join(str(o) for o in fr_offsets)});')
    floats = ', '.join(f'{v:.4f}' for v in fr_flat)
    lines.append(f'const float FRIEZE_DATA[{len(fr_flat)}] = float[]({floats});')

    return '\n'.join(lines)
