# Variation Wishlist

Candidates from jwildfire_variation_catalog.txt, filtered by:
- GPU-compatible (has `yes` in gpu column)
- Trivial to medium cost
- Prioritized for: structure, symmetry, tiling, identifiable shapes

## Priority 1: Tiling & Structure

| Name | Cost | Params | Category | Status |
|---|---|---|---|---|
| boarders | cheap | 3 | TILING | DONE (combined with boarders2) |
| hypertile | trivial | 2 | TILING | DONE |
| cell | cheap | 1 | TILING | DONE |
| crackle | medium | 6 | STOCHASTIC | TODO (needs Voronoi noise) |
| dc_carpet | trivial | 1 | TILING | TODO (needs DC color support) |
| stripes | trivial | 2 | TILING | DONE |
| xtrb | trivial | 6 | TILING | SKIP (massive trilinear coord system, not trivial) |
| splitbrdr | cheap | 4 | TILING | DONE |

## Priority 2: Polar & Spiral

| Name | Cost | Params | Category | Status |
|---|---|---|---|---|
| whorl | cheap | 2 | POLAR | DONE |
| disc2 | cheap | 3 | FRACTAL | DONE |
| layered_spiral | cheap | 1 | ROTATIONAL | DONE |
| spiralwing | cheap | 0 | ROTATIONAL | DONE |
| vortex | cheap | 0 | ROTATIONAL | SKIP (iterative flow field, too expensive) |
| phoenix_julia | medium | 4 | POLAR | DONE |
| juliaq | medium | 2 | POLAR | DONE |

## Priority 3: Reflective & Symmetry

| Name | Cost | Params | Category | Status |
|---|---|---|---|---|
| pyramid | trivial | 0 | REFLECTIVE | SKIP (3D, needs z-coordinate) |
| flipcircle | trivial | 0 | REFLECTIVE | DONE |
| auger | cheap | 4 | REFLECTIVE | DONE |
| eclipse | cheap | 1 | REFLECTIVE | DONE |
| minkowskope | cheap | 6 | REFLECTIVE | DONE |
| wallpaper_js | cheap | 0 | REFLECTIVE | SKIP (covered by existing wallpaper variation) |

## Priority 4: Interesting Shapes

| Name | Cost | Params | Category | Status |
|---|---|---|---|---|
| flower | medium | 2 | ALGEBRAIC | DONE |
| blade | medium | 0 | ALGEBRAIC | DONE |
| crown | medium | 2 | ALGEBRAIC | SKIP (base shape, ignores input coordinates) |
| dragon | medium | 0 | STOCHASTIC | SKIP (stateful turtle, doesn't fit chaos game) |
| glynnia | medium | 0 | POLAR | DONE |
| collideoscope | medium | 2 | POLAR | DONE |
| lissajous | cheap | 7 | FRACTAL | DONE |
| ripple | medium | 8 | FRACTAL | DONE |

## Status
- 127 variations implemented (NUM_VARIATIONS)
- 2 remaining TODO: crackle (Voronoi noise), dc_carpet (DC color support)
- DC color support would unlock dc_carpet and other dc_ variations
- See `variation_catalog.md` for the full exhaustive inventory
