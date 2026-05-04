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
| xtrb | trivial | 6 | TILING | TODO |
| splitbrdr | cheap | 4 | TILING | TODO |

## Priority 2: Polar & Spiral

| Name | Cost | Params | Category | Status |
|---|---|---|---|---|
| whorl | cheap | 2 | POLAR | DONE |
| disc2 | cheap | 3 | FRACTAL | DONE |
| layered_spiral | cheap | 1 | ROTATIONAL | DONE |
| spiralwing | cheap | 0 | ROTATIONAL | DONE |
| vortex | cheap | 0 | ROTATIONAL | SKIP (iterative flow field, too expensive) |
| phoenix_julia | medium | 4 | POLAR | TODO |
| juliaq | medium | 2 | POLAR | TODO |

## Priority 3: Reflective & Symmetry

| Name | Cost | Params | Category | Status |
|---|---|---|---|---|
| pyramid | trivial | 0 | REFLECTIVE | SKIP (3D, needs z-coordinate) |
| flipcircle | trivial | 0 | REFLECTIVE | DONE |
| auger | cheap | 4 | REFLECTIVE | DONE |
| eclipse | cheap | 1 | REFLECTIVE | DONE |
| minkowskope | cheap | 0 | REFLECTIVE | TODO |
| wallpaper_js | cheap | 0 | REFLECTIVE | TODO |

## Priority 4: Interesting Shapes

| Name | Cost | Params | Category | Status |
|---|---|---|---|---|
| flower | medium | 2 | ALGEBRAIC | DONE |
| blade | medium | 0 | ALGEBRAIC | DONE |
| crown | medium | 2 | ALGEBRAIC | TODO (complex number math) |
| dragon | medium | 0 | STOCHASTIC | TODO |
| glynnia | medium | 0 | POLAR | TODO |
| collideoscope | medium | 2 | POLAR | DONE |
| lissajous | cheap | 7 | FRACTAL | DONE |
| ripple | medium | 8 | FRACTAL | DONE |

## Status
- 70 variations currently implemented (was 54)
- 16 implemented this session
- ~14 remaining TODO candidates
- DC color support would unlock dc_carpet and other dc_ variations
