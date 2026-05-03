# Variation Wishlist

Candidates from jwildfire_variation_catalog.txt, filtered by:
- GPU-compatible (has `yes` in gpu column)
- Trivial to medium cost
- Not already implemented (54 current variations)
- Prioritized for: structure, symmetry, tiling, identifiable shapes

## Priority 1: Tiling & Structure
Best bang for buck — produce the structured, tiled visuals we want.

| Name | Cost | Params | Category | Notes |
|---|---|---|---|---|
| boarders | cheap | 0 | TILING | Creates bordered rectangular structures |
| boarders2 | cheap | 3 | TILING | Parameterized version |
| hypertile | trivial | 3 | TILING | Hyperbolic tiling — Escher-like |
| cell | cheap | 1 | TILING | Voronoi-like cells |
| crackle | medium | 6 | STOCHASTIC | Organic cell/crack patterns |
| dc_carpet | trivial | 1 | TILING | Sierpinski carpet tiling |
| stripes | trivial | 2 | TILING | Stripe patterns |
| xtrb | trivial | 6 | TILING | Complex tiling |
| splitbrdr | cheap | 4 | TILING | Split bordered tiling |

## Priority 2: Polar & Spiral
Enhance rotational/spiral structures.

| Name | Cost | Params | Category | Notes |
|---|---|---|---|---|
| whorl | cheap | 0 | POLAR | Spiral distortion |
| disc | medium | 0 | POLAR | Disc mapping |
| disc2 | cheap | 0 | FRACTAL | Variant disc |
| layered_spiral | cheap | 0 | ROTATIONAL | Multi-layer spirals |
| spiralwing | cheap | 0 | ROTATIONAL | Wing-like spirals |
| vortex | cheap | 0 | ROTATIONAL | Vortex distortion |
| phoenix_julia | medium | 4 | POLAR | Phoenix variant of Julia set |
| juliaq | medium | 2 | POLAR | Quaternion Julia |

## Priority 3: Reflective & Symmetry
Boost symmetry and mirror structures.

| Name | Cost | Params | Category | Notes |
|---|---|---|---|---|
| pyramid | trivial | 0 | REFLECTIVE | Pyramid reflection |
| flipcircle | trivial | 0 | REFLECTIVE | Circular flip |
| auger | cheap | 4 | REFLECTIVE | Ridged/augmented structures |
| eclipse | cheap | 1 | REFLECTIVE | Eclipse-like masking |
| minkowskope | cheap | 0 | REFLECTIVE | Minkowski space reflection |
| wallpaper_js | cheap | 0 | REFLECTIVE | Wallpaper group symmetry |

## Priority 4: Interesting Shapes
Distinctive visual character.

| Name | Cost | Params | Category | Notes |
|---|---|---|---|---|
| flower | medium | 4 | ALGEBRAIC | Petal/flower structures |
| blade | medium | 0 | ALGEBRAIC | Sharp radial blades |
| crown | medium | 0 | ALGEBRAIC | Crown/ring structures |
| dragon | medium | 0 | STOCHASTIC | Dragon curve |
| glynnia | medium | 0 | POLAR | Complex Glynn attractor |
| collideoscope | medium | 2 | POLAR | Kaleidoscopic collision |
| lissajous | cheap | 4 | FRACTAL | Lissajous curve patterns |
| ripple | medium | 0 | FRACTAL | Ripple/interference |

## Status
- 54 variations currently implemented
- ~40 candidates identified above
- Implement in priority order, test each batch visually
