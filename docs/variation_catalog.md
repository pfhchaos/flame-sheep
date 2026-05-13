# Flame Fractal Variation Catalog

Complete inventory of variations from JWildfire, Apophysis 7X, and Apophysis plugins,
cross-referenced against flame-sheep implementation status.

Generated 2026-05-10.

## Legend

- **DONE**: Implemented in flame-sheep (index shown in parens)
- **TODO**: Candidate for implementation
- **SKIP**: Not suitable (reason given)
- **NEEDS_DC**: Requires direct-color pipeline support first
- **GPU**: Has GPU code in JWildfire (SupportsGPU interface)

## Sources

- **flam3**: Original Scott Draves flam3 library (variations 0-48)
- **JWF**: JWildfire (Java, 670+ variations)
- **Apo**: Apophysis 7X (Delphi, 29 built-in + 47 registered variations)
- **ApoPlugin**: Apophysis plugin DLLs (C, ~30 plugins)

## Summary

| Metric | Count |
|--------|-------|
| flame-sheep implemented | 127 |
| JWildfire total variations | ~670 |
| 2D candidates (from prior analysis) | 412 |
| 3D-only (SKIP) | ~181 |
| DC/GLSL/simulation (SKIP or NEEDS_DC) | ~150 |

---

## Catalog

Sorted alphabetically. Every variation from JWildfire and Apophysis is listed.
For JWildfire variations, cost estimates come from the existing analysis in
`jwildfire_variation_catalog.txt`. GPU column indicates JWildfire SupportsGPU.

### Core flam3 variations (0-48)

These are the original Scott Draves variations, also present in Apophysis and JWildfire.

| Name | Source | GPU | Status | Notes |
|------|--------|-----|--------|-------|
| linear | flam3 | yes | DONE (0) | identity transform |
| sinusoidal | flam3 | yes | DONE (1) | sin(x), sin(y) |
| spherical | flam3 | yes | DONE (2) | 1/r^2 inversion |
| swirl | flam3 | yes | DONE (3) | sin/cos of r^2 rotation |
| horseshoe | flam3 | yes | DONE (4) | (x-y)(x+y)/r, 2xy/r |
| polar | flam3 | yes | DONE (5) | theta/pi, r-1 |
| handkerchief | flam3 | yes | DONE (6) | r*sin(theta+r), r*cos(theta-r) |
| heart | flam3 | yes | DONE (7) | r*sin(theta*r), -r*cos(theta*r) |
| disk | flam3 | yes | DONE (8) | theta/pi * sin(pi*r) |
| spiral | flam3 | yes | DONE (9) | (cos(theta)+sin(r))/r |
| hyperbolic | flam3 | yes | DONE (10) | sin(theta)/r, r*cos(theta) |
| diamond | flam3 | yes | DONE (11) | sin(theta)*cos(r), cos(theta)*sin(r) |
| ex | flam3 | yes | DONE (12) | p0^3*sin + p1^3*cos, power-of-angle |
| julia | flam3 | yes | DONE (13) | sqrt(r) with random +/- theta/2 |
| bent | flam3 | yes | DONE (14) | conditional fold on axes |
| waves | flam3 | yes | DONE (15) | affine-coeff driven sinusoidal offset |
| fisheye | flam3 | yes | DONE (16) | 2/(r+1) scaling, swap x/y |
| popcorn | flam3 | yes | DONE (17) | affine-coeff trig perturbation |
| exponential | flam3 | yes | DONE (18) | exp(x-1)*cos/sin(pi*y) |
| power | flam3 | yes | DONE (19) | r^sin(theta) scaling |
| cosine | flam3 | yes | DONE (20) | cos(pi*x)*cosh(y) |
| rings | flam3 | yes | DONE (21) | radial modular ring, affine-coeff |
| fan | flam3 | yes | DONE (22) | angular fan fold, affine-coeff |
| blob | flam3 | yes | DONE (23) | radial blob distortion, 3 params |
| pdj | flam3 | yes | DONE (24) | sin/cos of a,b,c,d scaled coords |
| fan2 | flam3 | yes | DONE (25) | parametric angular fan fold |
| rings2 | flam3 | yes | DONE (26) | parametric radial rings |
| eyefish | flam3 | yes | DONE (27) | 2/(r+1) fisheye without xy swap |
| bubble | flam3 | yes | DONE (28) | 4/(r^2+4) spherical projection |
| cylinder | flam3 | yes | DONE (29) | sin(x), y passthrough |
| blur | flam3 | yes | DONE (75) | uniform random disc |
| gaussian_blur | flam3 | yes | DONE (76) | gaussian random disc |
| radial_blur | flam3 | yes | DONE (77) | spin + zoom blur along angle |
| perspective | flam3 | yes | DONE (78) | perspective projection |
| noise | flam3 | yes | DONE (80) | random polar scatter |
| curl | flam3 | yes | DONE (37) | Moebius-like curl distortion |
| rectangles | flam3 | yes | DONE (38) | rectangular grid fold |
| arch | flam3 | yes | DONE (83) | random arch scatter |
| tangent | flam3 | yes | DONE (34) | tan(x)/cos(y) |
| square | flam3 | yes | DONE (91) | uniform random square |
| rays | flam3 | yes | DONE (85) | radial ray pattern |
| blade | flam3 | yes | DONE (60) | random blade scatter |
| secant2 | flam3 | yes | DONE (81) | improved secant |
| twintrian | flam3 | yes | DONE (93) | twin triangular scatter |
| cross | flam3 | yes | DONE (35) | 1/(x^2-y^2)^2 scaling |
| disc2 | flam3 | yes | DONE (58) | parametric disc with twist |
| super_shape | flam3 | yes | DONE (79) | superformula shape |
| flower | flam3 | yes | DONE (59) | petal pattern |
| conic | flam3 | yes | DONE (86) | conic section |
| parabola | flam3 | yes | DONE (84) | sin/cos of r for height/width |
| bent2 | flam3 | yes | DONE (99) | parametric bent |
| splits | flam3 | yes | DONE (30) | x/y split offset |
| stripes | flam3 | yes | DONE (67) | stripe warp pattern |
| wedge | flam3 | yes | DONE (95) | angular wedge fold |
| wedge_julia | flam3 | yes | DONE (94) | julia with wedge fold |
| wedge_sph | flam3 | yes | DONE (96) | spherical with wedge fold |
| pie | flam3 | yes | DONE (82) | pie slice scatter |
| ngon | flam3 | yes | DONE (49) | n-gon angular scaling |
| oscilloscope | flam3 | yes | DONE (89) | oscilloscope wave pattern |
| escher | flam3 | yes | DONE (87) | Escher-like tiling transform |
| loonie | flam3 | yes | DONE (50) | circle inversion variant |
| scry | flam3 | yes | DONE (51) | reciprocal r^2 variant |
| foci | flam3 | yes | DONE (105) | conformal focus mapping |
| modulus | flam3 | yes | DONE (98) | modular wrap |
| lazysusan | flam3 | yes | DONE (97) | lazy susan rotation |
| polar2 | flam3 | yes | DONE (104) | polar variant |
| bipolar | flam3 | yes | DONE (100) | bipolar coordinate transform |
| butterfly | flam3 | yes | DONE (36) | butterfly curve |
| cell | flam3 | yes | DONE (56) | cellular grid fold |
| cpow | flam3 | yes | DONE (48) | complex power |
| curve | flam3 | yes | DONE (92) | parametric curve offset |
| edisc | flam3 | yes | DONE (90) | elliptic disc |
| elliptic | flam3 | yes | DONE (88) | elliptic integral transform |
| flux | flam3 | yes | DONE (101) | flux spread mapping |
| separation | flam3 | yes | DONE (103) | separated axis fold |
| split | flam3 | yes | DONE (102) | sign-based split |
| popcorn2 | flam3 | yes | DONE (106) | parametric popcorn |
| waves2 | flam3 | yes | DONE (74) | parametric waves (different formula) |
| julian | flam3 | yes | DONE (32) | julian n-fold symmetry |
| juliascope | flam3 | yes | DONE (33) | julia with scope reflection |
| log | JWF | no | DONE (121) | complex logarithm |
| exp | JWF | yes | DONE (120) | complex exponential |
| secant | flam3 | yes | DONE (107) | original secant (pre-secant2) |

### Complex trig variations (flam3/JWildfire)

| Name | Source | GPU | Status | Notes |
|------|--------|-----|--------|-------|
| sin | flam3 | yes | DONE (108) | complex sin |
| cos | flam3 | yes | DONE (109) | complex cos |
| tan | flam3 | yes | DONE (110) | complex tan |
| sec | flam3 | yes | DONE (111) | complex sec |
| csc | flam3 | yes | DONE (112) | complex csc |
| cot | flam3 | yes | DONE (113) | complex cot |
| sinh | flam3 | yes | DONE (114) | complex sinh |
| cosh | flam3 | yes | DONE (115) | complex cosh |
| tanh | flam3 | yes | DONE (116) | complex tanh |
| sech | flam3 | yes | DONE (117) | complex sech |
| csch | flam3 | yes | DONE (118) | complex csch |
| coth | flam3 | yes | DONE (119) | complex coth |

### Extended flame-sheep variations

| Name | Source | GPU | Status | Notes |
|------|--------|-----|--------|-------|
| cloverleaf | JWF | yes | DONE (31) | clover leaf curve shape |
| checks | JWF | yes | DONE (39) | checkerboard tile pattern |
| hex_modulus | JWF | yes | DONE (40) | hexagonal modular grid |
| kaleidoscope | JWF | yes | DONE (41) | n-fold kaleidoscope |
| icon | custom | -- | DONE (42) | icon attractor (degree/lambda/alpha/beta/gamma/omega) |
| sattractor | custom | -- | DONE (43) | strange attractor |
| wallpaper | custom | -- | DONE (44) | wallpaper group symmetry |
| frieze | custom | -- | DONE (45) | frieze group symmetry |
| rings3 | JWF | yes | DONE (46) | parametric rings with n folds |
| mobius | JWF | yes | DONE (47) | Moebius transformation (8 params) |
| epispiral | JWF | yes | DONE (52) | epicycloid spiral |
| waves3 | JWF | yes | DONE (53) | waves with cross-frequency modulation |
| boarders | JWF | yes | DONE (54) | cell border pattern |
| hypertile | JWF | yes | DONE (55) | hyperbolic tiling |
| whorl | JWF | yes | DONE (57) | inside/outside whorl rotation |
| spiralwing | JWF | yes | DONE (61) | spiral wing pattern |
| collideoscope | JWF | yes | DONE (62) | colliding kaleidoscope |
| auger | JWF | yes | DONE (63) | auger drill pattern |
| flipcircle | JWF | yes | DONE (64) | circle-based flip reflection |
| eclipse | JWF | yes | DONE (65) | eclipse shift |
| layered_spiral | JWF | yes | DONE (66) | layered spiral rings |
| lissajous | JWF | yes | DONE (68) | Lissajous curve mapping |
| ripple | JWF | yes | DONE (69) | ripple wave pattern |
| waves_param | custom | -- | DONE (70) | waves with explicit params (not from affine) |
| popcorn_param | custom | -- | DONE (71) | popcorn with explicit params |
| rings_param | custom | -- | DONE (72) | rings with explicit param |
| fan_param | custom | -- | DONE (73) | fan with explicit params |
| super_shape | flam3 | yes | DONE (79) | superformula |
| splitbrdr | JWF | yes | DONE (122) | split border tiling |
| phoenix_julia | JWF | yes | DONE (123) | phoenix julia with distortion |
| juliaq | JWF | yes | DONE (124) | julia quotient |
| minkowskope | JWF | yes | DONE (125) | Minkowski metric kaleidoscope |
| glynnia | JWF | yes | DONE (126) | Glynn attractor variant |

---

## JWildfire / Apophysis Variations -- Full Catalog

### A

| Name | Source | GPU | Status | Notes |
|------|--------|-----|--------|-------|
| acosech | JWF | yes | TODO | inverse hyperbolic cosecant, trivial |
| acosh | JWF | yes | TODO | inverse hyperbolic cosine, trivial |
| acoth | JWF | yes | TODO | inverse hyperbolic cotangent, trivial |
| affine3D | JWF | yes | SKIP - 3D | 3D affine transform |
| anamorphcyl | JWF | yes | TODO | anamorphic cylinder projection, trivial |
| apocarpet_js | JWF | yes | TODO | Apollonian carpet IFS, cheap |
| apollony | JWF | yes | TODO | Apollony gasket IFS, stochastic |
| arch | JWF | yes | DONE (83) | random arch scatter |
| arcsinh | JWF | yes | TODO | inverse sinh, trivial |
| arcsech2 | JWF | yes | TODO | inverse sech variant, trivial |
| arctanh | JWF | yes | TODO | inverse tanh, trivial |
| arctruchet | JWF | yes | TODO | arc-based truchet pattern, medium |
| asteria | JWF | yes | TODO | star/pentagon fold, medium |
| atan | JWF | yes | TODO | atan-based distortion, 2 params, medium |
| atan2_spirals | JWF | yes | TODO | atan2 spiral patterns, 14 params, medium |
| attractorflow | JWF | yes | SKIP - 3D | 3D attractor flow, stateful |
| auger | JWF | yes | DONE (63) | auger drill pattern |

### B

| Name | Source | GPU | Status | Notes |
|------|--------|-----|--------|-------|
| bCollide | JWF | yes | TODO | hyperbolic collide, heavy |
| bMod | JWF | yes | TODO | hyperbolic modulus, heavy |
| bSwirl | JWF | yes | TODO | hyperbolic swirl, heavy |
| bTransform | JWF | yes | TODO | hyperbolic transform, heavy |
| barycentroid | JWF | yes | TODO | barycentric coordinate map, cheap |
| bbox3D | JWF | yes | SKIP - 3D | 3D bounding box |
| bent | JWF | yes | DONE (14) | conditional axis fold |
| bent2 | JWF | yes | DONE (99) | parametric bent |
| bi_linear | JWF | yes | TODO | bilinear mapping, trivial |
| bipolar | JWF/Apo | yes | DONE (100) | bipolar coordinate transform |
| bipolar2 | JWF | yes | TODO | extended bipolar with 9 params, medium |
| blade | JWF | yes | DONE (60) | random blade scatter |
| blade3D | JWF | yes | SKIP - 3D | 3D blade |
| blob | JWF | yes | DONE (23) | radial blob distortion |
| blob3D | JWF | yes | SKIP - 3D | 3D blob |
| blocky | JWF | yes | TODO | blocky tiling distortion, heavy |
| blur | JWF | yes | DONE (75) | uniform random disc |
| blur3D | JWF/Apo | yes | SKIP - 3D | 3D blur |
| blur_circle | JWF/Apo | yes | TODO | circle-bounded blur, cheap |
| blur_linear | JWF | yes | TODO | directional linear blur |
| blur_pixelize | JWF/Apo | yes | TODO | pixelated blur, cheap |
| blur_zoom | JWF/Apo | yes | TODO | zoom blur |
| boarders | JWF | yes | DONE (54) | cell border pattern |
| boarders2 | JWF | yes | TODO | extended boarders with params, cheap |
| box3D | JWF | yes | SKIP - 3D | 3D box shape |
| boxfold | JWF | yes | SKIP - 3D | 3D box fold for mandelbox |
| brownian_js | JWF | no | SKIP - STATEFUL | Brownian motion, stateful random walk |
| brushstroke_wf | JWF | no | SKIP - IMAGE | brush stroke effect, needs bitmap |
| bsplit | JWF | yes | TODO | basic split, cheap |
| bubble | JWF | yes | DONE (28) | 4/(r^2+4) spherical projection |
| bubble2 | JWF | yes | SKIP - 3D | bubble with z component |
| bubble_wf | JWF | yes | SKIP - 3D | bubble with z output |
| bubbleT3D | JWF | yes | SKIP - 3D | bubble with 3D torus |
| bulge | JWF | yes | TODO | radial bulge distortion, heavy |
| busybrad | JWF | no | TODO | complex tiling, 13 params, expensive |
| butterfly | JWF | yes | DONE (36) | butterfly curve |
| butterfly3D | JWF | yes | SKIP - 3D | 3D butterfly |
| butterfly_fay | JWF | yes | TODO | Fay butterfly curve variant |
| bwrands | JWF | yes | TODO | random boundary wraps, 12 params, medium |
| bwraps7 | JWF | yes | TODO | boundary wraps v7, 5 params, medium |

### C

| Name | Source | GPU | Status | Notes |
|------|--------|-----|--------|-------|
| c_symmetry | JWF | yes | TODO | complex plane symmetry, medium |
| c_var | JWF | no | TODO | complex variable distortion, trivial |
| cactusglobe | JWF | yes | SKIP - 3D | 3D cactus globe shape |
| camouflage | JWF | yes | SKIP - BASE_SHAPE | procedural camouflage pattern |
| cannabiscurve_wf | JWF | yes | SKIP - BASE_SHAPE | cannabis leaf curve shape |
| cappedcone3D | JWF | yes | SKIP - 3D | 3D capped cone SDF |
| cappedtorus3D | JWF | yes | SKIP - 3D | 3D capped torus SDF |
| capsule3D | JWF | yes | SKIP - 3D | 3D capsule SDF |
| cardioid | JWF | yes | TODO | cardioid curve, medium |
| cell | JWF | yes | DONE (56) | cellular grid fold |
| cell2 | JWF | no | TODO | extended cell with 16 params, cheap |
| cell3D | JWF | yes | SKIP - 3D | 3D cell |
| chaoscubes | JWF | yes | SKIP - 3D | 3D chaos cubes |
| checkerboard_wf | JWF | yes | SKIP - BASE_SHAPE | checkerboard base shape |
| checks | JWF | yes | DONE (39) | checkerboard tile pattern |
| chrysanthemum | JWF | yes | SKIP - BASE_SHAPE | chrysanthemum curve shape |
| chunk | JWF | yes | TODO | angular chunk fold, 7 params, cheap |
| circleLinear | JWF | yes | TODO | circle-linear hybrid tiling, cheap |
| circleRand | JWF | yes | TODO | random circle packing, heavy |
| circleTrans1 | JWF | yes | TODO | circle transform, 5 params, cheap |
| circleblur | JWF | yes | TODO | circular blur scatter |
| circlecrop | JWF | yes | SKIP - CROP | circle crop utility |
| circlesplit | JWF | yes | TODO | circle split distortion, medium |
| circlize | JWF | yes | TODO | square-to-circle mapping, cheap |
| circlize2 | JWF | yes | TODO | circlize variant, cheap |
| circular | JWF | yes | TODO | circular distortion, 2 params, medium |
| circular2 | JWF | yes | TODO | circular variant, 4 params, medium |
| circus | JWF | yes | TODO | circus circle fold, medium |
| clifford_js | JWF | yes | TODO | Clifford attractor, heavy |
| cloverleaf_wf | JWF | yes | DONE (31) | clover leaf curve (JWF name) |
| collideoscope | JWF/Apo | yes | DONE (62) | colliding kaleidoscope |
| colordomain | JWF | no | SKIP - DC | direct color domain coloring |
| colormap_wf | JWF | no | SKIP - IMAGE | external colormap |
| colorscale_wf | JWF | yes | SKIP - 3D | color scaling, 3D |
| combimirror | JWF | yes | SKIP - 3D | 3D combination mirror |
| complex | JWF | yes | TODO | generic complex function, 64 params, expensive |
| cone | JWF | yes | SKIP - 3D | 3D cone |
| cone3D | JWF | yes | SKIP - 3D | 3D cone SDF |
| conicalSpiral | JWF | yes | SKIP - 3D | 3D conical spiral |
| conic | JWF | yes | DONE (86) | conic section |
| corners | JWF | yes | TODO | corner distortion, 9 params, heavy |
| cos | JWF | yes | DONE (109) | complex cosine |
| cos2_bs | JWF | yes | TODO | cos with extra params, expensive |
| cosh | JWF | yes | DONE (115) | complex cosh |
| cosh2_bs | JWF | yes | TODO | cosh with extra params, expensive |
| cosine | JWF | yes | DONE (20) | cos(pi*x)*cosh(y) |
| cosq | JWF | yes | SKIP - 3D | quaternion cos |
| cot | JWF | yes | DONE (113) | complex cot |
| cot2_bs | JWF | yes | TODO | cot with extra params, expensive |
| coth | JWF | yes | DONE (119) | complex coth |
| coth2_bs | JWF | yes | TODO | coth with extra params, expensive |
| cothq | JWF | yes | SKIP - 3D | quaternion coth |
| cotq | JWF | yes | SKIP - 3D | quaternion cot |
| cpow | JWF/Apo | yes | DONE (48) | complex power |
| cpow2 | JWF | yes | TODO | complex power variant 2, medium |
| cpow3 | JWF | yes | TODO | complex power variant 3, heavy |
| cpow3_wf | JWF | yes | TODO | cpow3 with extra params, heavy |
| crob | JWF | yes | TODO | complex rotation/orbit |
| crop | JWF | yes | SKIP - CROP | rectangular crop utility |
| crop_box | JWF | yes | SKIP - CROP | box crop |
| crop_cross | JWF | yes | SKIP - CROP | cross-shaped crop |
| crop_polygon | JWF | yes | SKIP - CROP | polygon crop |
| crop_rhombus | JWF | yes | SKIP - CROP | rhombus crop |
| crop_stars | JWF | yes | SKIP - CROP | star-shaped crop |
| crop_trapezoid | JWF | yes | SKIP - CROP | trapezoid crop |
| crop_triangle | JWF | no | SKIP - CROP | triangle crop |
| crop_vesica | JWF | yes | SKIP - CROP | vesica crop |
| crop_x | JWF | yes | SKIP - CROP | x-axis crop |
| cross | JWF | yes | DONE (35) | 1/(x^2-y^2)^2 scaling |
| crown_js | JWF | yes | SKIP - 3D/BASE_SHAPE | crown shape |
| csc | JWF | yes | DONE (112) | complex csc |
| csc2_bs | JWF | yes | TODO | csc with extra params, expensive |
| csc_squared | JWF | yes | TODO | csc squared variant, 7 params, medium |
| csch | JWF | yes | DONE (118) | complex csch |
| csch2_bs | JWF | yes | TODO | csch with extra params, expensive |
| cschq | JWF | yes | SKIP - 3D | quaternion csch |
| cscq | JWF | yes | SKIP - 3D | quaternion csc |
| csin | JWF | yes | TODO | complex sin variant, trivial |
| cubic3D | JWF | yes | SKIP - 3D | 3D cubic distortion |
| cubicLattice_3D | JWF | yes | SKIP - 3D | 3D cubic lattice |
| curl | JWF/Apo | yes | DONE (37) | Moebius-like curl |
| curl3D | JWF/Apo | yes | SKIP - 3D | 3D curl |
| curl_sp | JWF/ApoPlugin | yes | SKIP - 3D | curl with special 3D params |
| curliecue | JWF | yes | SKIP - BASE_SHAPE | curlicue fractal, stateful |
| curliecue2 | JWF | yes | SKIP - BASE_SHAPE | curlicue variant |
| curve | JWF | yes | DONE (92) | parametric curve offset |
| custom_wf | JWF | yes | SKIP - CUSTOM | user-defined formula |
| custom_wf_full | JWF | yes | SKIP - CUSTOM | full custom formula |
| cylinder | JWF | yes | DONE (29) | sin(x), y passthrough |
| cylinder2 | JWF | yes | TODO | cylinder variant, cheap |
| cylinder3D | JWF | yes | SKIP - 3D | 3D cylinder SDF |
| cylinder_apo | JWF | yes | SKIP - 3D | Apophysis cylinder 3D compat |

### Cut/DC/GLSL variations

These are direct-color or shader-based variations that set the color directly
rather than transforming coordinates. They require DC pipeline support.

| Name | Source | GPU | Status | Notes |
|------|--------|-----|--------|-------|
| cut_2ewangtile | JWF | no | SKIP - DC/GLSL | Wang tile pattern |
| cut_alientext | JWF | no | SKIP - DC/GLSL | alien text pattern |
| cut_apollonian | JWF | no | SKIP - DC/GLSL | Apollonian gasket |
| cut_booleans | JWF | no | SKIP - DC/GLSL | boolean ops pattern |
| cut_bricks | JWF | no | SKIP - DC/GLSL | brick pattern |
| cut_btree | JWF | no | SKIP - DC/GLSL | binary tree |
| cut_btruchet | JWF | no | SKIP - DC/GLSL | basic truchet |
| cut_c | JWF | no | SKIP - DC/GLSL | complex plane coloring |
| cut_celtic | JWF | no | SKIP - DC/GLSL | Celtic knot pattern |
| cut_chains | JWF | no | SKIP - DC/GLSL | chain pattern |
| cut_circdes | JWF | no | SKIP - DC/GLSL | circle design |
| cut_fractal | JWF | no | SKIP - DC/GLSL | fractal coloring |
| cut_fun | JWF | no | SKIP - DC/GLSL | fun pattern |
| cut_glypho | JWF | no | SKIP - DC/GLSL | glyph pattern |
| cut_hexdots | JWF | no | SKIP - DC/GLSL | hex dot pattern |
| cut_hextruchetflow | JWF | no | SKIP - DC/GLSL | hex truchet flow |
| cut_jigsaw | JWF | no | SKIP - DC/GLSL | jigsaw pattern |
| cut_kaleido | JWF | no | SKIP - DC/GLSL | kaleidoscope coloring |
| cut_kleinian | JWF | no | SKIP - DC/GLSL | Kleinian group |
| cut_magfield | JWF | no | SKIP - DC/GLSL | magnetic field lines |
| cut_mandala | JWF | no | SKIP - DC/GLSL | mandala pattern |
| cut_metaballs | JWF | no | SKIP - DC/GLSL | metaball pattern |
| cut_pattern | JWF | no | SKIP - DC/GLSL | generic pattern |
| cut_randomtile | JWF | no | SKIP - DC/GLSL | random tile |
| cut_rgrid | JWF | no | SKIP - DC/GLSL | rotated grid |
| cut_sincos | JWF | no | SKIP - DC/GLSL | sin/cos pattern |
| cut_snowflake | JWF | no | SKIP - DC/GLSL | snowflake |
| cut_spiral | JWF | no | SKIP - DC/GLSL | spiral coloring |
| cut_spiralcb | JWF | no | SKIP - DC/GLSL | spiral checkerboard |
| cut_spots | JWF | no | SKIP - DC/GLSL | spot pattern |
| cut_sqcir | JWF | no | SKIP - DC/GLSL | square/circle transition |
| cut_sqsplits | JWF | no | SKIP - DC/GLSL | square split pattern |
| cut_swarp | JWF | no | SKIP - DC/GLSL | warped symmetry |
| cut_tileillusion | JWF | no | SKIP - DC/GLSL | tile illusion |
| cut_triantess | JWF | no | SKIP - DC/GLSL | triangle tessellation |
| cut_triskel | JWF | no | SKIP - DC/GLSL | triskelion |
| cut_truchet | JWF | no | SKIP - DC/GLSL | truchet coloring |
| cut_truchetweaving | JWF | no | SKIP - DC/GLSL | truchet weaving |
| cut_tstruchet | JWF | no | SKIP - DC/GLSL | tri-scale truchet |
| cut_vasarely | JWF | no | SKIP - DC/GLSL | Vasarely pattern |
| cut_web | JWF | no | SKIP - DC/GLSL | web pattern |
| cut_wood | JWF | no | SKIP - DC/GLSL | wood grain |
| cut_x | JWF | no | SKIP - DC/GLSL | x cut pattern |
| cut_yuebing | JWF | no | SKIP - DC/GLSL | yuebing pattern |
| cut_zigzag | JWF | no | SKIP - DC/GLSL | zigzag pattern |
| dc_acrilic | JWF | no | NEEDS_DC | acrylic paint simulation |
| dc_apollonian | JWF | no | NEEDS_DC | Apollonian with DC |
| dc_booleans | JWF | no | NEEDS_DC | boolean operations with DC |
| dc_bubble | JWF | yes | NEEDS_DC | bubble with direct color |
| dc_butterflies | JWF | no | NEEDS_DC | butterfly pattern with DC |
| dc_cairotiles | JWF | no | NEEDS_DC | Cairo tiling with DC |
| dc_carpet | JWF | yes | NEEDS_DC | carpet tiling with DC |
| dc_carpet3D | JWF | yes | SKIP - 3D/DC | 3D carpet with DC |
| dc_circlesblue | JWF | no | NEEDS_DC | blue circles with DC |
| dc_circuits | JWF | no | NEEDS_DC | circuit board pattern |
| dc_circlesblue | JWF | no | NEEDS_DC | blue circles |
| dc_code | JWF | no | NEEDS_DC | code pattern |
| dc_crackle_wf | JWF | yes | NEEDS_DC | Voronoi crackle with DC |
| dc_cracklep_wf | JWF | yes | NEEDS_DC | crackle variant |
| dc_cube | JWF | yes | SKIP - 3D/DC | 3D cube with DC |
| dc_cylinder | JWF | yes | NEEDS_DC | cylinder with DC |
| dc_cylinder2 | JWF | yes | NEEDS_DC | cylinder2 with DC |
| dc_dmodulus | JWF | no | NEEDS_DC | double modulus with DC |
| dc_ducks | JWF | no | NEEDS_DC | ducks fractal with DC |
| dc_fingerprint | JWF | no | NEEDS_DC | fingerprint pattern |
| dc_fractaldots | JWF | no | NEEDS_DC | fractal dots with DC |
| dc_fractcolor | JWF | no | NEEDS_DC | fractal coloring |
| dc_gabornoise | JWF | no | NEEDS_DC | Gabor noise with DC |
| dc_glypho | JWF | no | NEEDS_DC | glyph with DC |
| dc_gmandelbroot | JWF | no | NEEDS_DC | generic Mandelbrot with DC |
| dc_gnarly | JWF | yes | NEEDS_DC | gnarly pattern with DC, 18 params |
| dc_grid3D | JWF | no | SKIP - 3D/DC | 3D grid with DC |
| dc_hexagons | JWF | no | NEEDS_DC | hexagon tiling with DC |
| dc_hexes_wf | JWF | yes | NEEDS_DC | hexes with DC |
| dc_hoshi | JWF | no | NEEDS_DC | star pattern with DC |
| dc_hyperbolictile | JWF | no | NEEDS_DC | hyperbolic tiling with DC |
| dc_inversion | JWF | no | NEEDS_DC | inversion with DC |
| dc_kaleidocomplex | JWF | no | NEEDS_DC | complex kaleidoscope DC |
| dc_kaleidoscopic | JWF | no | NEEDS_DC | kaleidoscopic with DC |
| dc_kaleidotile | JWF | yes | NEEDS_DC | kaleidoscope tiling with DC |
| dc_kaliset | JWF | no | NEEDS_DC | Kali set with DC |
| dc_kaliset2 | JWF | no | NEEDS_DC | Kali set variant with DC |
| dc_layers | JWF | no | NEEDS_DC | layered pattern with DC |
| dc_linear | JWF | yes | NEEDS_DC | linear with DC |
| dc_mandala | JWF | no | NEEDS_DC | mandala with DC |
| dc_mandbrot | JWF | no | NEEDS_DC | Mandelbrot with DC |
| dc_mandelbox2D | JWF | no | NEEDS_DC | 2D mandelbox with DC |
| dc_menger | JWF | no | NEEDS_DC | Menger sponge 2D projection |
| dc_moebiuslog | JWF | no | NEEDS_DC | Moebius logarithm with DC |
| dc_pentatiles | JWF | no | NEEDS_DC | Penrose tiling with DC |
| dc_perlin | JWF | yes | NEEDS_DC | Perlin noise with DC |
| dc_poincaredisc | JWF | no | NEEDS_DC | Poincare disc with DC |
| dc_portal | JWF | no | NEEDS_DC | portal pattern with DC |
| dc_quadtree | JWF | no | NEEDS_DC | quadtree with DC |
| dc_randomoctree | JWF | no | NEEDS_DC | random octree with DC |
| dc_rotations | JWF | no | NEEDS_DC | rotation pattern with DC |
| dc_spacefold | JWF | no | NEEDS_DC | space fold with DC |
| dc_squares | JWF | no | NEEDS_DC | squares with DC |
| dc_starsfield | JWF | no | NEEDS_DC | star field with DC |
| dc_sunflower | JWF | no | NEEDS_DC | sunflower with DC |
| dc_tesla | JWF | no | NEEDS_DC | Tesla coil with DC |
| dc_tree | JWF | no | NEEDS_DC | tree with DC |
| dc_triantess | JWF | no | NEEDS_DC | triangle tessellation DC |
| dc_triangle | JWF | no | NEEDS_DC | triangle with DC |
| dc_triTile | JWF | yes | NEEDS_DC | triangle tiling with DC |
| dc_truchet | JWF | no | NEEDS_DC | truchet with DC |
| dc_turbulence | JWF | no | NEEDS_DC | turbulence with DC |
| dc_voronoise | JWF | no | NEEDS_DC | Voronoi noise with DC |
| dc_vortex | JWF | no | NEEDS_DC | vortex with DC |
| dc_warping | JWF | no | NEEDS_DC | warping with DC |
| dc_worley | JWF | no | NEEDS_DC | Worley noise with DC |
| dc_ztransl | JWF | yes | SKIP - 3D/DC | z-translate with DC |

### D

| Name | Source | GPU | Status | Notes |
|------|--------|-----|--------|-------|
| d_spherical | JWF | yes | TODO | displaced spherical, trivial |
| de_stijl | JWF | no | TODO | De Stijl art pattern, 5 params, cheap |
| deltaA | JWF | yes | TODO | delta function variant, medium |
| devil_warp | JWF | yes | TODO | devil warp distortion, 6 params, medium |
| diamond | JWF | yes | DONE (11) | sin(theta)*cos(r) |
| dinis_surface_wf | JWF | yes | SKIP - 3D | Dini's surface 3D |
| disc | JWF | yes | DONE (8) | theta/pi * sin(pi*r) |
| disc2 | JWF | yes | DONE (58) | parametric disc with twist |
| disc3 | JWF | no | TODO | disc variant, 8 params, medium |
| disc3d | JWF | yes | SKIP - 3D | 3D disc |
| displacemap_wf | JWF | no | SKIP - IMAGE | displacement map from image |
| dla_wf | JWF | no | SKIP - SIMULATION | diffusion-limited aggregation |
| dla3d_wf | JWF | yes | SKIP - 3D/SIMULATION | 3D DLA |
| dragon_js | JWF | yes | SKIP - BASE_SHAPE | dragon curve IFS, stateful |
| draw | JWF | no | SKIP - INTERACTIVE | interactive drawing tool |
| drunken_tiles | JWF | no | TODO | randomized tile pattern, 15 params, expensive |
| ducks | JWF | yes | TODO | ducks fractal (Petigen), complex iteration |
| dustpoint | JWF | yes | SKIP - 3D | 3D dust point |

### E

| Name | Source | GPU | Status | Notes |
|------|--------|-----|--------|-------|
| eCollide | JWF | yes | TODO | exponential collide, heavy |
| eJulia | JWF | yes | TODO | exponential Julia, heavy |
| eMod | JWF | yes | TODO | exponential modulus, heavy |
| eMotion | JWF | yes | TODO | exponential motion, heavy |
| ePush | JWF | no | TODO | exponential push, heavy |
| eRotate | JWF | no | TODO | exponential rotate, heavy |
| eScale | JWF | no | TODO | exponential scale, heavy |
| eSwirl | JWF/ApoPlugin | no | TODO | exponential swirl, heavy |
| eclipse | JWF | yes | DONE (65) | eclipse shift |
| edisc | JWF | yes | DONE (90) | elliptic disc |
| elliptic | JWF/Apo | yes | DONE (88) | elliptic integral transform |
| elliptic2 | JWF | yes | TODO | extended elliptic, 11 params, heavy |
| ellipsoid3D | JWF | yes | SKIP - 3D | 3D ellipsoid |
| ennepers | JWF | yes | TODO | Enneper's surface (2D projection), trivial |
| ennepers2 | JWF | yes | SKIP - 3D | Enneper's surface 3D |
| epispiral | JWF/Apo | yes | DONE (52) | epicycloid spiral |
| epispiral_wf | JWF | yes | TODO | epispiral variant with 1 param, medium |
| erf | JWF | no | TODO | error function, trivial |
| erf3D | JWF | yes | SKIP - 3D | 3D error function |
| escher | JWF/Apo | yes | DONE (87) | Escher-like tiling |
| estiq | JWF | yes | SKIP - 3D | 3D Escher tiling |
| ex | JWF | yes | DONE (12) | power-of-angle split |
| exblur | JWF | yes | SKIP - 3D | 3D exblur |
| exp | JWF | yes | DONE (120) | complex exponential |
| exp2_bs | JWF | no | TODO | exp with params, 3 params |
| exp_multi | JWF | no | TODO | multi exponential, 8 params, cheap |
| exponential | JWF | yes | DONE (18) | exp(x-1)*cos/sin(pi*y) |
| extrude | JWF/ApoPlugin | yes | SKIP - 3D | 3D extrusion |
| eyefish | JWF | yes | DONE (27) | 2/(r+1) fisheye |

### F

| Name | Source | GPU | Status | Notes |
|------|--------|-----|--------|-------|
| f_complex | JWF | yes | SKIP - 3D/SIMULATION | complex function, 3D + stateful |
| falloff2 | JWF/Apo | yes | SKIP - 3D | 3D falloff field |
| falloff3 | JWF | yes | SKIP - 3D | 3D falloff v3 |
| fan | JWF | yes | DONE (22) | angular fan fold |
| fan2 | JWF/Apo | yes | DONE (25) | parametric fan |
| farblur | JWF | yes | SKIP - 3D | 3D far blur |
| fdisc | JWF | yes | TODO | fancy disc, 8 params, medium |
| fibonacci2 | JWF | yes | TODO | Fibonacci spiral, heavy |
| fisheye | JWF | yes | DONE (16) | 2/(r+1) with xy swap |
| flame_bulb | JWF | yes | SKIP - 3D | 3D flame bulb |
| flatten | JWF/Apo | yes | SKIP - UTILITY | sets z=0, utility |
| flipcircle | JWF | yes | DONE (64) | circle-based flip |
| flipy | JWF | yes | TODO | y-axis flip, trivial |
| flora | JWF | yes | SKIP - BASE_SHAPE | procedural plant |
| flower | JWF | yes | DONE (59) | petal pattern |
| flower_db | JWF | no | TODO | flower with distance buffer, heavy |
| flux | JWF | yes | DONE (101) | flux spread mapping |
| foci | JWF/Apo | yes | DONE (105) | conformal focus mapping |
| foci_3D | JWF | yes | SKIP - 3D | 3D foci |
| fourth | JWF | yes | TODO | fourth-power distortion, 5 params, heavy |
| fract_dragon_wf | JWF | no | SKIP - FRACTAL_ITER | dragon fractal iterator |
| fract_formula_julia_wf | JWF | no | SKIP - FRACTAL_ITER | formula-based Julia iterator |
| fract_formula_mand_wf | JWF | no | SKIP - FRACTAL_ITER | formula-based Mandelbrot iterator |
| fract_julia_wf | JWF | no | SKIP - FRACTAL_ITER | Julia set iterator |
| fract_mandelbrot_wf | JWF | no | SKIP - FRACTAL_ITER | Mandelbrot iterator |
| fract_meteors_wf | JWF | no | SKIP - FRACTAL_ITER | meteors fractal iterator |
| fract_pearls_wf | JWF | no | SKIP - FRACTAL_ITER | pearls fractal iterator |
| fract_salamander_wf | JWF | yes | SKIP - FRACTAL_ITER | salamander fractal iterator |
| fresnel | JWF | yes | SKIP - 3D | Fresnel lens 3D |
| funnel | JWF | yes | TODO | funnel distortion, medium |
| flatten | JWF | yes | SKIP - UTILITY | z flattening utility |

### G

| Name | Source | GPU | Status | Notes |
|------|--------|-----|--------|-------|
| gamma | JWF | yes | TODO | gamma function, cheap |
| gaussian_blur | JWF | yes | DONE (76) | gaussian random disc |
| gdoffs | JWF | yes | TODO | generalized d-offsets, 8 params, trivial |
| gingerbread_man | JWF | yes | SKIP - SIMULATION | gingerbread man iterator |
| glitchy1 | JWF | yes | SKIP - 3D | 3D glitch effect |
| glitchy2 | JWF | no | TODO | 2D glitch, 53 params, cheap (but complex config) |
| glsl_* | JWF | no | SKIP - GLSL | GLSL shader variations (duplicate dc_*) |
| glynnia | JWF | yes | DONE (126) | Glynn attractor variant |
| glynnia3 | JWF | yes | TODO | extended Glynn with 4 params, heavy |
| glynnSim1 | JWF | yes | TODO | Glynn simulation 1, 6 params, medium |
| glynnSim2 | JWF | yes | TODO | Glynn simulation 2, 6 params, medium |
| glynnSim2B | JWF | yes | SKIP - 3D | Glynn sim 2B, 3D |
| glynnSim3 | JWF | yes | TODO | Glynn simulation 3, 4 params, medium |
| glynnSShape | JWF | no | TODO | Glynn supershape, 11 params, medium |
| glynnlissa | JWF | no | TODO | Glynn Lissajous, 11 params, medium |
| glynnspiro | JWF | no | TODO | Glynn spirograph, 13 params, heavy |
| glynns3subfl | JWF | yes | SKIP - 3D | Glynn 3D subflame |
| gosperisland_js | JWF | yes | SKIP - BASE_SHAPE | Gosper island IFS, stateful |
| gpattern | JWF | yes | SKIP - BASE_SHAPE | generic pattern, complex |
| gridout | JWF | yes | TODO | grid-out fold, cheap |
| gridout2 | JWF | yes | TODO | grid-out variant, 4 params, cheap |
| gridout3D | JWF | yes | SKIP - 3D | 3D grid out |
| grid3d_wf | JWF | yes | SKIP - 3D | 3D grid |
| gumowski_mira | JWF | no | SKIP - SIMULATION | Gumowski-Mira iterator |

### H

| Name | Source | GPU | Status | Notes |
|------|--------|-----|--------|-------|
| hadamard_js | JWF | yes | TODO | Hadamard matrix IFS, trivial |
| hamid_js | JWF | yes | SKIP - BASE_SHAPE | Hamid pattern, complex stateful |
| handkerchief | JWF | yes | DONE (6) | r*sin(theta+r), r*cos(theta-r) |
| harmonograph_js | JWF | yes | SKIP - BASE_SHAPE | harmonograph curves, stateful |
| heart | JWF | yes | DONE (7) | r*sin(theta*r) |
| heart_wf | JWF | yes | TODO | heart with extra params, medium |
| helicoid | JWF | yes | SKIP - 3D | 3D helicoid |
| helix | JWF | yes | SKIP - 3D | 3D helix |
| hemisphere | JWF/Apo | yes | SKIP - 3D | 3D hemisphere projection |
| henon | JWF | yes | TODO | Henon map, trivial |
| hex_modulus | JWF | yes | DONE (40) | hexagonal modular grid |
| hexaplay3D | JWF | yes | SKIP - 3D | 3D hexagonal play |
| hexes | JWF | yes | TODO | hexagonal Voronoi tiling, heavy |
| hexnix3D | JWF | yes | SKIP - 3D | 3D hex nix |
| hexprism3D | JWF | yes | SKIP - 3D | 3D hexagonal prism |
| hilbert_js | JWF | yes | SKIP - BASE_SHAPE | Hilbert curve IFS, stateful |
| ho | JWF | yes | SKIP - 3D | 3D Ho attractor |
| hole | JWF | yes | TODO | hole distortion, 2 params, medium |
| hole2 | JWF | no | TODO | hole variant, 6 params, expensive |
| holesq | JWF | yes | TODO | square hole, cheap |
| hopalong | JWF | yes | SKIP - SIMULATION | hopalong iterator |
| horseshoe | JWF | yes | DONE (4) | (x-y)(x+y)/r |
| hourglass3D | JWF | yes | SKIP - 3D | 3D hourglass |
| htree_js | JWF | yes | SKIP - BASE_SHAPE | H-tree IFS, stateful |
| hyperbolic | JWF | yes | DONE (10) | sin(theta)/r, r*cos(theta) |
| hyperbolicellipse | JWF | yes | TODO | hyperbolic ellipse, trivial |
| hypercrop | JWF | yes | SKIP - 3D | 3D hyperbolic crop |
| hypershift | JWF | yes | TODO | hyperbolic shift, 2 params, trivial |
| hypershift2 | JWF | yes | SKIP - 3D | 3D hypershift |
| hypertile | JWF | yes | DONE (55) | hyperbolic tiling |
| hypertile1 | JWF | yes | TODO | hypertile variant 1, cheap |
| hypertile2 | JWF | yes | TODO | hypertile variant 2, cheap |
| hypertile3D | JWF | yes | SKIP - 3D | 3D hypertile |
| hypertile3D1 | JWF | yes | SKIP - 3D | 3D hypertile v1 |
| hypertile3D2 | JWF | yes | SKIP - 3D | 3D hypertile v2 |
| hypertile3D2b | JWF | yes | SKIP - 3D | 3D hypertile v2b |
| hypertile3Db | JWF | yes | SKIP - 3D | 3D hypertile b |

### I

| Name | Source | GPU | Status | Notes |
|------|--------|-----|--------|-------|
| iconattractor_js | JWF | yes | SKIP - BASE_SHAPE | icon attractor, stateful |
| idisc | JWF | yes | TODO | inverted disc, cheap |
| iflames_wf | JWF | no | SKIP - SIMULATION | interactive flames, complex |
| inflateZ_1 | JWF | yes | SKIP - 3D | z-inflate variant 1 |
| inflateZ_2 | JWF | yes | SKIP - 3D | z-inflate variant 2 |
| inflateZ_3 | JWF | yes | SKIP - 3D | z-inflate variant 3 |
| inflateZ_4 | JWF | yes | SKIP - 3D | z-inflate variant 4 |
| inflateZ_5 | JWF | yes | SKIP - 3D | z-inflate variant 5 |
| inflateZ_6 | JWF | yes | SKIP - 3D | z-inflate variant 6 |
| intersection | JWF | yes | TODO | intersection tiling, 10 params, medium |
| inversion | JWF | no | TODO | circle inversion, 4 params, heavy |
| invsquircular | JWF | yes | TODO | inverse squircle mapping, medium |
| inverted_julia | JWF | yes | TODO | inverted julia, 9 params, medium |
| invpolar | JWF | yes | TODO | inverse polar, cheap |
| invtree_js | JWF | yes | TODO | inverse tree IFS, trivial |

### J

| Name | Source | GPU | Status | Notes |
|------|--------|-----|--------|-------|
| jac_asn | JWF | yes | TODO | Jacobi elliptic asn, expensive |
| jac_cn | JWF | yes | TODO | Jacobi elliptic cn, trivial |
| jac_dn | JWF | yes | TODO | Jacobi elliptic dn, trivial |
| jac_elk | JWF | yes | TODO | Jacobi elliptic integral, heavy |
| jac_sn | JWF | yes | TODO | Jacobi elliptic sn, trivial |
| japanese_maple_leaf | JWF | yes | SKIP - BASE_SHAPE | leaf shape |
| joukowski | JWF | yes | TODO | Joukowski transform, trivial |
| jubiQ | JWF | yes | SKIP - 3D | 3D Julia-bi-quaternion |
| julia | JWF | yes | DONE (13) | sqrt(r) random +/- theta/2 |
| julia3D | JWF/Apo | yes | SKIP - 3D | 3D Julia |
| julia3Dq | JWF | yes | SKIP - 3D | 3D Julia quaternion |
| julia3Dz | JWF/Apo | yes | SKIP - 3D | 3D Julia with z |
| juliac | JWF | yes | TODO | Julia-C, 3 params, medium |
| julian | JWF/Apo | yes | DONE (32) | julian n-fold symmetry |
| julian2 | JWF | yes | TODO | julian v2, 8 params, medium |
| julian3Dx | JWF | yes | TODO | julian 3D-x (2D usable), medium |
| juliaq | JWF | yes | DONE (124) | julia quotient |
| juliascope | JWF/Apo | yes | DONE (33) | julia with scope reflection |
| juliascope3Db | JWF | yes | SKIP - 3D | 3D juliascope |
| juliascopePlus | JWF | no | TODO | extended juliascope, 16 params |
| julia_outside | JWF | yes | TODO | outside-Julia variant, cheap |

### K

| Name | Source | GPU | Status | Notes |
|------|--------|-----|--------|-------|
| kaleidoscope | JWF | yes | DONE (41) | n-fold kaleidoscope |
| kaleidoimg | JWF | no | SKIP - IMAGE | kaleidoscope from image |
| kaplan | JWF | yes | SKIP - BASE_SHAPE | Kaplan tiling |
| klein_group | JWF | yes | SKIP - BASE_SHAPE | Klein group transform, complex |
| knots3D | JWF | yes | SKIP - 3D | 3D knots |
| koch_js | JWF | yes | SKIP - BASE_SHAPE | Koch curve IFS, stateful |

### L

| Name | Source | GPU | Status | Notes |
|------|--------|-----|--------|-------|
| lace_js | JWF | yes | TODO | lace IFS pattern, expensive |
| layered_spiral | JWF | yes | DONE (66) | layered spiral rings |
| lazyTravis | JWF | yes | TODO | lazy Travis pattern, 3 params, medium |
| lazyjess | JWF | no | TODO | lazy Jess pattern, 4 params, expensive |
| lazysensen | JWF | yes | SKIP - 3D | 3D lazy Mensen |
| lazysusan | JWF/Apo | yes | DONE (97) | lazy susan rotation |
| line | JWF | yes | SKIP - 3D | 3D line |
| linear | JWF | yes | DONE (0) | identity transform |
| linear3D | JWF/Apo | yes | SKIP - 3D | 3D linear |
| linearT | JWF | yes | TODO | linear with tilt params, cheap |
| linearT3D | JWF | yes | SKIP - 3D | 3D linearT |
| lissajous | JWF | yes | DONE (68) | Lissajous curve mapping |
| log | JWF | no | DONE (121) | complex logarithm |
| log_apo | JWF | yes | TODO | Apophysis-style log, cheap |
| log_db | JWF | no | TODO | log with distance buffer, heavy |
| log_tile2 | JWF | yes | SKIP - 3D | 3D log tiling |
| loonie | JWF/Apo | yes | DONE (50) | circle inversion variant |
| loonie2 | JWF | yes | TODO | loonie with params, expensive |
| loonie3 | JWF | yes | TODO | loonie variant 3, cheap |
| loonie_3D | JWF | yes | SKIP - 3D | 3D loonie |
| loq | JWF | yes | SKIP - 3D | 3D log variant |
| lorenz_js | JWF | yes | SKIP - 3D/SIMULATION | Lorenz attractor |
| lozi | JWF | yes | TODO | Lozi map, 3 params, trivial |
| lsystem_js | JWF | no | SKIP - SIMULATION | L-system IFS, very complex, stateful |

### M

| Name | Source | GPU | Status | Notes |
|------|--------|-----|--------|-------|
| macmillan | JWF | no | TODO | MacMillan map, trivial |
| mandala | JWF | yes | SKIP - BASE_SHAPE | mandala pattern, complex |
| mandala2 | JWF | yes | SKIP - BASE_SHAPE | mandala v2, very complex |
| mandelbrot | JWF | yes | SKIP - FRACTAL_ITER | Mandelbrot iterator |
| mandelbox2D | JWF | no | TODO | 2D Mandelbox, 12 params, expensive |
| maple_leaf | JWF | yes | SKIP - BASE_SHAPE | maple leaf shape |
| mask | JWF | yes | TODO | mask distortion, heavy |
| maurer_lines | JWF | yes | SKIP - BASE_SHAPE | Maurer line pattern, very complex |
| maurer_rose | JWF | yes | SKIP - BASE_SHAPE | Maurer rose pattern |
| mcarpet | JWF | yes | TODO | Minkowski carpet, 4 params, trivial |
| meeple | JWF | yes | SKIP - BASE_SHAPE | meeple shape |
| minkowskope | JWF | yes | DONE (125) | Minkowski metric kaleidoscope |
| minkQM | JWF | yes | TODO | Minkowski question mark, 6 params, cheap |
| mobiq | JWF | yes | SKIP - 3D | 3D Moebius quaternion |
| mobius | JWF/Apo | yes | DONE (47) | Moebius transformation |
| mobiusN | JWF | yes | TODO | Moebius N-fold, heavy |
| mobius3D_with_inverse | JWF | yes | SKIP - 3D | 3D Moebius with inverse |
| mobius_dragon_3D | JWF | yes | SKIP - 3D | 3D Moebius dragon |
| mobius_strip | JWF | yes | SKIP - 3D | 3D Moebius strip |
| modulus | JWF | yes | DONE (98) | modular wrap |
| msTruchet | JWF | yes | SKIP - BASE_SHAPE | multi-scale truchet, complex |
| multi_ifs | JWF | no | TODO | multi-IFS system, 7 params, medium |
| murl | JWF | yes | TODO | murl distortion, 2 params, medium |
| murl2 | JWF/ApoPlugin | yes | TODO | murl variant 2, heavy |

### N

| Name | Source | GPU | Status | Notes |
|------|--------|-----|--------|-------|
| nBlur | JWF | yes | SKIP - BASE_SHAPE | n-gon blur, complex |
| ngon | JWF/Apo | yes | DONE (49) | n-gon angular scaling |
| noise | JWF | yes | DONE (80) | random polar scatter |
| npolar | JWF | yes | TODO | n-polar, 2 params, heavy |
| nsudoku | JWF | yes | SKIP - BASE_SHAPE | Sudoku grid pattern |
| neuron3D | JWF | yes | SKIP - 3D | 3D neuron |

### O

| Name | Source | GPU | Status | Notes |
|------|--------|-----|--------|-------|
| oak_leaf | JWF | yes | SKIP - BASE_SHAPE | oak leaf shape |
| octagon | JWF | yes | SKIP - 3D | 3D octagon |
| octapol | JWF | yes | SKIP - 3D | 3D octapol |
| octahedron3D | JWF | yes | SKIP - 3D | 3D octahedron |
| octogonprism3D | JWF | yes | SKIP - 3D | 3D octagon prism |
| ocylinder3D | JWF | yes | SKIP - 3D | 3D oriented cylinder |
| ocappedcone3D | JWF | yes | SKIP - 3D | 3D oriented capped cone |
| onion | JWF | yes | SKIP - 3D | 3D onion layers |
| onion2 | JWF | yes | SKIP - 3D | 3D onion variant |
| oroundcone3D | JWF | yes | SKIP - 3D | 3D oriented round cone |
| ortho | JWF | yes | TODO | orthogonal mapping, 2 params, expensive |
| oscilloscope | JWF | yes | DONE (89) | oscilloscope wave pattern |
| oscilloscope2 | JWF | yes | TODO | oscilloscope v2, 6 params, medium |
| ovoid | JWF | yes | TODO | ovoid distortion, 2 params, trivial |
| ovoid3d | JWF | yes | SKIP - 3D | 3D ovoid |

### P

| Name | Source | GPU | Status | Notes |
|------|--------|-----|--------|-------|
| pTransform | JWF | yes | TODO | p-transform, 5 params, medium |
| panorama1 | JWF | yes | TODO | panorama projection 1, medium |
| panorama2 | JWF | yes | TODO | panorama projection 2, medium |
| parabola | JWF | yes | DONE (84) | sin/cos of r |
| parallel | JWF | yes | TODO | parallel lines, 12 params, medium |
| pdj | JWF | yes | DONE (24) | sin/cos of a,b,c,d |
| pdj3D | JWF | yes | SKIP - 3D | 3D pdj |
| perspective | JWF | yes | DONE (78) | perspective projection |
| petal | JWF | yes | TODO | petal pattern, heavy |
| petal3D | JWF | yes | SKIP - 3D | 3D petal |
| petal3D_apo | JWF | yes | SKIP - 3D | 3D petal Apo variant |
| phoenix_julia | JWF | yes | DONE (123) | phoenix julia with distortion |
| pie | JWF | yes | DONE (82) | pie slice scatter |
| pie3D | JWF | yes | SKIP - 3D | 3D pie |
| pixel_flow | JWF | yes | TODO | pixel flow effect, 5 params, cheap |
| plane_wf | JWF | yes | SKIP - 3D | 3D plane |
| plusrecip | JWF | yes | TODO | plus reciprocal, 2 params, medium |
| poincare3D | JWF | yes | SKIP - 3D | 3D Poincare |
| point_mirror_symmetry | JWF | yes | TODO | point mirror symmetry |
| pointgrid_wf | JWF | yes | TODO | point grid, 8 params, trivial |
| pointgrid3d_wf | JWF | yes | SKIP - 3D | 3D point grid |
| polar | JWF | yes | DONE (5) | theta/pi, r-1 |
| polar2 | JWF | yes | DONE (104) | polar variant |
| polylogarithm | JWF | yes | TODO | polylogarithm function, expensive |
| polySurf | JWF | yes | SKIP - 3D | 3D polynomial surface |
| popcorn | JWF | yes | DONE (17) | affine-coeff trig perturbation |
| popcorn2 | JWF/ApoPlugin | yes | DONE (106) | parametric popcorn |
| popcorn2_3D | JWF | yes | SKIP - 3D | 3D popcorn2 |
| pow_block | JWF | no | TODO | power block, 5 params, medium |
| power | JWF | yes | DONE (19) | r^sin(theta) scaling |
| pRose3D | JWF | yes | SKIP - 3D | 3D p-rose |
| pressure_wave | JWF | no | TODO | pressure wave, 6 params, cheap |
| primitives_wf | JWF | yes | SKIP - 3D/BASE_SHAPE | 3D primitive shapes |
| projective | JWF | yes | TODO | projective transform, 9 params, trivial |
| pulse | JWF | yes | TODO | pulse distortion, 4 params, cheap |
| pyramid | JWF | yes | TODO | pyramid reflection fold, trivial |
| pyramid3D | JWF | yes | SKIP - 3D | 3D pyramid |

### Pre/Post variations

Pre/post variations are applied before/after the main transform.
Most are SKIP for flame-sheep since we handle pre/post via the affine pipeline.

| Name | Source | GPU | Status | Notes |
|------|--------|-----|--------|-------|
| pre_blur | JWF/Apo | yes | SKIP - UTILITY | pre-transform blur |
| pre_blur3D | JWF | yes | SKIP - 3D | 3D pre-blur |
| pre_boarders2 | JWF | yes | TODO | pre-transform boarders2 |
| pre_bwraps2 | JWF/ApoPlugin | no | TODO | pre-transform bwraps2 |
| pre_c_symmetry | JWF | no | TODO | pre-transform c-symmetry |
| pre_c_var | JWF | no | TODO | pre-transform c-var |
| pre_circlecrop | JWF/Apo | yes | SKIP - CROP | pre-transform circle crop |
| pre_crop | JWF/Apo | yes | SKIP - CROP | pre-transform crop |
| pre_curl | JWF/Apo | yes | TODO | pre-transform curl |
| pre_custom_wf | JWF | yes | SKIP - CUSTOM | pre-transform custom |
| pre_dcztransl | JWF | yes | SKIP - 3D/DC | pre DC z-translate |
| pre_disc | JWF/Apo | yes | TODO | pre-transform disc, cheap |
| pre_disc3d | JWF | yes | SKIP - 3D | pre 3D disc |
| pre_falloff3 | JWF | yes | SKIP - 3D | pre 3D falloff |
| pre_flatten | JWF | yes | SKIP - UTILITY | pre z-flatten |
| pre_prepost_affine | JWF | yes | SKIP - UTILITY | pre/post affine utility |
| pre_rect_wf | JWF | yes | SKIP - BASE_SHAPE | pre rectangle |
| pre_recip | JWF | yes | SKIP - 3D | 3D pre-reciprocal |
| pre_rotate_x | JWF/Apo | yes | SKIP - 3D | 3D x-rotation |
| pre_rotate_y | JWF/Apo | yes | TODO | pre y-rotation, cheap |
| pre_sinusoidal3d | JWF | yes | SKIP - 3D | 3D pre-sinusoidal |
| pre_spherical | JWF/Apo | yes | TODO | pre-transform spherical |
| pre_spin_z | JWF/Apo | no | SKIP - 3D | 3D z-spin |
| pre_stabilize | JWF | yes | SKIP - BASE_SHAPE | pre-stabilize utility |
| pre_subflame_wf | JWF | yes | SKIP - 3D/SUBFLAME | sub-flame reference |
| pre_wave3D_wf | JWF | yes | SKIP - 3D | 3D pre-wave |
| pre_zscale | JWF/Apo | yes | SKIP - 3D | z-scale |
| pre_ztranslate | JWF/Apo | yes | SKIP - 3D | z-translate |
| prepost_affine | JWF | yes | SKIP - UTILITY | combined pre/post affine |
| prepost_blob | JWF | no | SKIP - UTILITY | combined pre/post blob |
| prepost_circlize | JWF | no | TODO | combined pre/post circlize |
| prepost_mobius | JWF | no | TODO | combined pre/post Moebius, trivial |
| post_affine3D | JWF | yes | SKIP - 3D | post 3D affine |
| post_aexion_crop | JWF | yes | SKIP - 3D | post 3D aexion crop |
| post_asurf_crop | JWF | yes | SKIP - 3D | post 3D amazing surf crop |
| post_axis_symmetry_wf | JWF | yes | TODO | post axis symmetry, medium |
| post_benesi_crop | JWF | yes | SKIP - 3D | post 3D Benesi crop |
| post_bristorbrot_crop | JWF | yes | SKIP - 3D | post 3D Bristorbrot crop |
| post_bulbtorus_crop | JWF | yes | SKIP - 3D | post 3D bulb torus crop |
| post_bumpmap_wf | JWF | yes | SKIP - 3D/IMAGE | post bump map from image |
| post_bwraps2 | JWF/Apo | yes | TODO | post boundary wraps, cheap |
| post_c_symmetry | JWF | no | TODO | post c-symmetry |
| post_c_var | JWF | no | TODO | post c-var |
| post_circlecrop | JWF/Apo | yes | SKIP - CROP | post circle crop |
| post_coastalbrot_crop | JWF | yes | SKIP - 3D | post coastalbrot crop |
| post_colormap_wf | JWF | no | SKIP - IMAGE | post color map |
| post_colorscale_wf | JWF | yes | SKIP - 3D | post color scale |
| post_crop | JWF/Apo | yes | SKIP - CROP | post crop |
| post_crop_box | JWF | yes | SKIP - CROP | post box crop |
| post_crop_cross | JWF | yes | SKIP - CROP | post cross crop |
| post_crop_polygon | JWF | yes | SKIP - CROP | post polygon crop |
| post_crop_rhombus | JWF | yes | SKIP - CROP | post rhombus crop |
| post_crop_stars | JWF | yes | SKIP - CROP | post star crop |
| post_crop_trapezoid | JWF | yes | SKIP - CROP | post trapezoid crop |
| post_crop_triangle | JWF | yes | SKIP - CROP | post triangle crop |
| post_crop_vesica | JWF | yes | SKIP - CROP | post vesica crop |
| post_crop_x | JWF | yes | SKIP - CROP | post x-axis crop |
| post_crosscrop | JWF | yes | SKIP - 3D | post cross crop |
| post_curl | JWF/Apo | yes | TODO | post curl |
| post_curl3D | JWF/Apo | yes | SKIP - 3D | post 3D curl |
| post_custom_wf | JWF | yes | SKIP - CUSTOM | post custom |
| post_depth | JWF | yes | SKIP - 3D | post depth |
| post_displacemap_wf | JWF | no | SKIP - IMAGE | post displacement map |
| post_falloff2 | JWF/Apo | yes | SKIP - 3D | post falloff2 |
| post_falloff3 | JWF | yes | SKIP - 3D | post falloff3 |
| post_flatten | JWF | yes | SKIP - UTILITY | post z-flatten |
| post_heat | JWF | yes | SKIP - 3D | post heat map |
| post_julia3Dq | JWF | no | SKIP - 3D | post 3D julia Q |
| post_juliaq | JWF | no | TODO | post julia quotient |
| post_log_tile2 | JWF | yes | SKIP - 3D | post log tile |
| post_mandelbox3d_crop | JWF | yes | SKIP - 3D | post mandelbox crop |
| post_mandelbulb3d_crop | JWF | yes | SKIP - 3D | post mandelbulb crop |
| post_mirror_wf | JWF | yes | TODO | post mirror, 11 params, cheap |
| post_point_crop | JWF | yes | SKIP - CROP | post point crop |
| post_point_symmetry_wf | JWF | yes | TODO | post point symmetry, 4 params, trivial |
| post_prepost_affine | JWF | yes | SKIP - UTILITY | post part of pre/post |
| post_rblur | JWF | yes | SKIP - 3D | post radial blur |
| post_rotate_x | JWF | yes | SKIP - 3D | post x-rotate |
| post_rotate_y | JWF | yes | SKIP - 3D | post y-rotate |
| post_rotate_z | JWF | yes | SKIP - 3D | post z-rotate |
| post_smartcrop | JWF | yes | TODO | post smart crop, heavy |
| post_spherical | JWF/ApoPlugin | yes | TODO | post spherical |
| post_spin_z | JWF | yes | SKIP - 3D | post z-spin |
| post_trig | JWF | yes | TODO | post trig function, 17 params, heavy |
| post_ztranslate_wf | JWF | yes | SKIP - 3D | post z-translate |

### Q

| Name | Source | GPU | Status | Notes |
|------|--------|-----|--------|-------|
| q_ode | JWF | yes | TODO | quadratic ODE, 12 params, trivial |
| quad | JWF | no | TODO | quad distortion, 29 params, heavy |
| quaternion | JWF | yes | SKIP - 3D | 3D quaternion transform |

### R

| Name | Source | GPU | Status | Notes |
|------|--------|-----|--------|-------|
| r_circleblur | JWF | yes | TODO | radial circle blur |
| radial_blur | JWF/Apo | yes | DONE (77) | spin + zoom blur |
| rational3 | JWF | yes | TODO | rational function degree 3, 8 params, trivial |
| rays | JWF | yes | DONE (85) | radial ray pattern |
| rays1 | JWF | yes | TODO | rays variant 1, cheap |
| rays2 | JWF | yes | TODO | rays variant 2, cheap |
| rays3 | JWF | yes | TODO | rays variant 3, medium |
| rbox3D | JWF | yes | SKIP - 3D | 3D rounded box |
| rectangles | JWF/Apo | yes | DONE (38) | rectangular grid fold |
| recurrenceplot | JWF | yes | SKIP - BASE_SHAPE | recurrence plot, complex |
| rhodonea | JWF | yes | SKIP - BASE_SHAPE | rhodonea (rose) curve |
| rhombus3D | JWF | yes | SKIP - 3D | 3D rhombus |
| ringer | JWF | no | TODO | ring distortion, 9 params, heavy |
| rings | JWF | yes | DONE (21) | radial ring fold |
| rings2 | JWF/Apo | yes | DONE (26) | parametric rings |
| rings3 | JWF | yes | DONE (46) | rings with n folds |
| ringsubflame | JWF | yes | SKIP - 3D/SUBFLAME | ring sub-flame |
| ringtile | JWF | yes | TODO | ring tiling |
| ripple | JWF | yes | DONE (69) | ripple wave pattern |
| rippled | JWF | yes | TODO | rippled distortion, cheap |
| romanesco | JWF | yes | SKIP - 3D/BASE_SHAPE | romanesco pattern |
| rose_wf | JWF | yes | SKIP - BASE_SHAPE | rose curve |
| rosoni | JWF | no | TODO | rosoni pattern, 7 params, expensive |
| roundspher | JWF | yes | TODO | rounded spherical, trivial |
| roundspher3D | JWF | yes | SKIP - 3D | 3D rounded spherical |
| rsquares_js | JWF | yes | SKIP - BASE_SHAPE | R-squares IFS |

### S

| Name | Source | GPU | Status | Notes |
|------|--------|-----|--------|-------|
| sattractor_js | JWF | yes | TODO | strange attractor scatter, trivial |
| scrambly | JWF | no | TODO | scramble distortion, 4 params, trivial |
| scry | JWF/Apo | yes | DONE (51) | reciprocal r^2 variant |
| scry2 | JWF | yes | TODO | scry variant 2, 3 params, heavy |
| scry_3D | JWF | yes | TODO | scry 3D (but also useful 2D), cheap |
| seashell3D | JWF | yes | SKIP - 3D | 3D seashell |
| sec | JWF | yes | DONE (111) | complex sec |
| sec2_bs | JWF | yes | TODO | sec with extra params, expensive |
| secant2 | JWF | yes | DONE (81) | improved secant |
| sech | JWF | yes | DONE (117) | complex sech |
| sech2_bs | JWF | yes | TODO | sech with extra params, expensive |
| sechq | JWF | yes | SKIP - 3D | quaternion sech |
| secq | JWF | yes | SKIP - 3D | quaternion sec |
| separation | JWF/Apo | yes | DONE (103) | separated axis fold |
| shift | JWF | yes | TODO | shift distortion, 3 params, cheap |
| shredded | JWF | yes | SKIP - 3D | 3D shred |
| shredlin | JWF | yes | TODO | linear shred, 4 params, trivial |
| shredrad | JWF | yes | TODO | radial shred, 2 params, cheap |
| siercarpet_js | JWF | yes | TODO | Sierpinski carpet IFS, expensive |
| sigmoid | JWF | yes | TODO | sigmoid function, 2 params, medium |
| sin | JWF | yes | DONE (108) | complex sin |
| sin2_bs | JWF | yes | TODO | sin with extra params, expensive |
| sineblur | JWF | yes | TODO | sine-weighted blur |
| sinh | JWF | yes | DONE (114) | complex sinh |
| sinh2_bs | JWF | yes | TODO | sinh with extra params, expensive |
| sinhq | JWF | yes | SKIP - 3D | quaternion sinh |
| sinq | JWF | yes | SKIP - 3D | quaternion sin |
| sinusgrid | JWF | yes | SKIP - 3D | 3D sinus grid |
| sinusoidal | JWF | yes | DONE (1) | sin(x), sin(y) |
| sinusoidal3d | JWF | yes | SKIP - 3D | 3D sinusoidal |
| sintrange | JWF | yes | TODO | sin-tan range, cheap |
| snowflake_wf | JWF | yes | SKIP - BASE_SHAPE | snowflake |
| solidangle3D | JWF | yes | SKIP - 3D | 3D solid angle |
| sph3D | JWF/ApoPlugin | yes | SKIP - 3D | 3D spherical |
| sphere_nja | JWF | yes | SKIP - 3D | 3D sphere |
| spherecrop | JWF | yes | SKIP - 3D | 3D sphere crop |
| spherical | JWF | yes | DONE (2) | 1/r^2 inversion |
| spherical3D | JWF | yes | SKIP - 3D | 3D spherical |
| spherical3D_wf | JWF | yes | SKIP - 3D | 3D spherical WF |
| sphericalN | JWF | yes | TODO | spherical with N-fold, 2 params, heavy |
| sphtiling3v2 | JWF | no | TODO | spherical tiling, 9 params, cheap |
| spiral | JWF | yes | DONE (9) | (cos(theta)+sin(r))/r |
| spiralwing | JWF | yes | DONE (61) | spiral wing pattern |
| spirograph | JWF | yes | TODO | spirograph, 9 params, medium |
| spirograph3D | JWF | yes | SKIP - 3D | 3D spirograph |
| spligon | JWF | yes | TODO | split polygon, 6 params, cheap |
| split | JWF | yes | DONE (102) | sign-based split |
| splitbrdr | JWF | yes | DONE (122) | split border tiling |
| splits | JWF/Apo | yes | DONE (30) | x/y split offset |
| splits3D | JWF | yes | SKIP - 3D | 3D splits |
| spliptic_bs | JWF | yes | TODO | spliptic variant, 2 params, heavy |
| square | JWF | yes | DONE (91) | uniform random square |
| square3D | JWF | yes | SKIP - 3D | 3D square |
| squarize | JWF | yes | TODO | circle-to-square mapping, medium |
| squircular | JWF | yes | TODO | squircle mapping, medium |
| squirrel | JWF | yes | TODO | squirrel distortion, 2 params, heavy |
| squish | JWF | yes | TODO | squish distortion, cheap |
| starblur | JWF | yes | TODO | star-shaped blur |
| starfractal | JWF | yes | SKIP - BASE_SHAPE | star fractal IFS |
| stripes | JWF | yes | DONE (67) | stripe warp pattern |
| stripfit | JWF | yes | TODO | strip fit tiling, cheap |
| stwin | JWF | yes | TODO | S-twin distortion, 7 params, cheap |
| subflame_wf | JWF | yes | SKIP - 3D/SUBFLAME | sub-flame reference |
| sunflower | JWF | yes | SKIP - BASE_SHAPE | sunflower pattern |
| sunvoroni | JWF | yes | SKIP - BASE_SHAPE | sunflower Voronoi |
| super_shape | JWF | yes | DONE (79) | superformula |
| superShape3d | JWF | yes | SKIP - 3D | 3D superformula |
| svg_wf | JWF | no | SKIP - IMAGE | SVG file rendering |
| svf | JWF | yes | SKIP - 3D | 3D surface |
| svensson_js | JWF | yes | TODO | Svensson attractor, heavy |
| swirl | JWF | yes | DONE (3) | sin/cos of r^2 rotation |
| swirl3 | JWF | yes | TODO | swirl variant 3, medium |
| swirl3D_wf | JWF | yes | SKIP - 3D | 3D swirl |
| sym_bg1..bg7 | JWF | yes | TODO | band group symmetry 1-7, trivial |
| sym_ng1..ng17 | JWF | yes | TODO | net group symmetry 1-17, trivial-medium |
| synth | JWF | no | TODO | synthesis variation, 35 params, heavy |
| szubieta | JWF | yes | SKIP - BASE_SHAPE | Szubieta pattern |

### T

| Name | Source | GPU | Status | Notes |
|------|--------|-----|--------|-------|
| tan | JWF | yes | DONE (110) | complex tan |
| tan2_bs | JWF | yes | TODO | tan with extra params, expensive |
| tancos | JWF | yes | TODO | tan*cos combination, cheap |
| tangent | JWF | yes | DONE (34) | tan(x)/cos(y) |
| tangent3D | JWF | yes | SKIP - 3D | 3D tangent |
| tanh | JWF | yes | DONE (116) | complex tanh |
| tanh2_bs | JWF | yes | TODO | tanh with extra params, expensive |
| tanhq | JWF | yes | SKIP - 3D | quaternion tanh |
| tanq | JWF | yes | SKIP - 3D | quaternion tan |
| taprats | JWF | yes | SKIP - BASE_SHAPE | Islamic tiling, complex |
| target | JWF | yes | TODO | target ring pattern, 3 params, heavy |
| target_sp | JWF | yes | TODO | target with spiral, 4 params, heavy |
| taurus | JWF | yes | SKIP - 3D | 3D torus |
| terrain3D | JWF | yes | SKIP - 3D | 3D terrain mesh |
| text_wf | JWF | no | SKIP - IMAGE | text rendering |
| threeply | JWF | yes | SKIP - SIMULATION | Threeply iterator |
| threepoint_js | JWF | yes | TODO | three-point IFS, trivial |
| tile_hlp | JWF | yes | TODO | tile helper, cheap |
| tile_log | JWF | yes | SKIP - 3D | 3D tile log |
| tile_reverse | JWF | yes | TODO | reverse tiling, 4 params, cheap |
| torus3D | JWF | yes | SKIP - 3D | 3D torus |
| tqmirror | JWF | yes | TODO | TQ mirror, 22 params, cheap |
| trade | JWF | yes | TODO | trade circle swap, 4 params, heavy |
| tree_js | JWF | no | SKIP - BASE_SHAPE | tree IFS, stateful |
| triantruchet | JWF | no | TODO | triangular truchet, 4 params, cheap |
| triangle | JWF | yes | SKIP - 3D | 3D triangle |
| triprism3D | JWF | yes | SKIP - 3D | 3D triangular prism |
| truchet | JWF | yes | SKIP - BASE_SHAPE | truchet tiling, complex |
| truchet2 | JWF | yes | SKIP - BASE_SHAPE | truchet v2 |
| truchet_ae | JWF | yes | SKIP - BASE_SHAPE | truchet artistic edition |
| truchet_fill | JWF | yes | TODO | truchet fill pattern, expensive |
| truchet_hex_crop | JWF | yes | TODO | hex truchet crop, expensive |
| truchet_hex_fill | JWF | yes | TODO | hex truchet fill, expensive |
| truchetflow | JWF | no | TODO | truchet flow lines, medium |
| tunnel | JWF | yes | SKIP - 3D/BASE_SHAPE | tunnel shape |
| twintrian | JWF | yes | DONE (93) | twin triangular scatter |
| twoface | JWF | yes | TODO | two-face distortion, trivial |

### U-V

| Name | Source | GPU | Status | Notes |
|------|--------|-----|--------|-------|
| unpolar | JWF | yes | TODO | inverse of polar, medium |
| vibration2 | JWF | yes | TODO | vibration effect, 26 params, heavy |
| vogel | JWF | yes | TODO | Vogel spiral, 2 params, cheap |
| voron | JWF | yes | TODO | Voronoi distortion, 5 params, heavy |
| vortex | JWF | yes | TODO | vortex rotation, cheap |

### W

| Name | Source | GPU | Status | Notes |
|------|--------|-----|--------|-------|
| w | JWF | no | TODO | W tiling, 13 params, expensive |
| waffle | JWF | yes | TODO | waffle grid pattern |
| wallpaper_js | JWF | no | TODO | wallpaper tiling, cheap |
| wangtiles | JWF | yes | SKIP - BASE_SHAPE | Wang tiling, complex |
| waveblur_wf | JWF | yes | SKIP - 3D | 3D wave blur |
| waves | JWF | yes | DONE (15) | affine-coeff sinusoidal offset |
| waves2 | JWF/Apo | yes | DONE (74) | parametric waves |
| waves22 | JWF | yes | TODO | waves v22, 8 params, heavy |
| waves23 | JWF | yes | TODO | waves v23, 4 params, cheap |
| waves2_3D | JWF | yes | SKIP - 3D | 3D waves2 |
| waves2_radial | JWF | yes | TODO | radial waves2, 6 params, medium |
| waves2_wf | JWF | yes | TODO | waves2 with extra params, medium |
| waves2b | JWF | yes | TODO | waves2 variant b, 10 params, medium |
| waves3 | JWF | yes | DONE (53) | waves with cross-freq modulation |
| waves3_wf | JWF | yes | TODO | waves3 variant, heavy |
| waves4 | JWF | yes | TODO | waves v4, 6 params, medium |
| waves42 | JWF | yes | TODO | waves v42, 7 params, medium |
| waves4_wf | JWF | yes | TODO | waves4 variant, heavy |
| wdisc | JWF | yes | TODO | weighted disc, cheap |
| wedge | JWF/Apo | yes | DONE (95) | angular wedge fold |
| wedge_julia | JWF | yes | DONE (94) | julia with wedge fold |
| wedge_sph | JWF/Apo | yes | DONE (96) | spherical with wedge fold |
| whirligig | JWF | yes | SKIP - BASE_SHAPE | whirligig pattern |
| whitney_umbrella | JWF | yes | SKIP - 3D | 3D Whitney umbrella |
| whorl | JWF/ApoPlugin | yes | DONE (57) | inside/outside whorl rotation |
| woggle_js | JWF | yes | TODO | woggle IFS, medium |

### X-Z

| Name | Source | GPU | Status | Notes |
|------|--------|-----|--------|-------|
| x | JWF | no | TODO | x tiling, 12 params, expensive |
| xerf | JWF | yes | SKIP - 3D | 3D error function |
| xheart | JWF | yes | TODO | heart variant, 2 params, trivial |
| xheart_blur_wf | JWF | yes | SKIP - BASE_SHAPE | xheart blur |
| xtrb | JWF | yes | TODO | extreme boundary, 6 params, trivial |
| y | JWF | no | TODO | y tiling, 12 params, expensive |
| yin_yang | JWF | yes | TODO | yin-yang fold, 5 params, medium |
| z | JWF | no | TODO | z tiling, 12 params, expensive |
| zblur | JWF/Apo | yes | SKIP - 3D | z-axis blur |
| zcone | JWF/Apo | yes | SKIP - 3D | z-cone |
| zscale | JWF/Apo | yes | SKIP - 3D | z-scale |
| ztranslate | JWF/Apo | yes | SKIP - 3D | z-translate |
| ztwister | JWF | yes | TODO | z-twister (2D usable), medium |

---

## Apophysis 7X Built-in Variations (29)

These are compiled into the Apophysis binary, not loaded as plugins.
All are also present in JWildfire.

| # | Name | flame-sheep Status |
|---|------|--------------------|
| 0 | linear | DONE (0) |
| 1 | flatten | SKIP - UTILITY |
| 2 | sinusoidal | DONE (1) |
| 3 | spherical | DONE (2) |
| 4 | swirl | DONE (3) |
| 5 | horseshoe | DONE (4) |
| 6 | polar | DONE (5) |
| 7 | disc | DONE (8) |
| 8 | spiral | DONE (9) |
| 9 | hyperbolic | DONE (10) |
| 10 | diamond | DONE (11) |
| 11 | eyefish | DONE (27) |
| 12 | bubble | DONE (28) |
| 13 | cylinder | DONE (29) |
| 14 | noise | DONE (80) |
| 15 | blur | DONE (75) |
| 16 | gaussian_blur | DONE (76) |
| 17 | zblur | SKIP - 3D |
| 18 | blur3D | SKIP - 3D |
| 19 | pre_blur | SKIP - UTILITY |
| 20 | pre_zscale | SKIP - 3D |
| 21 | pre_ztranslate | SKIP - 3D |
| 22 | pre_rotate_x | SKIP - 3D |
| 23 | pre_rotate_y | SKIP - 3D |
| 24 | zscale | SKIP - 3D |
| 25 | ztranslate | SKIP - 3D |
| 26 | zcone | SKIP - 3D |
| 27 | post_rotate_x | SKIP - 3D |
| 28 | post_rotate_y | SKIP - 3D |

## Apophysis 7X Registered Variations (plugins)

These are loaded from separate `.pas` files (built-in) or `.dll` plugins.

| File | Variation Name | flame-sheep Status |
|------|---------------|--------------------|
| varAuger.pas | auger | DONE (63) |
| varBipolar.pas | bipolar | DONE (100) |
| varBlurCircle.pas | blur_circle | TODO |
| varBlurPixelize.pas | blur_pixelize | TODO |
| varBlurZoom.pas | blur_zoom | TODO |
| varBwraps.pas | bwraps | TODO |
| varCrop.pas | crop | SKIP - CROP |
| varCross.pas | cross | DONE (35) |
| varCurl.pas | curl | DONE (37) |
| varCurl3D.pas | curl3D | SKIP - 3D |
| varElliptic.pas | elliptic | DONE (88) |
| varEpispiral.pas | epispiral | DONE (52) |
| varEscher.pas | escher | DONE (87) |
| varFalloff2.pas | falloff2 | SKIP - 3D |
| varFan2.pas | fan2 | DONE (25) |
| varFoci.pas | foci | DONE (105) |
| varHemisphere.pas | hemisphere | SKIP - 3D |
| varJulia3Djf.pas | julia3Djf | SKIP - 3D |
| varJulia3Dz.pas | julia3Dz | SKIP - 3D |
| varJuliaN.pas | julian | DONE (32) |
| varJuliaScope.pas | juliascope | DONE (33) |
| varLazysusan.pas | lazysusan | DONE (97) |
| varLog.pas | log | DONE (121) |
| varLoonie.pas | loonie | DONE (50) |
| varMobius.pas | mobius | DONE (47) |
| varNGon.pas | ngon | DONE (49) |
| varPolar2.pas | polar2 | DONE (104) |
| varPostBwraps.pas | post_bwraps | TODO |
| varPostCrop.pas | post_crop | SKIP - CROP |
| varPostCurl.pas | post_curl | TODO |
| varPostCurl3D.pas | post_curl3D | SKIP - 3D |
| varPostFalloff2.pas | post_falloff2 | SKIP - 3D |
| varPreBwraps.pas | pre_bwraps | TODO |
| varPreCrop.pas | pre_crop | SKIP - CROP |
| varPreDisc.pas | pre_disc | TODO |
| varPreFalloff2.pas | pre_falloff2 | SKIP - 3D |
| varPreSinusoidal.pas | pre_sinusoidal | SKIP - UTILITY |
| varPreSpherical.pas | pre_spherical | TODO |
| varRadialBlur.pas | radial_blur | DONE (77) |
| varRectangles.pas | rectangles | DONE (38) |
| varRings2.pas | rings2 | DONE (26) |
| varScry.pas | scry | DONE (51) |
| varSeparation.pas | separation | DONE (103) |
| varSplits.pas | splits | DONE (30) |
| varWaves2.pas | waves2 | DONE (74) |
| varWedge.pas | wedge | DONE (95) |
| varpdj.pas | pdj | DONE (24) |

## Apophysis Plugin DLLs

| Plugin Dir | Variation Name | flame-sheep Status |
|------------|---------------|--------------------|
| bcircle | bcircle | TODO |
| blur_circle | blur_circle | TODO (dup of varBlurCircle) |
| bwraps | bwraps2 | TODO |
| circlize | circlize | TODO |
| circlize2 | circlize2 | TODO |
| collideoscope | collideoscope | DONE (62) |
| curl2 | curl2 | TODO |
| curl_sp | curl_sp | SKIP - 3D |
| eSwirl | eSwirl | TODO |
| extrude | extrude | SKIP - 3D |
| falloff | falloff | SKIP - 3D |
| falloff3 | falloff3 | SKIP - 3D |
| gridout | gridout | TODO |
| julian3 | julian3 | TODO |
| julian3Dx | julian3Dx | TODO |
| juni | juni | TODO |
| murl | murl | TODO |
| murl2 | murl2 | TODO |
| petal | petal | TODO |
| popcorn2 | popcorn2 | DONE (106) |
| post_bwraps | post_bwraps2 | TODO |
| post_circlecrop | post_circlecrop | SKIP - CROP |
| post_curl3D | post_curl3D | SKIP - 3D |
| post_falloff3 | post_falloff3 | SKIP - 3D |
| post_julian2 | post_julian2 | TODO |
| post_log | post_log | TODO |
| post_mirror | post_mirror | TODO |
| post_murl | post_murl | TODO |
| post_spherical | post_spherical | TODO |
| post_stun | post_stun | TODO |
| pre_bwraps | pre_bwraps2 | TODO |
| pre_circlecrop | pre_circlecrop | SKIP - CROP |
| pre_curl | pre_curl | TODO |
| pre_falloff3 | pre_falloff3 | SKIP - 3D |
| pre_log | pre_log | TODO |
| pre_xfalloff2 | pre_xfalloff2 | SKIP - 3D |
| sph3D | sph3D | SKIP - 3D |
| stwins | stwins | TODO |
| whorl | whorl | DONE (57) |
| xcurl2 | xcurl2 | TODO |
| xheart | xheart | TODO |
| xhyperbol | xhyperbol | TODO |
| xtrb | xtrb | TODO |

---

## Priority Candidates for Implementation

Top variations to implement next, based on: visual distinctiveness, GPU support,
low complexity, presence in both JWildfire and Apophysis (wide usage).

### Tier 1: Trivial/Cheap, High Impact

| Name | Cost | Params | Notes |
|------|------|--------|-------|
| boarders2 | cheap | 3 | extended boarders, already have boarders |
| cylinder2 | cheap | 0 | cylinder variant |
| vortex | cheap | 0 | vortex rotation |
| gridout | cheap | 0 | grid-out fold |
| gridout2 | cheap | 4 | extended grid-out |
| holesq | cheap | 0 | square hole |
| idisc | cheap | 0 | inverted disc |
| loonie3 | cheap | 0 | loonie variant |
| wdisc | cheap | 0 | weighted disc |
| invpolar | cheap | 0 | inverse polar |
| bi_linear | trivial | 0 | bilinear mapping |
| twoface | trivial | 0 | two-face distortion |
| flipy | trivial | 0 | y-axis flip |
| hypershift | trivial | 2 | hyperbolic shift |
| pyramid | trivial | 0 | pyramid reflection fold |
| roundspher | trivial | 0 | rounded spherical |
| xheart | trivial | 2 | heart variant |
| anamorphcyl | trivial | 0 | anamorphic cylinder |
| acosech | trivial | 0 | inv hyperbolic cosecant |
| acosh | trivial | 0 | inv hyperbolic cosine |
| acoth | trivial | 0 | inv hyperbolic cotangent |
| arcsinh | trivial | 0 | inv sinh |
| arctanh | trivial | 0 | inv tanh |

### Tier 2: Medium, Visually Interesting

| Name | Cost | Params | Notes |
|------|------|--------|-------|
| cpow2 | medium | 4 | complex power variant |
| juliac | medium | 3 | Julia-C |
| murl | medium | 2 | murl distortion |
| npolar | heavy | 2 | n-polar symmetry |
| oscilloscope2 | medium | 6 | enhanced oscilloscope |
| circlesplit | medium | 2 | circle split |
| asteria | medium | 1 | star fold |
| sigmoid | medium | 2 | sigmoid function |
| epispiral_wf | medium | 1 | epispiral variant |
| heart_wf | medium | 4 | heart with params |
| unpolar | medium | 0 | inverse polar |
| squarize | medium | 0 | circle-to-square mapping |
| squircular | medium | 0 | squircle mapping |
| invsquircular | medium | 0 | inverse squircle |
| circleLinear | cheap | 6 | circle-linear hybrid |
| circlize | cheap | 1 | square-to-circle |
| linearT | cheap | 2 | linear with tilt |
| bCollide | heavy | 2 | hyperbolic collide |
| bMod | heavy | 2 | hyperbolic modulus |
| bSwirl | heavy | 2 | hyperbolic swirl |
| bTransform | heavy | 4 | hyperbolic transform |
| eCollide | heavy | 2 | exponential collide |
| eJulia | heavy | 1 | exponential Julia |
| eMod | heavy | 2 | exponential modulus |
| mobiusN | heavy | 1 | Moebius N-fold |

### Tier 3: Symmetry/Stochastic (Batch Add)

| Name | Cost | Params | Notes |
|------|------|--------|-------|
| sym_bg1..bg7 | trivial | 2 | 7 band group symmetries |
| sym_ng1..ng17 | trivial-medium | 2-5 | 17 net group symmetries |
| glynnSim1..3 | medium | 4-6 | Glynn simulations |
| apollony | medium | 0 | Apollonian gasket |
| yin_yang | medium | 5 | yin-yang fold |
