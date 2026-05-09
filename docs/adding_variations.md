# Adding New Variations

Target audience: Claude (or any AI writing variation code).

You WILL make mistakes implementing variations. The formulas look similar,
the conventions are subtle, and you'll confidently write code that passes
CPU/GPU agreement tests while being wrong in the same way on both sides.

This document exists because in one session we found 11 bugs in variations
you wrote, including a completely wrong butterfly formula, a missing sqrt
in cross, a factor-of-2 error in fan, and the wrong power of r in julian.
All of these passed CPU/GPU golden tests because you made the same error
in both implementations.

## The Checklist

For EVERY new variation:

### 1. Find the authoritative source

- If it's in flam3: read `~/projects/flam3/variations.c`. This is ground truth.
- If it's in JWildfire only: read `~/projects/JWildfire-src/src/org/jwildfire/create/tina/variation/`
- If it's in Apophysis: read `~/projects/apophysis-7x/src/Variations/`
- If you can't find a reference: SAY SO. Don't guess the formula.

### 2. Watch for these specific traps

**atan2 convention.** flam3 has TWO atan precalcs:
- `precalc_atan` = `atan2(tx, ty)` = `atan2(x, y)` — used by polar, handkerchief, heart, ex, fan, fan2, blob, disk, spiral, diamond
- `precalc_atanyx` = `atan2(ty, tx)` = `atan2(y, x)` — used by julian, juliascope, wedge, escher, cpow, ngon

Our Polar struct: `pc.theta = atan(p.x, p.y)` matches `precalc_atan`. `pc.phi = atan(p.y, p.x)` matches `precalc_atanyx`.

If you use the wrong one, the output rotates 90°. Visually it might still look like a fractal. You won't notice.

**pow(r, cn) vs pow(r², cn).** flam3 computes `pow(precalc_sumsq, cn)` which is `pow(r², cn) = pow(r, 2*cn)`. If you write `pow(r, cn)` you'll get the wrong radial scaling. This is what happened with julian — the output was structurally similar but with dampened recursion depth.

**GLSL mod() vs C fmod().** `mod(-0.3, 1.0)` = 0.7 in GLSL, = -0.3 in C. Use `x - y * trunc(x/y)` in GLSL if you need C fmod behavior. This bit us in fan.

**Precalc parameters.** Some variations (disc2, super_shape, perspective, radial_blur, wedge_julia) have precalculated values derived from their parameters. flam3 computes these once in a `_precalc` function. If you treat the derived values as independent parameters, you'll get wrong results. disc2's `cosadd` and `sinadd` are NOT independent — they're `cos(twist)-1` and `sin(twist)`.

**Weight convention.** Most variations: `p0 += weight * variation_output`. A few (whorl, loonie, arch, rays, blade, twintrian) use `weight` in non-standard ways inside the formula. Read the flam3 source carefully for these. Whorl uses `(weight - r)` as a denominator in BOTH branches.

**sin/cos of theta.** flam3's `precalc_sina/cosa` are sin/cos of `atan2(tx, ty)` — that's `atan2(x, y)`, the NON-standard one. These are used by spiral, hyperbolic, diamond, power, blob, rings, rings2. Our `pc.theta = atan(p.x, p.y)` matches this, so `sin(pc.theta)` and `cos(pc.theta)` are correct.

### 3. Implement in THREE places

1. **`flame_sheep/variations/_registry.py`** — add the index, param spec, add to PARAMETRIC_VARIATIONS if needed
2. **`flame_sheep/variations/_cpu.py`** — add the CPU implementation in `apply_variation_cpu()`
3. **`flame_sheep/shaders/variations.glsl`** — add the GLSL function AND the switch case in `apply_single_variation()`

Also update:
- **`flame_sheep/variations/_params.py`** — add `random_var_params()` entry and `_PARAM_RANGES` if parametric
- **`tests/variation_fixtures.py`** — add `VAR_POINTS` entry, `PARAM_FIXTURES` entry, `VAR_PARAM_SETS` extras, add to `RANDOM_VARIATIONS` if uses RNG

### 4. Verify against the reference source

Do NOT just verify CPU == GPU. That catches implementation typos but not formula errors.

**For flam3 variations:**
- Add the variation to `tools/flam3_variation_oracle.c`
- Compile and run: `cc -o /tmp/flam3_oracle tools/flam3_variation_oracle.c -lm && /tmp/flam3_oracle > tests/flam3_reference.csv`
- Add to `EXACT_VARIATIONS` or `RNG_VARIATIONS` in `tests/test_flam3_oracle.py`
- Run: `pytest tests/test_flam3_oracle.py`

**For JWildfire-only variations:**
- Write mathematical invariant tests where possible
- Cross-reference against the Java source for non-obvious formulas

### 5. Test with multiple parameter configurations

Default parameters hide bugs. Add at least 2 param configs to `VAR_PARAM_SETS` in `tests/variation_fixtures.py`:
- The default config
- A config that exercises conditional branches differently (negative values, extreme values, boundary cases)

This is what found HEX_MODULUS (param name mismatch), WEDGE (floor boundary), and disc2 (precalc model).

### 6. Choose test points carefully

Every variation needs its own entry in `VAR_POINTS`. No fallback to generic points. Choose points based on the variation's behavior:
- Has `if x < 0`: include points in all four quadrants
- Has `if r < 1`: include points inside and outside unit circle
- Uses `floor()`: include points near integer boundaries
- Divides by r: include near-zero points (but not exactly zero)
- Uses RNG: use `_RNG_COVERAGE` (24 points for branch diversity)

### 7. Regenerate and run

```bash
python tools/generate_golden_masters.py
pytest tests/test_golden_masters.py tests/test_gpu_golden.py tests/test_flam3_oracle.py tests/test_gpu_pipeline.py
```

All must pass. If a golden master test fails, regenerate AFTER verifying the new output is correct.

### 8. Update the XML parser if needed

If the variation has a different name in flam3 XML than in our registry, add an alias to `VAR_NAME_MAP` in `flame_sheep/esheep_parser.py`. Check how the sheep actually use it — parameter names in XML might differ from our internal names.

## Bug Patterns Found (2026-05-08/09)

| Bug | Variation | Root Cause | How Found |
|-----|-----------|-----------|-----------|
| vec2(0) default | pre_variations | Empty slots returned zero instead of input | Pipeline integration test |
| pow(r, cn) | julian, juliascope | Should be pow(r², cn) | Render comparison vs reference image |
| Wrong formula | butterfly | Completely different formula from flam3 | flam3 source comparison |
| Missing sqrt | cross | 1/(s²+eps) instead of sqrt(1/(s²+eps)) | flam3 source comparison |
| Branch sign | whorl | else branch: (weight-r) not (r-weight) | flam3 oracle agent |
| Param name | HEX_MODULUS | CPU read 'hex_modulus_size', spec says 'hex_size' | Extra param set test |
| Factor of 2 | fan, fan_param | Added dx instead of dx/2 | flam3 source comparison |
| mod vs fmod | fan (GLSL) | GLSL mod() != C fmod() for negatives | GPU/CPU comparison after fix |
| Wrong param model | disc2 | cosadd/sinadd are derived from twist, not independent | flam3 source comparison |
| Float boundary | SPLIT, WEDGE | cos(kπ)=0 or floor at exact integer | Extra param set test |
