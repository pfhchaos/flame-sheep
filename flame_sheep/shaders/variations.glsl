// ------------------------------------------------------------
// Variation functions
// Applied after affine transform to warp the point nonlinearly.
// Each function takes (x,y) and returns warped (x,y).
// Weight is applied by caller.
//
// Requires: rng_state, rng_next(), rng_float() from rng.glsl
// Requires: u_active_vars SSBO, PARAM_OFFSET define
// ------------------------------------------------------------

// Helper: r squared and r
float r2(vec2 p) { return dot(p, p); }
float r(vec2 p)  { return length(p); }
// atan2 — GLSL has atan(y,x) builtin
float theta(vec2 p) { return atan(p.x, p.y); }  // note: x/y intentional per flam3 spec
float phi(vec2 p)   { return atan(p.y, p.x); }

// --- Classic flam3 variations (0-14) ---

vec2 var_linear(vec2 p)       { return p; }
vec2 var_sinusoidal(vec2 p)   { return sin(p); }
vec2 var_spherical(vec2 p)    { return p / max(r2(p), 1e-6); }
vec2 var_swirl(vec2 p) {
    float rr = r2(p);
    return vec2(p.x * sin(rr) - p.y * cos(rr),
                p.x * cos(rr) + p.y * sin(rr));
}
vec2 var_horseshoe(vec2 p) {
    float ri = 1.0 / max(r(p), 1e-6);
    return ri * vec2((p.x - p.y) * (p.x + p.y), 2.0 * p.x * p.y);
}
vec2 var_polar(vec2 p) {
    return vec2(theta(p) / 3.14159265, r(p) - 1.0);
}
vec2 var_handkerchief(vec2 p) {
    float th = theta(p); float ri = r(p);
    return ri * vec2(sin(th + ri), cos(th - ri));
}
vec2 var_heart(vec2 p) {
    float th = theta(p); float ri = r(p);
    return ri * vec2(sin(th * ri), -cos(th * ri));
}
vec2 var_disk(vec2 p) {
    float th = theta(p); float ri = r(p);
    float piri = 3.14159265 * ri;
    return (th / 3.14159265) * vec2(sin(piri), cos(piri));
}
vec2 var_spiral(vec2 p) {
    float th = theta(p); float ri = r(p);
    return (1.0 / max(ri, 1e-6)) * vec2(cos(th) + sin(ri), sin(th) - cos(ri));
}
vec2 var_hyperbolic(vec2 p) {
    float th = theta(p); float ri = r(p);
    return vec2(sin(th) / max(ri, 1e-6), cos(th) * ri);
}
vec2 var_diamond(vec2 p) {
    float th = theta(p); float ri = r(p);
    return vec2(sin(th) * cos(ri), cos(th) * sin(ri));
}
vec2 var_ex(vec2 p) {
    float th = theta(p); float ri = r(p);
    float p0 = sin(th + ri); float p1 = cos(th - ri);
    return ri * vec2(p0*p0*p0 + p1*p1*p1, p0*p0*p0 - p1*p1*p1);
}
vec2 var_julia(vec2 p) {
    float sqr = sqrt(r(p));
    float th  = theta(p) * 0.5;
    // random sign flip — use rng for authentic julia behavior
    if (rng_next() % 2u == 0u) th += 3.14159265;
    return sqr * vec2(cos(th), sin(th));
}
vec2 var_bent(vec2 p) {
    vec2 q = p;
    if (q.x < 0.0) q.x *= 2.0;
    if (q.y < 0.0) q.y *= 0.5;
    return q;
}

// --- Parametric flam3 variations (15-29) ---

vec2 var_waves(vec2 p, int slot) {
    float freq_x = u_active_vars[slot + PARAM_OFFSET + 0];
    float freq_y = u_active_vars[slot + PARAM_OFFSET + 1];
    float amp_x  = u_active_vars[slot + PARAM_OFFSET + 2];
    float amp_y  = u_active_vars[slot + PARAM_OFFSET + 3];
    return vec2(p.x + freq_x * sin(p.y / max(amp_x*amp_x, 1e-6)),
                p.y + freq_y * sin(p.x / max(amp_y*amp_y, 1e-6)));
}
vec2 var_fisheye(vec2 p) {
    float ri = 2.0 / (r(p) + 1.0);
    return ri * vec2(p.y, p.x);  // note x/y swap
}
vec2 var_popcorn(vec2 p, int slot) {
    float cx = u_active_vars[slot + PARAM_OFFSET + 0];
    float cy = u_active_vars[slot + PARAM_OFFSET + 1];
    return vec2(p.x + cx * sin(tan(3.0 * p.y)),
                p.y + cy * sin(tan(3.0 * p.x)));
}
vec2 var_exponential(vec2 p) {
    return exp(p.x - 1.0) * vec2(cos(3.14159265 * p.y), sin(3.14159265 * p.y));
}
vec2 var_power(vec2 p) {
    float th = theta(p);
    return pow(r(p), sin(th)) * vec2(cos(th), sin(th));
}
vec2 var_cosine(vec2 p) {
    return vec2(cos(3.14159265 * p.x) * cosh(p.y),
               -sin(3.14159265 * p.x) * sinh(p.y));
}
vec2 var_rings(vec2 p, int slot) {
    float c = u_active_vars[slot + PARAM_OFFSET + 0];
    float cc = c * c + 1e-6;
    float ri = r(p);
    float th = theta(p);
    float rr = mod(ri + cc, 2.0 * cc) - cc + ri * (1.0 - cc);
    return rr * vec2(cos(th), sin(th));
}
vec2 var_fan(vec2 p, int slot) {
    float c = u_active_vars[slot + PARAM_OFFSET + 0];
    float f = u_active_vars[slot + PARAM_OFFSET + 1];
    float th = theta(p); float ri = r(p);
    float t = 3.14159265 * c * c + 1e-6;
    float th2 = (mod(th + f, 2.0*t) > t) ? th - t : th + t;
    return ri * vec2(cos(th2), sin(th2));
}
vec2 var_blob(vec2 p, int slot) {
    float low   = u_active_vars[slot + PARAM_OFFSET + 0];
    float high  = u_active_vars[slot + PARAM_OFFSET + 1];
    float waves = u_active_vars[slot + PARAM_OFFSET + 2];
    float ri = r(p); float th = theta(p);
    float rr = ri * (low + (high - low) * (0.5 + 0.5 * sin(waves * th)));
    return rr * vec2(sin(th), cos(th));
}
vec2 var_pdj(vec2 p, int slot) {
    float a = u_active_vars[slot + PARAM_OFFSET + 0];
    float b = u_active_vars[slot + PARAM_OFFSET + 1];
    float c = u_active_vars[slot + PARAM_OFFSET + 2];
    float d = u_active_vars[slot + PARAM_OFFSET + 3];
    return vec2(sin(a * p.y) - cos(b * p.x),
                sin(c * p.x) - cos(d * p.y));
}
vec2 var_fan2(vec2 p, int slot) {
    float fx = u_active_vars[slot + PARAM_OFFSET + 0];
    float fy = u_active_vars[slot + PARAM_OFFSET + 1];
    float ri = r(p); float th = theta(p);
    float dx = 3.14159265 * fx * fx + 1e-6;
    float dx2 = dx * 0.5;
    float t = th + fy - floor((th + fy) / dx) * dx;
    float a = (t > dx2) ? th - dx2 : th + dx2;
    return ri * vec2(sin(a), cos(a));
}
vec2 var_rings2(vec2 p, int slot) {
    float val = u_active_vars[slot + PARAM_OFFSET + 0];
    float _dx = val * val + 1e-6;
    float l = r(p);
    if (l < 1e-10) return p;
    float k = floor((l / _dx + 1.0) * 0.5);
    float rr = 2.0 - _dx * (k * 2.0 / l + 1.0);
    return rr * p;
}
vec2 var_eyefish(vec2 p)  { return 2.0 / (r(p) + 1.0) * p; }
vec2 var_bubble(vec2 p)   { return 4.0 / (r2(p) + 4.0) * p; }
vec2 var_cylinder(vec2 p) { return vec2(sin(p.x), p.y); }

// --- Extended variations (JWildfire / flam3 inspired, 30-37) ---

vec2 var_splits(vec2 p, int slot) {
    float sx = u_active_vars[slot + PARAM_OFFSET + 0];
    float sy = u_active_vars[slot + PARAM_OFFSET + 1];
    return vec2(p.x >= 0.0 ? p.x + sx : p.x - sx,
                p.y >= 0.0 ? p.y + sy : p.y - sy);
}

vec2 var_cloverleaf(vec2 p) {
    float a = phi(p);
    float ri = r(p);
    float rr = ri * (sin(2.0 * a) + 0.25 * sin(6.0 * a));
    return rr * vec2(cos(a), sin(a));
}

vec2 var_julian(vec2 p, int slot) {
    float power = u_active_vars[slot + PARAM_OFFSET + 0];
    float dist  = u_active_vars[slot + PARAM_OFFSET + 1];
    float abs_n = abs(power);
    float cn = dist / power * 0.5;
    float t_rand = floor(abs_n * rng_float());
    float a = (phi(p) + 6.28318530 * t_rand) / power;
    float ri = pow(max(r(p), 1e-6), cn);
    return ri * vec2(cos(a), sin(a));
}

vec2 var_juliascope(vec2 p, int slot) {
    float power = u_active_vars[slot + PARAM_OFFSET + 0];
    float dist  = u_active_vars[slot + PARAM_OFFSET + 1];
    float abs_n = abs(power);
    float cn = dist / power * 0.5;
    float t_rand = floor(abs_n * rng_float());
    float ang = phi(p);
    // "scope" part: randomly negate angle for reflective symmetry
    if (rng_next() % 2u == 0u) ang = -ang;
    float a = (ang + 6.28318530 * t_rand) / power;
    float ri = pow(max(r(p), 1e-6), cn);
    return ri * vec2(cos(a), sin(a));
}

vec2 var_tangent(vec2 p) {
    return vec2(sin(p.x) / max(abs(cos(p.y)), 1e-6),
                tan(p.y));
}

vec2 var_cross(vec2 p) {
    float d = p.x * p.x - p.y * p.y;
    float s = 1.0 / max(d * d, 1e-6);
    return s * p;
}

vec2 var_butterfly(vec2 p) {
    // weight factor from flam3 spec: 4 / (sqrt(3) * pi)
    float w = 1.3029400;
    float ri = r(p);
    float y2 = p.y * 2.0;
    return w * vec2(p.y * (2.0 * p.x / max(ri, 1e-6)), ri);
}

vec2 var_curl(vec2 p, int slot) {
    float c1 = u_active_vars[slot + PARAM_OFFSET + 0];
    float c2 = u_active_vars[slot + PARAM_OFFSET + 1];
    // Complex division: p / (1 + c1*p + c2*p^2)
    float x2 = p.x * p.x;
    float y2 = p.y * p.y;
    float re = 1.0 + c1 * p.x + c2 * (x2 - y2);
    float im =       c1 * p.y + c2 * 2.0 * p.x * p.y;
    float d = max(re * re + im * im, 1e-6);
    return vec2((p.x * re + p.y * im) / d,
                (p.y * re - p.x * im) / d);
}

// --- Tiling variations (38-41) ---

vec2 var_rectangles(vec2 p, int slot) {
    float rx = u_active_vars[slot + PARAM_OFFSET + 0];
    float ry = u_active_vars[slot + PARAM_OFFSET + 1];
    float ox = (abs(rx) < 1e-6) ? p.x : (2.0 * floor(p.x / rx) + 1.0) * rx - p.x;
    float oy = (abs(ry) < 1e-6) ? p.y : (2.0 * floor(p.y / ry) + 1.0) * ry - p.y;
    return vec2(ox, oy);
}

vec2 var_checks(vec2 p, int slot) {
    float cs = u_active_vars[slot + PARAM_OFFSET + 0];
    float cx = u_active_vars[slot + PARAM_OFFSET + 1];
    float cy = u_active_vars[slot + PARAM_OFFSET + 2];
    float ics = 1.0 / max(cs, 1e-6);
    int cell = int(round(p.x * ics)) + int(round(p.y * ics));
    if (cell % 2 == 0) {
        return vec2(p.x + cx, p.y);
    } else {
        return vec2(p.x, p.y + cy);
    }
}

vec2 var_hex_modulus(vec2 p, int slot) {
    float size = u_active_vars[slot + PARAM_OFFSET + 0];
    float hsize = 0.86602540 / max(size, 1e-6);  // sqrt(3)/2
    float weight = 1.0 / 0.86602540;
    // Convert to hex coords
    float hx = (0.57735027 * p.x * hsize - p.y * hsize / 3.0);
    float hz = (2.0 * p.y * hsize / 3.0);
    float hy = -hx - hz;
    // Round to nearest hex cell
    float rx = round(hx), ry = round(hy), rz = round(hz);
    float xd = abs(rx - hx), yd = abs(ry - hy), zd = abs(rz - hz);
    if (xd > yd && xd > zd) rx = -ry - rz;
    else if (yd > zd) ry = -rx - rz;
    // Hex-local coordinates
    float fx = hx - rx;
    float fy = hz - rz;
    return vec2(fx * weight, fy * weight);
}

vec2 var_kaleidoscope(vec2 p, int slot) {
    float pull = u_active_vars[slot + PARAM_OFFSET + 0];
    float rot = u_active_vars[slot + PARAM_OFFSET + 1];
    float n = max(u_active_vars[slot + PARAM_OFFSET + 2], 2.0);
    // Rotate input
    float cr = cos(rot), sr = sin(rot);
    vec2 rp = vec2(cr * p.x - sr * p.y, sr * p.x + cr * p.y);
    // Convert to polar
    float a = atan(rp.y, rp.x);
    float ri = length(rp) + pull;
    // Mirror into sector
    float sector = 6.28318530 / n;
    a = mod(a, sector);
    if (a > sector * 0.5) a = sector - a;
    return ri * vec2(cos(a), sin(a));
}

// --- Symmetry-generating variations (42-46) ---

vec2 var_icon(vec2 p, int slot) {
    // Icon attractor: complex polynomial with n-fold rotational symmetry
    // z' = lambda*z + alpha*conj(z)^(n-1) + beta*exp(i*omega)*conj(z)^(n-3)
    float n = u_active_vars[slot + PARAM_OFFSET + 0];      // degree (n-fold symmetry)
    float lam = u_active_vars[slot + PARAM_OFFSET + 1];    // lambda
    float alp = u_active_vars[slot + PARAM_OFFSET + 2];    // alpha
    float bet = u_active_vars[slot + PARAM_OFFSET + 3];    // beta
    float gam = u_active_vars[slot + PARAM_OFFSET + 4];    // gamma (unused in basic form)
    float ome = u_active_vars[slot + PARAM_OFFSET + 5];    // omega

    // z = p.x + i*p.y, conj(z) = p.x - i*p.y
    float zr = p.x, zi = p.y;
    float cr = p.x, ci = -p.y;  // conjugate

    // conj(z)^(n-1) via polar
    float cr_r = length(vec2(cr, ci));
    float cr_a = atan(ci, cr);
    float nm1 = n - 1.0;
    float cpow_r = pow(max(cr_r, 1e-10), nm1);
    float cpow_a = nm1 * cr_a;
    float c1r = cpow_r * cos(cpow_a);
    float c1i = cpow_r * sin(cpow_a);

    // conj(z)^(n-3) via polar
    float nm3 = n - 3.0;
    float cpow3_r = pow(max(cr_r, 1e-10), nm3);
    float cpow3_a = nm3 * cr_a;
    float c3r = cpow3_r * cos(cpow3_a);
    float c3i = cpow3_r * sin(cpow3_a);

    // exp(i*omega) * conj(z)^(n-3)
    float eor = cos(ome), eoi = sin(ome);
    float ec3r = eor * c3r - eoi * c3i;
    float ec3i = eor * c3i + eoi * c3r;

    // z' = lam*z + alp*conj(z)^(n-1) + bet*exp(i*ome)*conj(z)^(n-3)
    float rx = lam * zr + alp * c1r + bet * ec3r;
    float ry = lam * zi + alp * c1i + bet * ec3i;
    return vec2(rx, ry);
}

vec2 var_sattractor(vec2 p, int slot) {
    // Symmetric attractor: m-fold rotational symmetry from roots of unity
    // Pick random k in [0,m), rotate point by 2*pi*k/m
    float m = max(u_active_vars[slot + PARAM_OFFSET + 0], 2.0);
    float k = floor(rng_float() * m);
    float angle = k * 6.28318530 / m;
    float ca = cos(angle), sa = sin(angle);
    return vec2(ca * p.x - sa * p.y, sa * p.x + ca * p.y);
}

// {{SYMMETRY_GROUPS}}

vec2 var_wallpaper(vec2 p, int slot) {
    int group = int(u_active_vars[slot + PARAM_OFFSET + 0]);
    group = clamp(group, 0, 16);
    int count = WALLPAPER_COUNTS[group];
    int offset = WALLPAPER_OFFSETS[group];
    int elem = int(rng_float() * float(count));
    elem = clamp(elem, 0, count - 1);
    int base = (offset + elem) * 6;
    float a = WALLPAPER_DATA[base + 0];
    float b = WALLPAPER_DATA[base + 1];
    float c = WALLPAPER_DATA[base + 2];
    float d = WALLPAPER_DATA[base + 3];
    float e = WALLPAPER_DATA[base + 4];
    float f = WALLPAPER_DATA[base + 5];
    return vec2(a * p.x + b * p.y + c, d * p.x + e * p.y + f);
}

vec2 var_frieze(vec2 p, int slot) {
    int group = int(u_active_vars[slot + PARAM_OFFSET + 0]);
    group = clamp(group, 0, 6);
    int count = FRIEZE_COUNTS[group];
    int offset = FRIEZE_OFFSETS[group];
    int elem = int(rng_float() * float(count));
    elem = clamp(elem, 0, count - 1);
    int base = (offset + elem) * 6;
    float a = FRIEZE_DATA[base + 0];
    float b = FRIEZE_DATA[base + 1];
    float c = FRIEZE_DATA[base + 2];
    float d = FRIEZE_DATA[base + 3];
    float e = FRIEZE_DATA[base + 4];
    float f = FRIEZE_DATA[base + 5];
    return vec2(a * p.x + b * p.y + c, d * p.x + e * p.y + f);
}

vec2 var_rings3(vec2 p, int slot) {
    float val = u_active_vars[slot + PARAM_OFFSET + 0];
    float n   = u_active_vars[slot + PARAM_OFFSET + 1];
    float _dx = val * val + 1e-6;
    float c   = 2.0 * (_dx - _dx * _dx);
    float l = r(p);
    if (_dx < 1e-10 || l < 1e-10) return p;
    float k = floor((l / _dx + 1.0) * 0.5);
    float rr = 2.0 - _dx * (k * 2.0 / l + 1.0) - n * (k * c - 1.0) / l;
    return rr * p;
}

// ------------------------------------------------------------
// Apply single variation by index (switch-based dispatch)
// ------------------------------------------------------------
vec2 apply_single_variation(int var_idx, vec2 p, int slot) {
    switch (var_idx) {
        case  0: return var_linear(p);
        case  1: return var_sinusoidal(p);
        case  2: return var_spherical(p);
        case  3: return var_swirl(p);
        case  4: return var_horseshoe(p);
        case  5: return var_polar(p);
        case  6: return var_handkerchief(p);
        case  7: return var_heart(p);
        case  8: return var_disk(p);
        case  9: return var_spiral(p);
        case 10: return var_hyperbolic(p);
        case 11: return var_diamond(p);
        case 12: return var_ex(p);
        case 13: return var_julia(p);
        case 14: return var_bent(p);
        case 15: return var_waves(p, slot);
        case 16: return var_fisheye(p);
        case 17: return var_popcorn(p, slot);
        case 18: return var_exponential(p);
        case 19: return var_power(p);
        case 20: return var_cosine(p);
        case 21: return var_rings(p, slot);
        case 22: return var_fan(p, slot);
        case 23: return var_blob(p, slot);
        case 24: return var_pdj(p, slot);
        case 25: return var_fan2(p, slot);
        case 26: return var_rings2(p, slot);
        case 27: return var_eyefish(p);
        case 28: return var_bubble(p);
        case 29: return var_cylinder(p);
        case 30: return var_splits(p, slot);
        case 31: return var_cloverleaf(p);
        case 32: return var_julian(p, slot);
        case 33: return var_juliascope(p, slot);
        case 34: return var_tangent(p);
        case 35: return var_cross(p);
        case 36: return var_butterfly(p);
        case 37: return var_curl(p, slot);
        case 38: return var_rectangles(p, slot);
        case 39: return var_checks(p, slot);
        case 40: return var_hex_modulus(p, slot);
        case 41: return var_kaleidoscope(p, slot);
        case 42: return var_icon(p, slot);
        case 43: return var_sattractor(p, slot);
        case 44: return var_wallpaper(p, slot);
        case 45: return var_frieze(p, slot);
        case 46: return var_rings3(p, slot);
        default: return p;
    }
}

// ------------------------------------------------------------
// Apply active variations for transform tidx
// Loops over packed active slots: (var_idx, weight, p0..p5) per slot
// ------------------------------------------------------------
vec2 apply_variations(vec2 p, int tidx) {
    vec2 result = vec2(0.0);
    int base = tidx * MAX_ACTIVE_VARS * SLOT_SIZE;

    for (int i = 0; i < MAX_ACTIVE_VARS; i++) {
        int slot = base + i * SLOT_SIZE;
        int var_idx = int(u_active_vars[slot]);
        float w = u_active_vars[slot + 1];

        if (var_idx < 0) break;

        result += w * apply_single_variation(var_idx, p, slot);
    }

    return result;
}
