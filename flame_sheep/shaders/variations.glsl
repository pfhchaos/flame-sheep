// ------------------------------------------------------------
// Variation functions
// Applied after affine transform to warp the point nonlinearly.
// Each function takes (x,y) and returns warped (x,y).
// Weight is applied by caller.
//
// Requires: rng_state, rng_next(), rng_float() from rng.glsl
// Requires: u_active_vars SSBO, PARAM_OFFSET define
// ------------------------------------------------------------

// Precomputed polar coordinates — computed once per transform, shared
// across all variations in that transform's apply_variations() call.
struct Polar {
    float r_sq;    // dot(p, p)
    float r_val;   // length(p)
    float theta;   // atan(p.x, p.y)  — flam3 convention
    float phi;     // atan(p.y, p.x)  — standard atan2
};

Polar polar_compute(vec2 p) {
    Polar pc;
    pc.r_sq   = dot(p, p);
    pc.r_val  = sqrt(pc.r_sq);
    pc.theta  = atan(p.x, p.y);
    pc.phi    = atan(p.y, p.x);
    return pc;
}

// Legacy helpers — only used by variations that compute polar on a
// DIFFERENT point than the input (e.g. conjugate in var_icon).
float r2(vec2 p) { return dot(p, p); }
float r(vec2 p)  { return length(p); }

// --- Classic flam3 variations (0-14) ---

vec2 var_linear(vec2 p, Polar pc)       { return p; }
vec2 var_sinusoidal(vec2 p, Polar pc)   { return sin(p); }
vec2 var_spherical(vec2 p, Polar pc)    { return p / max(pc.r_sq, 1e-6); }
vec2 var_swirl(vec2 p, Polar pc) {
    float rr = pc.r_sq;
    return vec2(p.x * sin(rr) - p.y * cos(rr),
                p.x * cos(rr) + p.y * sin(rr));
}
vec2 var_horseshoe(vec2 p, Polar pc) {
    float ri = 1.0 / max(pc.r_val, 1e-6);
    return ri * vec2((p.x - p.y) * (p.x + p.y), 2.0 * p.x * p.y);
}
vec2 var_polar(vec2 p, Polar pc) {
    return vec2(pc.theta / 3.14159265, pc.r_val - 1.0);
}
vec2 var_handkerchief(vec2 p, Polar pc) {
    return pc.r_val * vec2(sin(pc.theta + pc.r_val), cos(pc.theta - pc.r_val));
}
vec2 var_heart(vec2 p, Polar pc) {
    return pc.r_val * vec2(sin(pc.theta * pc.r_val), -cos(pc.theta * pc.r_val));
}
vec2 var_disk(vec2 p, Polar pc) {
    float piri = 3.14159265 * pc.r_val;
    return (pc.theta / 3.14159265) * vec2(sin(piri), cos(piri));
}
vec2 var_spiral(vec2 p, Polar pc) {
    return (1.0 / max(pc.r_val, 1e-6)) * vec2(cos(pc.theta) + sin(pc.r_val), sin(pc.theta) - cos(pc.r_val));
}
vec2 var_hyperbolic(vec2 p, Polar pc) {
    return vec2(sin(pc.theta) / max(pc.r_val, 1e-6), cos(pc.theta) * pc.r_val);
}
vec2 var_diamond(vec2 p, Polar pc) {
    return vec2(sin(pc.theta) * cos(pc.r_val), cos(pc.theta) * sin(pc.r_val));
}
vec2 var_ex(vec2 p, Polar pc) {
    float p0 = sin(pc.theta + pc.r_val); float p1 = cos(pc.theta - pc.r_val);
    return pc.r_val * vec2(p0*p0*p0 + p1*p1*p1, p0*p0*p0 - p1*p1*p1);
}
vec2 var_julia(vec2 p, Polar pc) {
    float sqr = sqrt(pc.r_val);
    float th  = pc.theta * 0.5;
    if (rng_next() % 2u == 0u) th += 3.14159265;
    return sqr * vec2(cos(th), sin(th));
}
vec2 var_bent(vec2 p, Polar pc) {
    vec2 q = p;
    if (q.x < 0.0) q.x *= 2.0;
    if (q.y < 0.0) q.y *= 0.5;
    return q;
}

// --- Parametric flam3 variations (15-29) ---

vec2 var_waves(vec2 p, Polar pc, int tidx) {
    // flam3 waves: reads b,c,e,f from the live affine
    int abase = tidx * 6;
    float b = u_affines[abase + 1];
    float c = u_affines[abase + 2];
    float e = u_affines[abase + 4];
    float f = u_affines[abase + 5];
    return vec2(p.x + b * sin(p.y / max(c*c, 1e-6)),
                p.y + e * sin(p.x / max(f*f, 1e-6)));
}
vec2 var_fisheye(vec2 p, Polar pc) {
    float ri = 2.0 / (pc.r_val + 1.0);
    return ri * vec2(p.y, p.x);
}
vec2 var_popcorn(vec2 p, Polar pc, int tidx) {
    // flam3 popcorn: reads c,f from the live affine
    int abase = tidx * 6;
    float c = u_affines[abase + 2];
    float f = u_affines[abase + 5];
    return vec2(p.x + c * sin(tan(3.0 * p.y)),
                p.y + f * sin(tan(3.0 * p.x)));
}
vec2 var_exponential(vec2 p, Polar pc) {
    return exp(p.x - 1.0) * vec2(cos(3.14159265 * p.y), sin(3.14159265 * p.y));
}
vec2 var_power(vec2 p, Polar pc) {
    return pow(pc.r_val, sin(pc.theta)) * vec2(cos(pc.theta), sin(pc.theta));
}
vec2 var_cosine(vec2 p, Polar pc) {
    return vec2(cos(3.14159265 * p.x) * cosh(p.y),
               -sin(3.14159265 * p.x) * sinh(p.y));
}
vec2 var_rings(vec2 p, Polar pc, int tidx) {
    // flam3 rings: reads c from the live affine
    int abase = tidx * 6;
    float c = u_affines[abase + 2];
    float cc = c * c + 1e-6;
    float rr = mod(pc.r_val + cc, 2.0 * cc) - cc + pc.r_val * (1.0 - cc);
    return rr * vec2(cos(pc.theta), sin(pc.theta));
}
vec2 var_fan(vec2 p, Polar pc, int tidx) {
    // flam3 fan: reads c,f from the live affine
    // dx = π*c², dx2 = dx/2, a += (fmod(a+f,dx) > dx2) ? -dx2 : dx2
    int abase = tidx * 6;
    float c = u_affines[abase + 2];
    float f = u_affines[abase + 5];
    float dx = 3.14159265 * c * c + 1e-6;
    float dx2 = 0.5 * dx;
    float a = pc.theta;
    // Use fmod equivalent: val - dx * trunc(val/dx) to match C's fmod
    float fmod_val = (a + f) - dx * trunc((a + f) / dx);
    a += (fmod_val > dx2) ? -dx2 : dx2;
    return pc.r_val * vec2(cos(a), sin(a));
}
vec2 var_blob(vec2 p, Polar pc, int slot) {
    float low   = u_active_vars[slot + PARAM_OFFSET + 0];
    float high  = u_active_vars[slot + PARAM_OFFSET + 1];
    float waves = u_active_vars[slot + PARAM_OFFSET + 2];
    float rr = pc.r_val * (low + (high - low) * (0.5 + 0.5 * sin(waves * pc.theta)));
    return rr * vec2(sin(pc.theta), cos(pc.theta));
}
vec2 var_pdj(vec2 p, Polar pc, int slot) {
    float a = u_active_vars[slot + PARAM_OFFSET + 0];
    float b = u_active_vars[slot + PARAM_OFFSET + 1];
    float c = u_active_vars[slot + PARAM_OFFSET + 2];
    float d = u_active_vars[slot + PARAM_OFFSET + 3];
    return vec2(sin(a * p.y) - cos(b * p.x),
                sin(c * p.x) - cos(d * p.y));
}
vec2 var_fan2(vec2 p, Polar pc, int slot) {
    float fx = u_active_vars[slot + PARAM_OFFSET + 0];
    float fy = u_active_vars[slot + PARAM_OFFSET + 1];
    float dx = 3.14159265 * fx * fx + 1e-6;
    float dx2 = dx * 0.5;
    float t = pc.theta + fy - floor((pc.theta + fy) / dx) * dx;
    float a = (t > dx2) ? pc.theta - dx2 : pc.theta + dx2;
    return pc.r_val * vec2(sin(a), cos(a));
}
vec2 var_rings2(vec2 p, Polar pc, int slot) {
    float val = u_active_vars[slot + PARAM_OFFSET + 0];
    float _dx = val * val + 1e-6;
    if (pc.r_val < 1e-10) return p;
    float k = floor((pc.r_val / _dx + 1.0) * 0.5);
    float rr = 2.0 - _dx * (k * 2.0 / pc.r_val + 1.0);
    return rr * p;
}
vec2 var_eyefish(vec2 p, Polar pc)  { return 2.0 / (pc.r_val + 1.0) * p; }
vec2 var_bubble(vec2 p, Polar pc)   { return 4.0 / (pc.r_sq + 4.0) * p; }
vec2 var_cylinder(vec2 p, Polar pc) { return vec2(sin(p.x), p.y); }

// --- Extended variations (JWildfire / flam3 inspired, 30-37) ---

vec2 var_splits(vec2 p, Polar pc, int slot) {
    float sx = u_active_vars[slot + PARAM_OFFSET + 0];
    float sy = u_active_vars[slot + PARAM_OFFSET + 1];
    return vec2(p.x >= 0.0 ? p.x + sx : p.x - sx,
                p.y >= 0.0 ? p.y + sy : p.y - sy);
}

vec2 var_cloverleaf(vec2 p, Polar pc) {
    float rr = pc.r_val * (sin(2.0 * pc.phi) + 0.25 * sin(6.0 * pc.phi));
    return rr * vec2(cos(pc.phi), sin(pc.phi));
}

vec2 var_julian(vec2 p, Polar pc, int slot) {
    float power = u_active_vars[slot + PARAM_OFFSET + 0];
    float dist  = u_active_vars[slot + PARAM_OFFSET + 1];
    float abs_n = abs(power);
    float cn = dist / power * 0.5;
    float t_rand = floor(abs_n * rng_float());
    float a = (pc.phi + 6.28318530 * t_rand) / power;
    // flam3 uses pow(r², cn) = pow(sumsq, cn), NOT pow(r, cn)
    float ri = pow(max(pc.r_sq, 1e-6), cn);
    return ri * vec2(cos(a), sin(a));
}

vec2 var_juliascope(vec2 p, Polar pc, int slot) {
    float power = u_active_vars[slot + PARAM_OFFSET + 0];
    float dist  = u_active_vars[slot + PARAM_OFFSET + 1];
    float abs_n = abs(power);
    float cn = dist / power * 0.5;
    float t_rand = floor(abs_n * rng_float());
    float ang = pc.phi;
    if (rng_next() % 2u == 0u) ang = -ang;
    float a = (ang + 6.28318530 * t_rand) / power;
    // flam3 uses pow(r², cn), not pow(r, cn)
    float ri = pow(max(pc.r_sq, 1e-6), cn);
    return ri * vec2(cos(a), sin(a));
}

vec2 var_tangent(vec2 p, Polar pc) {
    return vec2(sin(p.x) / max(abs(cos(p.y)), 1e-6),
                tan(p.y));
}

vec2 var_cross(vec2 p, Polar pc) {
    // flam3: r = sqrt(1 / (s² + EPS))
    float s = p.x * p.x - p.y * p.y;
    float r = sqrt(1.0 / (s * s + 1e-10));
    return r * p;
}

vec2 var_butterfly(vec2 p, Polar pc) {
    // flam3: wx * sqrt(|y*x| / (eps + x² + (2y)²))
    float wx = 1.3029400317;
    float y2 = p.y * 2.0;
    float r = wx * sqrt(abs(p.y * p.x) / (1e-10 + p.x * p.x + y2 * y2));
    return vec2(r * p.x, r * y2);
}

vec2 var_curl(vec2 p, Polar pc, int slot) {
    float c1 = u_active_vars[slot + PARAM_OFFSET + 0];
    float c2 = u_active_vars[slot + PARAM_OFFSET + 1];
    float x2 = p.x * p.x;
    float y2 = p.y * p.y;
    float re = 1.0 + c1 * p.x + c2 * (x2 - y2);
    float im =       c1 * p.y + c2 * 2.0 * p.x * p.y;
    float d = max(re * re + im * im, 1e-6);
    return vec2((p.x * re + p.y * im) / d,
                (p.y * re - p.x * im) / d);
}

// --- Tiling variations (38-41) ---

vec2 var_rectangles(vec2 p, Polar pc, int slot) {
    float rx = u_active_vars[slot + PARAM_OFFSET + 0];
    float ry = u_active_vars[slot + PARAM_OFFSET + 1];
    float ox = (abs(rx) < 1e-6) ? p.x : (2.0 * floor(p.x / rx) + 1.0) * rx - p.x;
    float oy = (abs(ry) < 1e-6) ? p.y : (2.0 * floor(p.y / ry) + 1.0) * ry - p.y;
    return vec2(ox, oy);
}

vec2 var_checks(vec2 p, Polar pc, int slot) {
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

vec2 var_hex_modulus(vec2 p, Polar pc, int slot) {
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

vec2 var_kaleidoscope(vec2 p, Polar pc, int slot) {
    float pull = u_active_vars[slot + PARAM_OFFSET + 0];
    float rot = u_active_vars[slot + PARAM_OFFSET + 1];
    float n = max(u_active_vars[slot + PARAM_OFFSET + 2], 2.0);
    float cr = cos(rot), sr = sin(rot);
    vec2 rp = vec2(cr * p.x - sr * p.y, sr * p.x + cr * p.y);
    // Rotated point needs its own polar — can't use cache
    float a = atan(rp.y, rp.x);
    float ri = length(rp) + pull;
    // Mirror into sector
    float sector = 6.28318530 / n;
    a = mod(a, sector);
    if (a > sector * 0.5) a = sector - a;
    return ri * vec2(cos(a), sin(a));
}

// --- Symmetry-generating variations (42-46) ---

vec2 var_icon(vec2 p, Polar pc, int slot) {
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

vec2 var_sattractor(vec2 p, Polar pc, int slot) {
    // Symmetric attractor: m-fold rotational symmetry from roots of unity
    // Pick random k in [0,m), rotate point by 2*pi*k/m
    float m = max(u_active_vars[slot + PARAM_OFFSET + 0], 2.0);
    float k = floor(rng_float() * m);
    float angle = k * 6.28318530 / m;
    float ca = cos(angle), sa = sin(angle);
    return vec2(ca * p.x - sa * p.y, sa * p.x + ca * p.y);
}

// {{SYMMETRY_GROUPS}}

vec2 var_wallpaper(vec2 p, Polar pc, int slot) {
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

vec2 var_frieze(vec2 p, Polar pc, int slot) {
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

vec2 var_rings3(vec2 p, Polar pc, int slot) {
    float val = u_active_vars[slot + PARAM_OFFSET + 0];
    float n   = u_active_vars[slot + PARAM_OFFSET + 1];
    float _dx = val * val + 1e-6;
    float c   = 2.0 * (_dx - _dx * _dx);
    float l = pc.r_val;
    if (_dx < 1e-10 || l < 1e-10) return p;
    float k = floor((l / _dx + 1.0) * 0.5);
    float rr = 2.0 - _dx * (k * 2.0 / l + 1.0) - n * (k * c - 1.0) / l;
    return rr * p;
}

// --- Conformal / complex variations ---

vec2 var_mobius(vec2 p, Polar pc, int slot) {
    // Mobius transform: (az+b)/(cz+d) in complex arithmetic
    float re_a = u_active_vars[slot + PARAM_OFFSET + 0];
    float re_b = u_active_vars[slot + PARAM_OFFSET + 1];
    float re_c = u_active_vars[slot + PARAM_OFFSET + 2];
    float re_d = u_active_vars[slot + PARAM_OFFSET + 3];
    float im_a = u_active_vars[slot + PARAM_OFFSET + 4];
    float im_b = u_active_vars[slot + PARAM_OFFSET + 5];
    float im_c = u_active_vars[slot + PARAM_OFFSET + 6];
    float im_d = u_active_vars[slot + PARAM_OFFSET + 7];
    // numerator: a*z + b
    float re_u = re_a * p.x - im_a * p.y + re_b;
    float im_u = re_a * p.y + im_a * p.x + im_b;
    // denominator: c*z + d
    float re_v = re_c * p.x - im_c * p.y + re_d;
    float im_v = re_c * p.y + im_c * p.x + im_d;
    float d = max(re_v * re_v + im_v * im_v, 1e-6);
    float inv_d = 1.0 / d;
    return vec2((re_u * re_v + im_u * im_v) * inv_d,
                (im_u * re_v - re_u * im_v) * inv_d);
}

vec2 var_cpow(vec2 p, Polar pc, int slot) {
    // Complex power: z^(r+i*i) with n-fold symmetry from power
    float cr = u_active_vars[slot + PARAM_OFFSET + 0];
    float ci = u_active_vars[slot + PARAM_OFFSET + 1];
    float power = u_active_vars[slot + PARAM_OFFSET + 2];
    float a = pc.theta;
    float lnr = 0.5 * log(max(pc.r_sq, 1e-10));
    float va = 6.28318530 / power;
    float vc = cr / power;
    float vd = ci / power;
    float ang = vc * a + vd * lnr + va * floor(power * rng_float());
    float m = exp(vc * lnr - vd * a);
    return m * vec2(cos(ang), sin(ang));
}

// --- Geometric variations ---

vec2 var_ngon(vec2 p, Polar pc, int slot) {
    float circle = u_active_vars[slot + PARAM_OFFSET + 0];
    float corners = u_active_vars[slot + PARAM_OFFSET + 1];
    float power = u_active_vars[slot + PARAM_OFFSET + 2];
    float sides = u_active_vars[slot + PARAM_OFFSET + 3];
    float rf = pow(max(pc.r_sq, 1e-10), power * 0.5);
    float th = pc.phi;  // standard atan2 for ngon
    float b = 6.28318530 / sides;
    float ph = th - b * floor(th / b);
    if (ph > b * 0.5) ph -= b;
    float amp = (corners * (1.0 / max(cos(ph), 1e-6) - 1.0) + circle)
                / max(rf, 1e-6);
    return amp * p;
}

vec2 var_loonie(vec2 p, Polar pc) {
    // Bubble/lens effect — magnifies points inside weight radius
    float rr = pc.r_sq;
    // Weight is baked into the caller, but loonie uses w² as threshold.
    // With weight=1 (normalized), w²=1 so threshold = r²<1
    float w2 = 1.0;  // effective weight² (caller applies actual weight)
    if (rr < w2 && rr > 1e-10) {
        return sqrt(w2 / rr - 1.0) * p;
    }
    return p;
}

vec2 var_scry(vec2 p, Polar pc) {
    // Crystal ball effect — 1/(r*(r²+1))
    float rr = pc.r_sq;
    float ri = pc.r_val;
    float d = ri * (rr + 1.0);
    if (d < 1e-10) return p;
    return (1.0 / d) * p;
}

vec2 var_epispiral(vec2 p, Polar pc, int slot) {
    float n = u_active_vars[slot + PARAM_OFFSET + 0];
    float thickness = u_active_vars[slot + PARAM_OFFSET + 1];
    float holes = u_active_vars[slot + PARAM_OFFSET + 2];
    float th = pc.phi;
    float d = cos(n * th);
    if (abs(d) < 1e-6) return p;
    float t = -holes;
    if (abs(thickness) > 1e-6) {
        t += (rng_float() * thickness) / d;
    } else {
        t += 1.0 / d;
    }
    return t * vec2(cos(th), sin(th));
}

// --- Wave variations ---

vec2 var_waves3(vec2 p, Polar pc, int slot) {
    // waves3: waves2 + amplitude modulation via secondary sine
    float scalex  = u_active_vars[slot + PARAM_OFFSET + 0];
    float scaley  = u_active_vars[slot + PARAM_OFFSET + 1];
    float freqx   = u_active_vars[slot + PARAM_OFFSET + 2];
    float freqy   = u_active_vars[slot + PARAM_OFFSET + 3];
    float sx_freq = u_active_vars[slot + PARAM_OFFSET + 4];
    float sy_freq = u_active_vars[slot + PARAM_OFFSET + 5];
    float scalexx = 0.5 * scalex * (1.0 + sin(p.y * sx_freq));
    float scaleyy = 0.5 * scaley * (1.0 + sin(p.x * sy_freq));
    return vec2(p.x + sin(p.y * freqx) * scalexx,
                p.y + sin(p.x * freqy) * scaleyy);
}

// --- Tiling batch 2 ---

vec2 var_boarders(vec2 p, Polar pc, int slot) {
    // Boarders / Boarders2 combined — parameterized border tiling
    // c: border width scale (0.5 = original boarders)
    // cl: border offset (0.25 = original boarders)
    // cr: randomization threshold (0.75 = original boarders)
    float c  = u_active_vars[slot + PARAM_OFFSET + 0];
    float cl = u_active_vars[slot + PARAM_OFFSET + 1];
    float cr = u_active_vars[slot + PARAM_OFFSET + 2];
    float roundX = floor(p.x + 0.5);
    float roundY = floor(p.y + 0.5);
    float offX = p.x - roundX;
    float offY = p.y - roundY;
    if (rng_float() >= cr) {
        return vec2(offX * c + roundX, offY * c + roundY);
    }
    if (abs(offX) >= abs(offY)) {
        if (offX >= 0.0) {
            return vec2(offX * c + roundX + cl,
                        offY * c + roundY + cl * offY / (offX + 1e-10));
        } else {
            return vec2(offX * c + roundX - cl,
                        offY * c + roundY - cl * offY / (offX - 1e-10));
        }
    } else {
        if (offY >= 0.0) {
            return vec2(offX * c + roundX + cl * offX / (offY + 1e-10),
                        offY * c + roundY + cl);
        } else {
            return vec2(offX * c + roundX - cl * offX / (offY - 1e-10),
                        offY * c + roundY - cl);
        }
    }
}

vec2 var_hypertile(vec2 p, Polar pc, int slot) {
    // Hyperbolic tiling — Escher-like tessellation
    float ht_re = u_active_vars[slot + PARAM_OFFSET + 0];
    float ht_im = u_active_vars[slot + PARAM_OFFSET + 1];
    float a = p.x + ht_re;
    float b = p.y - ht_im;
    float c = ht_re * p.x - ht_im * p.y + 1.0;
    float d = ht_re * p.y + ht_im * p.x;
    float vr = 1.0 / max(c * c + d * d, 1e-6);
    return vec2(vr * (a * c + b * d), vr * (b * c - a * d));
}

vec2 var_cell(vec2 p, Polar pc, int slot) {
    // Cell — interleaved cell tiling
    float size = u_active_vars[slot + PARAM_OFFSET + 0];
    float inv_size = 1.0 / max(abs(size), 1e-6);
    int ix = int(floor(p.x * inv_size));
    int iy = int(floor(p.y * inv_size));
    float dx = p.x - float(ix) * size;
    float dy = p.y - float(iy) * size;
    // Interleave cells
    if (iy >= 0) {
        if (ix >= 0) { iy *= 2; ix *= 2; }
        else { iy *= 2; ix = -(2 * ix + 1); }
    } else {
        if (ix >= 0) { iy = -(2 * iy + 1); ix *= 2; }
        else { iy = -(2 * iy + 1); ix = -(2 * ix + 1); }
    }
    return vec2(dx + float(ix) * size, -dy - float(iy) * size);
}

vec2 var_whorl(vec2 p, Polar pc, int slot) {
    // Whorl — flam3 uses (weight - r) in BOTH branches
    // Weight is 1.0 (applied by caller), so denominator = 1.0 - r
    float inside  = u_active_vars[slot + PARAM_OFFSET + 0];
    float outside = u_active_vars[slot + PARAM_OFFSET + 1];
    float denom = 1.0 - pc.r_val;  // weight(1) - r, same both branches
    if (abs(denom) < 1e-6) denom = sign(denom) * 1e-6;
    float a;
    if (pc.r_val < 1.0) {
        a = pc.phi + inside / denom;
    } else {
        a = pc.phi + outside / denom;
    }
    return pc.r_val * vec2(cos(a), sin(a));
}

vec2 var_disc2(vec2 p, Polar pc, int slot) {
    // Disc2 — flam3 precalc: timespi = rot*π, cosadd = cos(twist)-1
    float rot   = u_active_vars[slot + PARAM_OFFSET + 0];
    float twist = u_active_vars[slot + PARAM_OFFSET + 1];
    float timespi = rot * 3.14159265;
    float sinadd = sin(twist);
    float cosadd = cos(twist) - 1.0;
    if (twist > 6.28318530) {
        float k = 1.0 + twist - 6.28318530;
        cosadd *= k; sinadd *= k;
    } else if (twist < -6.28318530) {
        float k = 1.0 + twist + 6.28318530;
        cosadd *= k; sinadd *= k;
    }
    float t = timespi * (p.x + p.y);
    // flam3 uses precalc_atan = atan2(tx,ty) = atan(x,y)
    float r_val = pc.theta / 3.14159265;
    return vec2((sin(t) + cosadd) * r_val,
                (cos(t) + sinadd) * r_val);
}

// --- Shape & spiral batch ---

vec2 var_flower(vec2 p, Polar pc, int slot) {
    // Flower — petal structures via cos(petals * theta) modulation
    float holes  = u_active_vars[slot + PARAM_OFFSET + 0];
    float petals = u_active_vars[slot + PARAM_OFFSET + 1];
    if (pc.r_val < 1e-6) return vec2(0.0);
    float r = (rng_float() - holes) * cos(petals * pc.phi) / pc.r_val;
    return r * p;
}

vec2 var_blade(vec2 p, Polar pc) {
    // Blade — sharp radial structures (stochastic)
    float r = rng_float() * pc.r_val;
    float sr = sin(r);
    float cr = cos(r);
    return vec2(p.x * (cr + sr), p.x * (cr - sr));
}

vec2 var_spiralwing(vec2 p, Polar pc) {
    // Spiralwing — wing-like spiral distortion
    float c1 = p.x * p.x;
    float c2 = p.y * p.y;
    float d = 1.0 / (c1 + c2 + 1e-6);
    float s2 = sin(c2);
    return vec2(d * cos(c1) * s2, d * sin(c1) * s2);
}

vec2 var_collideoscope(vec2 p, Polar pc, int slot) {
    // Collideoscope — kaleidoscopic angular folding
    float ka  = u_active_vars[slot + PARAM_OFFSET + 0];
    float num = u_active_vars[slot + PARAM_OFFSET + 1];
    num = max(num, 1.0);
    float kn_pi = num / 3.14159265;
    float pi_kn = 3.14159265 / num;
    float ka_kn = ka / num;
    float a = pc.phi;
    int alt;
    if (a >= 0.0) {
        alt = int(a * kn_pi);
        if (alt % 2 == 0)
            a = float(alt) * pi_kn + mod(ka_kn + a, pi_kn);
        else
            a = float(alt) * pi_kn + mod(-ka_kn + a, pi_kn);
    } else {
        alt = int(-a * kn_pi);
        if (alt % 2 != 0)
            a = -(float(alt) * pi_kn + mod(-ka_kn - a, pi_kn));
        else
            a = -(float(alt) * pi_kn + mod(ka_kn - a, pi_kn));
    }
    return pc.r_val * vec2(cos(a), sin(a));
}

vec2 var_auger(vec2 p, Polar pc, int slot) {
    // Auger — ridged/augmented distortion
    float freq   = u_active_vars[slot + PARAM_OFFSET + 0];
    float weight = u_active_vars[slot + PARAM_OFFSET + 1];
    float sym    = u_active_vars[slot + PARAM_OFFSET + 2];
    float scale  = u_active_vars[slot + PARAM_OFFSET + 3];
    float s = sin(freq * p.x);
    float t = sin(freq * p.y);
    float dy = p.y + weight * (scale * s * 0.5 + abs(p.y) * s);
    float dx = p.x + weight * (scale * t * 0.5 + abs(p.x) * t);
    return vec2(p.x + sym * (dx - p.x), dy);
}

// --- Reflective & misc batch ---

vec2 var_flipcircle(vec2 p, Polar pc) {
    // FlipCircle — flip y inside a circle
    if (dot(p, p) > 1.0)
        return vec2(p.x, p.y);
    else
        return vec2(p.x, -p.y);
}

vec2 var_eclipse(vec2 p, Polar pc, int slot) {
    // Eclipse — shifted circular masking
    float shift = u_active_vars[slot + PARAM_OFFSET + 0];
    if (abs(p.y) <= 1.0) {
        float c2 = sqrt(1.0 - p.y * p.y);
        if (abs(p.x) <= c2) {
            float x = p.x + shift;
            if (abs(x) >= c2)
                return vec2(-p.x, p.y);
            else
                return vec2(x, p.y);
        }
    }
    return p;
}

vec2 var_layered_spiral(vec2 p, Polar pc, int slot) {
    // LayeredSpiral — multi-layer spiral modulation
    float radius = u_active_vars[slot + PARAM_OFFSET + 0];
    float a = p.x * radius;
    float t = p.x * p.x + p.y * p.y + 1e-6;
    return vec2(a * cos(t), a * sin(t));
}

vec2 var_stripes(vec2 p, Polar pc, int slot) {
    // Stripes — stripe pattern with warp
    float space = u_active_vars[slot + PARAM_OFFSET + 0];
    float warp  = u_active_vars[slot + PARAM_OFFSET + 1];
    float roundx = floor(p.x + 0.5);
    float offx = p.x - roundx;
    return vec2(offx * (1.0 - space) + roundx,
                p.y + offx * offx * warp);
}

vec2 var_lissajous(vec2 p, Polar pc, int slot) {
    // Lissajous — parametric curve patterns (stochastic)
    float tmin = u_active_vars[slot + PARAM_OFFSET + 0];
    float tmax = u_active_vars[slot + PARAM_OFFSET + 1];
    float la   = u_active_vars[slot + PARAM_OFFSET + 2];
    float lb   = u_active_vars[slot + PARAM_OFFSET + 3];
    float lc   = u_active_vars[slot + PARAM_OFFSET + 4];
    float ld   = u_active_vars[slot + PARAM_OFFSET + 5];
    float le   = u_active_vars[slot + PARAM_OFFSET + 6];
    float t = (tmax - tmin) * rng_float() + tmin;
    float y = rng_float() - 0.5;
    return vec2(sin(la * t + ld) + lc * t + le * y,
                sin(lb * t) + lc * t + le * y);
}

vec2 var_ripple(vec2 p, Polar pc, int slot) {
    // Ripple — cosine wave distortion from center
    float freq   = u_active_vars[slot + PARAM_OFFSET + 0];
    float vel    = u_active_vars[slot + PARAM_OFFSET + 1];
    float amp    = u_active_vars[slot + PARAM_OFFSET + 2];
    float cx     = u_active_vars[slot + PARAM_OFFSET + 3];
    float cy     = u_active_vars[slot + PARAM_OFFSET + 4];
    float phase  = u_active_vars[slot + PARAM_OFFSET + 5];
    float scale  = u_active_vars[slot + PARAM_OFFSET + 6];
    float fixd   = u_active_vars[slot + PARAM_OFFSET + 7];
    float x = p.x * scale - cx;
    float y = p.y * scale + cy;
    float d = (fixd > 0.5) ? sqrt(x*x + y*y) : sqrt(x*x * y*y);
    d = max(d, 1e-6);
    float nx = x / d;
    float ny = y / d;
    float wave = cos(freq * d - vel + phase);
    float os = amp * wave;
    return vec2(p.x + nx * os, p.y + ny * os);
}

// --- Displaced parameterized variants (70-73) + waves2 (74) ---

vec2 var_waves_param(vec2 p, Polar pc, int slot) {
    // Same formula as waves, but reads from explicit params instead of affine
    float freq_x = u_active_vars[slot + PARAM_OFFSET + 0];
    float freq_y = u_active_vars[slot + PARAM_OFFSET + 1];
    float amp_x  = u_active_vars[slot + PARAM_OFFSET + 2];
    float amp_y  = u_active_vars[slot + PARAM_OFFSET + 3];
    return vec2(p.x + freq_x * sin(p.y / max(amp_x*amp_x, 1e-6)),
                p.y + freq_y * sin(p.x / max(amp_y*amp_y, 1e-6)));
}

vec2 var_popcorn_param(vec2 p, Polar pc, int slot) {
    float cx = u_active_vars[slot + PARAM_OFFSET + 0];
    float cy = u_active_vars[slot + PARAM_OFFSET + 1];
    return vec2(p.x + cx * sin(tan(3.0 * p.y)),
                p.y + cy * sin(tan(3.0 * p.x)));
}

vec2 var_rings_param(vec2 p, Polar pc, int slot) {
    float c = u_active_vars[slot + PARAM_OFFSET + 0];
    float cc = c * c + 1e-6;
    float rr = mod(pc.r_val + cc, 2.0 * cc) - cc + pc.r_val * (1.0 - cc);
    return rr * vec2(cos(pc.theta), sin(pc.theta));
}

vec2 var_fan_param(vec2 p, Polar pc, int slot) {
    float c = u_active_vars[slot + PARAM_OFFSET + 0];
    float f = u_active_vars[slot + PARAM_OFFSET + 1];
    float t = 3.14159265 * c * c + 1e-6;
    float th2 = (mod(pc.theta + f, 2.0*t) > t) ? pc.theta - t : pc.theta + t;
    return pc.r_val * vec2(cos(th2), sin(th2));
}

vec2 var_waves2(vec2 p, Polar pc, int slot) {
    // flam3 waves2: x + scalex*sin(y*freqx) — different formula from waves
    float scalex = u_active_vars[slot + PARAM_OFFSET + 0];
    float scaley = u_active_vars[slot + PARAM_OFFSET + 1];
    float freqx  = u_active_vars[slot + PARAM_OFFSET + 2];
    float freqy  = u_active_vars[slot + PARAM_OFFSET + 3];
    return vec2(p.x + scalex * sin(p.y * freqx),
                p.y + scaley * sin(p.x * freqy));
}

// --- flam3 standard batch 2 (75-121) ---

vec2 var_blur(vec2 p, Polar pc) {
    float a = rng_float() * 6.28318530;
    float r = rng_float();
    return r * vec2(cos(a), sin(a));
}

vec2 var_gaussian_blur(vec2 p, Polar pc) {
    float a = rng_float() * 6.28318530;
    float r = rng_float() + rng_float() + rng_float() + rng_float() - 2.0;
    return r * vec2(cos(a), sin(a));
}

vec2 var_radial_blur(vec2 p, Polar pc, int slot) {
    float ang = u_active_vars[slot + PARAM_OFFSET + 0];
    float rndg = rng_float() + rng_float() + rng_float() + rng_float() - 2.0;
    float spin = sin(ang * 1.57079632);
    float zoom = cos(ang * 1.57079632);
    float alpha = pc.phi + spin * rndg;
    float rz = zoom * rndg - 1.0;
    return vec2(pc.r_val * cos(alpha) + rz * p.x,
                pc.r_val * sin(alpha) + rz * p.y);
}

vec2 var_perspective(vec2 p, Polar pc, int slot) {
    float ang = u_active_vars[slot + PARAM_OFFSET + 0];
    float dist = u_active_vars[slot + PARAM_OFFSET + 1];
    float vsin = sin(ang * 1.57079632);
    float vfcos = dist * cos(ang * 1.57079632);
    float d = dist - p.y * vsin;
    if (abs(d) < 1e-10) return p;
    float t = 1.0 / d;
    return vec2(dist * p.x * t, vfcos * p.y * t);
}

vec2 var_super_shape(vec2 p, Polar pc, int slot) {
    float rnd = u_active_vars[slot + PARAM_OFFSET + 0];
    float m   = u_active_vars[slot + PARAM_OFFSET + 1];
    float n1  = u_active_vars[slot + PARAM_OFFSET + 2];
    float n2  = u_active_vars[slot + PARAM_OFFSET + 3];
    float n3  = u_active_vars[slot + PARAM_OFFSET + 4];
    float holes = u_active_vars[slot + PARAM_OFFSET + 5];
    float theta = m * 0.25 * pc.phi + 0.78539816;
    float t1 = pow(abs(cos(theta)), n2);
    float t2 = pow(abs(sin(theta)), n3);
    if (abs(n1) < 1e-10 || pc.r_val < 1e-10) return p;
    float r = ((rnd * rng_float() + (1.0 - rnd) * pc.r_val) - holes)
              * pow(t1 + t2, -1.0 / n1) / pc.r_val;
    return r * p;
}

vec2 var_noise(vec2 p, Polar pc) {
    float a = rng_float() * 6.28318530;
    float r = rng_float();
    return vec2(p.x * r * cos(a), p.y * r * sin(a));
}

vec2 var_secant2(vec2 p, Polar pc) {
    float r = pc.r_val;
    float cr = cos(r);
    if (abs(cr) < 1e-10) return vec2(p.x, p.y);
    float icr = 1.0 / cr;
    return vec2(p.x, cr < 0.0 ? icr + 1.0 : icr - 1.0);
}

vec2 var_pie(vec2 p, Polar pc, int slot) {
    float slices = u_active_vars[slot + PARAM_OFFSET + 0];
    float rot    = u_active_vars[slot + PARAM_OFFSET + 1];
    float thick  = u_active_vars[slot + PARAM_OFFSET + 2];
    float sl = floor(rng_float() * slices + 0.5);
    float a = rot + 6.28318530 * (sl + rng_float() * thick) / slices;
    float r = rng_float();
    return r * vec2(cos(a), sin(a));
}

vec2 var_arch(vec2 p, Polar pc) {
    float ang = rng_float() * 3.14159265;
    float sr = sin(ang);
    float cr = cos(ang);
    if (abs(cr) < 1e-10) return vec2(sr, sr);
    return vec2(sr, sr * sr / cr);
}

vec2 var_parabola(vec2 p, Polar pc, int slot) {
    float h = u_active_vars[slot + PARAM_OFFSET + 0];
    float w = u_active_vars[slot + PARAM_OFFSET + 1];
    float sr = sin(pc.r_val);
    float cr = cos(pc.r_val);
    return vec2(h * sr * sr * rng_float(), w * cr * rng_float());
}

vec2 var_rays(vec2 p, Polar pc) {
    float ang = rng_float() * 3.14159265;
    float r = 1.0 / max(pc.r_sq, 1e-10);
    float tanr = tan(ang) * r;
    return tanr * vec2(cos(p.x), sin(p.y));
}

vec2 var_conic(vec2 p, Polar pc, int slot) {
    float ecc = u_active_vars[slot + PARAM_OFFSET + 0];
    float holes = u_active_vars[slot + PARAM_OFFSET + 1];
    float ct = p.x / max(pc.r_val, 1e-10);
    float r = (rng_float() - holes) * ecc / (1.0 + ecc * ct) / max(pc.r_val, 1e-10);
    return r * p;
}

vec2 var_escher(vec2 p, Polar pc, int slot) {
    float beta = u_active_vars[slot + PARAM_OFFSET + 0];
    float seb = sin(beta), ceb = cos(beta);
    float vc = 0.5 * (1.0 + ceb);
    float vd = 0.5 * seb;
    float a = pc.phi;
    float lnr = 0.5 * log(max(pc.r_sq, 1e-10));
    float m = exp(vc * lnr - vd * a);
    float n = vc * a + vd * lnr;
    return m * vec2(cos(n), sin(n));
}

vec2 var_elliptic(vec2 p, Polar pc) {
    float sq = p.y * p.y + p.x * p.x;
    float x2 = 2.0 * p.x;
    float xmax = max(0.5 * (sqrt(max(sq + x2, 0.0) + 1e-10) +
                            sqrt(max(abs(sq - x2), 0.0) + 1e-10)), 1.0);
    float a = clamp(p.x / xmax, -1.0, 1.0);
    float ssx = sqrt(max(xmax - 1.0, 0.0));
    float sign = p.y > 0.0 ? 1.0 : -1.0;
    float v = 2.0 / 3.14159265;
    return vec2(v * asin(a), sign * v * log(xmax + ssx + 1e-10));
}

vec2 var_oscilloscope(vec2 p, Polar pc, int slot) {
    float sep  = u_active_vars[slot + PARAM_OFFSET + 0];
    float freq = u_active_vars[slot + PARAM_OFFSET + 1];
    float amp  = u_active_vars[slot + PARAM_OFFSET + 2];
    float damp = u_active_vars[slot + PARAM_OFFSET + 3];
    float t = amp * exp(-abs(p.x) * damp) * cos(6.28318530 * freq * p.x) + sep;
    float dy = abs(p.y) <= t ? -p.y : p.y;
    return vec2(p.x, dy);
}

vec2 var_edisc(vec2 p, Polar pc) {
    float tmp = pc.r_sq + 1.0;
    float tmp2 = 2.0 * p.x;
    float r1 = sqrt(max(tmp + tmp2, 0.0));
    float r2 = sqrt(max(tmp - tmp2, 0.0));
    float xmax = max((r1 + r2) * 0.5, 1.0);
    float a1 = log(xmax + sqrt(max(xmax - 1.0, 0.0)));
    float a2 = -acos(clamp(p.x / xmax, -1.0, 1.0));
    float snv = sin(a1);
    if (p.y > 0.0) snv = -snv;
    float w = 1.0 / 11.57034632;
    return vec2(w * cosh(a2) * cos(a1), w * sinh(a2) * snv);
}

vec2 var_square(vec2 p, Polar pc) {
    return vec2(rng_float() - 0.5, rng_float() - 0.5);
}

vec2 var_curve(vec2 p, Polar pc, int slot) {
    float xamp = u_active_vars[slot + PARAM_OFFSET + 0];
    float yamp = u_active_vars[slot + PARAM_OFFSET + 1];
    float xl   = u_active_vars[slot + PARAM_OFFSET + 2];
    float yl   = u_active_vars[slot + PARAM_OFFSET + 3];
    float xl2 = max(xl * xl, 1e-10);
    float yl2 = max(yl * yl, 1e-10);
    return vec2(p.x + xamp * exp(-p.y * p.y / xl2),
                p.y + yamp * exp(-p.x * p.x / yl2));
}

vec2 var_twintrian(vec2 p, Polar pc) {
    float r = rng_float() * pc.r_val;
    float sr = sin(r), cr = cos(r);
    float diff = log(max(sr * sr, 1e-20)) / 2.302585 + cr;  // log10
    return vec2(p.x * diff, p.x * (diff - sr * 3.14159265));
}

vec2 var_wedge_julia(vec2 p, Polar pc, int slot) {
    float ang   = u_active_vars[slot + PARAM_OFFSET + 0];
    float count = u_active_vars[slot + PARAM_OFFSET + 1];
    float power = u_active_vars[slot + PARAM_OFFSET + 2];
    float dist  = u_active_vars[slot + PARAM_OFFSET + 3];
    float abs_n = abs(power);
    float cn = dist / power * 0.5;
    float cf = 1.0 - ang * count / 3.14159265 * 0.5;
    float t_rnd = floor(abs_n * rng_float());
    float a = (pc.phi + 6.28318530 * t_rnd) / power;
    float c = floor((count * a + 3.14159265) / 3.14159265 * 0.5);
    a = a * cf + c * ang;
    // flam3 uses pow(r², cn), not pow(r, cn)
    float ri = pow(max(pc.r_sq, 1e-6), cn);
    return ri * vec2(cos(a), sin(a));
}

vec2 var_wedge(vec2 p, Polar pc, int slot) {
    float ang   = u_active_vars[slot + PARAM_OFFSET + 0];
    float hole  = u_active_vars[slot + PARAM_OFFSET + 1];
    float count = u_active_vars[slot + PARAM_OFFSET + 2];
    float swirl = u_active_vars[slot + PARAM_OFFSET + 3];
    float a = pc.phi + swirl * pc.r_val;
    float c = floor((count * a + 3.14159265) / 3.14159265 * 0.5);
    float cf = 1.0 - ang * count / 3.14159265 * 0.5;
    a = a * cf + c * ang;
    float r = pc.r_val + hole;
    return r * vec2(cos(a), sin(a));
}

vec2 var_wedge_sph(vec2 p, Polar pc, int slot) {
    float ang   = u_active_vars[slot + PARAM_OFFSET + 0];
    float hole  = u_active_vars[slot + PARAM_OFFSET + 1];
    float count = u_active_vars[slot + PARAM_OFFSET + 2];
    float swirl = u_active_vars[slot + PARAM_OFFSET + 3];
    float ri = 1.0 / max(pc.r_val, 1e-10);
    float a = pc.phi + swirl * ri;
    float c = floor((count * a + 3.14159265) / 3.14159265 * 0.5);
    float cf = 1.0 - ang * count / 3.14159265 * 0.5;
    a = a * cf + c * ang;
    float r = ri + hole;
    return r * vec2(cos(a), sin(a));
}

vec2 var_lazysusan(vec2 p, Polar pc, int slot) {
    float lx    = u_active_vars[slot + PARAM_OFFSET + 0];
    float ly    = u_active_vars[slot + PARAM_OFFSET + 1];
    float spin  = u_active_vars[slot + PARAM_OFFSET + 2];
    float space = u_active_vars[slot + PARAM_OFFSET + 3];
    float twist = u_active_vars[slot + PARAM_OFFSET + 4];
    float xx = p.x - lx;
    float yy = p.y + ly;
    float rr = sqrt(xx * xx + yy * yy);
    // amount is 1.0 (weight applied by caller)
    if (rr < 1.0) {
        float a = atan(yy, xx) + spin + twist * (1.0 - rr);
        rr = rr;
        return vec2(rr * cos(a) + lx, rr * sin(a) - ly);
    } else {
        rr = 1.0 + space / max(rr, 1e-10);
        return vec2(rr * xx + lx, rr * yy - ly);
    }
}

vec2 var_modulus_func(vec2 p, Polar pc, int slot) {
    float mx = u_active_vars[slot + PARAM_OFFSET + 0];
    float my = u_active_vars[slot + PARAM_OFFSET + 1];
    float xr = 2.0 * mx, yr = 2.0 * my;
    float ox, oy;
    if (p.x > mx) ox = -mx + mod(p.x + mx, xr);
    else if (p.x < -mx) ox = mx - mod(mx - p.x, xr);
    else ox = p.x;
    if (p.y > my) oy = -my + mod(p.y + my, yr);
    else if (p.y < -my) oy = my - mod(my - p.y, yr);
    else oy = p.y;
    return vec2(ox, oy);
}

vec2 var_bent2(vec2 p, Polar pc, int slot) {
    float bx = u_active_vars[slot + PARAM_OFFSET + 0];
    float by = u_active_vars[slot + PARAM_OFFSET + 1];
    float nx = p.x < 0.0 ? p.x * bx : p.x;
    float ny = p.y < 0.0 ? p.y * by : p.y;
    return vec2(nx, ny);
}

vec2 var_bipolar(vec2 p, Polar pc, int slot) {
    float shift = u_active_vars[slot + PARAM_OFFSET + 0];
    float x2y2 = pc.r_sq;
    float ps = -1.57079632 * shift;
    float y2 = 0.5 * atan(2.0 * p.y, x2y2 - 1.0) + ps;
    if (y2 > 1.57079632) y2 = -1.57079632 + mod(y2 + 1.57079632, 3.14159265);
    else if (y2 < -1.57079632) y2 = 1.57079632 - mod(1.57079632 - y2, 3.14159265);
    float t = x2y2 + 1.0;
    float tp = t + 2.0 * p.x;
    float tm = max(t - 2.0 * p.x, 1e-10);
    return vec2(0.159154943 * log(tp / tm), 0.636619772 * y2);
}

vec2 var_flux(vec2 p, Polar pc, int slot) {
    float spread = u_active_vars[slot + PARAM_OFFSET + 0];
    float xpw = p.x + 1.0;  // amount=1 (weight applied by caller)
    float xmw = p.x - 1.0;
    float avgr = (2.0 + spread) * sqrt(sqrt(max(p.y*p.y + xpw*xpw, 1e-10))
                                     / sqrt(max(p.y*p.y + xmw*xmw, 1e-10)));
    float avga = (atan(p.y, xmw) - atan(p.y, xpw)) * 0.5;
    return avgr * vec2(cos(avga), sin(avga));
}

vec2 var_split(vec2 p, Polar pc, int slot) {
    float xs = u_active_vars[slot + PARAM_OFFSET + 0];
    float ys = u_active_vars[slot + PARAM_OFFSET + 1];
    float sx = cos(p.x * xs * 3.14159265) >= 0.0 ? 1.0 : -1.0;
    float sy = cos(p.y * ys * 3.14159265) >= 0.0 ? 1.0 : -1.0;
    return vec2(p.x * sy, p.y * sx);
}

vec2 var_separation(vec2 p, Polar pc, int slot) {
    float sx  = u_active_vars[slot + PARAM_OFFSET + 0];
    float sy  = u_active_vars[slot + PARAM_OFFSET + 1];
    float sxi = u_active_vars[slot + PARAM_OFFSET + 2];
    float syi = u_active_vars[slot + PARAM_OFFSET + 3];
    float sx2 = sx * sx, sy2 = sy * sy;
    float ox, oy;
    if (p.x > 0.0) ox = sqrt(p.x*p.x + sx2) - p.x * sxi;
    else if (p.x < 0.0) ox = -(sqrt(p.x*p.x + sx2) + p.x * sxi);
    else ox = 0.0;
    if (p.y > 0.0) oy = sqrt(p.y*p.y + sy2) - p.y * syi;
    else if (p.y < 0.0) oy = -(sqrt(p.y*p.y + sy2) + p.y * syi);
    else oy = 0.0;
    return vec2(ox, oy);
}

vec2 var_polar2(vec2 p, Polar pc) {
    float v = 1.0 / 3.14159265;
    return vec2(v * pc.phi, v * 0.5 * log(max(pc.r_sq, 1e-10)));
}

vec2 var_foci(vec2 p, Polar pc) {
    float expx = exp(min(p.x, 20.0)) * 0.5;
    float expnx = 0.25 / max(expx, 1e-10);
    float tmp = expx + expnx - cos(p.y);
    if (abs(tmp) < 1e-10) return p;
    float d = 1.0 / tmp;
    return vec2((expx - expnx) * d, sin(p.y) * d);
}

vec2 var_popcorn2(vec2 p, Polar pc, int slot) {
    float px = u_active_vars[slot + PARAM_OFFSET + 0];
    float py = u_active_vars[slot + PARAM_OFFSET + 1];
    float c  = u_active_vars[slot + PARAM_OFFSET + 2];
    return vec2(p.x + px * sin(tan(p.y * c)),
                p.y + py * sin(tan(p.x * c)));
}

vec2 var_secant_func(vec2 p, Polar pc) {
    float r = pc.r_val;
    float cr = cos(r);
    if (abs(cr) < 1e-10) return vec2(p.x, p.y);
    return vec2(p.x, 1.0 / cr);
}

// --- Complex trig (z = x + iy) ---

vec2 var_sin_func(vec2 p, Polar pc)  { return vec2(sin(p.x)*cosh(p.y),  cos(p.x)*sinh(p.y)); }
vec2 var_cos_func(vec2 p, Polar pc)  { return vec2(cos(p.x)*cosh(p.y), -sin(p.x)*sinh(p.y)); }

vec2 var_tan_func(vec2 p, Polar pc) {
    float d = cos(2.0*p.x) + cosh(2.0*p.y);
    if (abs(d) < 1e-10) return p;
    return vec2(sin(2.0*p.x)/d, sinh(2.0*p.y)/d);
}
vec2 var_sec_func(vec2 p, Polar pc) {
    float d = cos(2.0*p.x) + cosh(2.0*p.y);
    if (abs(d) < 1e-10) return p;
    float s = 2.0/d;
    return vec2(s*cos(p.x)*cosh(p.y), s*sin(p.x)*sinh(p.y));
}
vec2 var_csc_func(vec2 p, Polar pc) {
    float d = cosh(2.0*p.y) - cos(2.0*p.x);
    if (abs(d) < 1e-10) return p;
    float s = 2.0/d;
    return vec2(s*sin(p.x)*cosh(p.y), -s*cos(p.x)*sinh(p.y));
}
vec2 var_cot_func(vec2 p, Polar pc) {
    float d = cosh(2.0*p.y) - cos(2.0*p.x);
    if (abs(d) < 1e-10) return p;
    return vec2(sin(2.0*p.x)/d, -sinh(2.0*p.y)/d);
}

vec2 var_sinh_func(vec2 p, Polar pc) { return vec2(sinh(p.x)*cos(p.y),  cosh(p.x)*sin(p.y)); }
vec2 var_cosh_func(vec2 p, Polar pc) { return vec2(cosh(p.x)*cos(p.y),  sinh(p.x)*sin(p.y)); }

vec2 var_tanh_func(vec2 p, Polar pc) {
    float d = cos(2.0*p.y) + cosh(2.0*p.x);
    if (abs(d) < 1e-10) return p;
    return vec2(sinh(2.0*p.x)/d, sin(2.0*p.y)/d);
}
vec2 var_sech_func(vec2 p, Polar pc) {
    float d = cos(2.0*p.y) + cosh(2.0*p.x);
    if (abs(d) < 1e-10) return p;
    float s = 2.0/d;
    return vec2(s*cos(p.y)*cosh(p.x), -s*sin(p.y)*sinh(p.x));
}
vec2 var_csch_func(vec2 p, Polar pc) {
    float d = cosh(2.0*p.x) - cos(2.0*p.y);
    if (abs(d) < 1e-10) return p;
    float s = 2.0/d;
    return vec2(s*sinh(p.x)*cos(p.y), -s*cosh(p.x)*sin(p.y));
}
vec2 var_coth_func(vec2 p, Polar pc) {
    float d = cosh(2.0*p.x) - cos(2.0*p.y);
    if (abs(d) < 1e-10) return p;
    return vec2(sinh(2.0*p.x)/d, sin(2.0*p.y)/d);
}

vec2 var_exp_func(vec2 p, Polar pc) {
    float e = exp(p.x);
    return e * vec2(cos(p.y), sin(p.y));
}

vec2 var_log_func(vec2 p, Polar pc) {
    return vec2(0.5 * log(max(pc.r_sq, 1e-10)), pc.phi);
}

// ------------------------------------------------------------
// Apply single variation by index (switch-based dispatch)
// ------------------------------------------------------------
vec2 apply_single_variation(int var_idx, vec2 p, Polar pc, int slot, int tidx) {
    switch (var_idx) {
        case  0: return var_linear(p, pc);
        case  1: return var_sinusoidal(p, pc);
        case  2: return var_spherical(p, pc);
        case  3: return var_swirl(p, pc);
        case  4: return var_horseshoe(p, pc);
        case  5: return var_polar(p, pc);
        case  6: return var_handkerchief(p, pc);
        case  7: return var_heart(p, pc);
        case  8: return var_disk(p, pc);
        case  9: return var_spiral(p, pc);
        case 10: return var_hyperbolic(p, pc);
        case 11: return var_diamond(p, pc);
        case 12: return var_ex(p, pc);
        case 13: return var_julia(p, pc);
        case 14: return var_bent(p, pc);
        case 15: return var_waves(p, pc, tidx);
        case 16: return var_fisheye(p, pc);
        case 17: return var_popcorn(p, pc, tidx);
        case 18: return var_exponential(p, pc);
        case 19: return var_power(p, pc);
        case 20: return var_cosine(p, pc);
        case 21: return var_rings(p, pc, tidx);
        case 22: return var_fan(p, pc, tidx);
        case 23: return var_blob(p, pc, slot);
        case 24: return var_pdj(p, pc, slot);
        case 25: return var_fan2(p, pc, slot);
        case 26: return var_rings2(p, pc, slot);
        case 27: return var_eyefish(p, pc);
        case 28: return var_bubble(p, pc);
        case 29: return var_cylinder(p, pc);
        case 30: return var_splits(p, pc, slot);
        case 31: return var_cloverleaf(p, pc);
        case 32: return var_julian(p, pc, slot);
        case 33: return var_juliascope(p, pc, slot);
        case 34: return var_tangent(p, pc);
        case 35: return var_cross(p, pc);
        case 36: return var_butterfly(p, pc);
        case 37: return var_curl(p, pc, slot);
        case 38: return var_rectangles(p, pc, slot);
        case 39: return var_checks(p, pc, slot);
        case 40: return var_hex_modulus(p, pc, slot);
        case 41: return var_kaleidoscope(p, pc, slot);
        case 42: return var_icon(p, pc, slot);
        case 43: return var_sattractor(p, pc, slot);
        case 44: return var_wallpaper(p, pc, slot);
        case 45: return var_frieze(p, pc, slot);
        case 46: return var_rings3(p, pc, slot);
        case 47: return var_mobius(p, pc, slot);
        case 48: return var_cpow(p, pc, slot);
        case 49: return var_ngon(p, pc, slot);
        case 50: return var_loonie(p, pc);
        case 51: return var_scry(p, pc);
        case 52: return var_epispiral(p, pc, slot);
        case 53: return var_waves3(p, pc, slot);
        case 54: return var_boarders(p, pc, slot);
        case 55: return var_hypertile(p, pc, slot);
        case 56: return var_cell(p, pc, slot);
        case 57: return var_whorl(p, pc, slot);
        case 58: return var_disc2(p, pc, slot);
        case 59: return var_flower(p, pc, slot);
        case 60: return var_blade(p, pc);
        case 61: return var_spiralwing(p, pc);
        case 62: return var_collideoscope(p, pc, slot);
        case 63: return var_auger(p, pc, slot);
        case 64: return var_flipcircle(p, pc);
        case 65: return var_eclipse(p, pc, slot);
        case 66: return var_layered_spiral(p, pc, slot);
        case 67: return var_stripes(p, pc, slot);
        case 68: return var_lissajous(p, pc, slot);
        case 69: return var_ripple(p, pc, slot);
        case 70: return var_waves_param(p, pc, slot);
        case 71: return var_popcorn_param(p, pc, slot);
        case 72: return var_rings_param(p, pc, slot);
        case 73: return var_fan_param(p, pc, slot);
        case 74: return var_waves2(p, pc, slot);
        case 75: return var_blur(p, pc);
        case 76: return var_gaussian_blur(p, pc);
        case 77: return var_radial_blur(p, pc, slot);
        case 78: return var_perspective(p, pc, slot);
        case 79: return var_super_shape(p, pc, slot);
        case 80: return var_noise(p, pc);
        case 81: return var_secant2(p, pc);
        case 82: return var_pie(p, pc, slot);
        case 83: return var_arch(p, pc);
        case 84: return var_parabola(p, pc, slot);
        case 85: return var_rays(p, pc);
        case 86: return var_conic(p, pc, slot);
        case 87: return var_escher(p, pc, slot);
        case 88: return var_elliptic(p, pc);
        case 89: return var_oscilloscope(p, pc, slot);
        case 90: return var_edisc(p, pc);
        case 91: return var_square(p, pc);
        case 92: return var_curve(p, pc, slot);
        case 93: return var_twintrian(p, pc);
        case 94: return var_wedge_julia(p, pc, slot);
        case 95: return var_wedge(p, pc, slot);
        case 96: return var_wedge_sph(p, pc, slot);
        case 97: return var_lazysusan(p, pc, slot);
        case 98: return var_modulus_func(p, pc, slot);
        case 99: return var_bent2(p, pc, slot);
        case 100: return var_bipolar(p, pc, slot);
        case 101: return var_flux(p, pc, slot);
        case 102: return var_split(p, pc, slot);
        case 103: return var_separation(p, pc, slot);
        case 104: return var_polar2(p, pc);
        case 105: return var_foci(p, pc);
        case 106: return var_popcorn2(p, pc, slot);
        case 107: return var_secant_func(p, pc);
        case 108: return var_sin_func(p, pc);
        case 109: return var_cos_func(p, pc);
        case 110: return var_tan_func(p, pc);
        case 111: return var_sec_func(p, pc);
        case 112: return var_csc_func(p, pc);
        case 113: return var_cot_func(p, pc);
        case 114: return var_sinh_func(p, pc);
        case 115: return var_cosh_func(p, pc);
        case 116: return var_tanh_func(p, pc);
        case 117: return var_sech_func(p, pc);
        case 118: return var_csch_func(p, pc);
        case 119: return var_coth_func(p, pc);
        case 120: return var_exp_func(p, pc);
        case 121: return var_log_func(p, pc);
        default: return p;
    }
}

// ------------------------------------------------------------
// Apply active variations for transform tidx
// Precomputes polar coordinates once, shared across all variations.
// ------------------------------------------------------------
vec2 apply_variations(vec2 p, int tidx) {
    Polar pc = polar_compute(p);
    vec2 result = vec2(0.0);
    int base = tidx * MAX_ACTIVE_VARS * SLOT_SIZE;

    for (int i = 0; i < MAX_ACTIVE_VARS; i++) {
        int slot = base + i * SLOT_SIZE;
        int var_idx = int(u_active_vars[slot]);
        float w = u_active_vars[slot + 1];

        if (var_idx < 0) break;

        result += w * apply_single_variation(var_idx, p, pc, slot, tidx);
    }

    return result;
}

// ------------------------------------------------------------
// Apply pre-variations (before affine). Same dispatch, reads from
// u_pre_active_vars SSBO. Empty slots (-1) = immediate break.
// ------------------------------------------------------------
vec2 apply_pre_variations(vec2 p, int tidx) {
    int base = tidx * MAX_ACTIVE_VARS * SLOT_SIZE;

    // Quick check: if first slot is empty, return input unchanged
    if (int(u_pre_active_vars[base]) < 0) return p;

    Polar pc = polar_compute(p);
    vec2 result = vec2(0.0);

    for (int i = 0; i < MAX_ACTIVE_VARS; i++) {
        int slot = base + i * SLOT_SIZE;
        int var_idx = int(u_pre_active_vars[slot]);
        float w = u_pre_active_vars[slot + 1];

        if (var_idx < 0) break;

        result += w * apply_single_variation(var_idx, p, pc, slot, tidx);
    }

    return result;
}
