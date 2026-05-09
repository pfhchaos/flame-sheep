/*
 * flam3_variation_oracle.c — ground truth variation outputs from flam3 math
 *
 * Extracts the exact computation from flam3's variations.c for comparison
 * against our CPU/GPU implementations. Tests include points that exercise
 * both sides of conditional branches.
 *
 * Build:
 *   cc -o /tmp/flam3_oracle tools/flam3_variation_oracle.c -lm -O2
 *
 * Usage:
 *   /tmp/flam3_oracle > tests/flam3_reference.csv
 */

#include <stdio.h>
#include <math.h>
#include <string.h>

#define EPS (1e-10)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define M_PI_2 (M_PI/2.0)

/* Deterministic xorshift32 RNG matching our GPU/CPU implementation */
static unsigned int _rng_state = 12345;
static void rng_seed(unsigned int s) { _rng_state = s; }
static double rng_float(void) {
    _rng_state ^= _rng_state << 13;
    _rng_state ^= _rng_state >> 17;
    _rng_state ^= _rng_state << 5;
    return (double)_rng_state / 4294967295.0;
}
static int rng_bit(void) { return rng_float() > 0.5 ? 1 : 0; }

/* Minimal xform struct for params */
typedef struct {
    double julian_power, julian_dist, julian_rN, julian_cn;
    double juliascope_power, juliascope_dist, juliascope_rN, juliascope_cn;
    double rings2_val;
    double blob_low, blob_high, blob_waves;
    double pdj_a, pdj_b, pdj_c, pdj_d;
    double fan2_x, fan2_y;
    double curl_c1, curl_c2;
    double rectangles_x, rectangles_y;
    double ngon_sides, ngon_power, ngon_circle, ngon_corners;
    double bent2_x, bent2_y;
    double bipolar_shift;
    double cell_size;
    double modulus_x, modulus_y;
    double splits_x, splits_y;
    double oscope_separation, oscope_frequency, oscope_amplitude, oscope_damping;
    double lazysusan_x, lazysusan_y, lazysusan_spin, lazysusan_twist, lazysusan_space;
    double separation_x, separation_y, separation_xinside, separation_yinside;
    double split_xsize, split_ysize;
    double wedge_angle, wedge_hole, wedge_count, wedge_swirl;
    double popcorn2_x, popcorn2_y, popcorn2_c;
    double conic_eccentricity, conic_holes;
    double escher_beta;
    double parabola_height, parabola_width;
    double perspective_angle, perspective_dist, persp_vsin, persp_vfcos;
    double flower_holes, flower_petals;
    double flux_spread;
    double cpow_r, cpow_i, cpow_power;
    double super_shape_rnd, super_shape_m, super_shape_n1, super_shape_n2, super_shape_n3, super_shape_holes;
    double disc2_rot, disc2_twist, disc2_cosadd, disc2_sinadd, disc2_timespi;
    double pie_slices, pie_rotation, pie_thickness;
    double radialBlur_spinvar, radialBlur_zoomvar;
    double whorl_inside, whorl_outside;
    double auger_sym, auger_weight, auger_freq, auger_scale;
    double stripes_space, stripes_warp;
} xform_t;

/* Precalculated values */
typedef struct {
    double tx, ty;
    double p0, p1;
    double sumsq, sqr, atanyx, atanxy;
    double sina, cosa; /* sin/cos of atan(tx,ty) = flam3 theta */
    xform_t *xf;
} helper_t;

static void setup(helper_t *f, double x, double y, xform_t *xf) {
    f->tx = x; f->ty = y;
    f->p0 = 0.0; f->p1 = 0.0;
    f->sumsq = x*x + y*y;
    f->sqr = sqrt(f->sumsq);
    f->atanyx = atan2(y, x);
    f->atanxy = atan2(x, y);
    double theta = atan2(x, y);
    f->sina = sin(theta);
    f->cosa = cos(theta);
    f->xf = xf;
}

/* ================================================================
 * Variation implementations copied from flam3 variations.c
 * Each one matches flam3 EXACTLY.
 * ================================================================ */

/* 0: linear */
static void v_linear(helper_t *f, double w) { f->p0 += w*f->tx; f->p1 += w*f->ty; }
/* 1: sinusoidal */
static void v_sinusoidal(helper_t *f, double w) { f->p0 += w*sin(f->tx); f->p1 += w*sin(f->ty); }
/* 2: spherical */
static void v_spherical(helper_t *f, double w) {
    double r2 = w / (f->sumsq + EPS);
    f->p0 += r2 * f->tx; f->p1 += r2 * f->ty;
}
/* 3: swirl */
static void v_swirl(helper_t *f, double w) {
    double r2 = f->sumsq;
    double c1 = sin(r2), c2 = cos(r2);
    f->p0 += w*(c1*f->tx - c2*f->ty);
    f->p1 += w*(c2*f->tx + c1*f->ty);
}
/* 4: horseshoe */
static void v_horseshoe(helper_t *f, double w) {
    double r = w / (f->sqr + EPS);
    f->p0 += (f->tx - f->ty) * (f->tx + f->ty) * r;
    f->p1 += 2.0 * f->tx * f->ty * r;
}
/* 5: polar */
static void v_polar(helper_t *f, double w) {
    /* flam3 uses precalc_atan = atan2(tx, ty) = atan2(x, y) */
    f->p0 += w * f->atanxy / M_PI;
    f->p1 += w * (f->sqr - 1.0);
}
/* 8: disk — note flam3 uses atan(tx,ty) not atan(ty,tx) */
static void v_disk(helper_t *f, double w) {
    double a = f->atanxy * (1.0/M_PI);
    double r = M_PI * f->sqr;
    f->p0 += w * sin(r) * a;
    f->p1 += w * cos(r) * a;
}
/* 13: julia — CONDITIONAL: random +π */
static void v_julia(helper_t *f, double w) {
    double a = 0.5 * f->atanyx;
    if (rng_bit()) a += M_PI;
    double r = w * sqrt(f->sqr);
    f->p0 += r * cos(a);
    f->p1 += r * sin(a);
}
/* 14: bent — CONDITIONAL: negative x/y */
static void v_bent(helper_t *f, double w) {
    double nx = f->tx, ny = f->ty;
    if (nx < 0.0) nx *= 2.0;
    if (ny < 0.0) ny /= 2.0;
    f->p0 += w*nx; f->p1 += w*ny;
}
/* 16: fisheye */
static void v_fisheye(helper_t *f, double w) {
    double r = 2.0 * w / (f->sqr + 1.0);
    f->p0 += r * f->ty;  /* NOTE: flam3 swaps x/y */
    f->p1 += r * f->tx;
}
/* 20: cosine */
static void v_cosine(helper_t *f, double w) {
    double nx = cos(f->tx * M_PI) * cosh(f->ty);
    double ny = -sin(f->tx * M_PI) * sinh(f->ty);
    f->p0 += w*nx; f->p1 += w*ny;
}
/* 22: fan — CONDITIONAL: mod threshold */
static void v_fan(helper_t *f, double w) {
    /* flam3 fan uses precalc c,f from affine coefs */
    /* For testing, use c=0.5, f=0.5 as if from identity affine */
    double dx = 0.5*0.5*M_PI + EPS;  /* c² * π */
    double dx2 = dx * 0.5;
    double a = f->atanxy;  /* flam3 uses atan(x,y) */
    double r = w * f->sqr;
    a += (fmod(a+0.5, 2*dx) > dx) ? -dx2 : dx2;
    f->p0 += r * cos(a); f->p1 += r * sin(a);
}
/* 26: rings2 — CONDITIONAL: floor-based mod */
static void v_rings2(helper_t *f, double w) {
    double val = f->xf->rings2_val;
    double dx = val*val + EPS;
    double r = f->sqr;
    if (r < 1e-10) { f->p0 += w*f->tx; f->p1 += w*f->ty; return; }
    double k = (int)((r/dx + 1)/2);
    double rr = w * (2.0 - dx*(k*2.0/r + 1.0));
    f->p0 += rr * f->tx; f->p1 += rr * f->ty;
}
/* 27: eyefish */
static void v_eyefish(helper_t *f, double w) {
    double r = 2.0 * w / (f->sqr + 1.0);
    f->p0 += r * f->tx; f->p1 += r * f->ty;
}
/* 28: bubble */
static void v_bubble(helper_t *f, double w) {
    double r = w / (0.25*(f->sumsq) + 1);
    f->p0 += r * f->tx; f->p1 += r * f->ty;
}
/* 32: julian */
static void v_julian(helper_t *f, double w) {
    int t_rnd = (int)(f->xf->julian_rN * rng_float());
    double a = (f->atanyx + 2*M_PI*t_rnd) / f->xf->julian_power;
    double r = w * pow(f->sumsq, f->xf->julian_cn);
    f->p0 += r * cos(a); f->p1 += r * sin(a);
}
/* 33: juliascope — CONDITIONAL: random angle sign */
static void v_juliascope(helper_t *f, double w) {
    int t_rnd = (int)(f->xf->juliascope_rN * rng_float());
    double tmpr;
    if (rng_bit())
        tmpr = (2*M_PI*t_rnd + f->atanyx) / f->xf->juliascope_power;
    else
        tmpr = (2*M_PI*t_rnd - f->atanyx) / f->xf->juliascope_power;
    double r = w * pow(f->sumsq, f->xf->juliascope_cn);
    f->p0 += r * cos(tmpr); f->p1 += r * sin(tmpr);
}
/* 38: ngon — CONDITIONAL: angle sector fold */
static void v_ngon(helper_t *f, double w) {
    double b = 2*M_PI / f->xf->ngon_sides;
    double ph = f->atanyx - b*floor(f->atanyx/b);
    if (ph > b/2) ph -= b;
    double r_factor = pow(f->sumsq, f->xf->ngon_power/2.0);
    double amp = (f->xf->ngon_corners * (1.0/cos(ph) - 1.0) + f->xf->ngon_circle);
    if (r_factor > 1e-10) amp /= r_factor;
    f->p0 += w * amp * f->tx; f->p1 += w * amp * f->ty;
}
/* 40: rectangles — CONDITIONAL: zero check */
static void v_rectangles(helper_t *f, double w) {
    double rx = f->xf->rectangles_x, ry = f->xf->rectangles_y;
    /* flam3: (2*floor(x/rx) + 1)*rx - x, weight applied outside */
    if (rx == 0) f->p0 += w * f->tx;
    else f->p0 += w * ((2*floor(f->tx/rx) + 1)*rx - f->tx);
    if (ry == 0) f->p1 += w * f->ty;
    else f->p1 += w * ((2*floor(f->ty/ry) + 1)*ry - f->ty);
}
/* 46: secant2 — CONDITIONAL: cos sign */
static void v_secant2(helper_t *f, double w) {
    double r = w * f->sqr;
    double cr = cos(r);
    double icr = 1.0 / (cr + EPS);
    f->p0 += w * f->tx;
    f->p1 += (cr < 0) ? w*(icr + 1) : w*(icr - 1);
}
/* 50: loonie — CONDITIONAL: r² vs w² */
static void v_loonie(helper_t *f, double w) {
    double r2 = f->sumsq;
    double w2 = w*w;
    if (r2 < w2) {
        double r = w * sqrt(w2/r2 - 1.0);
        f->p0 += r * f->tx; f->p1 += r * f->ty;
    } else {
        f->p0 += w * f->tx; f->p1 += w * f->ty;
    }
}
/* 54: bent2 — CONDITIONAL: negative x/y with params */
static void v_bent2(helper_t *f, double w) {
    double nx = f->tx, ny = f->ty;
    if (nx < 0.0) nx *= f->xf->bent2_x;
    if (ny < 0.0) ny *= f->xf->bent2_y;
    f->p0 += w*nx; f->p1 += w*ny;
}
/* 55: bipolar — CONDITIONAL: angle wrapping */
static void v_bipolar(helper_t *f, double w) {
    double x2y2 = f->sumsq;
    double t = x2y2 + 1;
    double x2 = 2*f->tx;
    double ps = -M_PI_2 * f->xf->bipolar_shift;
    double y = 0.5 * atan2(2*f->ty, x2y2 - 1) + ps;
    if (y > M_PI_2)
        y = -M_PI_2 + fmod(y + M_PI_2, M_PI);
    else if (y < -M_PI_2)
        y = M_PI_2 - fmod(M_PI_2 - y, M_PI);
    double f_val = t + x2;
    double g = t - x2;
    if (g == 0) g = EPS;
    f->p0 += w * 0.25 * 2.0/M_PI * log(f_val/g);
    f->p1 += w * 2.0/M_PI * y;
}
/* 58: cell — CONDITIONAL: quadrant-based interleave */
static void v_cell(helper_t *f, double w) {
    double sz = f->xf->cell_size;
    double inv = 1.0 / sz;
    int ix = (int)floor(f->tx * inv);
    int iy = (int)floor(f->ty * inv);
    double dx = f->tx - ix*sz;
    double dy = f->ty - iy*sz;
    if (iy >= 0) {
        if (ix >= 0) { iy *= 2; ix *= 2; }
        else { iy *= 2; ix = -(2*ix + 1); }
    } else {
        if (ix >= 0) { iy = -(2*iy + 1); ix *= 2; }
        else { iy = -(2*iy + 1); ix = -(2*ix + 1); }
    }
    f->p0 += w * (dx + ix*sz);
    f->p1 += w * (-dy - iy*sz);
}
/* 65: lazysusan — CONDITIONAL: inside vs outside radius */
static void v_lazysusan(helper_t *f, double w) {
    double xx = f->tx - f->xf->lazysusan_x;
    double yy = f->ty + f->xf->lazysusan_y;
    double rr = sqrt(xx*xx + yy*yy);
    if (rr < w) {
        double a = atan2(yy, xx) + f->xf->lazysusan_spin + f->xf->lazysusan_twist*(w - rr);
        rr = w * rr;
        f->p0 += rr*cos(a) + f->xf->lazysusan_x;
        f->p1 += rr*sin(a) - f->xf->lazysusan_y;
    } else {
        rr = w * (1.0 + f->xf->lazysusan_space / rr);
        f->p0 += rr*xx + f->xf->lazysusan_x;
        f->p1 += rr*yy - f->xf->lazysusan_y;
    }
}
/* 66: loonie */
/* (already done above as v_loonie) */

/* 68: modulus — CONDITIONAL: three-way x and y */
static void v_modulus(helper_t *f, double w) {
    double mx = f->xf->modulus_x, my = f->xf->modulus_y;
    double xr = 2*mx, yr = 2*my;
    double ox, oy;
    if (f->tx > mx) ox = -mx + fmod(f->tx + mx, xr);
    else if (f->tx < -mx) ox = mx - fmod(mx - f->tx, xr);
    else ox = f->tx;
    if (f->ty > my) oy = -my + fmod(f->ty + my, yr);
    else if (f->ty < -my) oy = my - fmod(my - f->ty, yr);
    else oy = f->ty;
    f->p0 += w*ox; f->p1 += w*oy;
}
/* 69: oscilloscope — CONDITIONAL: y threshold */
static void v_oscope(helper_t *f, double w) {
    double tpf = 2*M_PI*f->xf->oscope_frequency;
    double t;
    if (f->xf->oscope_damping == 0.0)
        t = f->xf->oscope_amplitude * cos(tpf*f->tx) + f->xf->oscope_separation;
    else
        t = f->xf->oscope_amplitude * exp(-fabs(f->tx)*f->xf->oscope_damping) * cos(tpf*f->tx) + f->xf->oscope_separation;
    if (fabs(f->ty) <= t)
        f->p1 -= w * f->ty;
    else
        f->p1 += w * f->ty;
    f->p0 += w * f->tx;
}
/* 70: polar2 */
static void v_polar2(helper_t *f, double w) {
    double p2v = w / M_PI;
    f->p0 += p2v * f->atanyx;
    f->p1 += p2v / 2.0 * log(f->sumsq);
}
/* 72: scry */
static void v_scry(helper_t *f, double w) {
    double t = f->sumsq;
    double r = 1.0 / (f->sqr * (t + 1.0/(w+EPS)));
    f->p0 += r * f->tx; f->p1 += r * f->ty;
}
/* 75: separation — CONDITIONAL: sign-based */
static void v_separation(helper_t *f, double w) {
    double sx2 = f->xf->separation_x * f->xf->separation_x;
    double sy2 = f->xf->separation_y * f->xf->separation_y;
    if (f->tx > 0.0)
        f->p0 += w*(sqrt(f->tx*f->tx + sx2) - f->tx*f->xf->separation_xinside);
    else
        f->p0 -= w*(sqrt(f->tx*f->tx + sx2) + f->tx*f->xf->separation_xinside);
    if (f->ty > 0.0)
        f->p1 += w*(sqrt(f->ty*f->ty + sy2) - f->ty*f->xf->separation_yinside);
    else
        f->p1 -= w*(sqrt(f->ty*f->ty + sy2) + f->ty*f->xf->separation_yinside);
}
/* 76: split — CONDITIONAL: cos-sign based */
static void v_split(helper_t *f, double w) {
    double sx = cos(f->ty * f->xf->split_ysize * M_PI) >= 0 ? 1 : -1;
    double sy = cos(f->tx * f->xf->split_xsize * M_PI) >= 0 ? 1 : -1;
    f->p0 += w * f->tx * sx;  /* NOTE: flam3 uses sx for p0 (y-based) */
    f->p1 += w * f->ty * sy;  /* and sy for p1 (x-based) */
}
/* 80: whorl — CONDITIONAL: inside vs outside r=1 */
static void v_whorl(helper_t *f, double w) {
    double r = f->sqr;
    double a;
    if (r < w)
        a = f->atanyx + f->xf->whorl_inside / (w - r);
    else
        a = f->atanyx + f->xf->whorl_outside / (r - w);
    f->p0 += w * r * cos(a);
    f->p1 += w * r * sin(a);
}

/* ================================================================
 * Test harness
 * ================================================================ */

/* Test points chosen to exercise conditional branches:
 * - positive/negative quadrants for bent, bent2, separation
 * - inside/outside unit circle for loonie, whorl, lazysusan
 * - various angles for fan, ngon, sector-based variations
 * - grid boundaries for rectangles, modulus, cell
 */
static double test_pts[][2] = {
    { 1.0,  0.5},  /* 0: Q1, r>1 */
    {-0.3,  0.7},  /* 1: Q2 */
    { 0.5, -0.5},  /* 2: Q4 */
    {-0.8, -0.4},  /* 3: Q3 */
    { 0.1,  0.1},  /* 4: near origin, r<1 */
    { 0.05, 0.02}, /* 5: very near origin */
    { 2.0,  1.0},  /* 6: far from origin */
    { 0.9,  0.3},  /* 7: r≈1 from inside */
    { 1.1,  0.3},  /* 8: r≈1 from outside */
    { 0.49, 0.49}, /* 9: grid boundary */
    { 0.51, 0.51}, /* 10: grid boundary other side */
    {-1.5,  0.8},  /* 11: negative, far */
};
#define N_PTS (sizeof(test_pts)/sizeof(test_pts[0]))

typedef void (*vfunc)(helper_t *, double);
typedef struct {
    int idx;
    const char *name;
    vfunc func;
    int rng;
} var_entry;

int main() {
    xform_t xf;
    memset(&xf, 0, sizeof(xf));

    /* Default params */
    xf.julian_power = 4; xf.julian_dist = 1;
    xf.julian_rN = 4; xf.julian_cn = 0.125;
    xf.juliascope_power = 4; xf.juliascope_dist = 1;
    xf.juliascope_rN = 4; xf.juliascope_cn = 0.125;
    xf.rings2_val = 0.5;
    xf.rectangles_x = 0.8; xf.rectangles_y = 0.6;
    xf.ngon_sides = 5; xf.ngon_power = 3; xf.ngon_circle = 1; xf.ngon_corners = 2;
    xf.bent2_x = 1.5; xf.bent2_y = -0.5;
    xf.bipolar_shift = 0.3;
    xf.cell_size = 1.0;
    xf.modulus_x = 0.5; xf.modulus_y = 0.5;
    xf.lazysusan_x = 0.1; xf.lazysusan_y = 0.1;
    xf.lazysusan_spin = 0.5; xf.lazysusan_twist = 0.3; xf.lazysusan_space = 0.2;
    xf.oscope_separation = 1.0; xf.oscope_frequency = M_PI;
    xf.oscope_amplitude = 1.0; xf.oscope_damping = 0.1;
    xf.separation_x = 0.5; xf.separation_y = 0.5;
    xf.separation_xinside = 0.2; xf.separation_yinside = 0.3;
    xf.split_xsize = 0.5; xf.split_ysize = 0.5;
    xf.whorl_inside = 0.5; xf.whorl_outside = 0.5;

    var_entry vars[] = {
        { 0, "linear",      v_linear,      0},
        { 1, "sinusoidal",  v_sinusoidal,  0},
        { 2, "spherical",   v_spherical,   0},
        { 3, "swirl",       v_swirl,       0},
        { 4, "horseshoe",   v_horseshoe,   0},
        { 5, "polar",       v_polar,       0},
        { 8, "disk",        v_disk,        0},
        {13, "julia",       v_julia,       1},
        {14, "bent",        v_bent,        0},
        {16, "fisheye",     v_fisheye,     0},
        {20, "cosine",      v_cosine,      0},
        {26, "rings2",      v_rings2,      0},
        {27, "eyefish",     v_eyefish,     0},
        {28, "bubble",      v_bubble,      0},
        {32, "julian",      v_julian,      1},
        {33, "juliascope",  v_juliascope,  1},
        {38, "rectangles",  v_rectangles,  0},
        {46, "secant2",     v_secant2,     0},
        {49, "ngon",        v_ngon,        0},
        {50, "loonie",      v_loonie,      0},
        {54, "bent2",       v_bent2,       0},
        {55, "bipolar",     v_bipolar,     0},
        {56, "cell",        v_cell,        0},
        {57, "whorl",       v_whorl,       0},
        {65, "lazysusan",   v_lazysusan,   0},
        {68, "modulus",     v_modulus,      0},
        {69, "oscilloscope",v_oscope,      0},
        {70, "polar2",      v_polar2,      0},
        {72, "scry",        v_scry,        0},
        {75, "separation",  v_separation,  0},
        {76, "split",       v_split,       0},
        {80, "whorl",       v_whorl,       0},
    };
    /* whorl is listed twice — once at our idx 57, once referencing flam3 idx 80.
     * Our idx for whorl is 57, fix the duplicate: */
    vars[31].idx = 57; /* second whorl entry — remove or fix */

    int n_vars = sizeof(vars)/sizeof(vars[0]) - 1; /* skip duplicate */

    printf("var_index,var_name,x_in,y_in,weight,x_out,y_out,rng_seed\n");

    for (int v = 0; v < n_vars; v++) {
        for (int p = 0; p < (int)N_PTS; p++) {
            helper_t f;
            setup(&f, test_pts[p][0], test_pts[p][1], &xf);
            unsigned int seed = (unsigned int)(p * 17 + 1);
            rng_seed(seed);
            vars[v].func(&f, 1.0);
            printf("%d,%s,%.15g,%.15g,1.0,%.15g,%.15g,%u\n",
                   vars[v].idx, vars[v].name,
                   test_pts[p][0], test_pts[p][1],
                   f.p0, f.p1, seed);
        }
    }

    /* Extra: julian with sheep 45928 params */
    xf.julian_power = 6; xf.julian_dist = -1.25244;
    xf.julian_rN = 6; xf.julian_cn = -1.25244/6.0/2.0;
    for (int p = 0; p < (int)N_PTS; p++) {
        helper_t f;
        setup(&f, test_pts[p][0], test_pts[p][1], &xf);
        rng_seed((unsigned int)(p * 17 + 1));
        v_julian(&f, 1.0);
        printf("32,julian_sheep45928,%.15g,%.15g,1.0,%.15g,%.15g,%u\n",
               test_pts[p][0], test_pts[p][1], f.p0, f.p1, p*17+1);
    }

    return 0;
}
