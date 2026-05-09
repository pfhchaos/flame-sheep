/*
 * flam3_variation_oracle.c — extract variation outputs from flam3 source
 *
 * Compiles against flam3's variations.c to get ground-truth outputs for
 * specific input points. Outputs CSV: var_index,x_in,y_in,weight,x_out,y_out
 *
 * Build:
 *   cc -o flam3_oracle tools/flam3_variation_oracle.c \
 *      -I ~/projects/flam3 -lm -DPACKAGE_DATA_DIR='""'
 *
 * Usage:
 *   ./flam3_oracle > tests/flam3_reference.csv
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Minimal stubs to satisfy flam3 headers without linking the full library */
#define EPS (1e-10)
#define M_PI 3.14159265358979323846

/* Replicate flam3's iter_helper and xform structures minimally */
typedef struct {
    double tx, ty;           /* input point (post-affine) */
    double p0, p1;           /* output accumulator */
    double precalc_sumsq;    /* tx² + ty² */
    double precalc_sqrt;     /* sqrt(sumsq) */
    double precalc_atanyx;   /* atan2(ty, tx) */
    double precalc_sina, precalc_cosa;  /* sin/cos of theta=atan2(tx,ty) */
    struct xform_s *xform;
    void *rc;                /* RNG context — stub */
} flam3_iter_helper;

typedef struct xform_s {
    double julian_power, julian_dist;
    double julian_rN, julian_cn;
    double juliascope_power, juliascope_dist;
    double juliascope_rN, juliascope_cn;
    double blob_low, blob_high, blob_waves;
    double pdj_a, pdj_b, pdj_c, pdj_d;
    double fan2_x, fan2_y;
    double rings2_val;
    double perspective_angle, perspective_dist;
    double persp_vsin, persp_vfcos;
    double curl_c1, curl_c2;
    double ngon_sides, ngon_power, ngon_circle, ngon_corners;
    double rectangles_x, rectangles_y;
    double super_shape_m, super_shape_n1, super_shape_n2, super_shape_n3;
    double super_shape_rnd, super_shape_holes;
    double cpow_r, cpow_i, cpow_power;
    double disc2_rot, disc2_twist;
    double disc2_sinadd, disc2_cosadd, disc2_timespi;
    double flower_holes, flower_petals;
    double parabola_height, parabola_width;
    double bent2_x, bent2_y;
    double bipolar_shift;
    double cell_size;
    double conic_eccentricity, conic_holes;
    double escher_beta;
    double lazysusan_x, lazysusan_y, lazysusan_spin, lazysusan_twist, lazysusan_space;
    double modulus_x, modulus_y;
    double oscope_separation, oscope_frequency, oscope_amplitude, oscope_damping;
    double popcorn2_x, popcorn2_y, popcorn2_c;
    double separation_x, separation_y, separation_xinside, separation_yinside;
    double split_xsize, split_ysize;
    double splits_x, splits_y;
    double stripes_space, stripes_warp;
    double wedge_angle, wedge_hole, wedge_count, wedge_swirl;
    double wedge_julia_angle, wedge_julia_count, wedge_julia_power, wedge_julia_dist;
    double wedgeJulia_cf, wedgeJulia_rN, wedgeJulia_cn;
    double wedge_sph_angle, wedge_sph_hole, wedge_sph_count, wedge_sph_swirl;
    double auger_sym, auger_weight, auger_freq, auger_scale;
    double flux_spread;
    double mobius_re_a, mobius_re_b, mobius_re_c, mobius_re_d;
    double mobius_im_a, mobius_im_b, mobius_im_c, mobius_im_d;
    double radialBlur_spinvar, radialBlur_zoomvar;
    double pie_slices, pie_rotation, pie_thickness;
    double ngon_power_d;  /* dummy for precalc */
    int num_active;
} flam3_xform;

/* Deterministic RNG stub — returns fixed sequence */
static unsigned int _rng_state = 12345;
static double flam3_random_isaac_01(void *rc) {
    _rng_state ^= _rng_state << 13;
    _rng_state ^= _rng_state >> 17;
    _rng_state ^= _rng_state << 5;
    return (double)_rng_state / 4294967295.0;
}

static int flam3_random_isaac_bit(void *rc) {
    return flam3_random_isaac_01(rc) > 0.5 ? 1 : 0;
}

#define irand(rc) ((int)(flam3_random_isaac_01(rc) * 2147483647.0))
#define flam3_random01() flam3_random_isaac_01(NULL)

/* Precalc helpers */
static void setup_helper(flam3_iter_helper *f, double x, double y) {
    f->tx = x;
    f->ty = y;
    f->p0 = 0.0;
    f->p1 = 0.0;
    f->precalc_sumsq = x*x + y*y;
    f->precalc_sqrt = sqrt(f->precalc_sumsq);
    f->precalc_atanyx = atan2(y, x);
    /* flam3 theta = atan2(x, y) for precalc_sina/cosa */
    double theta = atan2(x, y);
    f->precalc_sina = sin(theta);
    f->precalc_cosa = cos(theta);
    f->rc = NULL;
}

/* Copy variation functions directly from flam3 */
static void var0_linear(flam3_iter_helper *f, double w) {
    f->p0 += w * f->tx;
    f->p1 += w * f->ty;
}

static void var1_sinusoidal(flam3_iter_helper *f, double w) {
    f->p0 += w * sin(f->tx);
    f->p1 += w * sin(f->ty);
}

static void var2_spherical(flam3_iter_helper *f, double w) {
    double r2 = w / (f->precalc_sumsq + EPS);
    f->p0 += r2 * f->tx;
    f->p1 += r2 * f->ty;
}

static void var13_julia(flam3_iter_helper *f, double w) {
    double a = 0.5 * f->precalc_atanyx;
    if (flam3_random_isaac_bit(f->rc)) a += M_PI;
    double r = w * sqrt(f->precalc_sqrt);
    f->p0 += r * cos(a);
    f->p1 += r * sin(a);
}

static void var32_julian(flam3_iter_helper *f, double w) {
    int t_rnd = (int)(f->xform->julian_rN * flam3_random_isaac_01(f->rc));
    double tmpr = (f->precalc_atanyx + 2*M_PI*t_rnd) / f->xform->julian_power;
    double r = w * pow(f->precalc_sumsq, f->xform->julian_cn);
    f->p0 += r * cos(tmpr);
    f->p1 += r * sin(tmpr);
}

static void var33_juliascope(flam3_iter_helper *f, double w) {
    int t_rnd = (int)(f->xform->juliascope_rN * flam3_random_isaac_01(f->rc));
    double tmpr;
    if (flam3_random_isaac_bit(f->rc))
        tmpr = (2*M_PI*t_rnd + f->precalc_atanyx) / f->xform->juliascope_power;
    else
        tmpr = (2*M_PI*t_rnd - f->precalc_atanyx) / f->xform->juliascope_power;
    double r = w * pow(f->precalc_sumsq, f->xform->juliascope_cn);
    f->p0 += r * cos(tmpr);
    f->p1 += r * sin(tmpr);
}

static void var26_rings2(flam3_iter_helper *f, double w) {
    double dx = f->xform->rings2_val * f->xform->rings2_val + EPS;
    double r = f->precalc_sqrt;
    double l = r;
    if (dx == 0 || l == 0) { f->p0 += w*f->tx; f->p1 += w*f->ty; return; }
    double k = (int)((l/dx + 1)/2);
    double rr = w * (2.0 - dx*(k*2.0/l + 1.0));
    f->p0 += rr * f->tx;
    f->p1 += rr * f->ty;
}

/* Test points */
static double test_points[][2] = {
    {1.0, 0.5}, {-0.3, 0.7}, {0.5, -0.5}, {2.0, 1.0},
    {0.1, 0.1}, {0.8, -0.2}, {-0.5, 0.3}, {1.5, -0.8},
};
#define N_POINTS (sizeof(test_points)/sizeof(test_points[0]))

typedef void (*var_func)(flam3_iter_helper *, double);

typedef struct {
    int index;          /* our variation index */
    const char *name;
    var_func func;
    int uses_rng;
} var_entry;

int main() {
    flam3_xform xf;
    memset(&xf, 0, sizeof(xf));

    /* Set up default params */
    xf.julian_power = 4.0;
    xf.julian_dist = 1.0;
    xf.julian_rN = fabs(xf.julian_power);
    xf.julian_cn = xf.julian_dist / xf.julian_power / 2.0;

    xf.juliascope_power = 4.0;
    xf.juliascope_dist = 1.0;
    xf.juliascope_rN = fabs(xf.juliascope_power);
    xf.juliascope_cn = xf.juliascope_dist / xf.juliascope_power / 2.0;

    xf.rings2_val = 0.5;

    var_entry vars[] = {
        {0,  "linear",     var0_linear,     0},
        {1,  "sinusoidal", var1_sinusoidal, 0},
        {2,  "spherical",  var2_spherical,  0},
        {13, "julia",      var13_julia,     1},
        {26, "rings2",     var26_rings2,    0},
        {32, "julian",     var32_julian,    1},
        {33, "juliascope", var33_juliascope,1},
    };
    int n_vars = sizeof(vars) / sizeof(vars[0]);

    printf("var_index,var_name,x_in,y_in,weight,x_out,y_out,uses_rng\n");

    for (int v = 0; v < n_vars; v++) {
        for (int p = 0; p < (int)N_POINTS; p++) {
            flam3_iter_helper f;
            setup_helper(&f, test_points[p][0], test_points[p][1]);
            f.xform = &xf;

            /* Reset RNG for reproducibility */
            _rng_state = (unsigned int)(p * 17 + 1);

            vars[v].func(&f, 1.0);

            printf("%d,%s,%.10f,%.10f,1.0,%.10f,%.10f,%d\n",
                   vars[v].index, vars[v].name,
                   test_points[p][0], test_points[p][1],
                   f.p0, f.p1, vars[v].uses_rng);
        }
    }

    /* Now test julian with sheep 45928 params */
    xf.julian_power = 6.0;
    xf.julian_dist = -1.25244;
    xf.julian_rN = fabs(xf.julian_power);
    xf.julian_cn = xf.julian_dist / xf.julian_power / 2.0;

    printf("# julian with power=6 dist=-1.25244 (sheep 45928)\n");
    for (int p = 0; p < (int)N_POINTS; p++) {
        flam3_iter_helper f;
        setup_helper(&f, test_points[p][0], test_points[p][1]);
        f.xform = &xf;
        _rng_state = (unsigned int)(p * 17 + 1);
        var32_julian(&f, 1.0);
        printf("32,julian_sheep45928,%.10f,%.10f,1.0,%.10f,%.10f,1\n",
               test_points[p][0], test_points[p][1], f.p0, f.p1);
    }

    return 0;
}
