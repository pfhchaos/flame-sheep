#version 430 core

// ============================================================
// tonemap_flam3.frag — flam3-accurate tone mapping for offline rendering
// ============================================================
// Matches flam3's density estimation + gamma pipeline for producing
// images comparable to electricsheep.com renders.
//
// Key differences from tonemap.frag (live wallpaper):
//   - Uses absolute brightness scaling (k1/k2), not self-normalizing max
//   - pow(density, 1/gamma) gamma correction matching flam3
//   - Linear range blending near zero (gam_lin_thresh)
//   - Vibrancy controls color vs alpha-only blend
// ============================================================

in vec2 v_uv;
out vec4 frag_color;

#define PREFILTER_WHITE 255.0
#define COLOR_SCALE 1000000u

layout(std430, binding = 0) readonly buffer Histogram {
    uint histogram[];
};

uniform sampler2D u_palette;

uniform int u_width;
uniform int u_height;
uniform int u_viewport_x;
uniform int u_viewport_y;
uniform int u_viewport_w;
uniform int u_viewport_h;
uniform int u_surface_w;
uniform int u_surface_h;

// flam3 tonemap parameters
uniform float u_k1;             // brightness * contrast * 268 * PREFILTER_WHITE / 256
uniform float u_k2;             // 1 / (contrast * area * 255 * sample_density)
uniform float u_gamma;          // 1.0 / genome.gamma (pre-inverted on CPU)
uniform float u_vibrancy;       // 0..1, controls color saturation
uniform float u_lin_thresh;     // gamma linear range threshold (gam_lin_thresh)
uniform float u_highlight_power; // highlight power for hue preservation

// flam3_calc_alpha: gamma correction with linear range blending
float calc_alpha(float density, float gamma, float linrange) {
    if (density <= 0.0) return 0.0;

    if (linrange > 0.0 && density < linrange) {
        float funcval = pow(linrange, gamma);
        float frac = density / linrange;
        return (1.0 - frac) * density * (funcval / linrange) + frac * pow(density, gamma);
    }
    return pow(density, gamma);
}

void main() {
    // Convert UV to canvas pixel
    float scale_x = float(u_viewport_w) / float(u_surface_w);
    float scale_y = float(u_viewport_h) / float(u_surface_h);
    int px = u_viewport_x + int(v_uv.x * float(u_surface_w) * scale_x);
    int py = u_viewport_y + int((1.0 - v_uv.y) * float(u_surface_h) * scale_y);
    px = clamp(px, 0, u_width - 1);
    py = clamp(py, 0, u_height - 1);

    uint idx = uint(py * u_width + px);
    uint n_pixels = uint(u_width * u_height);
    uint hits = histogram[idx];
    uint color_acc = histogram[n_pixels + idx];

    if (hits == 0u) {
        frag_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // --------------------------------------------------------
    // Step 1: Density estimation (log scaling with k1/k2)
    //
    // In flam3 the bucket stores accumulated (r,g,b,a) where a = hit count.
    // ls = k1 * log(1 + hits * k2) / hits
    // This scales each pixel's color by its log-density.
    // --------------------------------------------------------
    float fhits = float(hits);
    float ls = u_k1 * log(1.0 + fhits * u_k2) / fhits;

    // --------------------------------------------------------
    // Step 2: Color lookup and scaling
    // --------------------------------------------------------
    float color_idx = float(color_acc) / (fhits * float(COLOR_SCALE));
    color_idx = clamp(color_idx, 0.0, 1.0);
    vec3 palette_color = texture(u_palette, vec2(color_idx, 0.5)).rgb;

    // Scale color channels by density estimation
    // In flam3: c[rgb] = bucket_color[rgb] * ls
    // Our buckets store hit counts and color indices, not RGB.
    // So we reconstruct: pixel_color = palette_lookup * ls
    vec3 scaled_color = palette_color * ls;

    // --------------------------------------------------------
    // Step 3: Gamma correction
    //
    // density = accumulated_alpha / PREFILTER_WHITE
    // alpha = pow(density, 1/gamma) with linear range blending
    // --------------------------------------------------------
    float density = fhits * u_k2;  // normalized density
    // Clamp density for alpha calculation (matches flam3's tmp = ac[3]/PREFILTER_WHITE)
    float alpha_density = ls * fhits / PREFILTER_WHITE;
    float alpha = calc_alpha(clamp(alpha_density, 0.0, 100.0), u_gamma, u_lin_thresh);
    alpha = clamp(alpha, 0.0, 1.0);

    // --------------------------------------------------------
    // Step 4: Vibrancy blend
    //
    // vibrancy=1: full color with alpha scaling
    // vibrancy=0: alpha-only (grayscale)
    //
    // flam3: a = vibrancy * newrgb + (1-vibrancy) * pow(color/255, 1/gamma) * 256
    // --------------------------------------------------------
    // Vibrancy component: colored, scaled by alpha
    vec3 vib_color = u_vibrancy * scaled_color / PREFILTER_WHITE;

    // Non-vibrancy component: gamma-corrected per channel
    vec3 nonvib_color = (1.0 - u_vibrancy) * pow(
        clamp(scaled_color / PREFILTER_WHITE, vec3(0.0), vec3(1.0)),
        vec3(u_gamma)
    );

    vec3 final_color = vib_color + nonvib_color;

    frag_color = vec4(clamp(final_color, 0.0, 1.0), 1.0);
}
