#version 430 core

// ============================================================
// tonemap.frag — log-density tone mapping + color
// ============================================================
// Runs once per pixel. Reads the histogram SSBO, applies
// log-density tone mapping (the key to flame fractal aesthetics),
// looks up color from palette, outputs final RGB.
//
// This is where the "flame" look comes from. Without log mapping
// you'd get a flat ugly blob. Log mapping reveals structure
// across many orders of magnitude of hit density simultaneously.
// ============================================================

// UV coords interpolated from vertex shader
in vec2 v_uv;

// Final pixel color output
out vec4 frag_color;

// Same histogram SSBO as compute shader
// Must match binding and layout exactly
#define COLOR_SCALE 1000000u

// Same packed layout as flame.comp:
//   [0        .. n_pixels-1] = hit_count
//   [n_pixels .. 2*n_pixels-1] = color_acc
layout(std430, binding = 0) readonly buffer Histogram {
    uint histogram[];
};

// Palette texture: 256x1 RGB — maps color index (0..1) to RGB
uniform sampler2D u_palette;

uniform int u_width;          // full virtual canvas render width
uniform int u_height;         // full virtual canvas render height
uniform int u_viewport_x;     // this window's left edge in canvas pixels
uniform int u_viewport_y;     // this window's top edge in canvas pixels
uniform int u_viewport_w;     // this window's width in canvas pixels
uniform int u_viewport_h;     // this window's height in canvas pixels
uniform int u_surface_w;      // actual EGL surface width (physical pixels)
uniform int u_surface_h;      // actual EGL surface height (physical pixels)

// Tone mapping parameters — tweak these to taste
uniform float u_gamma;              // audio-driven gamma (lower = brighter/vivid, higher = ghostly)
uniform float u_vibrancy  = 1.0;    // 0=desaturated, 1=full color
uniform int u_linear_mode = 0;      // 1 = skip log (for post-DE data already in log domain)
uniform uint u_hist_offset = 0u;   // offset into histogram[] for compare mode
uniform uint u_hist_stride = 0u;   // distance from hits to colors (set by renderer)


// Actual max hit count from GPU reduction pass (binding=8)
layout(std430, binding = 8) readonly buffer MaxBuf {
    uint max_hits;
};

void main() {
    // Convert UV to canvas pixel index.
    // v_uv is 0..1 over this window's quad (physical surface).
    // Map into the canvas slice, accounting for resolution scaling.
    // Flip Y because OpenGL origin is bottom-left.
    float scale_x = float(u_viewport_w) / float(u_surface_w);
    float scale_y = float(u_viewport_h) / float(u_surface_h);
    int px = u_viewport_x + int(v_uv.x * float(u_surface_w) * scale_x);
    int py = u_viewport_y + int((1.0 - v_uv.y) * float(u_surface_h) * scale_y);

    // Discard pixels outside canvas (overscan region)
    if (px < 0 || px >= u_width || py < 0 || py >= u_height) {
        frag_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    uint idx = uint(py * u_width + px);

    // Read histogram — u_hist_stride separates hits from colors
    uint oidx = u_hist_offset + idx;
    uint hits  = histogram[oidx];
    uint color = histogram[u_hist_stride + oidx];

    // Nothing hit this pixel — output black
    if (hits == 0u) {
        frag_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // --------------------------------------------------------
    // Log-density tone mapping
    //
    // Raw hits have huge dynamic range — center of attractor
    // might be hit millions of times, edges hit once or twice.
    // Linear mapping: edges disappear entirely.
    // Log mapping: edges are dim but visible, center is bright.
    //
    // alpha = log(hits) / log(max_hits)
    //
    // max_hits comes from a GPU reduction pass (reduce_max.comp).
    // --------------------------------------------------------

    // Density normalization
    float alpha;
    if (u_linear_mode != 0) {
        // Post-DE: values are already log-scaled, just normalize linearly
        alpha = (max_hits > 0u)
            ? clamp(float(hits) / float(max_hits), 0.0, 1.0)
            : 0.0;
    } else {
        // Normal: log-density normalization using actual max from GPU reduction pass
        float log_hits = log(float(hits) + 1.0);
        float log_max = log(float(max_hits) + 1.0);
        alpha = (log_max > 0.0)
            ? clamp(log_hits / log_max, 0.0, 1.0)
            : 0.0;
    }

    // Gamma: controls how much of the density range is visible.
    // Low gamma (1.2) = vivid, filaments pop. High gamma (2.5) = ghostly.
    // Driven by the brightness axis based on audio energy.
    alpha = pow(alpha, 1.0 / u_gamma);

    // --------------------------------------------------------
    // Color lookup
    //
    // Normal mode: color_acc / (hits * COLOR_SCALE)
    // Linear mode (post-DE): color_acc / hits (DE already normalized)
    // --------------------------------------------------------
    float color_idx;
    if (u_linear_mode != 0) {
        color_idx = float(color) / max(float(hits), 1.0);
    } else {
        color_idx = float(color) / (float(hits) * float(COLOR_SCALE));
    }
    color_idx = clamp(color_idx, 0.0, 1.0);

    // Sample palette — 1D lookup along X axis of the 2D texture
    vec3 palette_color = texture(u_palette, vec2(color_idx, 0.5)).rgb;

    // --------------------------------------------------------
    // Vibrancy
    //
    // Scott Draves' vibrancy parameter blends between:
    //   vibrancy=0: color applied to alpha only (like grayscale)
    //   vibrancy=1: full color with alpha
    //
    // This controls how "neon" vs "soft" the flame looks.
    // --------------------------------------------------------
    vec3 final_color = mix(vec3(alpha), palette_color * alpha, u_vibrancy);

    frag_color = vec4(clamp(final_color, 0.0, 1.0), 1.0);
}
