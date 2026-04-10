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

// Audio spectrum texture: N_BINS x 1 R — for reactive effects
uniform sampler2D u_audio;

uniform int u_width;
uniform int u_height;

// Tone mapping parameters — tweak these to taste
uniform float u_gamma     = 1.8;    // gamma correction (lower = brighter midtones)
uniform float u_brightness= 6.0;    // overall brightness multiplier
uniform float u_vibrancy  = 1.0;    // 0=desaturated, 1=full color

void main() {
    // Convert UV to pixel index
    // v_uv is 0..1, flip Y because OpenGL origin is bottom-left
    int px = int(v_uv.x * float(u_width));
    int py = int((1.0 - v_uv.y) * float(u_height));

    // Clamp to valid range
    px = clamp(px, 0, u_width  - 1);
    py = clamp(py, 0, u_height - 1);

    uint idx = uint(py * u_width + px);

    // Read histogram — packed array, color_acc offset by n_pixels
    uint n_pixels = uint(u_width * u_height);
    uint hits  = histogram[idx];
    uint color = histogram[n_pixels + idx];

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
    // We don't have max_hits as a uniform yet (TODO: compute it
    // in a reduction pass or approximate). For now we use a
    // reasonable approximation: clamp after scaling.
    // --------------------------------------------------------

    // log(hits) normalized — the +1 avoids log(0)
    float log_hits = log(float(hits) + 1.0);

    // Approximate normalization: scale by expected max
    // TODO: replace with actual max from reduction pass
    float expected_max_log = log(float(u_width * u_height) * 0.01 + 1.0);
    float alpha = clamp(log_hits / expected_max_log * u_brightness, 0.0, 1.0);

    // Gamma correction — same as display gamma, makes it look right on screen
    alpha = pow(alpha, 1.0 / u_gamma);

    // --------------------------------------------------------
    // Color lookup
    //
    // Average color index for this pixel = color_acc / hits
    // (both stored as uint, color_acc scaled by COLOR_SCALE)
    // --------------------------------------------------------
    float color_idx = float(color) / (float(hits) * float(COLOR_SCALE));
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

    // --------------------------------------------------------
    // Optional: audio reactivity
    //
    // Sample spectrum at a frequency corresponding to this
    // pixel's distance from center — makes the flame "pulse"
    // with the music without changing the geometry.
    // --------------------------------------------------------
    vec2 centered_uv = v_uv - 0.5;
    float dist_from_center = length(centered_uv) * 2.0;  // 0..1
    float audio_energy = texture(u_audio, vec2(dist_from_center * 0.5, 0.5)).r;

    // Subtle brightness pulse — scale by 1.0 + small audio fraction
    // Keeps the effect tasteful rather than strobing
    float audio_boost = 1.0 + audio_energy * 0.15;
    final_color *= audio_boost;

    frag_color = vec4(clamp(final_color, 0.0, 1.0), 1.0);
}
