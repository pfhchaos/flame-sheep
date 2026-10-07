#version 450
// Vulkan port of flame_sheep/rendering/shaders/tonemap.frag.
//
// Reads the histogram + max_buf, computes log-density alpha, maps the
// color_idx to a quick-and-dirty hue spectrum, outputs the final pixel
// color. The full GL version supports a palette texture, multi-monitor
// viewports, and a vibrancy parameter — this Phase 3 minimal cut skips
// them so the first end-to-end Vulkan render lands quickly. Each gap
// is a known follow-up.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

layout(std430, binding = 0) readonly buffer Histogram {
    uint histogram[];
};

layout(std430, binding = 1) readonly buffer MaxBuf {
    uint max_hits;
};

// Palette texture: 256x1 RGBA (the alpha channel is ignored). Color
// index ∈ [0,1] samples along U with linear filtering for smooth
// gradients between palette entries.
layout(set = 0, binding = 2) uniform sampler2D u_palette;

layout(push_constant) uniform PushConstants {
    // Canvas (= histogram resolution), shared across all outputs
    int canvas_w;
    int canvas_h;
    // Viewport: this output's slice of the canvas in canvas pixels.
    // Single-output / single-window: (0, 0, canvas_w, canvas_h).
    // Multi-monitor: each output renders its rectangular slice from
    // the same shared histogram, giving a continuous image.
    int viewport_x;
    int viewport_y;
    int viewport_w;
    int viewport_h;
    float gamma;
};

#define COLOR_SCALE 1000000u

void main() {
    // Map this output's UV [0,1]^2 into its viewport slice of the
    // full canvas. UV's origin is top-left, viewport is in canvas
    // pixel coords, sampling reads from the shared histogram.
    int px = viewport_x + int(v_uv.x * float(viewport_w));
    int py = viewport_y + int(v_uv.y * float(viewport_h));
    // Out-of-canvas → black (multi-monitor layouts where one output's
    // viewport extends beyond the canvas bounds — render the gap as
    // background rather than wrapping or clamping into garbage).
    if (px < 0 || px >= canvas_w || py < 0 || py >= canvas_h) {
        frag_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    uint n_pixels = uint(canvas_w * canvas_h);
    uint idx = uint(py * canvas_w + px);
    uint hits  = histogram[idx];
    uint color = histogram[n_pixels + idx];

    if (hits == 0u) {
        frag_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Log-density tone mapping. Without log, the histogram's huge
    // dynamic range (center ~millions of hits, edges ~1) makes
    // everything-not-center invisible.
    float log_hits = log(float(hits) + 1.0);
    float log_max  = log(float(max_hits) + 1.0);
    float alpha = (log_max > 0.0) ? clamp(log_hits / log_max, 0.0, 1.0)
                                   : 0.0;
    alpha = pow(alpha, 1.0 / gamma);

    // color_idx = color_acc / (hits * COLOR_SCALE)
    float color_idx = float(color) / (float(hits) * float(COLOR_SCALE));
    color_idx = clamp(color_idx, 0.0, 1.0);
    vec3 rgb = texture(u_palette, vec2(color_idx, 0.5)).rgb * alpha;
    frag_color = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}
