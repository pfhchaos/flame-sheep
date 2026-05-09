#version 430 core

// ============================================================
// test_pattern.frag — grid/calibration pattern for multi-monitor alignment
// ============================================================
// Renders a grid directly from UV coordinates, bypassing the chaos game
// and histogram. Goes through the same vertex shader (with skew) so
// perspective and viewport slicing behave identically to the fractal.
//
// Shows:
//   - Grid lines at regular intervals (alignment across screen edges)
//   - Color-coded quadrants (which part of the canvas each screen shows)
//   - Crosshair at center (viewport centering)
//   - Border lines at screen edges (gap/overlap detection)
// ============================================================

in vec2 v_uv;
out vec4 frag_color;

uniform int u_width;          // full virtual canvas render width
uniform int u_height;         // full virtual canvas render height
uniform int u_viewport_x;     // this window's left edge in canvas pixels
uniform int u_viewport_y;     // this window's top edge in canvas pixels
uniform int u_viewport_w;     // this window's width in canvas pixels
uniform int u_viewport_h;     // this window's height in canvas pixels
uniform int u_surface_w;      // actual surface width (physical pixels)
uniform int u_surface_h;      // actual surface height (physical pixels)
uniform float u_ppmm;         // canvas pixels per mm (for physical size grid)

void main() {
    // Map UV to canvas position
    float scale_x = float(u_viewport_w) / float(u_surface_w);
    float scale_y = float(u_viewport_h) / float(u_surface_h);
    float cx = float(u_viewport_x) + v_uv.x * float(u_surface_w) * scale_x;
    float cy = float(u_viewport_y) + (1.0 - v_uv.y) * float(u_surface_h) * scale_y;

    // Normalized canvas position (0..1)
    float nx = cx / float(u_width);
    float ny = cy / float(u_height);

    // Grid spacing based on physical size (10mm minor, 50mm major)
    // so you can verify dot pitch with a ruler
    float grid_major = 50.0 * u_ppmm;  // 50mm = ~2 inches
    float grid_minor = 10.0 * u_ppmm;  // 10mm = ~0.4 inches

    // Grid lines
    float mx = mod(cx, grid_major);
    float my = mod(cy, grid_major);
    float mnx = mod(cx, grid_minor);
    float mny = mod(cy, grid_minor);

    bool on_major = mx < 1.5 || my < 1.5;
    bool on_minor = mnx < 0.8 || mny < 0.8;

    // Center crosshair (canvas center)
    float center_x = float(u_width) * 0.5;
    float center_y = float(u_height) * 0.5;
    bool on_crosshair = (abs(cx - center_x) < 2.0) || (abs(cy - center_y) < 2.0);

    // Screen edge markers — lines at viewport boundaries
    bool on_left_edge = cx - float(u_viewport_x) < 3.0;
    bool on_right_edge = float(u_viewport_x + u_viewport_w) - cx < 3.0;
    bool on_top_edge = cy - float(u_viewport_y) < 3.0;
    bool on_bottom_edge = float(u_viewport_y + u_viewport_h) - cy < 3.0;
    bool on_edge = on_left_edge || on_right_edge || on_top_edge || on_bottom_edge;

    // Background color — obvious per-third tinting
    vec3 bg;
    if (nx < 0.333)
        bg = vec3(0.15, 0.0, 0.0);    // left third: RED
    else if (nx < 0.666)
        bg = vec3(0.0, 0.12, 0.0);    // center third: GREEN
    else
        bg = vec3(0.0, 0.0, 0.15);    // right third: BLUE

    // Compose
    vec3 color = bg;
    if (on_minor) color = vec3(0.15);
    if (on_major) color = vec3(0.3);
    if (on_crosshair) color = vec3(1.0, 1.0, 0.0);  // yellow crosshair
    if (on_edge) color = vec3(1.0, 0.0, 0.0);        // red screen borders

    frag_color = vec4(color, 1.0);
}
