#version 430 core

// ============================================================
// tonemap.vert — fullscreen quad vertex shader with perspective skew
// ============================================================
// Trapezoidal distortion for angled side monitors:
//   - Far side compressed vertically (narrower slice of canvas)
//   - Near side expanded (wider slice)
//   - Horizontal lines stay horizontal
//   - Vertical lines converge toward the far edge
//
// This matches viewing a flat wall through an angled window.
// ============================================================

in vec2 in_pos;
out vec2 v_uv;

uniform float u_skew = 0.0;

void main() {
    v_uv = in_pos * 0.5 + 0.5;

    if (u_skew == 0.0) {
        gl_Position = vec4(in_pos, 0.0, 1.0);
    } else {
        // Scale Y based on X position — trapezoid, not perspective warp.
        // t = 0..1 across screen, scale = 1 at center, compressed on far side.
        float t = in_pos.x * 0.5 + 0.5;
        float scale = 1.0 / (1.0 + u_skew * (t - 0.5));

        // Compress/expand Y around center (y=0 in clip space)
        float y_skewed = in_pos.y * scale;

        gl_Position = vec4(in_pos.x, y_skewed, 0.0, 1.0);
    }
}
