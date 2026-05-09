#version 430 core

// ============================================================
// tonemap.vert — fullscreen quad vertex shader with perspective skew
// ============================================================
// Uses the W coordinate for proper projective interpolation so
// straight lines stay straight on angled side monitors.
//
// The quad is split into 4 triangles (fan from center) to avoid
// the diagonal seam artifact that occurs with 2 triangles.
// ============================================================

in vec2 in_pos;
out vec2 v_uv;

uniform float u_skew = 0.0;

void main() {
    // Base UV
    v_uv = in_pos * 0.5 + 0.5;

    if (u_skew == 0.0) {
        gl_Position = vec4(in_pos, 0.0, 1.0);
    } else {
        float t = in_pos.x * 0.5 + 0.5;
        float w = 1.0 + u_skew * (t - 0.5);
        v_uv = v_uv * w;
        gl_Position = vec4(in_pos.x, in_pos.y, 0.0, w);
    }
}
