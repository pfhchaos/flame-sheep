#version 430 core

// ============================================================
// tonemap.vert — fullscreen quad vertex shader
// ============================================================
// Runs exactly 6 times (two triangles covering the screen).
// All it does is pass clip-space positions through and compute
// UV coords for the fragment shader.
//
// Clip space:  (-1,-1) bottom-left  →  (1,1) top-right
// UV space:    ( 0, 0) bottom-left  →  (1,1) top-right
// ============================================================

// 'in_pos' matches the VAO attribute name in renderer.py
// It's the clip-space XY from our quad vertex buffer
in vec2 in_pos;

// Passed to fragment shader — interpolated automatically across the quad
out vec2 v_uv;

// Perspective correction for angled side monitors.
// Skews UV sampling to simulate viewing the canvas from an angle.
// skew_amount: 0 = head-on, positive = viewer is to the left,
// negative = viewer is to the right.
uniform float u_skew = 0.0;

void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);

    // Base UV
    vec2 uv = in_pos * 0.5 + 0.5;

    // Perspective foreshortening: the far edge (away from viewer)
    // should sample a narrower slice of the canvas.
    // u_skew controls which side is "far": positive = right side far.
    // We compress the UV range on the far side.
    if (u_skew != 0.0) {
        // t goes 0..1 across the screen (left to right)
        float t = uv.x;
        // Perspective: far side compresses toward center
        float perspective = 1.0 + u_skew * (t - 0.5);
        uv.y = 0.5 + (uv.y - 0.5) / max(perspective, 0.1);
    }

    v_uv = uv;
}
