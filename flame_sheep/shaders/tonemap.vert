#version 430 core

// ============================================================
// tonemap.vert — fullscreen quad vertex shader with perspective skew
// ============================================================
// Uses the W coordinate for proper projective interpolation so
// straight lines stay straight on angled side monitors.
//
// u_skew controls perspective foreshortening:
//   0     = head-on (no distortion)
//   > 0   = right edge is further away (right monitor angled left)
//   < 0   = left edge is further away (left monitor angled right)
// ============================================================

in vec2 in_pos;
out vec2 v_uv;

// Perspective skew for angled monitors.
// Set from set_skew(angle_deg) — tan(angle) * 0.5
uniform float u_skew = 0.0;

void main() {
    // Base UV
    v_uv = in_pos * 0.5 + 0.5;

    if (u_skew == 0.0) {
        gl_Position = vec4(in_pos, 0.0, 1.0);
    } else {
        // Projective transform: use the W coordinate so the GPU
        // does perspective-correct interpolation. This keeps
        // straight lines straight (unlike UV-space warping).
        //
        // t = 0..1 across the screen (left to right)
        // The far side gets a larger W, which compresses its
        // contribution during perspective division (pos/W).
        float t = in_pos.x * 0.5 + 0.5;  // 0 = left, 1 = right
        // W > 1 on the far side compresses, W < 1 on near side expands
        float w = 1.0 - u_skew * (t - 0.5);

        // Multiply UV by W so perspective division recovers correct UVs
        v_uv = v_uv * w;

        gl_Position = vec4(in_pos.x, in_pos.y, 0.0, w);
    }
}
