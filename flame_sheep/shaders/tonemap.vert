#version 430 core

// ============================================================
// tonemap.vert — fullscreen quad with perspective UV mapping
// ============================================================
// The quad always fills the screen (no clipping, no black corners).
// The perspective skew is applied to UV coordinates only:
//   - Far side of angled monitor samples a WIDER vertical slice of canvas
//   - Near side samples a NARROWER slice
//   - Horizontal lines remain horizontal
//
// The canvas must be rendered tall enough to cover the widest slice.
// ============================================================

in vec2 in_pos;
out vec2 v_uv;

uniform float u_skew = 0.0;

void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);

    vec2 uv = in_pos * 0.5 + 0.5;

    if (u_skew != 0.0) {
        // Scale V (vertical UV) based on horizontal position.
        // Far side (higher W in viewing geometry) sees more wall = wider UV range.
        // Near side sees less wall = narrower UV range.
        float t = uv.x;  // 0..1 across screen
        float scale = 1.0 + u_skew * (t - 0.5);
        // Scale UV.y around 0.5 (vertical center)
        uv.y = 0.5 + (uv.y - 0.5) * scale;
    }

    v_uv = uv;
}
