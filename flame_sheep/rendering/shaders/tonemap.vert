#version 430 core

// ============================================================
// tonemap.vert — fullscreen quad with passthrough UV mapping
// ============================================================
// The quad always fills the screen (no clipping, no black corners).
// ============================================================

in vec2 in_pos;
out vec2 v_uv;

void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_pos * 0.5 + 0.5;
}
