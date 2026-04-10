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

void main() {
    // Pass position straight through to rasterizer.
    // z=0 (flat), w=1 (no perspective divide needed).
    gl_Position = vec4(in_pos, 0.0, 1.0);

    // Convert clip-space (-1..1) to UV (0..1)
    // Same as: uv = (pos + 1) / 2
    v_uv = in_pos * 0.5 + 0.5;
}
