#version 450

// Vulkan port of flame_sheep/rendering/shaders/tonemap.vert.
//
// Fullscreen-quad vertex shader using gl_VertexIndex — no vertex buffer
// needed. Same trick as the triangle demo. Emits NDC positions for two
// triangles forming a fullscreen quad, plus UV [0,1]^2 for the fragment
// shader to use as the histogram lookup coordinate.

layout(location = 0) out vec2 v_uv;

// Two triangles, 6 vertices, covering NDC [-1,1]^2.
//   t0: (-1,-1) (1,-1) (-1,1)
//   t1: (-1, 1) (1,-1) ( 1,1)
const vec2 positions[6] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 1.0, -1.0),
    vec2(-1.0,  1.0),
    vec2(-1.0,  1.0),
    vec2( 1.0, -1.0),
    vec2( 1.0,  1.0)
);

void main() {
    vec2 pos = positions[gl_VertexIndex];
    gl_Position = vec4(pos, 0.0, 1.0);
    // UV: (-1..1) → (0..1). Vulkan's Y is top-down (vs OpenGL bottom-up)
    // so the histogram's pixel-coord mapping needs to flip Y here OR in
    // the fragment lookup. We do it here so the fragment shader's
    // pixel index calc reads naturally as (y * width + x).
    v_uv = vec2((pos.x + 1.0) * 0.5, (1.0 - pos.y) * 0.5);
}
