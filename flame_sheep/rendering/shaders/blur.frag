#version 430 core

// ============================================================
// blur.frag — single-axis gaussian blur (separable)
// ============================================================
// Called twice per frame: once horizontal, once vertical.
// The direction is set by u_direction uniform.
//
// Uses a 9-tap gaussian kernel — light blur suitable for
// softening fractal detail behind transparent terminals.
// ============================================================

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_texture;
uniform vec2 u_direction;  // (1/w, 0) for horizontal, (0, 1/h) for vertical
uniform float u_radius;    // blur strength multiplier

void main() {
    // 9-tap gaussian weights (sigma ~= 2.0, normalized)
    const float weights[5] = float[](
        0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216
    );

    vec3 result = texture(u_texture, v_uv).rgb * weights[0];

    for (int i = 1; i < 5; i++) {
        vec2 offset = u_direction * float(i) * u_radius;
        result += texture(u_texture, v_uv + offset).rgb * weights[i];
        result += texture(u_texture, v_uv - offset).rgb * weights[i];
    }

    frag_color = vec4(result, 1.0);
}
