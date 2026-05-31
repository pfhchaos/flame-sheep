#version 430 core
// Blend current frame with previous frame for temporal blur.
// mix_factor: 0.0 = all previous (frozen), 1.0 = all current (no blur)
// comparison_mode: 0 = normal blend, 1 = split screen (4 vertical strips)

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_current;
uniform sampler2D u_previous;
uniform sampler2D u_gauss;       // gaussian-blurred current frame
uniform float u_mix_factor;      // how much of the new frame to show
uniform int u_comparison;        // 0 = normal, 1 = 4-strip comparison

void main() {
    vec4 curr = texture(u_current, v_uv);
    vec4 prev = texture(u_previous, v_uv);

    if (u_comparison == 0) {
        frag_color = mix(prev, curr, u_mix_factor);
        return;
    }

    // 4 vertical strips: raw | gaussian | temporal | both
    vec4 gauss = texture(u_gauss, v_uv);
    vec4 temporal = mix(prev, curr, u_mix_factor);
    // For "both" strip, we need temporal of gaussian. prev already
    // accumulated the blended result, so approximate by mixing prev with gauss.
    vec4 both = mix(prev, gauss, u_mix_factor);

    float x = v_uv.x;
    if      (x < 0.25) frag_color = curr;      // raw
    else if (x < 0.50) frag_color = gauss;     // gaussian only
    else if (x < 0.75) frag_color = temporal;  // temporal only
    else               frag_color = both;      // both
}
