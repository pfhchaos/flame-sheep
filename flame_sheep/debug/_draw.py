"""Batched 2D primitive renderer — filled rectangles and lines.

All geometry is collected via rect()/line() calls, then flushed in
one draw call per frame. Coordinates are in pixels, origin top-left.
"""

from __future__ import annotations

import numpy as np
import moderngl


# Inline shaders — no external files needed
_VERT = """
#version 330
in vec2 in_pos;
in vec4 in_color;
out vec4 v_color;
uniform vec2 u_resolution;

void main() {
    // Convert pixel coords (origin top-left) to clip space
    vec2 ndc = (in_pos / u_resolution) * 2.0 - 1.0;
    ndc.y = -ndc.y;  // flip Y
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_color = in_color;
}
"""

_FRAG = """
#version 330
in vec4 v_color;
out vec4 fragColor;

void main() {
    fragColor = v_color;
}
"""


class SolidRenderer:
    """Batched renderer for colored rectangles and lines.

    Usage each frame:
        draw.begin()
        draw.rect(x, y, w, h, color)
        draw.line(x1, y1, x2, y2, color, thickness)
        draw.flush(width, height)
    """

    def __init__(self, ctx: moderngl.Context) -> None:
        self._ctx = ctx
        self._prog = ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG)
        # Dynamic VBO — resized as needed
        self._vbo = ctx.buffer(reserve=1024 * 6 * 6 * 4)  # ~1K quads initial
        self._vao = ctx.vertex_array(
            self._prog,
            [(self._vbo, '2f 4f', 'in_pos', 'in_color')],
        )
        self._vertices: list[float] = []

    def begin(self) -> None:
        """Start a new frame batch."""
        self._vertices.clear()

    def rect(self, x: float, y: float, w: float, h: float,
             color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)) -> None:
        """Add a filled rectangle (two triangles)."""
        r, g, b, a = color
        v = self._vertices
        # Triangle 1
        v.extend([x, y, r, g, b, a])
        v.extend([x + w, y, r, g, b, a])
        v.extend([x + w, y + h, r, g, b, a])
        # Triangle 2
        v.extend([x, y, r, g, b, a])
        v.extend([x + w, y + h, r, g, b, a])
        v.extend([x, y + h, r, g, b, a])

    def line(self, x1: float, y1: float, x2: float, y2: float,
             color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
             thickness: float = 1.0) -> None:
        """Add a line as a thin rectangle."""
        dx = x2 - x1
        dy = y2 - y1
        length = (dx * dx + dy * dy) ** 0.5
        if length < 0.001:
            return
        # Perpendicular unit vector
        nx = -dy / length * thickness * 0.5
        ny = dx / length * thickness * 0.5
        r, g, b, a = color
        v = self._vertices
        v.extend([x1 + nx, y1 + ny, r, g, b, a])
        v.extend([x1 - nx, y1 - ny, r, g, b, a])
        v.extend([x2 - nx, y2 - ny, r, g, b, a])
        v.extend([x1 + nx, y1 + ny, r, g, b, a])
        v.extend([x2 - nx, y2 - ny, r, g, b, a])
        v.extend([x2 + nx, y2 + ny, r, g, b, a])

    def flush(self, width: int, height: int) -> None:
        """Upload batch and draw."""
        if not self._vertices:
            return
        data = np.array(self._vertices, dtype=np.float32)
        byte_size = data.nbytes
        if byte_size > self._vbo.size:
            self._vbo.orphan(byte_size * 2)
        self._vbo.write(data.tobytes())
        self._prog['u_resolution'].value = (float(width), float(height))
        n_verts = len(self._vertices) // 6  # 6 floats per vertex
        self._vao.render(moderngl.TRIANGLES, vertices=n_verts)
