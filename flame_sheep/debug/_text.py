"""Bitmap font text renderer — embedded 6x10 pixel font, no dependencies.

Renders ASCII text as batched textured quads. The font atlas is generated
at init time from a minimal built-in pixel font.
"""

from __future__ import annotations

import numpy as np
import moderngl


_VERT = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
in vec4 in_color;
out vec2 v_uv;
out vec4 v_color;
uniform vec2 u_resolution;

void main() {
    vec2 ndc = (in_pos / u_resolution) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = in_uv;
    v_color = in_color;
}
"""

_FRAG = """
#version 330
in vec2 v_uv;
in vec4 v_color;
out vec4 fragColor;
uniform sampler2D u_font;

void main() {
    float alpha = texture(u_font, v_uv).r;
    if (alpha < 0.5) discard;
    fragColor = v_color;
}
"""

# Glyph dimensions
GLYPH_W = 6
GLYPH_H = 10
_COLS = 16  # glyphs per row in atlas
_ROWS = 6  # rows in atlas (chars 32..127)


def _build_font_atlas() -> np.ndarray:
    """Build a minimal 6x10 bitmap font atlas covering ASCII 32-127.

    Returns an (rows*10, cols*6) uint8 array suitable for a GL texture.
    Each pixel is 0 or 255.
    """
    # Minimal 6x10 glyphs for printable ASCII (32-127)
    # Each glyph is 10 rows of 6-bit patterns (MSB = leftmost pixel)
    _GLYPHS: dict[int, list[int]] = {
        32: [0]*10,  # space
        33: [0b001000, 0b001000, 0b001000, 0b001000, 0b001000, 0b001000, 0b000000, 0b001000, 0b000000, 0],  # !
        35: [0b010100, 0b010100, 0b111110, 0b010100, 0b111110, 0b010100, 0b010100, 0b000000, 0, 0],  # #
        40: [0b000100, 0b001000, 0b010000, 0b010000, 0b010000, 0b010000, 0b001000, 0b000100, 0, 0],  # (
        41: [0b010000, 0b001000, 0b000100, 0b000100, 0b000100, 0b000100, 0b001000, 0b010000, 0, 0],  # )
        43: [0, 0, 0b001000, 0b001000, 0b111110, 0b001000, 0b001000, 0, 0, 0],  # +
        44: [0]*6 + [0b001000, 0b010000, 0, 0],  # ,
        45: [0]*4 + [0b111110] + [0]*5,  # -
        46: [0]*7 + [0b001000, 0, 0],  # .
        47: [0b000010, 0b000100, 0b000100, 0b001000, 0b001000, 0b010000, 0b010000, 0b100000, 0, 0],  # /
        48: [0b011100, 0b100010, 0b100110, 0b101010, 0b110010, 0b100010, 0b011100, 0, 0, 0],  # 0
        49: [0b001000, 0b011000, 0b001000, 0b001000, 0b001000, 0b001000, 0b011100, 0, 0, 0],  # 1
        50: [0b011100, 0b100010, 0b000010, 0b000100, 0b001000, 0b010000, 0b111110, 0, 0, 0],  # 2
        51: [0b011100, 0b100010, 0b000010, 0b001100, 0b000010, 0b100010, 0b011100, 0, 0, 0],  # 3
        52: [0b000100, 0b001100, 0b010100, 0b100100, 0b111110, 0b000100, 0b000100, 0, 0, 0],  # 4
        53: [0b111110, 0b100000, 0b111100, 0b000010, 0b000010, 0b100010, 0b011100, 0, 0, 0],  # 5
        54: [0b011100, 0b100000, 0b111100, 0b100010, 0b100010, 0b100010, 0b011100, 0, 0, 0],  # 6
        55: [0b111110, 0b000010, 0b000100, 0b001000, 0b010000, 0b010000, 0b010000, 0, 0, 0],  # 7
        56: [0b011100, 0b100010, 0b100010, 0b011100, 0b100010, 0b100010, 0b011100, 0, 0, 0],  # 8
        57: [0b011100, 0b100010, 0b100010, 0b011110, 0b000010, 0b000100, 0b011000, 0, 0, 0],  # 9
        58: [0, 0, 0b001000, 0, 0, 0b001000, 0, 0, 0, 0],  # :
        60: [0b000100, 0b001000, 0b010000, 0b010000, 0b001000, 0b000100, 0, 0, 0, 0],  # <
        61: [0, 0, 0b111110, 0, 0b111110, 0, 0, 0, 0, 0],  # =
        62: [0b010000, 0b001000, 0b000100, 0b000100, 0b001000, 0b010000, 0, 0, 0, 0],  # >
        91: [0b011100, 0b010000, 0b010000, 0b010000, 0b010000, 0b010000, 0b011100, 0, 0, 0],  # [
        93: [0b011100, 0b000100, 0b000100, 0b000100, 0b000100, 0b000100, 0b011100, 0, 0, 0],  # ]
    }
    # Letters A-Z (uppercase)
    _LETTERS = {
        65: [0b011100, 0b100010, 0b100010, 0b111110, 0b100010, 0b100010, 0b100010, 0, 0, 0],
        66: [0b111100, 0b100010, 0b100010, 0b111100, 0b100010, 0b100010, 0b111100, 0, 0, 0],
        67: [0b011100, 0b100010, 0b100000, 0b100000, 0b100000, 0b100010, 0b011100, 0, 0, 0],
        68: [0b111100, 0b100010, 0b100010, 0b100010, 0b100010, 0b100010, 0b111100, 0, 0, 0],
        69: [0b111110, 0b100000, 0b100000, 0b111100, 0b100000, 0b100000, 0b111110, 0, 0, 0],
        70: [0b111110, 0b100000, 0b100000, 0b111100, 0b100000, 0b100000, 0b100000, 0, 0, 0],
        71: [0b011100, 0b100010, 0b100000, 0b100110, 0b100010, 0b100010, 0b011110, 0, 0, 0],
        72: [0b100010, 0b100010, 0b100010, 0b111110, 0b100010, 0b100010, 0b100010, 0, 0, 0],
        73: [0b011100, 0b001000, 0b001000, 0b001000, 0b001000, 0b001000, 0b011100, 0, 0, 0],
        74: [0b000010, 0b000010, 0b000010, 0b000010, 0b000010, 0b100010, 0b011100, 0, 0, 0],
        75: [0b100010, 0b100100, 0b101000, 0b110000, 0b101000, 0b100100, 0b100010, 0, 0, 0],
        76: [0b100000, 0b100000, 0b100000, 0b100000, 0b100000, 0b100000, 0b111110, 0, 0, 0],
        77: [0b100010, 0b110110, 0b101010, 0b101010, 0b100010, 0b100010, 0b100010, 0, 0, 0],
        78: [0b100010, 0b110010, 0b101010, 0b100110, 0b100010, 0b100010, 0b100010, 0, 0, 0],
        79: [0b011100, 0b100010, 0b100010, 0b100010, 0b100010, 0b100010, 0b011100, 0, 0, 0],
        80: [0b111100, 0b100010, 0b100010, 0b111100, 0b100000, 0b100000, 0b100000, 0, 0, 0],
        81: [0b011100, 0b100010, 0b100010, 0b100010, 0b101010, 0b100100, 0b011010, 0, 0, 0],
        82: [0b111100, 0b100010, 0b100010, 0b111100, 0b101000, 0b100100, 0b100010, 0, 0, 0],
        83: [0b011100, 0b100010, 0b100000, 0b011100, 0b000010, 0b100010, 0b011100, 0, 0, 0],
        84: [0b111110, 0b001000, 0b001000, 0b001000, 0b001000, 0b001000, 0b001000, 0, 0, 0],
        85: [0b100010, 0b100010, 0b100010, 0b100010, 0b100010, 0b100010, 0b011100, 0, 0, 0],
        86: [0b100010, 0b100010, 0b100010, 0b100010, 0b010100, 0b010100, 0b001000, 0, 0, 0],
        87: [0b100010, 0b100010, 0b100010, 0b101010, 0b101010, 0b110110, 0b100010, 0, 0, 0],
        88: [0b100010, 0b100010, 0b010100, 0b001000, 0b010100, 0b100010, 0b100010, 0, 0, 0],
        89: [0b100010, 0b100010, 0b010100, 0b001000, 0b001000, 0b001000, 0b001000, 0, 0, 0],
        90: [0b111110, 0b000010, 0b000100, 0b001000, 0b010000, 0b100000, 0b111110, 0, 0, 0],
    }
    # Lowercase a-z (same patterns shifted to 97-122)
    _LOWER = {}
    for code, pattern in _LETTERS.items():
        _LOWER[code + 32] = pattern
    # Override a few lowercase that differ significantly
    _LOWER[97]  = [0, 0, 0b011100, 0b000010, 0b011110, 0b100010, 0b011110, 0, 0, 0]   # a
    _LOWER[98]  = [0b100000, 0b100000, 0b111100, 0b100010, 0b100010, 0b100010, 0b111100, 0, 0, 0]  # b
    _LOWER[99]  = [0, 0, 0b011100, 0b100000, 0b100000, 0b100010, 0b011100, 0, 0, 0]   # c
    _LOWER[100] = [0b000010, 0b000010, 0b011110, 0b100010, 0b100010, 0b100010, 0b011110, 0, 0, 0]  # d
    _LOWER[101] = [0, 0, 0b011100, 0b100010, 0b111110, 0b100000, 0b011100, 0, 0, 0]   # e
    _LOWER[103] = [0, 0, 0b011110, 0b100010, 0b100010, 0b011110, 0b000010, 0b011100, 0, 0]  # g
    _LOWER[105] = [0b001000, 0, 0b011000, 0b001000, 0b001000, 0b001000, 0b011100, 0, 0, 0]  # i
    _LOWER[110] = [0, 0, 0b111100, 0b100010, 0b100010, 0b100010, 0b100010, 0, 0, 0]   # n
    _LOWER[111] = [0, 0, 0b011100, 0b100010, 0b100010, 0b100010, 0b011100, 0, 0, 0]   # o
    _LOWER[112] = [0, 0, 0b111100, 0b100010, 0b100010, 0b111100, 0b100000, 0b100000, 0, 0]  # p
    _LOWER[114] = [0, 0, 0b101100, 0b110010, 0b100000, 0b100000, 0b100000, 0, 0, 0]   # r
    _LOWER[115] = [0, 0, 0b011110, 0b100000, 0b011100, 0b000010, 0b111100, 0, 0, 0]   # s
    _LOWER[116] = [0b001000, 0b001000, 0b011100, 0b001000, 0b001000, 0b001000, 0b000110, 0, 0, 0]  # t
    _LOWER[117] = [0, 0, 0b100010, 0b100010, 0b100010, 0b100110, 0b011010, 0, 0, 0]   # u

    _GLYPHS.update(_LETTERS)
    _GLYPHS.update(_LOWER)

    atlas_w = _COLS * GLYPH_W
    atlas_h = _ROWS * GLYPH_H
    atlas = np.zeros((atlas_h, atlas_w), dtype=np.uint8)

    for code in range(32, 128):
        pattern = _GLYPHS.get(code, [0] * 10)
        col = (code - 32) % _COLS
        row = (code - 32) // _COLS
        for gy, bits in enumerate(pattern):
            for gx in range(GLYPH_W):
                if bits & (1 << (GLYPH_W - 1 - gx)):
                    atlas[row * GLYPH_H + gy, col * GLYPH_W + gx] = 255

    return atlas


class TextRenderer:
    """Bitmap font text renderer — batched textured quads.

    Usage each frame:
        text.begin()
        text.draw("Hello", x, y, color, scale)
        text.flush(width, height)
    """

    def __init__(self, ctx: moderngl.Context) -> None:
        self._ctx = ctx
        self._prog = ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG)

        # Build font atlas texture
        atlas = _build_font_atlas()
        self._atlas_h, self._atlas_w = atlas.shape
        self._tex = ctx.texture((self._atlas_w, self._atlas_h), 1, atlas.tobytes())
        self._tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

        # Dynamic VBO for text quads
        self._vbo = ctx.buffer(reserve=4096 * 8 * 4)  # ~500 chars initial
        self._vao = ctx.vertex_array(
            self._prog,
            [(self._vbo, '2f 2f 4f', 'in_pos', 'in_uv', 'in_color')],
        )
        self._vertices: list[float] = []

    def begin(self) -> None:
        """Start a new frame batch."""
        self._vertices.clear()

    def draw(self, text: str, x: float, y: float,
             color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
             scale: int = 1) -> None:
        """Add text to the batch."""
        r, g, b, a = color
        sw = GLYPH_W * scale
        sh = GLYPH_H * scale
        v = self._vertices

        for ch in text:
            code = ord(ch)
            if code < 32 or code > 127:
                code = 32  # fallback to space
            idx = code - 32
            col = idx % _COLS
            row = idx // _COLS

            # UV coordinates in atlas
            u0 = col * GLYPH_W / self._atlas_w
            v0 = row * GLYPH_H / self._atlas_h
            u1 = (col + 1) * GLYPH_W / self._atlas_w
            v1 = (row + 1) * GLYPH_H / self._atlas_h

            # Two triangles per glyph
            v.extend([x, y, u0, v0, r, g, b, a])
            v.extend([x + sw, y, u1, v0, r, g, b, a])
            v.extend([x + sw, y + sh, u1, v1, r, g, b, a])

            v.extend([x, y, u0, v0, r, g, b, a])
            v.extend([x + sw, y + sh, u1, v1, r, g, b, a])
            v.extend([x, y + sh, u0, v1, r, g, b, a])

            x += sw

    def text_width(self, text: str, scale: int = 1) -> float:
        """Return pixel width of text at given scale."""
        return len(text) * GLYPH_W * scale

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
        self._tex.use(0)
        self._prog['u_font'].value = 0
        n_verts = len(self._vertices) // 8  # 8 floats per vertex
        self._vao.render(moderngl.TRIANGLES, vertices=n_verts)
