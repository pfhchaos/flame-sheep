"""GpuContext + Viewport — framework-side GPU plumbing shared across
visualizations.

GpuContext wraps the moderngl context, canvas dimensions, physical pixel
scale, shader compilation, resource allocation, the shared fullscreen-
quad mesh, framework render helpers (test pattern, debug overlay), and
the framebuffer→PNG snapshot pipeline. A visualizer composes with a
GpuContext rather than inheriting from one; viz code owns its own
dispatch logic and reaches into GpuContext only for the shared
infrastructure.

Eventual destination is the viz_authoring sibling package; today this
lives in flame_sheep/ because that's where the consumers are. The
public API (method names + signatures) is designed to survive a
Vulkan port — the underlying moderngl handle is exposed as ``self.ctx``
today as a migration shim, will go private once everything that needs
GL primitives goes through GpuContext methods.
"""
from __future__ import annotations

import ctypes
import os
import time as _time
from pathlib import Path

import moderngl
import numpy as np

GPU_TIMING_ENABLED = os.environ.get('FLAME_SHEEP_GPU_TIMING') == '1'


SHADER_DIR = Path(__file__).parent / 'shaders'


class GpuRingTimer:
    """Synchronous GPU-time measurement via glFinish + wall-clock.

    Gated on FLAME_SHEEP_GPU_TIMING=1. When the env var is unset (the
    default), this is a no-op context manager — call sites don't need
    to branch. Enabling it makes every wrapped dispatch serialize on
    glFinish, inflating frame time significantly; only use for
    diagnosing per-stage GPU cost.

    Why glFinish + wall-clock: GL_TIME_ELAPSED queries are unreliable
    on Mesa Xe / Intel Arc — most reads return 0 with no error. The
    ring-buffered async approach this class is named after didn't
    survive contact with the driver.
    """
    __slots__ = ('_ctx', 'last_ns', '_t')

    def __init__(self, ctx: moderngl.Context, logger=None):
        # None when timing is disabled — __enter__/__exit__ short-circuit.
        self._ctx = ctx if GPU_TIMING_ENABLED else None
        self.last_ns = 0
        self._t = 0.0

    def __enter__(self):
        if self._ctx is None:
            return self
        # Wait for any prior GPU work to complete so we time only the work
        # inside this block, not work that was already in flight.
        self._ctx.finish()
        self._t = _time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self._ctx is None:
            return
        # Wait for the GPU work in this block to complete before timing.
        self._ctx.finish()
        self.last_ns = int((_time.perf_counter() - self._t) * 1e9)


# glBindFramebuffer(GL_FRAMEBUFFER, 0) to return to the windowing system's
# default framebuffer. PyOpenGL preferred; ctypes fallback for environments
# where it isn't available.
try:
    from OpenGL import GL as _GL
    def _bind_default_framebuffer() -> None:
        _GL.glBindFramebuffer(_GL.GL_FRAMEBUFFER, 0)
except ImportError:
    _libGL = ctypes.CDLL('libGL.so.1')
    def _bind_default_framebuffer() -> None:
        _libGL.glBindFramebuffer(0x8D40, 0)  # GL_FRAMEBUFFER = 0x8D40


def _resolve_includes(source: str, shader_dir: Path) -> str:
    """Resolve #include "file.glsl" directives by inlining file contents."""
    import re
    def _replace(m: re.Match[str]) -> str:
        path = shader_dir / m.group(1)
        return path.read_text()
    return re.sub(r'#include\s+"(.+?)"', _replace, source)


class Viewport:
    """Describes one window's rectangle within the virtual canvas (render coords)."""
    __slots__ = ('x', 'y', 'w', 'h')

    def __init__(self, x: int, y: int, w: int, h: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __repr__(self) -> str:
        return f'Viewport(x={self.x}, y={self.y}, w={self.w}, h={self.h})'


class GpuContext:
    """Shared GPU plumbing for visualizers — moderngl context, canvas
    dimensions, physical scale, shader compilation, shared quad mesh,
    resource allocation, framework render helpers, snapshot encoding.

    A visualizer composes with this rather than inheriting — ``viz.gpu``
    is the framework handle, viz code owns its own dispatch logic.

    Backend-agnostic surface: today wraps moderngl, future versions may
    wrap Vulkan. The underlying ctx is _ctx (private) so consumer code
    goes through public methods that survive a backend change. ``self.ctx``
    is currently exposed as a migration shim — moves to private once all
    resource allocation has been factored through methods.
    """

    def __init__(self, ctx: moderngl.Context, canvas_w: int, canvas_h: int,
                 ppmm: float = 1.0) -> None:
        self._ctx = ctx
        self.ctx = ctx  # migration shim — same object, will go private later
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.ppmm = ppmm

        # Shared fullscreen-quad mesh: NX×NY grid of triangles, not a single
        # rectangle. Per-triangle bilinear UV interpolation is bounded by the
        # cell size; a single huge triangle would accumulate float error toward
        # the corners. Used by any program drawing a fullscreen pass (tonemap,
        # blur, test pattern, post-processing). Allocated once at GpuContext
        # construction so all consumers share the same VBO.
        NX, NY = 8, 2  # 8 columns × 2 rows = 32 triangles
        verts = []
        for iy in range(NY):
            y0 = -1.0 + 2.0 * iy / NY
            y1 = -1.0 + 2.0 * (iy + 1) / NY
            for ix in range(NX):
                x0 = -1.0 + 2.0 * ix / NX
                x1 = -1.0 + 2.0 * (ix + 1) / NX
                verts.extend([x0, y0, x1, y0, x0, y1])
                verts.extend([x1, y0, x1, y1, x0, y1])
        self.quad_vbo = self._ctx.buffer(
            np.array(verts, dtype=np.float32).tobytes())

        # Lazy slots for framework render helpers (compiled on first use —
        # most callers never touch test pattern or debug overlays).
        self._test_pattern_program = None
        self._test_pattern_vao = None
        self._debug_circle_program = None
        self._debug_circle_vao = None

    def make_quad_vao(self, program: moderngl.Program,
                      in_attr: str = 'in_pos') -> moderngl.VertexArray:
        """Bind the shared fullscreen-quad VBO to `program`'s vertex
        attribute named `in_attr` (default ``in_pos``). One VAO per
        program — the binding is shader-specific, even though the data
        isn't."""
        return self._ctx.vertex_array(program, [(self.quad_vbo, '2f', in_attr)])

    # --- Resource allocation -----------------------------------------------

    def allocate_storage_buffer(self, size_bytes: int,
                                initial: bytes | None = None
                                ) -> moderngl.Buffer:
        """Allocate an SSBO of size_bytes. ``initial`` (if given) is
        uploaded; otherwise the buffer is uninitialized (caller responsible
        for writing before first read).

        moderngl's ctx.buffer needs SOMETHING to size from — there's no
        "allocate empty of size N" API. We zero-fill in the uninitialized
        path to keep callers from seeing garbage."""
        if initial is None:
            initial = bytes(size_bytes)
        elif len(initial) != size_bytes:
            raise ValueError(
                f'initial bytes ({len(initial)}) != requested size ({size_bytes})')
        return self._ctx.buffer(initial)

    def allocate_texture(self, size: tuple[int, int], components: int,
                         dtype: str, initial: bytes | None = None,
                         filter: tuple | None = None
                         ) -> moderngl.Texture:
        """Allocate a 2D texture. dtype matches moderngl's spec ('f4', 'f2',
        'f1', 'u1', etc.). ``filter`` is an optional (min, mag) filter tuple
        — defaults to whatever moderngl picks (nearest)."""
        tex = self._ctx.texture(size, components=components, dtype=dtype,
                                 data=initial)
        if filter is not None:
            tex.filter = filter
        return tex

    # --- Surface state -----------------------------------------------------

    def bind_default_framebuffer(self) -> None:
        """Bind the windowing system's default framebuffer for direct screen
        rendering. Equivalent to glBindFramebuffer(GL_FRAMEBUFFER, 0)."""
        _bind_default_framebuffer()

    # --- Framework render helpers ------------------------------------------

    def render_test_pattern(self, viewport: Viewport,
                             surface_w: int, surface_h: int) -> None:
        """Render an alignment grid for multi-monitor calibration. Used to
        debug the framework's own viewport math — viz authors don't have
        to think about it. Programs + VAO compiled lazily on first call."""
        if self._test_pattern_program is None:
            self._test_pattern_program = self.compile_program(
                SHADER_DIR / 'tonemap.vert',
                SHADER_DIR / 'test_pattern.frag')
            self._test_pattern_vao = self.make_quad_vao(self._test_pattern_program)

        self.bind_default_framebuffer()
        self._ctx.viewport = (0, 0, surface_w, surface_h)

        p = self._test_pattern_program
        p['u_width']      = self.canvas_w
        p['u_height']     = self.canvas_h
        p['u_viewport_x'] = viewport.x
        p['u_viewport_y'] = viewport.y
        p['u_viewport_w'] = viewport.w
        p['u_viewport_h'] = viewport.h
        p['u_surface_w']  = surface_w
        p['u_surface_h']  = surface_h
        if 'u_ppmm' in p:
            p['u_ppmm']   = self.ppmm

        self._test_pattern_vao.render(moderngl.TRIANGLES)

    def draw_debug_circle(self, ndc_x: float, ndc_y: float,
                           surface_w: int, surface_h: int,
                           radius_px: float = 30.0,
                           color: tuple[float, float, float] = (1.0, 0.0, 0.0)
                           ) -> None:
        """Draw a debug circle overlay at NDC coords (-1..1) on the
        currently-bound framebuffer. Inline shader (no .frag file —
        debug-only convenience). Lazy compile on first call."""
        if self._debug_circle_program is None:
            self._debug_circle_program = self._ctx.program(
                vertex_shader="""
                #version 430
                in vec2 in_pos;
                void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
                """,
                fragment_shader="""
                #version 430
                uniform vec2 u_center;
                uniform float u_radius;
                uniform vec2 u_resolution;
                uniform vec3 u_color;
                out vec4 fragColor;
                void main() {
                    vec2 pixel = gl_FragCoord.xy;
                    vec2 center_px = (u_center * 0.5 + 0.5) * u_resolution;
                    float dist = length(pixel - center_px);
                    float ring = smoothstep(u_radius - 2.0, u_radius - 1.0, dist)
                               * (1.0 - smoothstep(u_radius + 1.0, u_radius + 2.0, dist));
                    if (ring < 0.01) discard;
                    fragColor = vec4(u_color * ring, ring);
                }
                """,
            )
            self._debug_circle_vao = self.make_quad_vao(self._debug_circle_program)

        self._ctx.enable(moderngl.BLEND)
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        p = self._debug_circle_program
        p['u_center'] = (ndc_x, ndc_y)
        p['u_radius'] = radius_px
        p['u_resolution'] = (float(surface_w), float(surface_h))
        p['u_color'] = color
        self._debug_circle_vao.render(moderngl.TRIANGLES)
        self._ctx.disable(moderngl.BLEND)

    # --- Snapshot helpers --------------------------------------------------

    def framebuffer_to_png_bytes(self, fbo: moderngl.Framebuffer,
                                  width: int, height: int) -> bytes:
        """Read an FBO's color attachment, flip Y for image conventions,
        encode as PNG. Doesn't release the FBO — caller manages lifetime
        of the resources it created."""
        import io
        from PIL import Image
        data = fbo.read(components=4)
        img = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
        img = img[::-1].copy()  # OpenGL origin is bottom-left
        buf = io.BytesIO()
        Image.fromarray(img, 'RGBA').save(buf, format='PNG', optimize=True)
        return buf.getvalue()

    # --- Shader compilation -------------------------------------------------

    def compile_compute_shader(self, path: Path,
                               defines: dict[str, str] | None = None,
                               source_transform=None
                               ) -> moderngl.ComputeShader:
        """Load + #include-resolve + (optionally) #define-inject +
        (optionally) apply source_transform + compile a compute shader.

        Returns a moderngl.ComputeShader today; future Vulkan port will
        return a backend-agnostic wrapper of the same shape.

        defines values get rendered as ``#define KEY VALUE`` and inserted
        right after the first newline (which puts them after #version).
        For bare ``#define KEY`` form, use value=``''``.

        source_transform runs BEFORE define injection and is for shader-
        specific source manipulation (e.g. flame.comp's symmetry-group
        GLSL substitution).
        """
        src = path.read_text()
        src = _resolve_includes(src, path.parent)
        if source_transform is not None:
            src = source_transform(src)
        if defines:
            block = '\n'.join(
                f'#define {k}' if v == '' else f'#define {k} {v}'
                for k, v in defines.items())
            src = src.replace('\n', '\n' + block + '\n', 1)
        return self._ctx.compute_shader(src)

    def compile_program(self, vert_path: Path,
                        frag_path: Path) -> moderngl.Program:
        """Load + #include-resolve both shader files + compile a vert/frag
        program. Sources are read fresh per call — no caching today; if
        repeated reads of tonemap.vert show up as a hotspot we can add it."""
        vs = _resolve_includes(vert_path.read_text(), vert_path.parent)
        fs = _resolve_includes(frag_path.read_text(), frag_path.parent)
        return self._ctx.program(vertex_shader=vs, fragment_shader=fs)
