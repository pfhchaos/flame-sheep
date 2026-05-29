"""SnapshotPipeline — render the current chaos-game histogram to a PNG.

Used by the offline render workers (rendering candidate genomes for
scoring + display) and by ad-hoc save-to-file paths in tools/. Does
NOT touch the live wallpaper render path — that goes through
TonemapPipeline directly to the swap-chain surface.

Three variants:
  - snapshot_png: standard tonemap (log-density + gamma), to PNG
  - snapshot_de_png: apply density-estimation blur to histogram first,
    then standard tonemap (linear mode since DE already log-scaled)
  - snapshot_flam3_png: tonemap with flam3-accurate k1/k2 brightness +
    pow(density, 1/gamma) curve, matching scottdraves's rect.c math

Also owns apply_density_estimation, since DE is a snapshot-only
operation today (live renders skip it for speed).

Sub-component of FlameRenderer. Holds a back-reference (`self._r`) for
shared resource access; doesn't own buffers or shaders itself.

Stage 0 of `docs/reorg_plan.md`. Unit description: "renders the
current histogram state to a PNG via various tonemap recipes (standard,
density-estimated, flam3-accurate)."
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import moderngl

from .gpu_context import Viewport

if TYPE_CHECKING:
    from .renderer import FlameRenderer


class SnapshotPipeline:
    """PNG-snapshot rendering, backed by a FlameRenderer's shared
    resources (palette_tex, tonemap_program, tonemap_flam3_program,
    de_shader, _de_in_buf, _de_out_buf, histogram_buf, quad_vao,
    framebuffer_to_png_bytes helper)."""

    def __init__(self, renderer: 'FlameRenderer') -> None:
        self._r = renderer

    def apply_density_estimation(self, max_radius: int = 9,
                                  curve: float = 0.4,
                                  min_density: float = 0.0) -> None:
        """Apply flam3-style density estimation to the histogram.

        Spatially-varying gaussian blur: sparse pixels get wide kernels,
        dense pixels stay sharp. Reveals filament structure that would
        otherwise be invisible single-pixel dots.

        Modifies the histogram in-place (copies to DE buffers, runs shader,
        copies result back).

        Args:
            max_radius: maximum blur kernel radius (flam3 estimator_radius)
            curve: density falloff curve (flam3 estimator_curve, typically 0.4-0.6)
            min_density: minimum hits to consider (below = noise)
        """
        r = self._r
        w, h = r.canvas_w, r.canvas_h

        # Copy current histogram → DE input buffer
        data = r.histogram_buf.read()
        r._de_in_buf.write(data)

        # Zero the output buffer (scatter accumulates from zero)
        r._de_out_buf.write(b'\x00' * len(data))

        # Run DE scatter shader
        de = r.de_shader
        de['u_width'] = w
        de['u_height'] = h
        de['u_max_radius'] = max_radius
        de['u_curve'] = float(curve)

        groups_x = (w + 15) // 16
        groups_y = (h + 15) // 16
        de.run(group_x=groups_x, group_y=groups_y)
        r.ctx.memory_barrier()

        # Copy DE output → main histogram buffer
        # FP_SCALE is baked into the values but the tonemap normalizes by
        # max_hits anyway, so the scale cancels out — just copy directly.
        result = r._de_out_buf.read()
        r.histogram_buf.write(result)

    def png(self, brightness: float = 6.0,
            linear_mode: bool = False) -> bytes:
        """Tonemap the current histogram to an RGBA PNG and return as bytes.

        Uses a temporary FBO — does not affect screen output.

        Args:
            brightness: gamma parameter for tone mapping
            linear_mode: if True, skip log-density (for post-DE histograms)
        """
        r = self._r
        ctx = r.ctx
        w, h = r.canvas_w, r.canvas_h
        fbo_tex = ctx.texture((w, h), components=4, dtype='f1')
        fbo = ctx.framebuffer(color_attachments=[fbo_tex])
        fbo.use()
        ctx.viewport = (0, 0, w, h)

        # Compute max hit count for tonemap normalization
        r.chaos.reduce_histogram_max()

        viewport = Viewport(0, 0, w, h)
        r.palette_tex.use(location=0)

        p = r.tonemap_program
        p['u_palette']       = 0
        p['u_width']         = w
        p['u_height']        = h
        p['u_viewport_x']    = 0
        p['u_viewport_y']    = 0
        p['u_viewport_w']    = w
        p['u_viewport_h']    = h
        p['u_surface_w']     = w
        p['u_surface_h']     = h
        p['u_gamma']         = brightness
        if 'u_linear_mode' in p:
            p['u_linear_mode'] = 1 if linear_mode else 0

        r.quad_vao.render(moderngl.TRIANGLES)

        png_bytes = r.gpu.framebuffer_to_png_bytes(fbo, w, h)
        fbo.release()
        fbo_tex.release()
        return png_bytes

    def de_png(self, brightness: float = 6.0,
               max_radius: int = 9, curve: float = 0.4) -> bytes:
        """Render with density estimation, then tonemap to PNG.

        Applies spatially-varying DE blur to the histogram before
        tonemapping. Uses linear mode in tonemap since DE scatter
        already applies log-density scaling.
        """
        self.apply_density_estimation(max_radius=max_radius, curve=curve)
        return self.png(brightness=brightness)

    def flam3_png(self, brightness: float = 4.0, gamma: float = 4.0,
                  vibrancy: float = 1.0, contrast: float = 1.0,
                  sample_density: float = 1.0,
                  highlight_power: float = -1.0) -> bytes:
        """Tonemap using flam3-accurate pipeline and return as PNG bytes.

        Uses absolute brightness scaling (k1/k2) and pow(density, 1/gamma)
        gamma correction, matching flam3's rendering math.

        Args:
            brightness: flam3 brightness parameter (typically 2-100)
            gamma: flam3 gamma parameter (typically 1-5)
            vibrancy: color saturation blend (0=grayscale, 1=full color)
            contrast: flam3 contrast parameter (typically 1.0)
            sample_density: total samples per pixel (for k2 normalization)
            highlight_power: hue preservation power (-1 = disabled)
        """
        r = self._r
        ctx = r.ctx
        w, h = r.canvas_w, r.canvas_h

        # Compute k1 and k2 matching flam3's rect.c
        # k1 = contrast * brightness * PREFILTER_WHITE * 268 / 256
        # k2 = oversample² * nbatches / (contrast * area * WHITE_LEVEL * sample_density * sumfilt)
        #
        # area = image_w * image_h / (ppux * ppuy) — world-space area of viewport
        # For us: viewport spans (-1/zoom, 1/zoom) in each axis, so:
        #   world_area = (2/zoom)² = 4/zoom²
        # ppux = ppuy = zoom * w/2, so area = w*h / (zoom*w/2)² = 4/zoom²
        # oversample=1, nbatches=1, sumfilt=1 for our simple case
        k1 = contrast * brightness * 255.0 * 268.0 / 256.0
        area = 4.0 / max(r._last_zoom ** 2, 1e-10) if hasattr(r, '_last_zoom') else 4.0
        k2 = 1.0 / max(contrast * area * 255.0 * max(sample_density, 0.01), 1e-10)

        fbo_tex = ctx.texture((w, h), components=4, dtype='f1')
        fbo = ctx.framebuffer(color_attachments=[fbo_tex])
        fbo.use()
        ctx.viewport = (0, 0, w, h)

        r.palette_tex.use(location=0)

        p = r.tonemap_flam3_program
        p['u_palette']       = 0
        p['u_width']         = w
        p['u_height']        = h
        p['u_viewport_x']    = 0
        p['u_viewport_y']    = 0
        p['u_viewport_w']    = w
        p['u_viewport_h']    = h
        p['u_surface_w']     = w
        p['u_surface_h']     = h
        p['u_k1']            = float(k1)
        p['u_k2']            = float(k2)
        p['u_gamma']         = 1.0 / max(gamma, 0.01)  # pre-invert for shader
        p['u_vibrancy']      = float(vibrancy)
        # These may be optimized out by the compiler
        if 'u_lin_thresh' in p:
            p['u_lin_thresh']    = 0.01
        if 'u_highlight_power' in p:
            p['u_highlight_power'] = float(highlight_power)

        r.quad_vao.render(moderngl.TRIANGLES)

        png_bytes = r.gpu.framebuffer_to_png_bytes(fbo, w, h)
        fbo.release()
        fbo_tex.release()
        return png_bytes
