"""TonemapPipeline — palette-mapped tonemap render of the chaos-game
histogram, with optional gaussian blur and temporal blend.

Reads from the histogram SSBO (binding=0, written by ChaosGame) and
the palette texture. Outputs to a target framebuffer (typically the
swap-chain surface, sometimes an off-screen FBO for compare-mode
split-screen). Maintains per-size caches of intermediate FBOs:

  - `_blur_fbos[(w, h)]`: two FBOs A/B for the two-pass separable
    gaussian blur (horizontal A→B, vertical B→target).
  - `_temporal_fbos[(w, h)]`: three FBOs for temporal blend (accumulator,
    current frame, blend output — rotated each frame).

Sub-component of FlameRenderer. Holds a back-reference (`self._r`) for
shared resource access; doesn't own buffers or shaders itself, just
the per-size FBO caches and the per-frame logic that wires them up.

Stage 0 of `docs/reorg_plan.md`. Unit description: "renders the
chaos-game histogram to a target framebuffer via the tonemap shader,
optionally adding spatial blur and temporal blend."
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import moderngl

from .gpu_context import _bind_default_framebuffer, Viewport

if TYPE_CHECKING:
    from .renderer import FlameRenderer


class TonemapPipeline:
    """Tonemap render + optional blur + temporal blend, backed by a
    FlameRenderer's shared resources (palette_tex, tonemap_program,
    blur_program, _temporal_mix_prog, quad_vao, blur_vao,
    _temporal_mix_vao)."""

    def __init__(self, renderer: 'FlameRenderer') -> None:
        self._r = renderer
        # Per-size FBO caches — instantiated lazily on first request.
        self._blur_fbos: dict[tuple[int, int], tuple] = {}
        self._temporal_fbos: dict[tuple[int, int], list[tuple]] = {}

    def _get_blur_fbos(self, w: int, h: int) -> tuple[
            moderngl.Framebuffer, moderngl.Texture,
            moderngl.Framebuffer, moderngl.Texture]:
        """Get or create a pair of FBOs for two-pass blur at the given size."""
        ctx = self._r.ctx
        key = (w, h)
        if key not in self._blur_fbos:
            tex_a = ctx.texture((w, h), components=4, dtype='f2')
            tex_a.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex_a.repeat_x = False
            tex_a.repeat_y = False
            fbo_a = ctx.framebuffer(color_attachments=[tex_a])

            tex_b = ctx.texture((w, h), components=4, dtype='f2')
            tex_b.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex_b.repeat_x = False
            tex_b.repeat_y = False
            fbo_b = ctx.framebuffer(color_attachments=[tex_b])

            self._blur_fbos[key] = (fbo_a, tex_a, fbo_b, tex_b)
        return self._blur_fbos[key]

    def _get_temporal_fbos(self, w: int, h: int):
        """Get or create FBOs for temporal blending at given size.
        Returns three (fbo, tex) tuples: current frame, accumulated
        result, blend output."""
        ctx = self._r.ctx
        key = (w, h)
        if key not in self._temporal_fbos:
            fbos = []
            for _ in range(3):
                tex = ctx.texture((w, h), components=4, dtype='f2')
                tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                fbo = ctx.framebuffer(color_attachments=[tex])
                fbo.use()
                ctx.clear(0.0, 0.0, 0.0, 1.0)
                fbos.append((fbo, tex))
            self._temporal_fbos[key] = fbos
        return self._temporal_fbos[key]

    def render(self, viewport: Viewport, surface_w: int, surface_h: int,
               brightness: float = 6.0, dt: float = 1/60,
               screen_rect: tuple[int, int, int, int] | None = None,
               target_fbo: 'moderngl.Framebuffer | None' = None) -> None:
        """Render the tonemap pass for one window's viewport slice,
        optionally with gaussian blur and temporal blend.

        viewport    — the slice of the canvas this surface displays (in canvas coords)
        surface_w/h — the actual EGL surface size (physical pixels)
        brightness  — tone mapping brightness (driven by audio RMS)
        dt          — frame time in seconds (for frame-rate-independent temporal blend)

        The caller must have already made the target window's EGL surface
        current before calling this. After this returns, call win.swap().

        screen_rect — optional (x, y, w, h) GL viewport override for split-screen.
                      When set, renders to a sub-region of the surface instead of
                      the full surface. Used by compare mode for side-by-side display.
        """
        r = self._r
        ctx = r.ctx

        # Override surface dimensions for GL viewport if screen_rect provided
        if screen_rect is not None:
            _gl_viewport = screen_rect
            # The tonemap shader maps UVs across the GL viewport, so surface_w/h
            # must match the viewport size for correct aspect ratio
            surface_w = screen_rect[2]
            surface_h = screen_rect[3]
        else:
            _gl_viewport = (0, 0, surface_w, surface_h)

        # Frame-rate-independent temporal blend.
        # temporal_decay: blend strength in seconds (half-life of previous frame).
        # 0 = off, 0.05 = subtle speckle smoothing, 0.5 = visible trails.
        # mix_factor is how much NEW frame to show (1 = all new, 0 = frozen).
        use_temporal = r.temporal_decay > 0.0
        if use_temporal:
            # Exponential decay: after temporal_decay seconds, previous
            # frame has faded to 50%. Per-frame retention = 0.5^(dt/half_life).
            temporal_mix = 1.0 - 0.5 ** (dt / r.temporal_decay)
        else:
            temporal_mix = 1.0
        if use_temporal:
            # 3 FBOs: accum (previous blended result), current (this frame),
            # blend (output of mix → becomes new accum next frame)
            (accum_fbo, accum_tex), (current_fbo, current_tex), (blend_fbo, blend_tex) = \
                self._get_temporal_fbos(surface_w, surface_h)

        if r.blur_radius <= 0:
            # No spatial blur — tonemap directly
            target = current_fbo if use_temporal else target_fbo
            if target:
                target.use()
            else:
                _bind_default_framebuffer()
            ctx.viewport = _gl_viewport

            r.palette_tex.use(location=0)
            p = r.tonemap_program
            p['u_palette']       = 0
            p['u_width']         = r.canvas_w
            p['u_height']        = r.canvas_h
            p['u_viewport_x']    = viewport.x
            p['u_viewport_y']    = viewport.y
            p['u_viewport_w']    = viewport.w
            p['u_viewport_h']    = viewport.h
            p['u_surface_w']     = surface_w
            p['u_surface_h']     = surface_h
            p['u_gamma']         = brightness

            r.quad_vao.render(moderngl.TRIANGLES)
        else:
            # Spatial blur: tonemap → FBO A → blur H → FBO B → blur V → temp/screen
            fbo_a, tex_a, fbo_b, tex_b = self._get_blur_fbos(surface_w, surface_h)

            # Pass 1: tonemap into FBO A
            fbo_a.use()
            ctx.viewport = (0, 0, surface_w, surface_h)
            r.palette_tex.use(location=0)
            p = r.tonemap_program
            p['u_palette']       = 0
            p['u_width']         = r.canvas_w
            p['u_height']        = r.canvas_h
            p['u_viewport_x']    = viewport.x
            p['u_viewport_y']    = viewport.y
            p['u_viewport_w']    = viewport.w
            p['u_viewport_h']    = viewport.h
            p['u_surface_w']     = surface_w
            p['u_surface_h']     = surface_h
            p['u_gamma']         = brightness

            r.quad_vao.render(moderngl.TRIANGLES)

            # Pass 2: horizontal blur A → B
            fbo_b.use()
            tex_a.use(location=0)
            bp = r.blur_program
            bp['u_texture']   = 0
            bp['u_direction'] = (1.0 / surface_w, 0.0)
            bp['u_radius']    = r.blur_radius
            r.blur_vao.render(moderngl.TRIANGLES)

            # Pass 3: vertical blur B → temp/screen
            if use_temporal:
                current_fbo.use()
                ctx.viewport = (0, 0, surface_w, surface_h)
            elif target_fbo:
                target_fbo.use()
                ctx.viewport = _gl_viewport
            else:
                _bind_default_framebuffer()
                ctx.viewport = _gl_viewport
            tex_b.use(location=0)
            bp['u_texture']   = 0
            bp['u_direction'] = (0.0, 1.0 / surface_h)
            bp['u_radius']    = r.blur_radius
            r.blur_vao.render(moderngl.TRIANGLES)

        # Temporal blend: mix current + accumulated → blend_fbo, display,
        # then rotate FBOs so blend becomes new accumulator.
        if use_temporal:
            # Mix current + accum → blend_fbo
            blend_fbo.use()
            ctx.viewport = (0, 0, surface_w, surface_h)
            current_tex.use(location=0)
            accum_tex.use(location=1)
            tp = r._temporal_mix_prog
            tp['u_current']    = 0
            tp['u_previous']   = 1
            tp['u_mix_factor'] = temporal_mix
            tp['u_comparison'] = 0
            r._temporal_mix_vao.render(moderngl.TRIANGLES)

            # Blit blend result to screen/target
            if target_fbo:
                target_fbo.use()
            else:
                _bind_default_framebuffer()
            ctx.viewport = _gl_viewport
            blend_tex.use(location=0)
            bp = r.blur_program
            bp['u_texture']   = 0
            bp['u_direction'] = (0.0, 0.0)
            bp['u_radius']    = 0.0
            r.blur_vao.render(moderngl.TRIANGLES)

            # Rotate: blend → accum for next frame
            key = (surface_w, surface_h)
            self._temporal_fbos[key] = [
                (blend_fbo, blend_tex),     # new accum
                (accum_fbo, accum_tex),      # new current (scratch)
                (current_fbo, current_tex),  # new blend (scratch)
            ]
