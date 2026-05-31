"""Vulkan port of CompareRenderer (the visual half of compare mode).

CompareMode (pair selection / voting logic) lives in compare.py and is
reused unchanged — this module only ports the rendering side.

Differences from the GL version (compare.py:CompareRenderer):
- Two independent ChaosGame instances instead of a single shared
  histogram with offsets. GL needed the offset trick to work around
  Mesa's per-program SSBO binding cache on Arc; Vulkan's explicit
  descriptor sets make that workaround unnecessary.
- No blur pass (GL also disables blur in compare mode).
- Per-frame command buffer recording instead of pre-recorded buffers.
  Compare mode is interactive (user voting), not the wallpaper's hot
  path — the extra ~ms of CPU each frame is invisible. Skipping
  pre-record means we can record different tonemap descriptor sets
  per frame without managing a 2× set of pre-recorded buffers.

Geometry: the compare view replaces the normal flame on the largest
output surface. Left/right halves of that surface show the left/right
genome respectively. Other outputs continue to show... whatever the
caller chooses (currently the wallpaper just shows the normal frame
on them; the integration point in wallpaper_vk decides).
"""
from __future__ import annotations

import logging
import math
import struct
from pathlib import Path

import cffi
import numpy as np
import vulkan as vk

from viz_authoring.vk.chaos_game import ChaosGame
from viz_authoring.vk.image import (create_image_rgba8, create_linear_sampler,
                                       upload_image_rgba8)
from viz_authoring.vk.pipeline import GraphicsPipeline

log = logging.getLogger(__name__)

SHADER_DIR = (Path(__file__).resolve().parents[2]
              / 'viz_authoring/src/viz_authoring/vk/shaders')

# Compare canvas scaling — half-resolution per side. Mirrors GL
# CompareRenderer.CMP_SCALE so render fidelity matches.
CMP_SCALE = 2

# Walker count per side. Matches GL CompareRenderer.CMP_WALKERS:
# quarter of the wallpaper's count so per-pixel walker density is
# roughly the same as the live wallpaper at full res.
CMP_WALKERS = 65536 // 4


def _palette_to_rgba8(palette: np.ndarray) -> np.ndarray:
    """Same convention HeadlessVkRenderer uses — accept catalog
    (1, 256, 4) uint8 or (256, 3) float32, normalize to (1, 256, 4)
    uint8 with alpha=255."""
    if palette.dtype == np.uint8 and palette.shape == (1, 256, 4):
        return palette
    p = np.clip(palette, 0.0, 1.0)
    rgba = np.zeros((1, 256, 4), dtype=np.uint8)
    if p.ndim == 3 and p.shape == (1, 256, 4):
        rgba = (p * 255.0).astype(np.uint8)
        rgba[0, :, 3] = 255
    else:
        rgba[0, :, :3] = (p[:, :3] * 255.0).astype(np.uint8)
        rgba[0, :, 3] = 255
    return rgba


class CompareRendererVk:
    """Holds two half-resolution ChaosGames and the per-side tonemap
    pipelines. wallpaper_vk owns the swapchain + render passes and
    drives recording.

    Lifecycle:
      - Construct once at wallpaper startup (cheap — just buffer
        allocation + two ChaosGames + two pipelines).
      - set_pair(left_g, right_g) before each new pair.
      - tick(frame) each frame to advance both chaos games.
      - record_tonemap(cb, render_pass, framebuffer, surf_w, surf_h,
                        brightness) once per output that wants the
        compare view.
    """

    def __init__(self, ctx,
                  main_canvas_w: int, main_canvas_h: int,
                  swapchain_render_pass):
        """main_canvas_w / main_canvas_h: the wallpaper's canvas dims.
        Compare's per-side render dimension is the largest output's
        half-width / CMP_SCALE × full-height / CMP_SCALE — same as
        GL CompareRenderer derives it.

        For the MVP we use main_canvas dims directly. If the largest
        output isn't the same shape as the canvas, the compare view
        will be slightly mis-sized; refinement later.
        """
        self.ctx = ctx
        # Per-side render dimensions. The eventual viewport on the
        # actual swapchain may differ — these are the histogram /
        # walker sizes, not the on-screen pixels.
        self.side_w = (main_canvas_w // 2) // CMP_SCALE
        self.side_h = main_canvas_h // CMP_SCALE
        self._frame_seed = 0
        self._scoring_render_pass = swapchain_render_pass

        log.info(f'[compare_vk] init: side={self.side_w}x{self.side_h} '
                 f'walkers={CMP_WALKERS}')

        # Two independent chaos games — no offset trickery needed in
        # Vulkan. Each owns its own histogram, walkers, transform_hits.
        self._left_chaos = ChaosGame(ctx, self.side_w, self.side_h,
                                       n_walkers=CMP_WALKERS)
        self._right_chaos = ChaosGame(ctx, self.side_w, self.side_h,
                                        n_walkers=CMP_WALKERS)

        # Palette resources — single image shared by both sides, since
        # the GL code uploads the same frame.palette to both anyway.
        self._palette_img = create_image_rgba8(ctx, 256, 1)
        upload_image_rgba8(ctx, self._palette_img,
                            np.zeros((1, 256, 4), dtype=np.uint8))
        self._palette_sampler = create_linear_sampler(ctx)

        # Two tonemap pipelines, each bound to one side's histogram.
        # Same shader; descriptor sets differ. Target the same
        # swapchain render pass the wallpaper uses (passed in by
        # caller) so we can record into the same command buffer.
        self._left_tonemap = GraphicsPipeline(
            ctx, swapchain_render_pass,
            SHADER_DIR / 'tonemap.vert',
            SHADER_DIR / 'tonemap.frag',
            extent=vk.VkExtent2D(width=self.side_w, height=self.side_h),
            storage_buffers=[self._left_chaos.histogram,
                              self._left_chaos.max_buf],
            sampled_images=[(self._palette_img, self._palette_sampler)],
            push_constant_size=28,
            dynamic_viewport=True,
        )
        self._right_tonemap = GraphicsPipeline(
            ctx, swapchain_render_pass,
            SHADER_DIR / 'tonemap.vert',
            SHADER_DIR / 'tonemap.frag',
            extent=vk.VkExtent2D(width=self.side_w, height=self.side_h),
            storage_buffers=[self._right_chaos.histogram,
                              self._right_chaos.max_buf],
            sampled_images=[(self._palette_img, self._palette_sampler)],
            push_constant_size=28,
            dynamic_viewport=True,
        )

        self._left_genome = None
        self._right_genome = None

    def set_pair(self, left_genome, right_genome) -> None:
        """Upload new pair. Done once on pair change, not per frame."""
        from ..runtime.wallpaper_vk import _genome_to_chaos_kwargs
        self._left_chaos.set_genome(**_genome_to_chaos_kwargs(left_genome))
        self._right_chaos.set_genome(**_genome_to_chaos_kwargs(right_genome))
        self._left_chaos.reset_walkers()
        self._right_chaos.reset_walkers()
        self._left_genome = left_genome
        self._right_genome = right_genome
        log.info('[compare_vk] new pair loaded')

    def set_palette(self, palette: np.ndarray) -> None:
        """Update palette for both sides (per-frame, cheap memcpy)."""
        upload_image_rgba8(self.ctx, self._palette_img,
                            _palette_to_rgba8(palette))

    def tick(self, frame, rotation_phase: float = 0.0) -> None:
        """Run one chaos-game frame for each side. Mirrors wallpaper's
        chaos.frame() but for the compare pair. Uses fence-synced
        render_frame for both — accept the two fence waits since
        compare mode isn't FPS-critical."""
        if self._left_genome is None or self._right_genome is None:
            return
        # Rotation phase animates the rendered orientation; per-side
        # render uses the same rotation. GL version rotates the genome
        # affines by `rot`, but Vk passes rotation as a per-frame push
        # constant — equivalent effect for our purposes.
        cos_r = math.cos(rotation_phase)
        sin_r = math.sin(rotation_phase)
        for cg in (self._left_chaos, self._right_chaos):
            cg.frame(iterations=frame.iterations,
                      zoom=(self._left_genome.zoom,
                             self._left_genome.zoom),
                      rotation=rotation_phase,
                      center=tuple(self._left_genome.center),
                      decay=0.3)

    def record_split_tonemap(self, cb, swapchain_pass,
                                framebuffer, surf_w: int, surf_h: int,
                                brightness: float) -> None:
        """Record the split-screen tonemap into command buffer `cb`.
        Caller is responsible for begin/end of the command buffer
        itself + queue submission. We just emit the render pass +
        draws.

        Layout: left half of surf gets left tonemap, right half gets
        right tonemap. Each tonemap maps its own (side_w × side_h)
        histogram into surf_w/2 × surf_h pixels.
        """
        half_w = surf_w // 2
        clear = vk.VkClearValue(
            color=vk.VkClearColorValue(float32=[0, 0, 0, 1]))
        rp_begin = vk.VkRenderPassBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            renderPass=swapchain_pass, framebuffer=framebuffer,
            renderArea=vk.VkRect2D(
                offset=vk.VkOffset2D(x=0, y=0),
                extent=vk.VkExtent2D(width=surf_w, height=surf_h),
            ),
            clearValueCount=1, pClearValues=[clear],
        )
        vk.vkCmdBeginRenderPass(cb, rp_begin,
                                  vk.VK_SUBPASS_CONTENTS_INLINE)

        ffi = cffi.FFI()
        for half, pipeline, chaos in (
            (0,       self._left_tonemap,  self._left_chaos),
            (half_w,  self._right_tonemap, self._right_chaos),
        ):
            # reduce_max for THIS side's histogram. Must run before
            # the tonemap reads max_buf. We compress it into the same
            # command buffer via a compute dispatch before the render
            # pass — but the render pass is already open. So instead
            # we rely on the caller having called tick() which itself
            # ran reduce_max via chaos.frame(). Good.

            # Bind pipeline + descriptor set
            vk.vkCmdBindPipeline(cb, vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  pipeline.pipeline)
            vk.vkCmdBindDescriptorSets(
                cb, vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipeline.layout, 0, 1, [pipeline.descriptor_set],
                0, None)

            # Dynamic viewport for our half of the surface
            viewport = vk.VkViewport(
                x=float(half), y=0.0,
                width=float(half_w), height=float(surf_h),
                minDepth=0.0, maxDepth=1.0)
            scissor = vk.VkRect2D(
                offset=vk.VkOffset2D(x=half, y=0),
                extent=vk.VkExtent2D(width=half_w, height=surf_h))
            vk.vkCmdSetViewport(cb, 0, 1, [viewport])
            vk.vkCmdSetScissor(cb, 0, 1, [scissor])

            # Push constants: canvas=side dim, viewport=full side,
            # gamma=brightness (the tonemap shader treats the float as
            # the gamma exponent — same as wallpaper_vk).
            push = struct.pack(
                '6if', chaos.canvas_w, chaos.canvas_h,
                0, 0, chaos.canvas_w, chaos.canvas_h, brightness)
            pc = ffi.new('char[]', push)
            vk.vkCmdPushConstants(
                cb, pipeline.layout,
                vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                0, len(push), ffi.cast('void*', pc))
            vk.vkCmdDraw(cb, 3, 1, 0, 0)

        vk.vkCmdEndRenderPass(cb)

    def cleanup(self) -> None:
        self._left_tonemap.cleanup()
        self._right_tonemap.cleanup()
        self._palette_img.destroy()
        self._palette_sampler.destroy()
        self._left_chaos.cleanup()
        self._right_chaos.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.cleanup()
