"""Headless Vulkan renderer — ChaosGame + tonemap → host buffer.

Wraps the chaos-game compute pipeline and the tonemap graphics
pipeline in a class that:
  - has no swapchain / surface dependency (subprocess-safe; no Wayland
    socket required)
  - exposes histogram + transform_hits download (for scorers operating
    on raw data)
  - exposes snapshot_png() that renders tonemap into an offscreen
    image, copies to host-visible buffer, encodes as PNG bytes

Intended consumers:
  - flame_sheep.genome.render_worker (auto-spawned by wallpaper for
    background scoring renders)
  - flame_sheep.storage.catalog (genome catalog image generation)
  - flame_sheep.genome.scoring.gpu (standalone scorer)
  - tools/* one-shot rendering

All three currently use the GL FlameRenderer with moderngl + EGL. The
GL path has been crashing under subprocess.Popen on Mesa-Xe + Intel
Arc; multiprocessing.fork has caused kernel panics historically.
Vulkan + subprocess.Popen has been stable in the precompile_worker
since shipping, so the headless Vk path is the right portage target.
"""
from __future__ import annotations

import io
import struct
from pathlib import Path

import cffi
import numpy as np
import vulkan as vk

from .chaos_game import ChaosGame
from viz_authoring.vk.context import VkContext
from viz_authoring.vk.image import create_color_attachment_image, create_image_rgba8, \
    create_linear_sampler, upload_image_rgba8
from viz_authoring.vk.pipeline import GraphicsPipeline, make_transfer_src_render_pass

SHADER_DIR = Path(__file__).parent / 'shaders'


def _palette_to_rgba8(palette: np.ndarray) -> np.ndarray:
    """Normalize palette to (1, 256, 4) uint8 alpha=255.
    Accepts:
      - (256, 3) float32 [0,1]     — raw RGB palette
      - (256, 4) float32 [0,1]     — raw RGBA palette
      - (1, 256, 4) uint8          — already in upload format
        (catalog loader returns this)
    """
    if palette.dtype == np.uint8 and palette.shape == (1, 256, 4):
        return palette
    p = np.clip(palette, 0.0, 1.0)
    if p.ndim == 3 and p.shape == (1, 256, 4):
        # float32 in the upload layout already
        rgba = (p * 255.0).astype(np.uint8)
        rgba[0, :, 3] = 255
        return rgba
    # 2-D RGB or RGBA palette → reshape to upload layout
    rgba = np.zeros((1, 256, 4), dtype=np.uint8)
    rgba[0, :, :3] = (p[:, :3] * 255.0).astype(np.uint8)
    rgba[0, :, 3] = 255
    return rgba


class HeadlessVkRenderer:
    """Headless equivalent of FlameRenderer for scoring / catalog use.

    Lifecycle:
        renderer = HeadlessVkRenderer(ctx, 512, 512, n_walkers=14720)
        renderer.set_genome(**genome_kwargs)
        renderer.set_palette(palette)         # (256, 3) float32
        renderer.reset_walkers()
        renderer.clear_histogram()
        renderer.dispatch_chaos_game(iterations=400)
        hits, colors = renderer.download_histogram()
        png_bytes = renderer.snapshot_png()
        renderer.cleanup()

    Buffers are owned by ChaosGame and exposed via .chaos for direct
    access. The headless additions are: tonemap pipeline, render
    target image, host-visible readback buffer, render pass.
    """

    def __init__(self, ctx: VkContext, width: int, height: int,
                  n_walkers: int = 14720):
        self.ctx = ctx
        self.width = width
        self.height = height

        # Chaos game owns histogram, walkers, affines, variations,
        # transform_hits, etc. scoring_mode=True so transform_hits
        # actually get written — extra atomic per inner-loop iter,
        # but scoring renders aren't latency-sensitive.
        self.chaos = ChaosGame(ctx, width, height, n_walkers=n_walkers,
                                scoring_mode=True)

        # Palette texture for the tonemap shader.
        self._palette_img = create_image_rgba8(ctx, 256, 1)
        upload_image_rgba8(ctx, self._palette_img,
                            np.zeros((1, 256, 4), dtype=np.uint8))
        self._palette_sampler = create_linear_sampler(ctx)

        # Render pass that ends in TRANSFER_SRC_OPTIMAL so we can
        # vkCmdCopyImageToBuffer immediately after.
        self._render_pass = make_transfer_src_render_pass(
            ctx, vk.VK_FORMAT_R8G8B8A8_UNORM)

        # The render target: DEVICE_LOCAL color attachment we can
        # later copy to a host-visible buffer.
        self._target = create_color_attachment_image(
            ctx, width, height,
            extra_usage=vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
        fb_create = vk.VkFramebufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            renderPass=self._render_pass,
            attachmentCount=1, pAttachments=[self._target.view],
            width=width, height=height, layers=1,
        )
        self._framebuffer = vk.vkCreateFramebuffer(ctx.device, fb_create, None)

        # Tonemap pipeline. Same shaders as wallpaper path — just
        # targeting our offscreen render pass instead of the
        # intermediate / swapchain ones.
        extent = vk.VkExtent2D(width=width, height=height)
        self._tonemap = GraphicsPipeline(
            ctx, self._render_pass,
            SHADER_DIR / 'tonemap.vert',
            SHADER_DIR / 'tonemap.frag',
            extent=extent,
            storage_buffers=[self.chaos.histogram, self.chaos.max_buf],
            sampled_images=[(self._palette_img, self._palette_sampler)],
            push_constant_size=28,   # 6 ints + 1 float
        )

        # Host-visible readback buffer that vkCmdCopyImageToBuffer
        # writes into. RGBA8 → 4 bytes per pixel.
        self._readback = ctx.create_buffer(
            width * height * 4, 'transfer_dst')

    # ----- delegate-to-chaos passthrough ----------------------------
    # Pass-through helpers so callers don't reach into self.chaos
    # everywhere — keeps the headless renderer the single API surface.

    def set_genome(self, **kwargs) -> None:
        self.chaos.set_genome(**kwargs)

    def reset_walkers(self, seed: int | None = None) -> None:
        self.chaos.reset_walkers(seed=seed)

    def clear_histogram(self, decay: float = 0.0) -> None:
        self.chaos.clear_histogram(decay=decay)

    def clear_transform_hits(self) -> None:
        self.chaos.clear_transform_hits()

    def dispatch_chaos_game(self, iterations: int,
                              zoom: tuple[float, float] = (1.0, 1.0),
                              rotation: float = 0.0,
                              center: tuple[float, float] = (0.0, 0.0),
                              ) -> None:
        """One chaos-game pass. Does NOT clear the histogram first —
        caller can decide whether to accumulate (swept render) or
        start fresh (single static render)."""
        self.chaos.render_frame(iterations=iterations, zoom=zoom,
                                  rotation=rotation, center=center)

    def download_histogram(self):
        return self.chaos.download_histogram()

    def download_transform_hits(self):
        return self.chaos.download_transform_hits()

    # ----- palette -------------------------------------------------

    def set_palette(self, palette: np.ndarray) -> None:
        """palette: (256, 3) float32 in [0,1]."""
        upload_image_rgba8(self.ctx, self._palette_img,
                            _palette_to_rgba8(palette))

    # ----- the headless PNG snapshot path --------------------------

    def snapshot_rgba(self, gamma: float = 6.0) -> np.ndarray:
        """Render the current histogram + palette through the tonemap
        pipeline into an offscreen image, copy to a host buffer,
        return as (H, W, 4) uint8 numpy array. Use this when you need
        the raw pixels — snapshot_png() wraps this with PIL encode.

        Before calling: ensure max_buf is populated. The chaos game's
        batched frame() does that; render_frame() alone doesn't, so a
        scoring caller should run reduce_max_hits() first (we do this
        unconditionally below for safety).
        """
        # The chaos game's reduce_max sequence is what populates
        # max_buf. Run it once so the tonemap sees a valid max.
        self.chaos.reduce_max_hits()

        ffi = cffi.FFI()
        cb = self.ctx.allocate_command_buffers(1)[0]
        vk.vkBeginCommandBuffer(cb, vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        ))

        clear = vk.VkClearValue(
            color=vk.VkClearColorValue(float32=[0, 0, 0, 1]))
        rp_begin = vk.VkRenderPassBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            renderPass=self._render_pass,
            framebuffer=self._framebuffer,
            renderArea=vk.VkRect2D(
                offset=vk.VkOffset2D(x=0, y=0),
                extent=vk.VkExtent2D(width=self.width, height=self.height),
            ),
            clearValueCount=1, pClearValues=[clear],
        )
        vk.vkCmdBeginRenderPass(cb, rp_begin, vk.VK_SUBPASS_CONTENTS_INLINE)
        vk.vkCmdBindPipeline(cb, vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                              self._tonemap.pipeline)
        vk.vkCmdBindDescriptorSets(
            cb, vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            self._tonemap.layout, 0, 1,
            [self._tonemap.descriptor_set], 0, None)

        # Tonemap push constants: canvas_w, canvas_h, viewport_x,y,w,h, gamma
        push_bytes = struct.pack(
            '6if', self.width, self.height,
            0, 0, self.width, self.height, gamma)
        pc = ffi.new('char[]', push_bytes)
        vk.vkCmdPushConstants(
            cb, self._tonemap.layout,
            vk.VK_SHADER_STAGE_FRAGMENT_BIT,
            0, len(push_bytes), ffi.cast('void*', pc))
        # Fullscreen triangle: 3 verts, no vertex buffer (vert shader
        # generates positions from gl_VertexIndex).
        vk.vkCmdDraw(cb, 3, 1, 0, 0)
        vk.vkCmdEndRenderPass(cb)

        # Copy the now-rendered image to the host buffer. Image is in
        # TRANSFER_SRC_OPTIMAL layout (set by the render pass's
        # finalLayout). Pack tightly: no row padding, RGBA8.
        copy_region = vk.VkBufferImageCopy(
            bufferOffset=0,
            bufferRowLength=0,   # tightly packed
            bufferImageHeight=0,
            imageSubresource=vk.VkImageSubresourceLayers(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                mipLevel=0, baseArrayLayer=0, layerCount=1,
            ),
            imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
            imageExtent=vk.VkExtent3D(
                width=self.width, height=self.height, depth=1),
        )
        vk.vkCmdCopyImageToBuffer(
            cb, self._target.image,
            vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            self._readback.buffer, 1, [copy_region])

        vk.vkEndCommandBuffer(cb)

        # Submit + wait. Same fence pattern ChaosGame.frame() uses.
        submit = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1, pCommandBuffers=[cb],
        )
        fence = vk.vkCreateFence(self.ctx.device, vk.VkFenceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO), None)
        vk.vkQueueSubmit(self.ctx.graphics_queue, 1, [submit], fence)
        vk.vkWaitForFences(self.ctx.device, 1, [fence], vk.VK_TRUE,
                            0xFFFFFFFFFFFFFFFF)
        vk.vkDestroyFence(self.ctx.device, fence, None)
        vk.vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool,
                                 1, [cb])

        # Read back. ctx.download is parameterized by dtype + element
        # count — RGBA8 = uint8, 4 channels per pixel.
        rgba = self.ctx.download(
            self._readback, np.uint8, self.width * self.height * 4)
        return rgba.reshape(self.height, self.width, 4)

    def snapshot_png(self, gamma: float = 6.0) -> bytes:
        """RGBA snapshot encoded as PNG bytes. Wraps snapshot_rgba()
        with PIL encode. Callers wanting raw pixels should use
        snapshot_rgba() directly."""
        rgba = self.snapshot_rgba(gamma=gamma)
        # PIL imported lazily — keeps this module importable without
        # PIL for callers that only use raw data.
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(rgba, mode='RGBA').save(buf, format='PNG')
        return buf.getvalue()

    # ----- cleanup -------------------------------------------------

    def cleanup(self) -> None:
        vk.vkDestroyFramebuffer(self.ctx.device, self._framebuffer, None)
        self._tonemap.cleanup()
        vk.vkDestroyRenderPass(self.ctx.device, self._render_pass, None)
        self._target.destroy()
        self._palette_img.destroy()
        self._palette_sampler.destroy()
        self.chaos.cleanup()
        # buffers (palette + readback) auto-freed by ctx.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.cleanup()
