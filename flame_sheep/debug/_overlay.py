"""Debug overlay window — moderngl-window WindowConfig.

Second orchestrator consumer that visualizes real-time audio analysis
in a separate GLFW window.
"""

from __future__ import annotations

import logging
import time

import moderngl
import moderngl_window as mglw

from flame_sheep.orchestrator import Orchestrator
from flame_sheep.config import cfg
from ._draw import SolidRenderer
from ._text import TextRenderer
from ._timeline import TimelineBuffer
from ._panels import build_panels

log = logging.getLogger(__name__)


class DebugOverlay(mglw.WindowConfig):
    """Debug overlay window — visualizes audio analysis state."""
    title = 'flame-sheep debug'
    gl_version = (3, 3)
    resizable = True
    vsync = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Orchestrator is set by run_debug_overlay before window creation
        orch = _ORCH
        self._orch = orch
        self._consumer_id = orch.register('debug_overlay')
        orch.start()

        # Renderers
        self._draw = SolidRenderer(self.ctx)
        self._text = TextRenderer(self.ctx)

        # Timeline
        band_config = orch.band_config
        self._timeline = TimelineBuffer(
            band_names=band_config.detection_band_names,
            max_seconds=cfg.debug.timeline_seconds,
        )

        # Panels
        enabled = list(cfg.debug.panels) if hasattr(cfg.debug, 'panels') else None
        self._panels = build_panels(band_config, enabled=enabled)

        log.info(f'debug overlay started ({len(self._panels)} panels)')

    def on_render(self, time_val: float, frame_time: float) -> None:
        self.ctx.clear(0.12, 0.14, 0.17)
        self.ctx.enable(moderngl.BLEND)

        # Tick orchestrator (we own it in this process)
        self._orch.tick()

        # Drain events into timeline
        events = self._orch.drain_events(self._consumer_id)
        for te in events:
            self._timeline.push(te)

        # Update panels
        snap = self._orch.audio_state
        for panel in self._panels:
            if panel.visible:
                panel.update(snap, self._timeline, frame_time)

        # Render panels (vertical stack)
        w, h = self.window_size
        self._draw.begin()
        self._text.begin()
        y = 0
        for panel in self._panels:
            if panel.visible:
                panel.render(self._draw, self._text, 0, y, w, panel.height)
                y += panel.height
        self._draw.flush(w, h)
        self._text.flush(w, h)

    def close(self) -> None:
        self._orch.stop()


# Module-level orchestrator reference — set before window creation
# (moderngl-window doesn't support passing args to WindowConfig.__init__)
_ORCH: Orchestrator | None = None


def run_debug_overlay(audio_device: str | int | None = None,
                      test_audio: bool = False) -> None:
    """Entry point: create orchestrator + open debug window."""
    from flame_sheep_audio import DEFAULT_DEVICE
    from moderngl_window import settings
    import sys

    global _ORCH
    _ORCH = Orchestrator(audio_device=audio_device or DEFAULT_DEVICE,
                         test_audio=test_audio)

    settings.WINDOW['class'] = 'moderngl_window.context.glfw.Window'
    settings.WINDOW['size'] = (cfg.debug.window_width, cfg.debug.window_height)
    settings.WINDOW['title'] = 'flame-sheep debug'

    # Strip unknown args so moderngl-window doesn't choke
    sys.argv = [sys.argv[0]]

    mglw.run_window_config(DebugOverlay)
