"""Multi-monitor composition — framework-side screen layout.

Future home for everything currently scattered across display.py and
wallpaper.py's setup block that answers "given the user's monitors,
where does the visualization land on each one?" The hard parts —
Wayland output discovery, monitor configs, ppi vs physical-mm
disambiguation, multi-monitor alignment, canvas-size derivation —
become one-time framework work that a future viz author never has
to touch.

Today: just PhysicalViewport (the unit-aware viewport descriptor
used by wallpaper to convert physical-mm rectangles into pixel
Viewports against a known canvas ppmm). More moves in here as the
framework matures — discover_outputs, compute_canvas, etc.

Eventual destination is the viz_authoring sibling package; for now
it lives in flame_sheep/ because that's where the consumers are.
"""
from __future__ import annotations

from dataclasses import dataclass

from .rendering import Viewport


@dataclass(frozen=True)
class PhysicalViewport:
    """A viewport described in physical millimeters.

    Convert to a pixel-coordinate Viewport at a known ppmm scale for
    actual rendering. The mm description is anchored to real screen
    geometry; the pixel Viewport is what shaders consume.
    """
    x_mm: float
    y_mm: float
    w_mm: float
    h_mm: float

    def to_viewport(self, ppmm: float) -> Viewport:
        return Viewport(
            int(self.x_mm * ppmm),
            int(self.y_mm * ppmm),
            int(self.w_mm * ppmm),
            int(self.h_mm * ppmm),
        )
