"""GPU rendering for flame fractals.

Internal layout (post Stage 0 decomposition):
  - gpu_context.py: framework-side GPU plumbing (moderngl context, shader
    compilation, viewports, buffer/texture allocation). Could be reused by
    any viz; future viz_authoring extraction target.
  - renderer.py: flame-specific dispatch (chaos game, histogram, tonemap,
    snapshot). FlameRenderer class lives here.
  - window.py: Wayland layer-shell window management (was wayland_window.py).
  - surface.py: monitor discovery, layout config, singleton lock (was
    display.py).
  - shaders/: GLSL shaders for the flame fractal pipeline.

This __init__.py re-exports the public API. External code should import
from `flame_sheep.rendering`, not from submodules directly — the submodule
layout is an implementation detail of Stage 3b of docs/reorg_plan.md.
"""

from .gpu_context import (
    SHADER_DIR,
    Viewport,
    GpuContext,
    _resolve_includes,
    _bind_default_framebuffer,
)
from .renderer import (
    FlameRenderer,
    N_WALKERS,
    N_ITERS,
    LIVE_ITER_MIN,
    LIVE_ITER_MAX,
)
from .surface import (
    _monitor_cfg,
    _get_output_layout,
    _ensure_singleton,
)
from .window import (
    WallpaperSession,
    OutputSurface,
    WallpaperWindow,
)

__all__ = [
    # GPU context (framework-side, future viz_authoring)
    'SHADER_DIR',
    'Viewport',
    'GpuContext',
    '_resolve_includes',
    '_bind_default_framebuffer',
    # Flame-specific rendering
    'FlameRenderer',
    'N_WALKERS',
    'N_ITERS',
    'LIVE_ITER_MIN',
    'LIVE_ITER_MAX',
    # Wayland window
    'WallpaperSession',
    'OutputSurface',
    'WallpaperWindow',
    # Monitor / surface
    '_monitor_cfg',
    '_get_output_layout',
    '_ensure_singleton',
]
