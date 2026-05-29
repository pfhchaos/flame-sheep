"""Shim re-exporting from the viz_authoring package, plus the
flame-specific `SHADER_DIR` pointing at this package's shader sources.

The GPU framework (GpuContext, Viewport, etc.) was extracted into
viz_authoring in Stage 9 of `docs/reorg_plan.md`. SHADER_DIR
intentionally stays flame-side — it points at `flame_sheep/rendering/
shaders/`, which is flame-fractal-specific GLSL (chaos game, tonemap,
density estimation, etc.), not generic infrastructure.

External callers can use either:
  `from flame_sheep.rendering import GpuContext, SHADER_DIR`  (preferred)
  `from flame_sheep.rendering.gpu_context import GpuContext, SHADER_DIR`  (works via this shim)
"""
from pathlib import Path

from viz_authoring.gpu_context import (  # noqa: F401
    Viewport,
    GpuContext,
    GpuRingTimer,
    GPU_TIMING_ENABLED,
    _resolve_includes,
    _bind_default_framebuffer,
)

# Flame-specific shader root. Lives here (not in viz_authoring) because
# the shaders are flame-fractal-specific.
SHADER_DIR = Path(__file__).parent / 'shaders'
