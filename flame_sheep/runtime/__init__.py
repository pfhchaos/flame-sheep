"""Live-wallpaper runtime coordination.

Contains the components that orchestrate the running wallpaper:
  - core.py: FlameSheepCore (the main per-frame coordinator)
  - orchestrator.py: Orchestrator + TimestampedEvent (audio event pipeline)
  - control.py: ControlPipe + ControlEvent (external commands)
  - session.py: SessionMonitor (login session presence/idle)
  - mpris.py: MprisListener (media player metadata)
  - role_mapper.py: RoleMapper + role constants (DOWNBEAT/BACKBEAT/...)
  - wallpaper_vk.py: _run_wallpaper_vk (the wallpaper's main loop)
  - wayland_outputs.py: monitor discovery + singleton lock

This __init__.py re-exports the public API. External code should
import from `flame_sheep.runtime`, not from submodules directly.
"""

# role_mapper loads first — it has no internal dependencies and is imported
# by axes/* modules during core.py's transitive load. Loading it later would
# leave it unbound when axes try `from flame_sheep.runtime import RoleMapper`
# during partial init.
from .role_mapper import (
    RoleMapper,
    DOWNBEAT, BACKBEAT, SUBDIVISION, ENERGY,
    ALL_ROLES, DEFAULT_MAPPING,
)
from .control import ControlPipe, ControlEvent
from .session import SessionMonitor
from .mpris import MprisListener
from .orchestrator import Orchestrator, TimestampedEvent
from .core import FlameSheepCore

__all__ = [
    'FlameSheepCore',
    'Orchestrator',
    'TimestampedEvent',
    'ControlPipe',
    'ControlEvent',
    'SessionMonitor',
    'MprisListener',
    'RoleMapper',
    'DOWNBEAT', 'BACKBEAT', 'SUBDIVISION', 'ENERGY',
    'ALL_ROLES', 'DEFAULT_MAPPING',
]
