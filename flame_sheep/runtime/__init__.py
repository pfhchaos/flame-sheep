"""Live-wallpaper runtime coordination.

Contains the components that orchestrate the running wallpaper:
  - core.py: FlameSheepCore (the main per-frame coordinator)
  - orchestrator.py: Orchestrator + TimestampedEvent (audio event pipeline)
  - control.py: ControlPipe + ControlEvent (external commands)
  - session.py: SessionMonitor (login session presence/idle)
  - mpris.py: MprisListener (media player metadata)
  - command_handlers.py: WallpaperCommands (control-pipe command dispatch)
  - role_mapper.py: RoleMapper + role constants (DOWNBEAT/BACKBEAT/...)
  - wallpaper.py: _run_wallpaper (the wallpaper command's main loop)

This __init__.py re-exports the public API per the convention used by
storage/, rendering/, etc. External code should import from
`flame_sheep.runtime`, not from submodules directly — the submodule
layout is an implementation detail of Stage 3a of docs/reorg_plan.md.
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
from .command_handlers import WallpaperCommands
from .wallpaper import _run_wallpaper

__all__ = [
    'FlameSheepCore',
    'Orchestrator',
    'TimestampedEvent',
    'ControlPipe',
    'ControlEvent',
    'SessionMonitor',
    'MprisListener',
    'WallpaperCommands',
    'RoleMapper',
    'DOWNBEAT', 'BACKBEAT', 'SUBDIVISION', 'ENERGY',
    'ALL_ROLES', 'DEFAULT_MAPPING',
    '_run_wallpaper',
]
