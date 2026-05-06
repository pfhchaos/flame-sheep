"""Role-based axis routing — maps audio band names to visual roles.

Decouples visualization axes from specific band names. Axes consume
abstract roles (downbeat, backbeat, subdivision, energy) rather than
hardcoded band names like 'low' or 'mid'.

Currently uses static config-based mapping. Designed to be replaced
with automatic role detection (based on tempo + onset density) when
the tempo tracker is reliable enough.

Future considerations for arbitrary bin counts:
  - Multiple bands may map to the same role (e.g., 3 low-freq bins
    all contribute to 'downbeat')
  - A role may have no matching band (graceful fallback)
  - Auto-detection would rank bands by density relative to tempo
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from flame_sheep_audio._types import BeatEvent, BandState

if TYPE_CHECKING:
    from flame_sheep_audio._types import AudioState

log = logging.getLogger(__name__)

# Standard role names
DOWNBEAT = 'downbeat'
BACKBEAT = 'backbeat'
SUBDIVISION = 'subdivision'
ENERGY = 'energy'

ALL_ROLES = (DOWNBEAT, BACKBEAT, SUBDIVISION, ENERGY)

# Default mapping (matches the default BandConfig)
DEFAULT_MAPPING = {
    DOWNBEAT: 'low',
    BACKBEAT: 'mid',
    SUBDIVISION: 'high',
    ENERGY: 'subbass',
}


class RoleMapper:
    """Maps band names to visual roles.

    Axes call band_for_role('downbeat') to get the band name they
    should listen to. Events are translated via role_for_event().

    With the default mapping this is a trivial dict lookup. The
    interface is designed so an AutoRoleMapper subclass can swap in
    later without changing any axis code.
    """

    def __init__(self, mapping: dict[str, str] | None = None) -> None:
        self._role_to_band: dict[str, str] = dict(mapping or DEFAULT_MAPPING)
        self._band_to_role: dict[str, str] = {v: k for k, v in self._role_to_band.items()}

    def band_for_role(self, role: str) -> str:
        """Get the band name assigned to a role.

        Raises KeyError if the role has no mapping — this is intentional,
        axes should fail loudly if a required role is unmapped.
        """
        return self._role_to_band[role]

    def role_for_band(self, band: str) -> str | None:
        """Get the role assigned to a band, or None if unmapped."""
        return self._band_to_role.get(band)

    def role_for_event(self, event: BeatEvent) -> str | None:
        """Get the role for a beat event based on its kind (band name)."""
        return self._band_to_role.get(event.kind)

    def band_state(self, audio: AudioState, role: str) -> BandState:
        """Get the BandState for a role from an AudioState.

        Returns a default BandState if the band doesn't exist in this
        audio frame (graceful fallback for missing bands).
        """
        band = self._role_to_band.get(role)
        if band is None:
            return BandState()
        return audio.bands.get(band, BandState())

    def has_role(self, role: str) -> bool:
        """Check if a role has a mapped band."""
        return role in self._role_to_band

    @property
    def mapping(self) -> dict[str, str]:
        """Current role → band mapping (read-only copy)."""
        return dict(self._role_to_band)
