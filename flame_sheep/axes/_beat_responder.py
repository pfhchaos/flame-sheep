"""Beat responder — processes beat events and returns actions for GenomeAxis.

Owns section change detection, break damping, downbeat energy tracking.
Does NOT mutate morph state — returns actions that the coordinator executes.
"""

from __future__ import annotations

import enum
import logging
from typing import TYPE_CHECKING

from flame_sheep.config import cfg
from flame_sheep.role_mapper import RoleMapper, BACKBEAT
from flame_sheep_audio import BeatEvent

if TYPE_CHECKING:
    from flame_sheep_audio import AudioState

log = logging.getLogger(__name__)


class BeatAction(enum.Enum):
    """Actions that BeatResponder requests from the coordinator."""
    RELEASE_DWELL = 'release_dwell'
    COMMIT_SWAP = 'commit_swap'
    SECTION_CHANGE = 'section_change'


class BeatResponder:
    """Processes beat events and returns actions. Does not mutate morph state."""

    SECTION_CHANGE_THRESHOLD = 0.6
    SECTION_COOLDOWN = 480

    def __init__(self, role: RoleMapper) -> None:
        self._role = role
        self._section_change_pending = False
        self._section_warmup = 0
        self._section_cooldown = 0
        self._recent_downbeat_energy = 0.5
        self._break_damping = 1.0
        self._last_downbeat_time = 0.0
        # Per-event data passed back with actions
        self._last_kick_energy = 0.0
        self._last_morph_nudge = 0.0

    @property
    def break_damping(self) -> float:
        return self._break_damping

    @break_damping.setter
    def break_damping(self, value: float) -> None:
        self._break_damping = value

    @property
    def section_change_pending(self) -> bool:
        return self._section_change_pending

    @section_change_pending.setter
    def section_change_pending(self, value: bool) -> None:
        self._section_change_pending = value

    @property
    def section_warmup(self) -> int:
        return self._section_warmup

    @section_warmup.setter
    def section_warmup(self, value: int) -> None:
        self._section_warmup = value

    @property
    def section_cooldown(self) -> int:
        return self._section_cooldown

    @section_cooldown.setter
    def section_cooldown(self, value: int) -> None:
        self._section_cooldown = value

    @property
    def recent_downbeat_energy(self) -> float:
        return self._recent_downbeat_energy

    @recent_downbeat_energy.setter
    def recent_downbeat_energy(self, value: float) -> None:
        self._recent_downbeat_energy = value

    @property
    def last_downbeat_time(self) -> float:
        return self._last_downbeat_time

    @last_downbeat_time.setter
    def last_downbeat_time(self, value: float) -> None:
        self._last_downbeat_time = value

    @property
    def last_morph_nudge(self) -> float:
        """Amount to nudge morph_t from the last downbeat processing."""
        return self._last_morph_nudge

    def tick(self, audio: AudioState) -> None:
        """Per-frame updates: section detection, break damping, counters."""
        self._section_warmup += 1
        self._section_cooldown += 1

        # Section change detection
        WARMUP_FRAMES = 1800
        if (
            audio.section_change > self.SECTION_CHANGE_THRESHOLD
            and self._section_warmup > WARMUP_FRAMES
            and self._section_cooldown > self.SECTION_COOLDOWN
        ):
            if not self._section_change_pending:
                self._section_change_pending = True
                log.info(
                    f"[section] change detected ({audio.section_change:.3f}), "
                    f"will swap loop on next strong beat"
                )

        # Break damping
        if audio.break_intensity > 0:
            self._break_damping *= cfg.genome.break_decay
        else:
            self._break_damping = min(1.0, self._break_damping / cfg.genome.break_decay)

    def process_kick(self, event: BeatEvent, morph_in_swap_ready: bool,
                     morph_in_dwell: bool) -> BeatAction | None:
        """Process a kick event. Returns an action or None."""
        self._last_kick_energy = event.energy
        if morph_in_swap_ready:
            return BeatAction.COMMIT_SWAP
        elif morph_in_dwell:
            log.debug(f'[kick] dwell released, energy={event.energy:.2f}')
            return BeatAction.RELEASE_DWELL
        return None

    def process_downbeat(self, event: BeatEvent, clock: float,
                         audio: AudioState, has_lib: bool) -> BeatAction | None:
        """Process a downbeat event. Returns an action or None."""
        self._last_downbeat_time = clock
        self._recent_downbeat_energy = self._recent_downbeat_energy * 0.8 + event.energy * 0.2

        backbeat_density = self._role.band_state(audio, BACKBEAT).onset_density
        density_scale = 1.0 / (1.0 + backbeat_density * cfg.genome.density_damping)

        # Section change pending → consume
        if self._section_change_pending and has_lib:
            self._section_change_pending = False
            self._section_cooldown = 0
            return BeatAction.SECTION_CHANGE

        bpm = max(audio.effective_bpm, 60.0)
        kick = event.energy * density_scale

        # Morph nudge + rotation boost (rotation handled by caller)
        self._last_morph_nudge = kick * 0.05
        log.debug(f"[backbeat] energy={event.energy:.2f}")
        return None

    def reset_for_song(self, clock: float) -> None:
        """Reset state for a new song."""
        self._last_downbeat_time = clock
        self._recent_downbeat_energy = 0.5
        self._break_damping = 1.0
        self._section_change_pending = False
        self._section_warmup = 0
