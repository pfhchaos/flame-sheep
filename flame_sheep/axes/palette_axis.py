"""Palette axis — snare events drive palette graph traversal."""

import logging

log = logging.getLogger(__name__)


import numpy as np

from flame_sheep_audio import AudioState
from flame_sheep.genome import _lerp_arr
from flame_sheep.config import cfg


class PaletteAxis:
    """Snare -> palette walk. Energy controls jump distance."""

    PALETTE_HISTORY_SIZE = 8

    @property
    def DRIFT_MORPH_SPEED(self): return cfg.palette.drift_morph_speed
    @property
    def DENSITY_DAMPING(self): return cfg.palette.density_damping

    def __init__(self, initial_palette: np.ndarray, lib=None, rng=None):
        self.enabled = True
        self._lib = lib
        self._rng = rng or np.random.default_rng()
        self._current_palette_id: int | None = None
        self.palette_history: list[int] = []  # recent palette IDs for vote propagation

        self.palette_current = initial_palette.copy()
        self.palette_target  = initial_palette.copy()
        self.palette_t       = 0.0
        self.palette_speed   = self.DRIFT_MORPH_SPEED

    def tick(self, audio: AudioState, dt: float, clock: float) -> None:
        # Advance palette morph
        self.palette_t = min(1.0, self.palette_t + self.palette_speed)

        if self.palette_t >= 1.0:
            self.palette_current = self.palette_target.copy()
            self.palette_t       = 0.0
            self.palette_speed   = self.DRIFT_MORPH_SPEED

        # Handle snare events (scaled by density)
        snare_density = audio.bands['snare'].onset_density
        density_scale = 1.0 / (1.0 + snare_density * self.DENSITY_DAMPING)
        for event in audio.events:
            if event.kind == 'snare':
                self.palette_current = _lerp_arr(
                    self.palette_current, self.palette_target, self.palette_t)
                next_palette = self._pick_next_palette(event.energy)
                if next_palette is not None:
                    self.palette_target = next_palette
                self.palette_t     = 0.0
                self.palette_speed = 0.03 + event.energy * 0.1 * density_scale
                log.debug(f'[snare] energy={event.energy:.2f}')

        # Decay speed toward drift
        self.palette_speed = max(self.DRIFT_MORPH_SPEED, self.palette_speed * 0.98)

    def contribute(self, frame) -> None:
        frame.palette = _lerp_arr(
            self.palette_current, self.palette_target, self.palette_t)

    def _pick_next_palette(self, energy: float) -> np.ndarray | None:
        if self._lib is None or self._lib.palette_count() < 2:
            from flame_sheep.genome import _random_palette
            return _random_palette(self._rng)

        min_dist = 0.02 + energy * 0.13
        max_dist = 0.15 + energy * 0.45

        if self._current_palette_id is not None:
            neighbors = self._lib.palette_neighbors(
                self._current_palette_id, min_dist=min_dist, max_dist=max_dist,
                min_fitness=0.5)
            if neighbors:
                ids = [n[0] for n in neighbors]
                dists = np.array([n[1] for n in neighbors])
                # Weight by proximity, boosted by user votes
                vote_bonus = np.array([
                    max(0.1, 1.0 + self._lib.net_rating('palette', pid) * 0.3)
                    for pid in ids])
                weights = vote_bonus / (dists + 0.01)
                weights /= weights.sum()
                choice = int(self._rng.choice(ids, p=weights))
                self._current_palette_id = choice
                self._track_palette(choice)
                return self._lib.load_palette(choice)

        all_ids = self._lib.all_palette_ids()
        if all_ids:
            choice = int(self._rng.choice(all_ids))
            self._current_palette_id = choice
            self._track_palette(choice)
            return self._lib.load_palette(choice)

        from flame_sheep.genome import _random_palette
        return _random_palette(self._rng)

    def _track_palette(self, palette_id: int):
        """Add palette to recent history, deduplicating consecutive repeats."""
        if not self.palette_history or self.palette_history[-1] != palette_id:
            self.palette_history.append(palette_id)
            if len(self.palette_history) > self.PALETTE_HISTORY_SIZE:
                self.palette_history = self.palette_history[-self.PALETTE_HISTORY_SIZE:]
