"""Palette axis — snare events drive palette graph traversal."""

import numpy as np

from flame_sheep.audio._types import BeatEvent
from flame_sheep.genome import _lerp_arr


class PaletteAxis:
    """Snare → palette walk. Energy controls jump distance."""

    DRIFT_MORPH_SPEED = 0.003

    def __init__(self, initial_palette: np.ndarray, lib=None, rng=None):
        self.enabled = True
        self._lib = lib
        self._rng = rng or np.random.default_rng()
        self._current_palette_id: int | None = None

        self.palette_current = initial_palette.copy()
        self.palette_target  = initial_palette.copy()
        self.palette_t       = 0.0
        self.palette_speed   = self.DRIFT_MORPH_SPEED

    def tick(self, events: list[BeatEvent], rms: float,
             dt: float, clock: float) -> None:
        # Advance palette morph
        self.palette_t = min(1.0, self.palette_t + self.palette_speed)

        if self.palette_t >= 1.0:
            self.palette_current = self.palette_target.copy()
            self.palette_t       = 0.0
            self.palette_speed   = self.DRIFT_MORPH_SPEED

        # Handle snare events
        for event in events:
            if event.kind == 'snare':
                self.palette_current = _lerp_arr(
                    self.palette_current, self.palette_target, self.palette_t)
                next_palette = self._pick_next_palette(event.energy)
                if next_palette is not None:
                    self.palette_target = next_palette
                self.palette_t     = 0.0
                self.palette_speed = 0.03 + event.energy * 0.1
                print(f'[snare] energy={event.energy:.2f}')

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
                weights = 1.0 / (dists + 0.01)
                weights /= weights.sum()
                choice = int(self._rng.choice(ids, p=weights))
                self._current_palette_id = choice
                return self._lib.load_palette(choice)

        all_ids = self._lib.all_palette_ids()
        if all_ids:
            choice = int(self._rng.choice(all_ids))
            self._current_palette_id = choice
            return self._lib.load_palette(choice)

        from flame_sheep.genome import _random_palette
        return _random_palette(self._rng)
