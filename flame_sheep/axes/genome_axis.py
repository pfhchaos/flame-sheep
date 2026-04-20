"""Genome axis — kick events drive genome morphing and swapping."""

import logging
import threading
import numpy as np

log = logging.getLogger(__name__)

from flame_sheep.audio._types import BeatEvent
from flame_sheep.genome import Genome


class GenomeAxis:
    """Kick → genome morph/swap. Manages genome lifecycle, loops, prefetch.

    State:
        current_genome, target_genome: endpoints of the current morph
        morph_t: 0..1 progress through current morph
        morph_speed: how fast morph_t advances per frame
    """

    MIN_GENOME_DISTANCE = 0.15
    DRIFT_MORPH_SPEED   = 0.003
    KICK_SWAP_EVERY     = 4
    KICK_MORPH_PULSE    = 0.03

    def __init__(self, genome_factory, lib=None, rng=None):
        self.enabled = True
        self._genome_factory = genome_factory
        self._lib = lib
        self.rng = rng or np.random.default_rng()

        # Morph state
        self.current_genome = self._genome_factory()
        self.target_genome  = self._genome_factory()
        self.morph_t        = 0.0
        self.morph_speed    = self.DRIFT_MORPH_SPEED

        # Prefetch
        self._next_genome: Genome | None = None
        self._genome_lock = threading.Lock()

        # Kick counting
        self._kick_count = 0
        self._recent_kick_energy = 0.5

        # Loop playback
        self._loop_genomes: list[Genome] = []
        self._loop_pos: int = 0
        self.active_loop_id: int | None = None

        # Walker reset flag (consumed by renderer)
        self.needs_walker_reset = False

        # Last beat times for interval tracking
        self._last_kick_time = 0.0

        if lib is not None and lib.loop_count() > 0:
            self._load_top_loop()
        else:
            self._prefetch_genome()

    def tick(self, events: list[BeatEvent], rms: float,
             dt: float, clock: float) -> None:
        # Handle kick events
        for event in events:
            if event.kind == 'kick':
                self._handle_kick(event, clock)

        # Advance morph
        self.morph_t = min(1.0, self.morph_t + self.morph_speed)

        if self.morph_t >= 1.0:
            self.current_genome = self.target_genome
            self.morph_t        = 0.0
            self._swap_next_genome()
            self.morph_speed    = self.DRIFT_MORPH_SPEED

        # Decay morph speed toward drift
        self.morph_speed = max(self.DRIFT_MORPH_SPEED, self.morph_speed * 0.98)

    def contribute(self, frame) -> None:
        frame.genome = self.current_genome.lerp(self.target_genome, self.morph_t)

    def _handle_kick(self, event: BeatEvent, clock: float):
        since = clock - self._last_kick_time
        self._last_kick_time = clock

        self._recent_kick_energy = (
            self._recent_kick_energy * 0.8 + event.energy * 0.2)

        if (event.energy > self._recent_kick_energy * 1.5
                and self._kick_count > 1):
            self._kick_count = 0

        self._kick_count += 1
        if self._kick_count >= self.KICK_SWAP_EVERY:
            self._kick_count = 0
            self.current_genome = self.current_genome.lerp(
                self.target_genome, self.morph_t)
            self._swap_next_genome()
            self.morph_t = 0.0
            kick_period = since if since < 2.0 else 0.5
            frames_until_swap = (kick_period * self.KICK_SWAP_EVERY * 60) * 0.8
            self.morph_speed = max(0.01, 1.0 / frames_until_swap)
            dist = self.current_genome.distance(self.target_genome)
            log.debug(f'[SWAP]  +{since:.3f}s  energy={event.energy:.2f}  '
                  f'dist={dist:.3f}  spd={self.morph_speed:.3f}')
        else:
            self.morph_speed = min(0.15,
                self.morph_speed + event.energy * self.KICK_MORPH_PULSE)
            log.debug(f'[kick]  +{since:.3f}s  energy={event.energy:.2f}  '
                  f'beat={self._kick_count}/{self.KICK_SWAP_EVERY}')

    def force_swap(self):
        """Immediately swap to a new genome."""
        self.current_genome = self.current_genome.lerp(
            self.target_genome, self.morph_t)
        self._swap_next_genome()
        self.morph_t     = 0.0
        self.morph_speed = 0.15
        self.needs_walker_reset = True
        log.info('force swap')

    # --- Loop management ---

    def _load_top_loop(self):
        top = self._lib.top_loops(n=1)
        if not top:
            self._loop_genomes = []
            self.active_loop_id = None
            self._prefetch_genome()
            return
        self._start_loop(top[0][0])

    def load_loop(self, loop_id: int):
        self._start_loop(loop_id)

    def _start_loop(self, loop_id: int):
        items = self._lib.load_loop(loop_id)
        self._loop_genomes = [genome for _, genome, _ in items]
        n = len(self._loop_genomes)
        start = int(self.rng.integers(0, n))
        self._loop_pos = (start + 1) % n
        self.active_loop_id = loop_id
        self.current_genome = self._loop_genomes[start]
        self.target_genome = self._loop_genomes[self._loop_pos]
        self.morph_t = 0.0
        self.morph_speed = 0.05
        self.needs_walker_reset = True
        log.info(f'[loop] loaded #{loop_id} ({n} genomes, start={start})')

    def next_loop(self):
        if self._lib is None or self._lib.loop_count() < 1:
            return
        top = self._lib.top_loops(n=10)
        for lid, _ in top:
            if lid != self.active_loop_id:
                self.load_loop(lid)
                return
        if top:
            self.load_loop(top[0][0])

    # --- Genome prefetch ---

    def _prefetch_genome(self):
        current_snapshot = self.current_genome
        factory = self._genome_factory
        def _gen():
            for _ in range(20):
                g = factory()
                if g.distance(current_snapshot) >= self.MIN_GENOME_DISTANCE:
                    break
            with self._genome_lock:
                self._next_genome = g
        threading.Thread(target=_gen, daemon=True).start()

    def _swap_next_genome(self):
        if self._loop_genomes:
            self._loop_pos = (self._loop_pos + 1) % len(self._loop_genomes)
            self.target_genome = self._loop_genomes[self._loop_pos]
        else:
            with self._genome_lock:
                if self._next_genome is not None:
                    self.target_genome = self._next_genome
                    self._next_genome  = None
                else:
                    self.target_genome = self._genome_factory()
            self._prefetch_genome()
