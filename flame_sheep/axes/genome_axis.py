"""Genome axis — kick/drop events drive genome morphing and swapping."""

import logging
import threading
from collections import deque
import numpy as np

log = logging.getLogger(__name__)

from flame_sheep.audio._types import AudioState, BeatEvent
from flame_sheep.genome import Genome


class GenomeAxis:
    """Kick -> genome morph/swap. Manages genome lifecycle, loops, prefetch.

    Responds to event types:
      - kick: count toward swap, pulse morph speed
      - drop: force swap + freeze morph for N bars
      - song_start: reset drop/kick state

    Continuous features:
      - percussiveness: scales morph speed (low = slow drift, high = snappy)
      - centroid_delta: triggers swaps in low-percussiveness mode

    State:
        current_genome, target_genome: endpoints of the current morph
        morph_t: 0..1 progress through current morph
        morph_speed: how fast morph_t advances per frame
    """

    MIN_GENOME_DISTANCE = 0.15
    DRIFT_MORPH_SPEED   = 0.003
    KICK_SWAP_EVERY     = 4
    KICK_MORPH_PULSE    = 0.03
    LOOP_HISTORY_SIZE   = 8

    # Centroid delta threshold for triggering a swap in low-percussiveness mode
    CENTROID_SWAP_THRESHOLD = 200.0

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
        self._loop_history: deque[int] = deque(maxlen=self.LOOP_HISTORY_SIZE)

        # Walker reset flag (consumed by renderer)
        self.needs_walker_reset = False

        # Timing
        self._last_kick_time = 0.0
        self._drop_freeze_remaining = 0.0

        if lib is not None and lib.loop_count() > 0:
            self._load_top_loop()
        else:
            self._prefetch_genome()

    def tick(self, audio: AudioState, dt: float, clock: float) -> None:
        # Handle discrete events
        for event in audio.events:
            if event.kind == 'kick':
                self._handle_kick(event, clock)
            elif event.kind == 'drop':
                self._handle_drop(event)
            elif event.kind == 'song_start':
                self._handle_song_start()

        # In low-percussiveness mode, a large centroid shift triggers a swap
        if (audio.percussiveness < 0.3
                and audio.centroid_delta > self.CENTROID_SWAP_THRESHOLD
                and self.morph_t > 0.3
                and self._drop_freeze_remaining <= 0):
            self.current_genome = self.current_genome.lerp(
                self.target_genome, self.morph_t)
            self._swap_next_genome()
            self.morph_t = 0.0
            self.morph_speed = 0.02
            log.debug(f'[centroid swap] delta={audio.centroid_delta:.0f}Hz '
                      f'perc={audio.percussiveness:.2f}')

        # Drop freeze: hold the genome, don't morph
        if self._drop_freeze_remaining > 0:
            self._drop_freeze_remaining -= dt
            return  # brightness/zoom still respond

        # Scale morph speed by percussiveness
        perc_scale = 0.2 + 0.8 * min(1.0, audio.percussiveness / 0.5)

        # Advance morph
        self.morph_t = min(1.0, self.morph_t + self.morph_speed * perc_scale)

        if self.morph_t >= 1.0:
            self.current_genome = self.target_genome
            self.morph_t        = 0.0
            self._swap_next_genome()
            self.morph_speed    = self.DRIFT_MORPH_SPEED

        # Decay morph speed toward drift
        self.morph_speed = max(self.DRIFT_MORPH_SPEED, self.morph_speed * 0.98)

    def contribute(self, frame) -> None:
        frame.genome = self.current_genome.lerp(self.target_genome, self.morph_t)

    # --- Event handlers ---

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

    def _handle_drop(self, event: BeatEvent):
        """Drop event: force swap + freeze. energy field = freeze duration."""
        self.current_genome = self.current_genome.lerp(
            self.target_genome, self.morph_t)
        self._swap_next_genome()
        self.morph_t = 0.0
        self._drop_freeze_remaining = event.energy  # encoded as freeze seconds
        self.needs_walker_reset = True
        log.info(f'[DROP] freeze {event.energy:.1f}s')

    def _handle_song_start(self):
        """Reset state for new song — swap loop + reset counters."""
        self._kick_count = 0
        self._last_kick_time = 0.0
        self._drop_freeze_remaining = 0.0
        # New song, new loop
        if self._lib is not None and self._lib.loop_count() > 1:
            self.next_loop()
            log.info(f'[song_start] switched to loop #{self.active_loop_id}')

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
        if self._lib is None or self._lib.loop_count() < 1:
            self._loop_genomes = []
            self.active_loop_id = None
            self._prefetch_genome()
            return
        self.next_loop()

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
        top = self._lib.top_loops(n=20)
        candidates = [(lid, info) for lid, info in top
                      if lid not in self._loop_history]
        if not candidates:
            candidates = [(lid, info) for lid, info in top
                          if lid != self.active_loop_id]
        if not candidates:
            candidates = top
        fitnesses = np.array([max(info['fitness'], 0.01)
                              for _, info in candidates])
        weights = fitnesses / fitnesses.sum()
        idx = self.rng.choice(len(candidates), p=weights)
        lid, info = candidates[idx]
        self._load_and_track(lid, info['fitness'])

    def _load_and_track(self, loop_id: int, fitness: float | None = None):
        self._loop_history.append(loop_id)
        self.load_loop(loop_id)
        if fitness is not None:
            log.info(f'[loop] fitness={fitness:.3f}')

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
