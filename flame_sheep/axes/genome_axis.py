"""Genome axis — kick/drop events drive genome morphing and swapping."""

import logging
import threading
from collections import deque
import numpy as np

log = logging.getLogger(__name__)

from flame_sheep.audio._types import AudioState, BeatEvent
from flame_sheep.genome import Genome
from flame_sheep.variations import Variation


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
    KICK_MORPH_PULSE    = 0.03
    LOOP_HISTORY_SIZE   = 8
    BREAK_DECAY         = 0.97   # damping per frame during break (~half speed in 0.4s)
    DENSITY_MORPH_SCALE = 0.015  # morph_speed baseline += kick_density * this
    STRONG_BEAT_THRESHOLD = 1.5  # swap when energy > recent_avg * this

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

        # Symmetry scoring: set when a morph completes naturally
        self.score_ready = False

        # Kick energy tracking (for strong beat detection)
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
        self._break_damping = 1.0

        if lib is not None and lib.loop_count() > 0:
            self._load_top_loop()
        else:
            self._prefetch_genome()

    def tick(self, audio: AudioState, dt: float, clock: float) -> None:
        # Handle discrete events
        for event in audio.events:
            if event.kind == 'kick':
                self._handle_kick(event, clock)
            elif event.kind == 'song_start':
                self._handle_song_start()

        # In low-percussiveness mode, a large centroid shift triggers a swap
        if (audio.percussiveness < 0.3
                and audio.centroid_delta > self.CENTROID_SWAP_THRESHOLD
                and self.morph_t > 0.3):
            self.current_genome = self.current_genome.lerp(
                self.target_genome, self.morph_t)
            self._swap_next_genome()
            self.morph_t = 0.0
            self.morph_speed = 0.02
            log.debug(f'[centroid swap] delta={audio.centroid_delta:.0f}Hz '
                      f'perc={audio.percussiveness:.2f}')

        # Break damping: exponential slowdown during breaks, symmetric recovery
        if audio.breaking:
            self._break_damping *= self.BREAK_DECAY
        else:
            self._break_damping = min(1.0, self._break_damping / self.BREAK_DECAY)

        # Scale morph speed by percussiveness and break damping
        perc_scale = 0.2 + 0.8 * min(1.0, audio.percussiveness / 0.5)

        # Advance morph
        self.morph_t = min(1.0,
            self.morph_t + self.morph_speed * perc_scale * self._break_damping)

        if self.morph_t >= 1.0:
            self.score_ready    = True
            self.current_genome = self.target_genome
            self.morph_t        = 0.0
            self._swap_next_genome()
            self.morph_speed    = self.DRIFT_MORPH_SPEED

        # Decay morph speed toward density-driven baseline
        density_speed = (self.DRIFT_MORPH_SPEED
                         + audio.onset_density.get('kick', 0) * self.DENSITY_MORPH_SCALE)
        self.morph_speed = max(density_speed, self.morph_speed * 0.98)

    def contribute(self, frame) -> None:
        frame.genome = self.current_genome.lerp(self.target_genome, self.morph_t)

    # --- Event handlers ---

    def _handle_kick(self, event: BeatEvent, clock: float):
        since = clock - self._last_kick_time
        self._last_kick_time = clock

        self._recent_kick_energy = (
            self._recent_kick_energy * 0.8 + event.energy * 0.2)

        # Strong beat → swap direction (downbeat detection)
        if event.energy > self._recent_kick_energy * self.STRONG_BEAT_THRESHOLD:
            self.current_genome = self.current_genome.lerp(
                self.target_genome, self.morph_t)
            self._swap_next_genome()
            self.morph_t = 0.0
            dist = self.current_genome.distance(self.target_genome)
            log.debug(f'[SWAP]  +{since:.3f}s  energy={event.energy:.2f}  '
                      f'avg={self._recent_kick_energy:.2f}  dist={dist:.3f}')
        else:
            # Normal kick — pulse morph speed
            self.morph_speed = min(0.15,
                self.morph_speed + event.energy * self.KICK_MORPH_PULSE)
            log.debug(f'[kick]  +{since:.3f}s  energy={event.energy:.2f}')

    def _handle_song_start(self):
        """Reset state for new song — swap loop + reset energy tracking."""
        self._last_kick_time = 0.0
        self._recent_kick_energy = 0.5
        self._break_damping = 1.0
        # New song, new loop
        if self._lib is not None and self._lib.loop_count() > 1:
            self.next_loop()
            log.info(f'[song_start] switched to loop #{self.active_loop_id}')

    def accept_handoff(self, genome: Genome, loop_id: int | None = None):
        """Receive genome from drift mode on transition back to active."""
        self.current_genome = genome
        self.target_genome = genome
        self.morph_t = 0.0
        self.morph_speed = self.DRIFT_MORPH_SPEED
        self._recent_kick_energy = 0.5
        self._break_damping = 1.0
        if loop_id is not None and loop_id != self.active_loop_id:
            self.load_loop(loop_id)
        self._prefetch_genome()

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
        log.info(f'[genome] {_describe_genome(self.target_genome)}')


# Reverse map: variation index -> name
_VAR_NAMES = {v: k.lower() for k, v in vars(Variation).items()
              if isinstance(v, int)}


def _describe_genome(g: Genome) -> str:
    """One-line summary: top variations per transform, sorted by weight."""
    parts = []
    for i, t in enumerate(sorted(g.transforms, key=lambda t: -t.weight)):
        if t.weight < 0.05:
            continue
        active = [(w, _VAR_NAMES.get(j, f'v{j}'))
                  for j, w in enumerate(t.variations) if w > 0.01]
        active.sort(key=lambda x: -x[0])
        names = '+'.join(n for _, n in active[:3])
        parts.append(f'{names}({t.weight:.2f})')
    return '  '.join(parts) if parts else '(empty)'
