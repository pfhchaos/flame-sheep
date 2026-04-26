"""Drift mode — independent state machine for silence visualization.

When audio goes quiet, drift mode takes over genome morphing at its own
fixed pace, independent of the active-audio GenomeAxis.  FlameSheepCore
acts as compositor, switching which state machine drives FrameState.genome.
"""

import logging
import threading

import numpy as np

from flame_sheep_audio import AudioState
from flame_sheep.genome import Genome
from flame_sheep.config import cfg

log = logging.getLogger(__name__)


class DriftMode:
    """Self-contained genome morph engine for sustained silence.

    Owns its own current/target genome and morph state.  Never touches
    GenomeAxis internals — transitions are handled by the compositor
    via enter()/exit() handoff.
    """

    @property
    def RMS_THRESHOLD(self): return cfg.drift.rms_threshold
    @property
    def ENTER_FRAMES(self): return cfg.drift.enter_frames
    @property
    def MORPH_SPEED(self): return cfg.drift.morph_speed
    @property
    def MIN_GENOME_DISTANCE(self): return cfg.drift.min_genome_distance
    @property
    def CYCLES_PER_LOOP(self): return cfg.drift.cycles_per_loop

    def __init__(self, genome_factory, lib=None, rng=None):
        self._genome_factory = genome_factory
        self._lib = lib
        self.rng = rng or np.random.default_rng()

        self.active = False
        self._quiet_frames = 0

        # Own morph state (populated on enter())
        self.current_genome: Genome | None = None
        self.target_genome: Genome | None = None
        self.morph_t = 0.0

        # Loop state (copied from genome_axis on enter())
        self._loop_genomes: list[Genome] = []
        self._loop_pos = 0
        self._loop_cycles = 0
        self.active_loop_id: int | None = None
        self._loop_history: list[int] = []

        # Prefetch
        self._next_genome: Genome | None = None
        self._genome_lock = threading.Lock()

    def tick(self, audio: AudioState, dt: float, clock: float) -> None:
        """Monitor RMS for activation; advance morph when active."""
        if audio.bands['subbass'].rms < self.RMS_THRESHOLD:
            self._quiet_frames += 1
            if self._quiet_frames >= self.ENTER_FRAMES and not self.active:
                self.active = True
        else:
            self._quiet_frames = 0
            if self.active:
                self.active = False

        if not self.active:
            return

        # Fixed-speed morph — no perc_scale
        self.morph_t = min(1.0, self.morph_t + self.MORPH_SPEED)

        if self.morph_t >= 1.0:
            self.current_genome = self.target_genome
            self.morph_t = 0.0
            self._swap_next_genome()

            # Track loop cycles for loop switching
            if (self._loop_genomes
                    and self._loop_pos == 0
                    and self._lib is not None):
                self._loop_cycles += 1
                if self._loop_cycles >= self.CYCLES_PER_LOOP:
                    self._loop_cycles = 0
                    self._next_loop()

    def contribute(self, frame) -> None:
        """Write interpolated genome to frame."""
        frame.genome = self.current_genome.lerp(self.target_genome, self.morph_t)

    def enter(self, genome_axis) -> None:
        """Snapshot genome state from genome_axis — no visual pop."""
        self.current_genome = genome_axis.current_genome.lerp(
            genome_axis.target_genome, genome_axis.morph_t)
        self.morph_t = 0.0
        self._loop_cycles = 0

        # Copy loop state
        self._loop_genomes = list(genome_axis._loop_genomes)
        self._loop_pos = genome_axis._loop_pos
        self.active_loop_id = genome_axis.active_loop_id

        # Pick next target
        self._swap_next_genome()
        log.info('[drift] entered')

    def exit(self) -> Genome:
        """Return exact visual genome for handoff back to active mode."""
        genome = self.current_genome.lerp(self.target_genome, self.morph_t)
        log.info('[drift] exited')
        return genome

    # --- Internal genome management ---

    def _swap_next_genome(self):
        if self._loop_genomes:
            self._loop_pos = (self._loop_pos + 1) % len(self._loop_genomes)
            self.target_genome = self._loop_genomes[self._loop_pos]
        else:
            with self._genome_lock:
                if self._next_genome is not None:
                    self.target_genome = self._next_genome
                    self._next_genome = None
                else:
                    self.target_genome = self._genome_factory()
            self._prefetch_genome()
        dist = self.current_genome.distance(self.target_genome)
        log.debug(f'[drift] swap  dist={dist:.3f}')

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

    def _next_loop(self):
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
        self._loop_history.append(lid)

        # Load loop genomes
        items = self._lib.load_loop(lid)
        self._loop_genomes = [genome for _, genome, _ in items]
        n = len(self._loop_genomes)
        start = int(self.rng.integers(0, n))
        self._loop_pos = (start + 1) % n
        self.active_loop_id = lid
        self.current_genome = self._loop_genomes[start]
        self.target_genome = self._loop_genomes[self._loop_pos]
        self.morph_t = 0.0
        log.info(f'[drift] switched to loop #{lid} '
                 f'(fitness={info["fitness"]:.3f}, {n} genomes)')
