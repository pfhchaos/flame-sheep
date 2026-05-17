"""Loop player — loop sequence, graph walk, prefetch, and loop progression.

Owns the loop lifecycle: loading, stepping, cycle counting, history tracking,
graph-based next-loop selection, and the loop exit probability model.
Also handles genome prefetch for non-loop (graph walk) mode.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from flame_sheep.config import cfg
from flame_sheep.loops import loop_sequence, cycle_length
from flame_sheep.genome import Genome

if TYPE_CHECKING:
    from flame_sheep.storage import Library

log = logging.getLogger(__name__)


class LoopPlayer:
    """Manages loop playback, graph walk, and genome prefetch."""

    LOOP_HISTORY_SIZE = 8
    MIN_EXIT_CYCLES = 3  # Minimum full cycles before exit probability kicks in

    def __init__(self, lib: Library | None,
                 genome_factory, current_genome: Genome,
                 rng: np.random.Generator) -> None:
        self._lib = lib
        self._genome_factory = genome_factory
        self._rng = rng

        # Loop state
        self._loop_genomes: list[Genome] = []
        self._loop_structure: str = 'cyclic'
        self._loop_sequence = loop_sequence([], 'cyclic')
        self._loop_step: int = 0
        self._loop_cycle_len: int = 0
        self.active_loop_id: int | None = None
        self._loop_history: deque[int] = deque(maxlen=self.LOOP_HISTORY_SIZE)
        self._cycle_count: int = 0

        # Prefetch (for non-loop graph walk mode)
        self._next_genome: Genome | None = None
        self._genome_lock = threading.Lock()

    @property
    def has_loop(self) -> bool:
        return len(self._loop_genomes) > 0

    @property
    def loop_genomes(self) -> list[Genome]:
        return self._loop_genomes

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    def load_loop(self, loop_id: int, current_genome: Genome) -> Genome:
        """Load a specific loop. Returns the target genome to morph into."""
        return self._start_loop(loop_id, current_genome)

    def next_in_sequence(self) -> Genome:
        """Advance loop sequence, return next genome. Tracks step + cycles."""
        genome = next(self._loop_sequence)
        self._loop_step += 1
        if self._loop_step >= self._loop_cycle_len:
            self._loop_step = 0
            self._cycle_count += 1
            log.debug(f"[loop] cycle {self._cycle_count} complete "
                      f"({self._loop_structure}, {len(self._loop_genomes)} genomes)")
        return genome

    def next_loop(self, current_genome: Genome,
                  exclude_loop: int | None = None) -> Genome | None:
        """Pick next loop via graph, return entry genome. Returns None if no loops."""
        if self._lib is None or self._lib.loop_count() < 1:
            return None

        ref_id = current_genome.db_id

        if ref_id is not None:
            candidates = self._lib.nearest_loop_transitions(
                ref_id, exclude_loop=exclude_loop or self.active_loop_id,
                exclude_loops=set(self._loop_history), n=3)
            if candidates:
                weights = np.array([max(f, 0.01) / (1.0 + d)
                                    for _, f, d, _ in candidates])
                weights /= weights.sum()
                idx = self._rng.choice(len(candidates), p=weights)
                lid, fitness, dist, entry_gid = candidates[idx]
                log.info(f'[next_loop] -> #{lid} (fitness={fitness:.3f}, dist={dist:.3f})')
                self._loop_history.append(lid)
                return self._start_loop(lid, current_genome, entry_genome_id=entry_gid)

        # Fallback: weighted random from top loops
        top = self._lib.top_loops(n=20)
        candidates = [(lid, info) for lid, info in top
                      if lid not in self._loop_history and lid != self.active_loop_id]
        if not candidates:
            candidates = [(lid, info) for lid, info in top
                          if lid != self.active_loop_id]
        if not candidates:
            candidates = top
        fitnesses = np.array([max(info['fitness'], 0.01) for _, info in candidates])
        weights = fitnesses / fitnesses.sum()
        idx = self._rng.choice(len(candidates), p=weights)
        lid, info = candidates[idx]
        log.info(f'[next_loop] fallback -> #{lid} (fitness={info["fitness"]:.3f})')
        self._loop_history.append(lid)
        return self._start_loop(lid, current_genome)

    def graph_walk(self, current_genome: Genome) -> Genome:
        """Idle/energy mode graph walk. Returns next target genome."""
        current_id = current_genome.db_id
        if current_id is not None and self._lib is not None:
            neighbors = self._lib.nearest_transitions(current_id, n=10)
            if neighbors:
                scored = []
                for gid, dist in neighbors:
                    try:
                        g = self._lib.load_genome(gid)
                        scores = self._lib.genome_scores(gid)
                        cnn = scores.get('cnn_score') or scores.get('coverage', 0.5)
                        score = max(cnn, 0.01) / (1.0 + dist)
                        scored.append((g, score))
                    except (KeyError, Exception):
                        continue

                if scored:
                    weights = np.array([s for _, s in scored])
                    weights /= weights.sum()
                    idx = self._rng.choice(len(scored), p=weights)
                    target = scored[idx][0]
                    log.debug(f'[graph walk] -> genome #{target.db_id}')
                    return target

        # Fallback: random genome
        target = self._genome_factory()
        log.debug('[graph walk] fallback to random')
        return target

    def swap_next(self, current_genome: Genome) -> Genome:
        """Get next genome — from loop sequence or prefetch/factory."""
        if self._loop_genomes:
            return self.next_in_sequence()
        else:
            with self._genome_lock:
                if self._next_genome is not None:
                    target = self._next_genome
                    self._next_genome = None
                else:
                    target = self._genome_factory()
            self._prefetch_genome(current_genome)
            from flame_sheep.axes.genome_axis import _describe_genome
            log.debug(f"[genome] {_describe_genome(target)}")
            return target

    def should_exit_loop(self) -> bool:
        """Loop progression: after MIN_EXIT_CYCLES, increasing exit probability.

        Exit probability at genome swap i within a cycle is i / loop_length.
        This biases toward exiting near the end of a cycle (completing the
        phrase) while still allowing mid-cycle exits to avoid hub-node ruts
        in graph traversal.
        """
        if not self._loop_genomes:
            return False
        if self._cycle_count < self.MIN_EXIT_CYCLES:
            return False
        n = len(self._loop_genomes)
        if n <= 1:
            return True
        # Position within current cycle (0-indexed)
        pos = self._loop_step
        exit_prob = pos / n
        if self._rng.random() < exit_prob:
            log.info(f"[loop] exiting after {self._cycle_count} cycles "
                     f"(pos={pos}/{n}, prob={exit_prob:.2f})")
            return True
        return False

    def _load_top_loop(self, current_genome: Genome) -> Genome | None:
        """Load the best available loop at startup."""
        if self._lib is None or self._lib.loop_count() < 1:
            self._loop_genomes = []
            self.active_loop_id = None
            self._prefetch_genome(current_genome)
            return None
        return self.next_loop(current_genome)

    def _start_loop(self, loop_id: int, current_genome: Genome,
                    entry_genome_id: int | None = None) -> Genome:
        """Load loop and return the entry genome as target."""
        items = self._lib.load_loop(loop_id)
        self._loop_genomes = [genome for _, genome, _ in items]
        self._loop_structure = self._lib.loop_type(loop_id)
        n = len(self._loop_genomes)

        self._loop_sequence = loop_sequence(self._loop_genomes, self._loop_structure)
        self._loop_cycle_len = cycle_length(n, self._loop_structure)

        # Pick start position: use graph-provided entry genome, or find closest
        if entry_genome_id is not None:
            start = next((i for i, g in enumerate(self._loop_genomes)
                          if g.db_id == entry_genome_id), 0)
        else:
            start = self._best_entry_position(self._loop_genomes, current_genome)

        for _ in range(start):
            next(self._loop_sequence)
        self._loop_step = start

        self.active_loop_id = loop_id
        self._cycle_count = 0

        target = self._loop_genomes[start]
        dist = current_genome.distance(target)
        target_id = target.db_id
        log.info(f"[loop] loaded #{loop_id} ({self._loop_structure}, "
                 f"{n} genomes, start={start}, target=#{target_id}, entry_dist={dist:.3f})")
        for i, g in enumerate(self._loop_genomes):
            from flame_sheep.axes.genome_axis import _describe_genome
            marker = " <--" if i == start else ""
            log.debug(f"  [{i}] {_describe_genome(g)}{marker}")

        return target

    def _best_entry_position(self, loop_genomes: list, current_genome: Genome) -> int:
        """Pick the loop genome closest to the current genome."""
        if current_genome is None or len(loop_genomes) <= 1:
            return int(self._rng.integers(0, max(1, len(loop_genomes))))

        best_idx = 0
        best_dist = float('inf')
        for i, g in enumerate(loop_genomes):
            d = current_genome.distance(g)
            if d < best_dist:
                best_dist = d
                best_idx = i

        return best_idx

    def _prefetch_genome(self, current_genome: Genome) -> None:
        """Background-prefetch next genome for non-loop mode."""
        current_snapshot = current_genome
        factory = self._genome_factory
        min_dist = cfg.drift.min_genome_distance

        def _gen():
            for _ in range(20):
                g = factory()
                if g.distance(current_snapshot) >= min_dist:
                    break
            with self._genome_lock:
                self._next_genome = g

        threading.Thread(target=_gen, daemon=True).start()
