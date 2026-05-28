"""Loop composition — assemble genomes into coherent visual cycles.

A loop is an ordered sequence of genomes where morphing from each to the
next produces a consistent directional flow, and the last genome wraps
back to the first. The 3x3 motion field between adjacent genomes
characterises the visual motion of each transition; good loops have high
coherence between consecutive motion fields (the flow keeps going the
same way rather than jerking back and forth).

This package re-exports the public loop API. External code should import
from `flame_sheep.loops`, not from submodules directly — the submodule
layout is an implementation detail of Stage 4c of `docs/reorg_plan.md`.
"""

from __future__ import annotations

from .candidate import LoopCandidate
from .prune import prune_duplicate_loops, prune_low_fitness_loops, prune_loops
from .sequence import STRUCTURES, cycle_length, loop_sequence
from .compose import (
    compose_loops,
    compose_loops_graph,
    save_best_loops,
    # Private helpers re-exported for test access (test_loops.py)
    _build_one_loop,
    _too_similar,
)
from .evolve import (
    crossover,
    delete_genome,
    evolve_loops,
    explore_breed,
    insert_genome,
    jitter_loop,
    mutate_loop,
    mutate_structure,
    refine_loop,
)
from .scoring import score_loop

__all__ = [
    'LoopCandidate',
    'STRUCTURES',
    'compose_loops',
    'compose_loops_graph',
    'crossover',
    'cycle_length',
    'delete_genome',
    'evolve_loops',
    'explore_breed',
    'insert_genome',
    'jitter_loop',
    'loop_sequence',
    'mutate_loop',
    'mutate_structure',
    'prune_duplicate_loops',
    'prune_loops',
    'prune_low_fitness_loops',
    'refine_loop',
    'save_best_loops',
    'score_loop',
]
