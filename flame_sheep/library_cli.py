"""CLI handlers for library management: --generate-genomes, --compose-loops, etc.

Pure CLI entry point — no GL, no audio, no render loop. Dispatched from
main()'s argparse when any of the relevant flags are set.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

log = logging.getLogger(__name__)


def _run_library_commands(args: argparse.Namespace) -> None:
    """Handle --generate-genomes, --compose-loops, --evolve, --stats."""
    from .genome import Genome
    from .storage import Library
    from .loops import save_best_loops, evolve_loops

    lib = Library()

    if args.generate_genomes:
        n = args.generate_genomes
        print(f'Generating {n} genomes...')
        rng = np.random.default_rng()
        for i in range(n):
            g = Genome.random(rng)
            lib.save_genome(g)
            if (i + 1) % 10 == 0 or i == n - 1:
                print(f'  {i + 1}/{n}')
        print(f'Library now has {lib.genome_count()} genomes')

    if args.generate_palettes is not None:
        from .genome import _random_palette
        n = args.generate_palettes
        print(f'Generating {n} palettes...')
        rng = np.random.default_rng()
        for i in range(n):
            lib.save_palette(_random_palette(rng))
            if (i + 1) % 50 == 0 or i == n - 1:
                print(f'  {i + 1}/{n}')
        print(f'Library now has {lib.palette_count()} palettes')

    if args.compose_loops is not None:
        from .loops import compose_loops_graph
        n = args.compose_loops
        print(f'Composing loops (target: {n}, length: {args.loop_length})...')
        candidates = compose_loops_graph(
            lib, loop_length=args.loop_length, n_attempts=n * 10,
        )
        if candidates:
            ids = save_best_loops(lib, candidates, n_keep=n)
            print(f'Saved {len(ids)} loops')
            for lid in ids:
                scores = lib.loop_fitness(lid)
                print(f'  #{lid}: fitness={scores["fitness"]:.3f}, '
                      f'coherence={scores["mean_coherence"]:.3f}')
        else:
            print('No viable loops found — try generating more genomes')

    if args.evolve:
        n_loops = lib.loop_count()
        if n_loops < 2:
            print('Need at least 2 loops to evolve. Run --compose-loops first.')
        else:
            print(f'Evolving from {n_loops} loops...')
            top = evolve_loops(lib, n_generations=3, n_offspring=8,
                               pool_size=min(30, lib.genome_count()),
                               loop_length=args.loop_length)
            print(f'Top {len(top)} loops after evolution ({lib.loop_count()} total):')
            for lid in top:
                scores = lib.loop_fitness(lid)
                print(f'  #{lid}: fitness={scores["fitness"]:.3f}')

    if getattr(args, 'prune_loops', False):
        from .loops import prune_loops
        before = lib.loop_count()
        n = prune_loops(lib)
        print(f'Pruned {n} loops ({before} -> {lib.loop_count()})')

    if args.stats:
        n_genomes = lib.genome_count()
        n_loops = lib.loop_count()
        n_palettes = lib.palette_count()
        print(f'\nLibrary: {n_genomes} genomes, {n_loops} loops, {n_palettes} palettes')
        if n_loops > 0:
            top = lib.top_loops(n=5)
            print(f'Top loops:')
            for lid, scores in top:
                gids = lib.loop_genome_ids(lid)
                rating = lib.net_rating('loop', lid)
                print(f'  #{lid}: {len(gids)} genomes, '
                      f'fitness={scores["fitness"]:.3f}, '
                      f'smoothness={scores.get("smoothness", 0):.3f}, '
                      f'coherence={scores["mean_coherence"]:.3f}, '
                      f'diversity={scores["diversity"]:.3f}, '
                      f'votes={rating:+d}')

    lib.close()
