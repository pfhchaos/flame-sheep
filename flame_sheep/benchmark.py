"""GPU benchmarks for variations and library genomes.

CLI entry point: `python -m flame_sheep --benchmark-variations`. Runs two
phases:
  1. Per-variation: each variation solo with 3 transforms (isolates cost)
  2. Library genomes: actual genomes from the library (real-world cost)

Outputs a cost table with budget classification (ok/HEAVY/COSTLY).
"""

from __future__ import annotations

import logging
import time

import numpy as np

log = logging.getLogger(__name__)


def _run_variation_benchmark() -> None:
    """Benchmark variations and library genomes on the GPU."""
    import moderngl
    from .genome import (Genome, Variation, NUM_VARIATIONS)
    from .renderer import FlameRenderer, LIVE_ITER_MAX

    # Variation names for display
    var_names = {}
    for name in dir(Variation):
        if name.startswith('_'):
            continue
        val = getattr(Variation, name)
        if isinstance(val, int) and 0 <= val < NUM_VARIATIONS:
            var_names[val] = name

    ctx = moderngl.create_context(standalone=True)
    renderer = FlameRenderer(ctx, 1920, 1080)

    n_warmup = 5
    n_frames = 30
    rng = np.random.default_rng(42)

    def _bench_genome(g: Genome) -> float:
        """Returns ms/frame for a genome at LIVE_ITER_MAX (worst-case live render)."""
        renderer.upload_genome(g)
        for _ in range(n_warmup):
            renderer.clear_histogram(decay=0.3)
            renderer.dispatch_chaos_game(iterations=LIVE_ITER_MAX)
            ctx.finish()
        t0 = time.perf_counter()
        for _ in range(n_frames):
            renderer.clear_histogram(decay=0.3)
            renderer.dispatch_chaos_game(iterations=LIVE_ITER_MAX)
            ctx.finish()
        return (time.perf_counter() - t0) / n_frames * 1000

    # --- Phase 1: Per-variation ---
    print('\n=== Per-Variation Benchmark (3 transforms, solo) ===\n')
    results = []
    for var_idx in range(NUM_VARIATIONS):
        g = Genome.random(rng, n_transforms=3)
        for tr in g.transforms:
            tr.variations[:] = 0.0
            tr.variations[var_idx] = 1.0
            if var_idx in (Variation.JULIAN, Variation.JULIASCOPE):
                tr.var_params = {'julian_power': 3.0, 'julian_dist': 1.0}
            elif var_idx == Variation.SPLITS:
                tr.var_params = {'splits_x': 0.5, 'splits_y': 0.5}
            elif var_idx == Variation.CURL:
                tr.var_params = {'curl_c1': 0.5, 'curl_c2': 0.0}

        ms = _bench_genome(g)
        name = var_names.get(var_idx, f'var_{var_idx}')
        results.append((var_idx, name, ms))

    baseline = next(r[2] for r in results if r[0] == 0)  # LINEAR
    print(f'{"idx":>3}  {"variation":<20}  {"ms/frame":>9}  {"rel":>6}  {"budget"}')
    print('-' * 58)

    for idx, name, ms in sorted(results, key=lambda r: -r[2]):
        rel = ms / baseline if baseline > 0 else 0
        if rel > 4.0:
            budget = 'COSTLY'
        elif rel > 2.0:
            budget = 'HEAVY'
        else:
            budget = 'ok'
        print(f'{idx:3d}  {name:<20}  {ms:9.3f}  {rel:5.2f}x  {budget}')

    print(f'\nBaseline (linear): {baseline:.3f} ms/frame')
    print(f'Budget at 60fps: {16.7:.1f} ms/frame')

    # --- Phase 2: Library genomes ---
    from .storage import Library
    lib = Library()
    n_genomes = lib.genome_count()
    if n_genomes > 0:
        print(f'\n=== Library Genome Benchmark ({min(n_genomes, 50)} genomes) ===\n')
        top = lib.top_genomes(n=50)
        genome_results = []
        for gid, _score in top:
            g = lib.load_genome(gid)
            ms = _bench_genome(g)
            # Identify dominant variations
            dom_vars = []
            for tr in g.transforms:
                if tr.weight < 0.05:
                    continue
                for j, w in enumerate(tr.variations):
                    if w > 0.1:
                        dom_vars.append(var_names.get(j, f'v{j}'))
            genome_results.append((gid, ms, dom_vars))

        print(f'{"id":>5}  {"ms/frame":>9}  {"fps":>5}  {"budget":<8}  variations')
        print('-' * 70)

        for gid, ms, dom_vars in sorted(genome_results, key=lambda r: -r[1]):
            fps = 1000 / ms if ms > 0 else 999
            if ms > 16.7:
                budget = 'COSTLY'
            elif ms > 10.0:
                budget = 'HEAVY'
            else:
                budget = 'ok'
            vars_str = ', '.join(sorted(set(dom_vars)))[:40]
            print(f'{gid:5d}  {ms:9.3f}  {fps:5.1f}  {budget:<8}  {vars_str}')

        avg = np.mean([r[1] for r in genome_results])
        worst = max(r[1] for r in genome_results)
        best = min(r[1] for r in genome_results)
        costly = sum(1 for r in genome_results if r[1] > 16.7)
        print(f'\nAvg: {avg:.1f}ms  Best: {best:.1f}ms  Worst: {worst:.1f}ms')
        print(f'{costly}/{len(genome_results)} genomes over 60fps budget (16.7ms)')

    lib.close()
    ctx.release()
