"""GPU benchmarks for variations and library genomes.

CLI entry point: `python -m flame_sheep --benchmark-variations`. Runs two
phases:
  1. Per-variation: each variation solo with 3 transforms (isolates cost)
  2. Library genomes: actual genomes from the library (real-world cost)

Library-genome sampling mode is configurable via --bench-sample:
  - random (default): uniform sample from the library — what you want
    for measuring cost variance across the whole library.
  - top: highest-rated genomes (the original behavior). Biased toward
    well-formed genomes; tells you cost in the regime the wallpaper
    actually plays from.
  - stratified: equal-sized buckets across cnn_score percentiles, so
    cost-vs-score correlation is visible.

Outputs a cost table with budget classification (ok/HEAVY/COSTLY) plus
distribution stats (min/median/p90/p99/max + per-quartile breakdown if
stratified).
"""

from __future__ import annotations

import logging
import time

import numpy as np

log = logging.getLogger(__name__)


def _run_variation_benchmark(sample_mode: str = 'random',
                             n_genomes: int = 50) -> None:
    """Benchmark variations and library genomes on the GPU."""
    from ..genome import (Genome, Variation, NUM_VARIATIONS,
                           genome_to_chaos_kwargs)
    from ..render_params import LIVE_ITER_MAX
    from viz_authoring.vk.context import VkContext
    from viz_authoring.vk.headless import HeadlessVkRenderer

    # Variation names for display
    var_names = {}
    for name in dir(Variation):
        if name.startswith('_'):
            continue
        val = getattr(Variation, name)
        if isinstance(val, int) and 0 <= val < NUM_VARIATIONS:
            var_names[val] = name

    ctx = VkContext(instance_extensions=[])
    ctx.select_device()
    renderer = HeadlessVkRenderer(ctx, 1920, 1080)

    n_warmup = 5
    n_frames = 30
    rng = np.random.default_rng(42)

    def _bench_genome(g: Genome) -> float:
        """Returns ms/frame for a genome at LIVE_ITER_MAX (worst-case live render)."""
        renderer.set_genome(**genome_to_chaos_kwargs(g))
        # Vk's render_frame() is fence-synced; each call returns after
        # GPU work completes. No explicit ctx.finish() needed.
        for _ in range(n_warmup):
            renderer.clear_histogram(decay=0.3)
            renderer.dispatch_chaos_game(iterations=LIVE_ITER_MAX)
        t0 = time.perf_counter()
        for _ in range(n_frames):
            renderer.clear_histogram(decay=0.3)
            renderer.dispatch_chaos_game(iterations=LIVE_ITER_MAX)
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
    from ..storage import Library
    lib = Library()
    if lib.genome_count() > 0:
        if sample_mode == 'top':
            sample = lib.top_genomes(n=n_genomes)
        elif sample_mode == 'stratified':
            sample = lib.stratified_genomes(n=n_genomes)
        else:
            sample = lib.random_genomes(n=n_genomes)
        print(f'\n=== Library Genome Benchmark '
              f'({len(sample)} genomes, sample={sample_mode}) ===\n')
        genome_results = []
        for gid, meta in sample:
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
            genome_results.append((gid, ms, dom_vars, meta.get('cnn_score')))

        print(f'{"id":>5}  {"score":>6}  {"ms/frame":>9}  {"fps":>5}  '
              f'{"budget":<8}  variations')
        print('-' * 80)

        for gid, ms, dom_vars, score in sorted(genome_results, key=lambda r: -r[1]):
            fps = 1000 / ms if ms > 0 else 999
            if ms > 16.7:
                budget = 'COSTLY'
            elif ms > 10.0:
                budget = 'HEAVY'
            else:
                budget = 'ok'
            vars_str = ', '.join(sorted(set(dom_vars)))[:40]
            score_str = f'{score:6.3f}' if score is not None else '   n/a'
            print(f'{gid:5d}  {score_str}  {ms:9.3f}  {fps:5.1f}  '
                  f'{budget:<8}  {vars_str}')

        # Distribution: tells us at a glance whether cost varies enough
        # to warrant cost-aware iteration budgeting in compare-mode.
        msec = np.array([r[1] for r in genome_results])
        p = np.percentile(msec, [50, 90, 99])
        print(f'\nmin={msec.min():.1f}ms  median={p[0]:.1f}ms  '
              f'p90={p[1]:.1f}ms  p99={p[2]:.1f}ms  max={msec.max():.1f}ms  '
              f'spread={msec.max()/max(msec.min(), 0.01):.1f}x')
        costly = sum(1 for v in msec if v > 16.7)
        print(f'{costly}/{len(msec)} genomes over 60fps budget (16.7ms)')

        # Cost-vs-score correlation (only meaningful if we have scores)
        scored = [(r[1], r[3]) for r in genome_results if r[3] is not None]
        if len(scored) >= 8:
            ms_arr = np.array([s[0] for s in scored])
            sc_arr = np.array([s[1] for s in scored])
            r = float(np.corrcoef(ms_arr, sc_arr)[0, 1])
            print(f'Cost vs cnn_score correlation: r={r:+.3f} '
                  f'(positive = expensive genomes score higher)')

    lib.close()
    renderer.cleanup()
    ctx.cleanup()
