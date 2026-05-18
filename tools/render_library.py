#!/usr/bin/env python3
"""Render library genomes with histogram export for domain-native scoring.

Produces .npz files alongside existing PNG renders in the DB.
Saves to a directory for training/benchmarking.

Usage:
    python tools/render_library.py --output ~/datasets/library-renders/
    python tools/render_library.py --output ~/datasets/library-renders/ --limit 100
"""
from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S')

    parser = argparse.ArgumentParser(description='Render library genomes with histogram export')
    parser.add_argument('--output', type=str, required=True, help='Output directory for .npz files')
    parser.add_argument('--size', type=int, default=512, help='Render resolution')
    parser.add_argument('--limit', type=int, default=None, help='Max genomes to render')
    parser.add_argument('--resume', action='store_true', help='Skip already rendered')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load library
    from flame_sheep.storage import Library
    lib = Library()

    # Get all genome IDs with renders
    rows = lib.conn.execute(
        '''SELECT genome_id FROM genome_blobs
           WHERE render_static IS NOT NULL
           ORDER BY genome_id'''
    ).fetchall()
    genome_ids = [r[0] for r in rows]

    if args.limit:
        genome_ids = genome_ids[:args.limit]

    if args.resume:
        before = len(genome_ids)
        genome_ids = [gid for gid in genome_ids
                      if not (output_dir / f'genome_{gid}_hist.npz').exists()]
        log.info('Resume: %d remaining (skipped %d)', len(genome_ids), before - len(genome_ids))

    if not genome_ids:
        log.info('Nothing to render')
        return

    log.info('Rendering %d genomes to %s', len(genome_ids), output_dir)

    # Create renderer
    from flame_sheep.renderer import FlameRenderer, N_ITERS
    import moderngl
    ctx = moderngl.create_context(standalone=True, backend='egl')
    renderer = FlameRenderer(ctx, args.size, args.size)

    from flame_sheep_audio import N_BINS
    from flame_sheep.scoring_channels import save_raw_histograms

    LIVE_MAX_ITERS = 500  # match detail axis max
    n_dispatches = max(1, LIVE_MAX_ITERS // N_ITERS)
    remainder = LIVE_MAX_ITERS - n_dispatches * N_ITERS
    swept_steps = 36

    rendered = 0
    failed = 0
    t0 = time.time()

    for gid in genome_ids:
        try:
            genome = lib.load_genome(gid)

            # --- Static render with first-hit snapshots ---
            renderer.upload_genome(genome)
            renderer.reset_walkers()
            renderer.upload_audio(np.zeros(N_BINS, dtype=np.float32))
            renderer.clear_histogram()

            # 16 snapshots across the live iteration range (100-500)
            min_iters = 100
            max_iters = LIVE_MAX_ITERS
            n_snapshots = 16
            snap_iters = np.linspace(min_iters, max_iters, n_snapshots, dtype=int)
            first_hit = None
            total_dispatched = 0

            for snap_idx, target_iter in enumerate(snap_iters):
                while total_dispatched < target_iter:
                    dispatch_count = min(N_ITERS, target_iter - total_dispatched)
                    renderer.dispatch_chaos_game(iterations=dispatch_count)
                    ctx.memory_barrier()
                    total_dispatched += dispatch_count
                hits, _ = renderer.histogram_data()
                if first_hit is None:
                    first_hit = np.full(hits.shape, 255, dtype=np.uint8)
                mapped_idx = snap_idx * 255 // max(n_snapshots - 1, 1)
                newly_hit = (hits > 0) & (first_hit == 255)
                first_hit[newly_hit] = mapped_idx

            static_hits, static_colors = renderer.histogram_data()

            # --- Swept render ---
            renderer.upload_genome(genome)
            renderer.reset_walkers()
            renderer.upload_audio(np.zeros(N_BINS, dtype=np.float32))
            renderer.clear_histogram()

            base_rotation = genome.rotation
            frames_per_step = max(1, n_dispatches // swept_steps)
            for i in range(swept_steps):
                angle = base_rotation + (2.0 * math.pi * i / swept_steps)
                renderer.set_rotation(angle)
                for _ in range(frames_per_step):
                    renderer.dispatch_chaos_game(iterations=N_ITERS)
                    ctx.memory_barrier()
            renderer.set_rotation(base_rotation)

            swept_hits, _ = renderer.histogram_data()

            # --- Save ---
            hist_path = output_dir / f'genome_{gid}_hist.npz'
            save_raw_histograms(str(hist_path), static_hits, static_colors,
                                swept_hits, first_hit=first_hit)

            rendered += 1
            if rendered % 50 == 0:
                elapsed = time.time() - t0
                rate = rendered / elapsed
                eta = (len(genome_ids) - rendered) / rate
                log.info('Rendered %d/%d (%.1f/s, ETA %.0fs)',
                         rendered, len(genome_ids), rate, eta)

        except Exception as e:
            log.warning('Failed genome #%d: %s', gid, e)
            failed += 1

    elapsed = time.time() - t0
    log.info('Done: %d rendered, %d failed, %.1fs (%.1f/s)',
             rendered, failed, elapsed, rendered / max(elapsed, 1))
    lib.close()


if __name__ == '__main__':
    main()
