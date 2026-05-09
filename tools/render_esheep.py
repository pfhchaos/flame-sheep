#!/usr/bin/env python3
"""Batch render Electric Sheep genomes for CNN training.

Renders each sheep genome to two images:
  - static: single-frame render (what the fractal looks like still)
  - swept:  rotation-accumulated render (what the viewer actually sees)

Output structure:
  output_dir/
    gen247_52506_r192_static.png
    gen247_52506_r192_swept.png
    manifest.csv   (generation, sheep_id, rating, static_path, swept_path)

Usage:
    python tools/render_esheep.py --output ~/datasets/esheep-renders/
    python tools/render_esheep.py --output ~/datasets/esheep-renders/ --size 256 --min-rating 10
    python tools/render_esheep.py --output ~/datasets/esheep-renders/ --generation 247
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger(__name__)

DEFAULT_SIZE = 256
DEFAULT_FRAMES = 30        # frames to accumulate for static render
DEFAULT_SWEPT_STEPS = 36   # 36 steps × 10° = full rotation


def create_renderer(size: int):
    """Create a headless GPU renderer."""
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    import moderngl
    ctx = moderngl.create_context(standalone=True, backend='egl')
    log.info('GPU: %s', ctx.info['GL_RENDERER'])

    from flame_sheep.renderer import FlameRenderer
    renderer = FlameRenderer(ctx, size, size)
    return ctx, renderer


def _auto_zoom(genome) -> None:
    """Auto-fit the genome's zoom/center to frame the attractor.

    Runs a quick CPU chaos game to find where points land, then adjusts
    zoom and center so the attractor fills the viewport.
    """
    from flame_sheep.variations._cpu import apply_variations_cpu
    rng = np.random.default_rng(42)
    x, y = 0.0, 0.0
    weights = np.array([t.weight for t in genome.transforms], dtype=np.float64)
    weights /= weights.sum()
    cumw = np.cumsum(weights)

    xs, ys = [], []
    for i in range(5000):
        r = rng.random()
        tidx = min(int(np.searchsorted(cumw, r)), len(genome.transforms) - 1)
        tr = genome.transforms[tidx]
        a, b, c, d, e, f = tr.affine
        nx, ny = a * x + b * y + c, d * x + e * y + f
        nx, ny = apply_variations_cpu(tr.variations, nx, ny, tr.affine)
        if tr.post_affine is not None:
            pa, pb, pc, pd, pe, pf = tr.post_affine
            nx, ny = pa * nx + pb * ny + pc, pd * nx + pe * ny + pf
        x, y = nx, ny
        if not (np.isfinite(x) and np.isfinite(y)):
            x, y = 0.0, 0.0
            continue
        if i > 50 and abs(x) < 1e6 and abs(y) < 1e6:
            xs.append(x)
            ys.append(y)

    if len(xs) < 100:
        return  # degenerate, leave zoom as-is

    xs, ys = np.array(xs), np.array(ys)
    cx, cy = xs.mean(), ys.mean()
    # Use 95th percentile radius for framing (ignore outliers)
    dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    radius = max(np.percentile(dists, 95), 0.01)
    # zoom maps (-1,1) to viewport, so zoom = 1/radius with padding
    genome.center = np.array([cx, cy], dtype=np.float32)
    genome.zoom = 0.9 / radius  # 0.9 = 10% padding


def render_genome(genome, renderer, ctx,
                  n_frames: int = DEFAULT_FRAMES,
                  swept_steps: int = DEFAULT_SWEPT_STEPS) -> tuple[bytes, bytes]:
    """Render a genome to static + swept PNG bytes."""
    from flame_sheep.renderer import N_ITERS

    # --- Static render ---
    renderer.upload_genome(genome)
    renderer.reset_walkers()

    # Zero audio input
    from flame_sheep_audio import N_BINS
    renderer.upload_audio(np.zeros(N_BINS, dtype=np.float32))

    for _ in range(n_frames):
        renderer.clear_histogram()
        renderer.dispatch_chaos_game(iterations=N_ITERS)
        ctx.memory_barrier()

    # Our tonemap uses pow(alpha, 1/brightness) — higher = brighter.
    # flam3 uses pow(density, gamma) where gamma ~4 darkens.
    # For training images, use a fixed high value to reveal structure.
    brightness = 8.0
    static_png = renderer.snapshot_png(brightness=brightness)

    # --- Swept render (rotation-accumulated) ---
    renderer.upload_genome(genome)
    renderer.reset_walkers()
    renderer.upload_audio(np.zeros(N_BINS, dtype=np.float32))
    renderer.clear_histogram()

    base_rotation = genome.rotation
    for i in range(swept_steps):
        angle = base_rotation + (2.0 * math.pi * i / swept_steps)
        renderer.set_rotation(angle)
        renderer.dispatch_chaos_game(iterations=N_ITERS)
        ctx.memory_barrier()
    renderer.set_rotation(base_rotation)

    swept_png = renderer.snapshot_png(brightness=brightness)

    return static_png, swept_png


def main():
    parser = argparse.ArgumentParser(description='Batch render Electric Sheep genomes')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for rendered images')
    parser.add_argument('--db', type=str,
                        default=str(Path.home() / '.local/share/flame-sheep/esheep.db'),
                        help='Electric Sheep database path')
    parser.add_argument('--size', type=int, default=DEFAULT_SIZE,
                        help=f'Render size in pixels (default: {DEFAULT_SIZE})')
    parser.add_argument('--frames', type=int, default=DEFAULT_FRAMES,
                        help=f'Frames to accumulate for static render (default: {DEFAULT_FRAMES})')
    parser.add_argument('--swept-steps', type=int, default=DEFAULT_SWEPT_STEPS,
                        help=f'Rotation steps for swept render (default: {DEFAULT_SWEPT_STEPS})')
    parser.add_argument('--min-rating', type=int, default=0,
                        help='Minimum rating to include (default: 0)')
    parser.add_argument('--generation', type=int, default=None,
                        help='Only render a specific generation')
    parser.add_argument('--limit', type=int, default=None,
                        help='Maximum number of genomes to render')
    parser.add_argument('--resume', action='store_true',
                        help='Skip genomes that already have rendered images')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load genomes
    from flame_sheep.esheep_parser import load_esheep_genomes
    log.info('Loading genomes from %s...', args.db)
    genomes = load_esheep_genomes(
        args.db,
        min_rating=args.min_rating,
        generation=args.generation,
    )
    log.info('Loaded %d genomes', len(genomes))

    if args.limit:
        genomes = genomes[:args.limit]

    # Filter already-rendered if resuming
    if args.resume:
        before = len(genomes)
        genomes = [
            (g, r, gen, sid) for g, r, gen, sid in genomes
            if not (output_dir / f'gen{gen}_{sid}_r{r}_static.png').exists()
        ]
        log.info('Resume: %d remaining (skipped %d already rendered)',
                 len(genomes), before - len(genomes))

    if not genomes:
        log.info('Nothing to render')
        return

    # Create renderer
    ctx, renderer = create_renderer(args.size)

    # Open manifest
    manifest_path = output_dir / 'manifest.csv'
    manifest_exists = manifest_path.exists()
    manifest_file = open(manifest_path, 'a', newline='')
    writer = csv.writer(manifest_file)
    if not manifest_exists:
        writer.writerow(['generation', 'sheep_id', 'rating',
                         'static_path', 'swept_path'])

    # Render loop
    rendered = 0
    failed = 0
    t0 = time.time()

    for genome, rating, gen, sid in genomes:
        prefix = f'gen{gen}_{sid}_r{rating}'
        static_path = output_dir / f'{prefix}_static.png'
        swept_path = output_dir / f'{prefix}_swept.png'

        try:
            static_png, swept_png = render_genome(
                genome, renderer, ctx,
                n_frames=args.frames,
                swept_steps=args.swept_steps,
            )

            static_path.write_bytes(static_png)
            swept_path.write_bytes(swept_png)

            writer.writerow([gen, sid, rating,
                             static_path.name, swept_path.name])
            manifest_file.flush()

            rendered += 1

            if rendered % 50 == 0:
                elapsed = time.time() - t0
                rate = rendered / elapsed
                eta = (len(genomes) - rendered) / rate
                log.info('Rendered %d/%d (%.1f/s, ETA %.0fs)',
                         rendered, len(genomes), rate, eta)

        except Exception as e:
            log.error('Failed gen %d sheep %d: %s', gen, sid, e)
            failed += 1

    manifest_file.close()
    elapsed = time.time() - t0
    log.info('Done: %d rendered, %d failed, %.1fs (%.1f genomes/s)',
             rendered, failed, elapsed, rendered / max(elapsed, 1))

    ctx.release()


if __name__ == '__main__':
    main()
