"""Genome catalog — render genomes to images for offline evaluation.

Generates a batch of genomes, optionally evolves them, renders each
to a PNG, and writes them to a directory for human review.

Workflow:
  1. Generate/evolve genomes → render to catalog/unsorted/
  2. Human moves images to catalog/good/ or catalog/bad/
  3. Import votes back into the database

Filenames encode genome ID and fitness for traceability:
  genome_0042_f3.271.png
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def render_genome_to_image(genome, size: int = 512,
                           n_iterations: int = 2_000_000) -> np.ndarray:
    """Render a genome via CPU chaos game to an RGBA image.

    Returns uint8 array of shape (size, size, 4).
    """
    import warnings
    from .variations import apply_variations_cpu

    bound = 4.0
    hit_grid = np.zeros((size, size), dtype=np.float64)
    color_grid = np.zeros((size, size), dtype=np.float64)
    rng = np.random.default_rng()

    weights = np.array([tr.weight for tr in genome.transforms], dtype=np.float64)
    if weights.sum() == 0:
        return np.zeros((size, size, 4), dtype=np.uint8)
    weights /= weights.sum()
    cumw = np.cumsum(weights)

    x, y, c = 0.0, 0.0, 0.5
    fuse = 20

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for i in range(fuse + n_iterations):
            r = rng.random()
            tidx = min(int(np.searchsorted(cumw, r)), len(genome.transforms) - 1)
            tr = genome.transforms[tidx]
            a, b, cc, d, e, f = tr.affine
            nx = a * x + b * y + cc
            ny = d * x + e * y + f
            nx, ny = apply_variations_cpu(tr.variations, nx, ny)
            x, y = nx, ny
            c = (c + tr.color) * 0.5

            if not (np.isfinite(x) and np.isfinite(y)):
                x, y, c = 0.0, 0.0, 0.5  # reset walker instead of breaking
                continue

            if i >= fuse and abs(x) < bound and abs(y) < bound:
                gx = max(0, min(size - 1, int((x + bound) / (2 * bound) * size)))
                gy = max(0, min(size - 1, int((y + bound) / (2 * bound) * size)))
                hit_grid[gy, gx] += 1.0
                color_grid[gy, gx] += c

    # Tone map: log density display
    if hit_grid.max() == 0:
        return np.zeros((size, size, 4), dtype=np.uint8)

    log_hits = np.log1p(hit_grid)
    brightness = log_hits / log_hits.max()

    # Color from palette
    hit_mask = hit_grid > 0
    avg_color = np.zeros((size, size), dtype=np.float64)
    avg_color[hit_mask] = color_grid[hit_mask] / hit_grid[hit_mask]

    # Map palette index to RGB
    palette = genome.palette  # (256, 3) float32
    color_idx = np.clip((avg_color * 255).astype(int), 0, 255)
    rgb = palette[color_idx]  # (size, size, 3)

    # Apply brightness
    img = np.zeros((size, size, 4), dtype=np.uint8)
    for ch in range(3):
        img[:, :, ch] = np.clip(rgb[:, :, ch] * brightness * 255, 0, 255).astype(np.uint8)
    img[:, :, 3] = np.clip(brightness * 255, 0, 255).astype(np.uint8)

    return img


def render_genome_gpu(genome, ctx, size: int = 2048, n_frames: int = 60,
                      brightness: float = 6.0,
                      _renderer_cache: dict = {}) -> np.ndarray | None:
    """Render a genome using the GPU pipeline. Returns RGBA uint8 array or None.

    Uses a headless moderngl context and the full FlameRenderer pipeline:
    chaos game → histogram → tonemap → readback.

    Caches the renderer to avoid recompiling shaders per genome.
    """
    from .renderer import FlameRenderer, Viewport, N_ITERS
    import moderngl

    cache_key = (id(ctx), size)
    if cache_key not in _renderer_cache:
        _renderer_cache[cache_key] = FlameRenderer(ctx, size, size)
    renderer = _renderer_cache[cache_key]
    renderer.upload_genome(genome)
    renderer.reset_walkers()
    renderer.blur_radius = 0  # no blur for catalog images

    # Upload a flat audio texture (no audio reactivity for stills)
    from flame_sheep_audio import N_BINS
    renderer.upload_audio(np.zeros(N_BINS, dtype=np.float32))

    # Run multiple frames to accumulate density
    for _ in range(n_frames):
        renderer.clear_histogram()
        renderer.dispatch_chaos_game(iterations=N_ITERS)
        ctx.memory_barrier()

    # Tonemap to FBO
    fbo_tex = ctx.texture((size, size), components=4, dtype='f1')
    fbo = ctx.framebuffer(color_attachments=[fbo_tex])
    fbo.use()
    ctx.viewport = (0, 0, size, size)

    viewport = Viewport(0, 0, size, size)
    renderer.palette_tex.use(location=0)
    renderer.audio_tex.use(location=1)

    p = renderer.tonemap_program
    p['u_palette'] = 0
    p['u_audio'] = 1
    p['u_width'] = size
    p['u_height'] = size
    p['u_viewport_x'] = 0
    p['u_viewport_y'] = 0
    p['u_viewport_w'] = size
    p['u_viewport_h'] = size
    p['u_surface_w'] = size
    p['u_surface_h'] = size
    p['u_brightness'] = brightness

    renderer.quad_vao.render(moderngl.TRIANGLES)

    # Read back pixels
    data = fbo.read(components=4)
    img = np.frombuffer(data, dtype=np.uint8).reshape(size, size, 4)
    # Flip vertically (OpenGL origin is bottom-left)
    img = img[::-1].copy()

    fbo.release()
    fbo_tex.release()

    return img


def generate_catalog(
    output_dir: str | Path,
    n_genomes: int = 50,
    n_evolve: int = 3,
    image_size: int = 2048,
    n_iterations: int = 200_000,
) -> None:
    """Generate a catalog of rendered genomes for evaluation.

    Creates output_dir/unsorted/ with rendered PNGs.
    Creates output_dir/good/ and output_dir/bad/ for sorting.
    """
    from .storage import Library
    from .genome import Genome, _score_from_histogram

    output = Path(output_dir)
    for subdir in ['unsorted', 'good', 'bad']:
        (output / subdir).mkdir(parents=True, exist_ok=True)

    lib = Library()
    rng = np.random.default_rng()

    # Generate genomes
    genomes = []
    for i in range(n_genomes):
        g = Genome.random(rng=rng)
        if g.is_viable():
            gid = lib.save_genome(g)
            genomes.append((gid, g))

    log.info(f'Generated {len(genomes)} viable genomes')

    # Evolve: keep top half by fitness, breed, repeat
    for gen in range(n_evolve):
        # Score all
        scored = []
        for gid, g in genomes:
            fitness = lib.genome_fitness(gid)
            scored.append((fitness, gid, g))
        scored.sort(reverse=True)

        # Keep top half
        survivors = scored[:len(scored) // 2]
        children = []

        # Breed pairs
        for _ in range(len(scored) - len(survivors)):
            if len(survivors) < 2:
                break
            idx_a, idx_b = rng.choice(len(survivors), size=2, replace=False)
            parent_a = survivors[idx_a][2]
            parent_b = survivors[idx_b][2]
            blend = float(rng.uniform(0.2, 0.8))
            child = parent_a.lerp(parent_b, blend).jitter(rng, scale=0.1)
            if child.is_viable():
                cid = lib.save_genome(child)
                children.append((cid, child))

        genomes = [(gid, g) for _, gid, g in survivors] + children
        log.info(f'Evolution gen {gen + 1}: {len(genomes)} genomes')

    # Create GPU context if available
    gpu_ctx = None
    try:
        import moderngl
        gpu_ctx = moderngl.create_standalone_context(require=430)
        log.info('GPU rendering enabled')
    except Exception as e:
        log.info(f'GPU not available ({e}), using CPU renderer')

    # Render all survivors
    rendered = 0
    for gid, g in genomes:
        fitness = lib.genome_fitness(gid)

        if gpu_ctx is not None:
            img = render_genome_gpu(g, gpu_ctx, size=image_size)
        else:
            img = render_genome_to_image(g, size=image_size, n_iterations=n_iterations)

        if img is None or img[:, :, :3].max() == 0:
            continue

        filename = f'genome_{gid:04d}_f{fitness:.3f}.png'
        filepath = output / 'unsorted' / filename

        try:
            from PIL import Image
            Image.fromarray(img).save(filepath)
        except ImportError:
            np.save(filepath.with_suffix('.npy'), img)

        rendered += 1
        if rendered % 5 == 0:
            log.info(f'Rendered {rendered}/{len(genomes)}...')

    if gpu_ctx is not None:
        gpu_ctx.release()

    log.info(f'Rendered {rendered} genomes to {output / "unsorted"}')
    log.info(f'Sort into good/ and bad/, then run --import-catalog')


def import_catalog(catalog_dir: str | Path) -> None:
    """Import votes from sorted catalog folders.

    Reads genome IDs from filenames in good/ and bad/ directories
    and records votes in the database.
    """
    from .storage import Library

    catalog = Path(catalog_dir)
    lib = Library()

    for folder, rating in [('good', 1), ('bad', -1)]:
        folder_path = catalog / folder
        if not folder_path.exists():
            continue
        for f in folder_path.iterdir():
            if not f.stem.startswith('genome_'):
                continue
            # Parse genome ID from filename: genome_0042_f3.271.png
            parts = f.stem.split('_')
            if len(parts) >= 2:
                try:
                    gid = int(parts[1])
                    lib.conn.execute(
                        'INSERT INTO ratings (target_type, target_id, rating) VALUES (?, ?, ?)',
                        ('genome', gid, rating),
                    )
                    log.info(f'Voted {rating:+d} on genome #{gid}')
                except (ValueError, Exception) as e:
                    log.warning(f'Skipping {f.name}: {e}')

    lib.conn.commit()
    log.info('Catalog votes imported')
