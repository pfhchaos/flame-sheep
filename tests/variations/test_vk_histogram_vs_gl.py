"""Vulkan vs GL histogram cross-check via stored catalog renders.

The catalog DB stores `hist_static` blobs — raw 512×512 uint32
(hits, colors) pairs from the GL renderer at known params
(LIVE_ITER_MAX=500 iterations, walker count = walker_density × pixels).
Render the same genome through Vulkan with matching params; assert
the totals match deterministically (same walker count × same iters
× same fuse × same bounds check → same total energy) and the spatial
structure overlaps substantially (per-pixel placement varies with
RNG seed but the attractor shape is the same).

This is the CPU-independent cross-validation we couldn't get from
the per-variation golden tests — that suite verifies each variation
function in isolation; this verifies the FULL chaos game composition
matches between backends on real (non-trivial, multi-transform, post-
affine, optionally final-xform) genomes.

What this catches:
  - Composition bugs (transform sequencing, post-affine handling,
    final-xform handling, color blend)
  - Bounds / world-to-pixel mismatches between backends
  - Walker-state persistence bugs
  - Subtle differences in the chaos game's bounds-check semantics

What this does NOT verify:
  - Per-pixel exact values (RNG / atomic-add ordering differs)
  - That the result is "correct" — just that GL and Vulkan agree
"""
from __future__ import annotations

import pytest

import numpy as np


# Walker-count formula matches flame_sheep/genome/render_worker.py.
# render_worker.py:67-72: walker_density = live_walkers / (live_w*live_h)
# scoring_walkers = (walker_density * render_size**2 // 64) * 64.
# These constants are pinned to the cfg defaults — if scoring_cfg
# changes in production, this test would need to follow.
_LIVE_W = 1080
_LIVE_H = 1080
_LIVE_WALKERS = 65536
_RENDER_SIZE = 512
_LIVE_ITER_MAX = 500


def _scoring_walker_count() -> int:
    density = _LIVE_WALKERS / (_LIVE_W * _LIVE_H)
    n = int(density * _RENDER_SIZE * _RENDER_SIZE)
    return (n // 64) * 64


@pytest.fixture(scope='module')
def lib():
    try:
        from flame_sheep.storage.library import Library
    except ImportError:
        pytest.skip('flame_sheep.storage.library not available')
    return Library()


@pytest.fixture(scope='module')
def vk_chaos():
    try:
        from viz_authoring.vk.context import VkContext
        from viz_authoring.vk.chaos_game import ChaosGame
    except ImportError:
        pytest.skip('viz_authoring.vk not available')
    try:
        ctx = VkContext(instance_extensions=[],
                         app_name='vk-hist-vs-gl',
                         pipeline_cache_path=None)
        ctx.select_device()
    except Exception as e:
        pytest.skip(f'No Vulkan device: {e}')
    cg = ChaosGame(ctx, _RENDER_SIZE, _RENDER_SIZE,
                    n_walkers=_scoring_walker_count())
    yield ctx, cg
    cg.cleanup()
    ctx.cleanup()


def _pick_test_genomes(lib, n: int = 5) -> list[int]:
    """Genomes with stored hist_static, picked by CNN score (highest
    scoring → most-likely-rendered-correctly genomes are the better
    cross-check baseline — if they don't agree, deeper genomes won't
    either). Limit to a manageable count for a unit test."""
    rows = lib.conn.execute(
        'SELECT gb.genome_id FROM genome_blobs gb '
        'JOIN genomes g ON g.id = gb.genome_id '
        'WHERE gb.hist_static IS NOT NULL AND g.cnn_score IS NOT NULL '
        'ORDER BY g.cnn_score DESC LIMIT ?',
        (n,)
    ).fetchall()
    if not rows:
        pytest.skip('no genomes with hist_static + cnn_score in catalog')
    return [r[0] for r in rows]


def _render_vulkan(vk_chaos, genome_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Render `genome_id` through the Vulkan chaos game at catalog params,
    return (hits, colors) numpy arrays matching the stored shape."""
    from viz_authoring.vk.wallpaper_demo import _genome_from_catalog
    ctx, cg = vk_chaos
    kwargs, _palette, zoom, rotation, center = _genome_from_catalog(genome_id)
    cg.set_genome(**kwargs)
    cg.reset_walkers(seed=42)  # deterministic across test runs
    cg.clear_histogram()
    cg.render_frame(iterations=_LIVE_ITER_MAX, zoom=zoom,
                     rotation=rotation, center=center)
    return cg.download_histogram()


# ----- the actual cross-check ---------------------------------------

class TestVulkanMatchesGLHistogram:
    """Per-test thresholds chosen empirically:

    Total hits ratio: identical chaos-game machinery → ratio within 5%
        (typical is < 0.5%; 5% absorbs noise on small renders where
        a few walkers escaping early can swing the count meaningfully).

    Coverage ratio: same generous 5% — coverage = unique pixels touched
        is even less sensitive than total hits.

    IoU: 0.4 is empirically the floor for healthy renders — per-pixel
        placement varies with RNG so the attractor's stochastic edges
        contribute IoU loss, but the structural core overlaps heavily.
        Values seen on a sample of 5 catalog genomes: 0.53-0.85.
    """

    @pytest.mark.parametrize('genome_id', [347, 1489, 85, 602, 2821])
    def test_genome_renders_match_catalog(self, vk_chaos, lib, genome_id):
        from flame_sheep.genome.scoring.scoring_channels import (
            unpack_static_histogram)
        blob = lib.load_genome_blobs(
            genome_id, cols=['hist_static'])['hist_static']
        if blob is None:
            pytest.skip(f'genome {genome_id} has no stored hist_static')
        gl_hits, gl_colors = unpack_static_histogram(blob)

        vk_hits, vk_colors = _render_vulkan(vk_chaos, genome_id)

        # Total energy: same walkers × iters × fuse → identical to noise.
        ratio = vk_hits.sum() / max(int(gl_hits.sum()), 1)
        assert 0.95 <= ratio <= 1.05, (
            f'genome {genome_id}: total hits drift {ratio:.3f}× '
            f'(GL {gl_hits.sum()}, Vk {vk_hits.sum()})')

        # Spatial coverage: same chaos-game machinery + same bounds check
        # → same number of pixels touched (to small RNG variance).
        gl_cov = int((gl_hits > 0).sum())
        vk_cov = int((vk_hits > 0).sum())
        cov_ratio = vk_cov / max(gl_cov, 1)
        assert 0.95 <= cov_ratio <= 1.05, (
            f'genome {genome_id}: coverage drift {cov_ratio:.3f}× '
            f'(GL {gl_cov}, Vk {vk_cov})')

        # Spatial overlap: per-pixel placement is stochastic so IoU is
        # loose, but the attractor's structural core overlaps substantially.
        gl_mask = gl_hits > 0
        vk_mask = vk_hits > 0
        union = int((gl_mask | vk_mask).sum())
        intersection = int((gl_mask & vk_mask).sum())
        iou = intersection / max(union, 1)
        assert iou > 0.4, (
            f'genome {genome_id}: spatial IoU {iou:.3f} suggests the '
            f'attractor lands in significantly different pixels — '
            f'check world_to_pixel / zoom / rotation / center handling')


def test_walker_count_matches_render_worker_default():
    """Pin: the walker-count formula here must track render_worker's.
    If render_worker.py changes its defaults the cross-check fixture
    needs to follow, OR the test would render against the wrong
    baseline and the per-genome assertions would fail confusingly."""
    expected = _scoring_walker_count()
    # Sanity: matches what we observed when calibrating the test.
    assert expected == 14720, (
        f'walker count {expected} != 14720 — render_worker defaults '
        f'(or this test pin) changed; revisit which is right')
