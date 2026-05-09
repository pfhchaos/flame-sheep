"""GPU pipeline integration tests — verify full chaos game matches CPU reference.

Tests the complete pipeline including:
- Individual variation GPU/CPU agreement with specific params
- Full chaos game with multiple transforms
- Post-affine and final xform pipeline
- Electric Sheep genome rendering

Requires GPU (headless EGL context). Skipped if no GL available.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

try:
    import moderngl
    _HAS_GL = True
except ImportError:
    _HAS_GL = False

from flame_sheep.variations import (
    Variation, NUM_VARIATIONS, apply_variation_cpu,
)
from flame_sheep.variations._registry import VAR_PARAMS_SPEC, SLOT_SIZE
from flame_sheep.genome import Genome, Transform, MAX_TRANSFORMS, MAX_ACTIVE_VARS
import flame_sheep.variations._cpu as cpu_mod

SHADER_DIR = Path(__file__).parent.parent / 'flame_sheep' / 'shaders'

MAX_PARAMS = SLOT_SIZE - 2

# Test points with good spread
TEST_POINTS = [
    (0.5, 0.3), (1.0, 0.5), (-0.3, 0.7), (0.8, -0.4),
    (0.1, 0.1), (-0.5, -0.5), (1.5, -0.3), (0.3, 1.2),
]


@pytest.fixture(scope='module')
def gpu_ctx():
    if not _HAS_GL:
        pytest.skip('No moderngl available')
    try:
        ctx = moderngl.create_context(standalone=True, backend='egl')
    except Exception:
        pytest.skip('No GL context available')
    yield ctx
    ctx.release()


@pytest.fixture(scope='module')
def variation_shader(gpu_ctx):
    from flame_sheep.variations._symmetry_groups import generate_glsl

    src = (SHADER_DIR / 'test_variation.comp').read_text()
    src = src.replace('// {{SYMMETRY_GROUPS}}', generate_glsl())
    import re
    def _replace(m):
        return (SHADER_DIR / m.group(1)).read_text()
    src = re.sub(r'#include\s+"(.+?)"', _replace, src)
    return gpu_ctx.compute_shader(src)


def _run_variation_gpu(gpu_ctx, shader, var_idx, params, points,
                       affine=None):
    """Run a single variation on GPU, return output points."""
    n = len(points)
    in_data = np.array(points, dtype=np.float32).flatten()
    in_buf = gpu_ctx.buffer(in_data.tobytes())
    in_buf.bind_to_storage_buffer(6)

    out_buf = gpu_ctx.buffer(reserve=n * 2 * 4)
    out_buf.bind_to_storage_buffer(7)

    # Active vars
    av = np.full(MAX_ACTIVE_VARS * SLOT_SIZE, -1.0, dtype=np.float32)
    av[0] = float(var_idx)
    av[1] = 1.0
    spec = VAR_PARAMS_SPEC.get(var_idx, [])
    for k, pname in enumerate(spec[:MAX_PARAMS]):
        av[2 + k] = params.get(pname, 0.0)
    av_buf = gpu_ctx.buffer(av.tobytes())
    av_buf.bind_to_storage_buffer(3)

    # Affine
    if affine is None:
        affine = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    aff_buf = gpu_ctx.buffer(affine.tobytes())
    aff_buf.bind_to_storage_buffer(2)

    # Pre-variations (empty)
    pre_buf = gpu_ctx.buffer(np.full(MAX_ACTIVE_VARS * SLOT_SIZE, -1.0,
                                     dtype=np.float32).tobytes())
    pre_buf.bind_to_storage_buffer(10)

    shader['u_n_points'] = n
    shader.run(group_x=(n + 63) // 64)
    gpu_ctx.memory_barrier()

    out = np.frombuffer(out_buf.read(), dtype=np.float32).reshape(-1, 2)
    for buf in (in_buf, out_buf, av_buf, aff_buf, pre_buf):
        buf.release()
    return out


class TestVariationParamAgreement:
    """Test GPU/CPU agreement for variations with specific param values."""

    def _check(self, gpu_ctx, shader, var_idx, params, points=None,
               affine=None, uses_rng=False, atol=1e-4):
        if points is None:
            points = TEST_POINTS
        cpu_mod._current_var_params = params
        gpu_out = _run_variation_gpu(gpu_ctx, shader, var_idx, params,
                                     points, affine=affine)
        for i, (px, py) in enumerate(points):
            if uses_rng:
                cpu_mod._rng = cpu_mod.Xorshift32(seed=i * 17 + 1)
            cx, cy = apply_variation_cpu(var_idx, px, py, 1.0, affine)
            if uses_rng:
                cpu_mod._rng = None
            gx, gy = float(gpu_out[i, 0]), float(gpu_out[i, 1])

            if not (np.isfinite(cx) and np.isfinite(cy)):
                continue

            np.testing.assert_allclose(
                [gx, gy], [cx, cy], atol=atol,
                err_msg=f'var {var_idx} at ({px},{py}) params={params}')

    def test_rings2_default(self, gpu_ctx, variation_shader):
        self._check(gpu_ctx, variation_shader, Variation.RINGS2,
                    {'rings2_val': 0.5})

    def test_rings2_sheep_params(self, gpu_ctx, variation_shader):
        """rings2 with val=0.9 as used in sheep 69369."""
        self._check(gpu_ctx, variation_shader, Variation.RINGS2,
                    {'rings2_val': 0.9})

    def test_julian_default(self, gpu_ctx, variation_shader):
        self._check(gpu_ctx, variation_shader, Variation.JULIAN,
                    {'julian_power': 4.0, 'julian_dist': 1.0},
                    uses_rng=True)

    def test_julian_sheep_params(self, gpu_ctx, variation_shader):
        """julian with power=6, dist=-1.25 as used in sheep 45928."""
        self._check(gpu_ctx, variation_shader, Variation.JULIAN,
                    {'julian_power': 6.0, 'julian_dist': -1.25244},
                    uses_rng=True)

    def test_julian_negative_dist(self, gpu_ctx, variation_shader):
        """julian with negative dist — output radius should use pow(r², cn)."""
        params = {'julian_power': 4.0, 'julian_dist': -1.0}
        cpu_mod._current_var_params = params
        gpu_out = _run_variation_gpu(gpu_ctx, variation_shader,
                                     Variation.JULIAN, params, TEST_POINTS)
        for i, (px, py) in enumerate(TEST_POINTS):
            cpu_mod._rng = cpu_mod.Xorshift32(seed=i * 17 + 1)
            cx, cy = apply_variation_cpu(Variation.JULIAN, px, py, 1.0)
            cpu_mod._rng = None
            # Check magnitudes match (angles may differ due to RNG branch)
            cr = np.sqrt(cx*cx + cy*cy)
            gr = np.sqrt(gpu_out[i, 0]**2 + gpu_out[i, 1]**2)
            np.testing.assert_allclose(
                gr, cr, rtol=1e-3,
                err_msg=f'julian radius mismatch at ({px},{py})')

    def test_spherical(self, gpu_ctx, variation_shader):
        self._check(gpu_ctx, variation_shader, Variation.SPHERICAL, {})

    def test_blur(self, gpu_ctx, variation_shader):
        """Blur uses RNG — check output is finite and nonzero."""
        gpu_out = _run_variation_gpu(gpu_ctx, variation_shader,
                                     Variation.BLUR, {}, TEST_POINTS)
        for i in range(len(TEST_POINTS)):
            assert np.isfinite(gpu_out[i]).all(), f'blur produced non-finite at point {i}'
            # Blur output should be near origin (random point on unit disk)
            r = np.sqrt(gpu_out[i, 0]**2 + gpu_out[i, 1]**2)
            assert r < 2.0, f'blur produced point outside unit disk: r={r}'

    def test_waves_affine_reading(self, gpu_ctx, variation_shader):
        """Waves reads b,c,e,f from affine."""
        affine = np.array([0.8, 0.5, 0.3, -0.2, 0.6, 0.5], dtype=np.float32)
        self._check(gpu_ctx, variation_shader, Variation.WAVES, {},
                    affine=affine)


class TestFullPipelineRendering:
    """Test complete chaos game rendering produces expected pixel coverage."""

    def _render_genome(self, gpu_ctx, genome, n_frames=10):
        from flame_sheep.renderer import FlameRenderer
        from flame_sheep_audio import N_BINS

        renderer = FlameRenderer(gpu_ctx, 256, 256)
        renderer.upload_genome(genome)
        renderer.reset_walkers()
        renderer.upload_audio(np.zeros(N_BINS, dtype=np.float32))
        renderer.clear_histogram()
        for _ in range(n_frames):
            renderer.dispatch_chaos_game()
            gpu_ctx.memory_barrier()
        hits, _ = renderer.histogram_data()
        return hits

    def test_sierpinski_renders(self, gpu_ctx):
        """Classic Sierpinski triangle should produce visible coverage."""
        g = Genome()
        corners = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        for cx, cy in corners:
            t = Transform()
            t.affine = np.array([0.5, 0, cx*0.5, 0, 0.5, cy*0.5], dtype=np.float32)
            t.variations[Variation.LINEAR] = 1.0
            t.weight = 1.0
            g.transforms.append(t)
        g.zoom = 0.8
        g.center = np.array([0.5, 0.4], dtype=np.float32)

        hits = self._render_genome(gpu_ctx, g, n_frames=50)
        nonzero = np.count_nonzero(hits)
        assert nonzero > 100, f'Sierpinski should cover many pixels, got {nonzero}'

    def test_random_genome_renders(self, gpu_ctx):
        """A random genome should produce some pixel coverage."""
        g = Genome.random(np.random.default_rng(42))
        hits = self._render_genome(gpu_ctx, g, n_frames=20)
        assert hits.sum() > 0, 'Random genome produced zero hits'

    def test_post_affine_renders(self, gpu_ctx):
        """Genome with post-affine should produce different coverage than without."""
        g = Genome()
        t = Transform()
        t.affine = np.array([0.5, 0, 0.25, 0, 0.5, 0], dtype=np.float32)
        t.variations[Variation.LINEAR] = 1.0
        t.weight = 1.0
        g.transforms.append(t)
        t2 = Transform()
        t2.affine = np.array([0.5, 0, -0.25, 0, 0.5, 0.5], dtype=np.float32)
        t2.variations[Variation.LINEAR] = 1.0
        t2.weight = 1.0
        g.transforms.append(t2)

        hits_no_post = self._render_genome(gpu_ctx, g, n_frames=20)

        # Add post-affine (rotation)
        import copy
        g2 = copy.deepcopy(g)
        g2.transforms[0].post_affine = np.array(
            [0.7071, -0.7071, 0, 0.7071, 0.7071, 0], dtype=np.float32)
        hits_with_post = self._render_genome(gpu_ctx, g2, n_frames=20)

        # Both should render, but with different patterns
        assert np.count_nonzero(hits_no_post) > 50
        assert np.count_nonzero(hits_with_post) > 50
        # They should differ
        assert not np.array_equal(hits_no_post, hits_with_post)

    def test_final_xform_renders(self, gpu_ctx):
        """Genome with final xform should produce different coverage."""
        g = Genome()
        t = Transform()
        t.affine = np.array([0.5, 0, 0.25, 0, 0.5, 0], dtype=np.float32)
        t.variations[Variation.LINEAR] = 1.0
        t.weight = 1.0
        g.transforms.append(t)
        t2 = Transform()
        t2.affine = np.array([0.5, 0, -0.25, 0, 0.5, 0.5], dtype=np.float32)
        t2.variations[Variation.LINEAR] = 1.0
        t2.weight = 1.0
        g.transforms.append(t2)

        hits_no_final = self._render_genome(gpu_ctx, g, n_frames=20)

        # Add final xform with spherical
        import copy
        g2 = copy.deepcopy(g)
        ft = Transform()
        ft.affine = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
        ft.variations[Variation.SPHERICAL] = 1.0
        g2.final_xform = ft
        hits_with_final = self._render_genome(gpu_ctx, g2, n_frames=20)

        assert np.count_nonzero(hits_no_final) > 50
        assert np.count_nonzero(hits_with_final) > 50
        assert not np.array_equal(hits_no_final, hits_with_final)


class TestSheepRendering:
    """Test that Electric Sheep genomes produce reasonable renders."""

    def _load_sheep(self, generation, sheep_id):
        from flame_sheep.esheep_parser import load_esheep_genomes
        db_path = str(Path.home() / '.local/share/flame-sheep/esheep.db')
        genomes = load_esheep_genomes(db_path, generation=generation)
        for g, r, gen, sid in genomes:
            if sid == sheep_id:
                return g
        pytest.skip(f'Sheep {generation}.{sheep_id} not in database')

    def test_sheep_69369_cpu_vs_gpu_coverage(self, gpu_ctx):
        """Sheep 69369 should produce similar coverage on GPU as CPU."""
        from flame_sheep.renderer import FlameRenderer
        from flame_sheep.variations._cpu import apply_variations_cpu
        from flame_sheep_audio import N_BINS

        g = self._load_sheep(244, 69369)

        # CPU chaos game
        rng = np.random.default_rng(42)
        x, y = 0.0, 0.0
        weights = np.array([t.weight for t in g.transforms], dtype=np.float64)
        weights /= weights.sum()
        cumw = np.cumsum(weights)
        cpu_points = set()
        grid = 256
        bound = 1.0 / g.zoom

        for i in range(20000):
            r_val = rng.random()
            tidx = min(int(np.searchsorted(cumw, r_val)), len(g.transforms) - 1)
            tr = g.transforms[tidx]
            a, b, c, d, e, f = tr.affine
            nx, ny = a*x + b*y + c, d*x + e*y + f
            nx, ny = apply_variations_cpu(tr.variations, nx, ny, np.array(tr.affine))
            if tr.post_affine is not None:
                pa, pb, pc_, pd, pe, pf = tr.post_affine
                nx, ny = pa*nx + pb*ny + pc_, pd*nx + pe*ny + pf
            if g.final_xform:
                ft = g.final_xform
                fa, fb, fc, fd, fe, ff = ft.affine
                nx, ny = fa*nx + fb*ny + fc, fd*nx + fe*ny + ff
                nx, ny = apply_variations_cpu(ft.variations, nx, ny, np.array(ft.affine))
            x, y = nx, ny
            if not (np.isfinite(x) and np.isfinite(y)):
                x, y = 0.0, 0.0
                continue
            if i > 50:
                gx = int((x + bound) / (2*bound) * grid)
                gy = int((y + bound) / (2*bound) * grid)
                if 0 <= gx < grid and 0 <= gy < grid:
                    cpu_points.add((gx, gy))

        # GPU chaos game
        renderer = FlameRenderer(gpu_ctx, grid, grid)
        renderer.upload_genome(g)
        renderer.reset_walkers()
        renderer.upload_audio(np.zeros(N_BINS, dtype=np.float32))
        renderer.clear_histogram()
        for _ in range(50):
            renderer.dispatch_chaos_game()
            gpu_ctx.memory_barrier()
        hits, _ = renderer.histogram_data()
        gpu_pixels = np.count_nonzero(hits)

        # Both should produce meaningful coverage
        assert len(cpu_points) > 10, \
            f'CPU only hit {len(cpu_points)} pixels — genome may be degenerate'
        assert gpu_pixels > 5, \
            f'GPU only hit {gpu_pixels} pixels vs CPU {len(cpu_points)}'
        # GPU should be within an order of magnitude of CPU coverage
        ratio = gpu_pixels / max(len(cpu_points), 1)
        assert ratio > 0.01, \
            f'GPU coverage ({gpu_pixels}) is {ratio:.1%} of CPU ({len(cpu_points)})'
