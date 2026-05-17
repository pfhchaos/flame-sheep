"""Test that all shader permutations compile successfully.

Requires a GPU context (EGL standalone). Skipped if no GPU available.
"""

import pytest

try:
    import os
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    import moderngl
    ctx = moderngl.create_context(standalone=True, backend='egl')
    HAS_GPU = True
except Exception:
    HAS_GPU = False
    ctx = None

pytestmark = pytest.mark.skipif(not HAS_GPU, reason='No GPU context available')


@pytest.fixture(scope='module')
def gpu_ctx():
    return ctx


class TestFlameShaderCompile:
    """flame.comp must compile in all permutations."""

    def _compile_flame(self, gpu_ctx, scoring: bool):
        from flame_sheep.renderer import SHADER_DIR, _resolve_includes
        from flame_sheep.variations._symmetry_groups import generate_glsl

        src = (SHADER_DIR / 'flame.comp').read_text()
        src = _resolve_includes(src, SHADER_DIR)
        src = src.replace('// {{SYMMETRY_GROUPS}}', generate_glsl())
        if scoring:
            src = src.replace('\n', '\n#define SCORING_MODE\n', 1)
        return gpu_ctx.compute_shader(src)

    def test_flame_normal(self, gpu_ctx):
        shader = self._compile_flame(gpu_ctx, scoring=False)
        assert shader is not None

    def test_flame_scoring(self, gpu_ctx):
        shader = self._compile_flame(gpu_ctx, scoring=True)
        assert shader is not None


class TestSupportShaderCompile:
    """All support shaders must compile."""

    def test_clear(self, gpu_ctx):
        from flame_sheep.renderer import SHADER_DIR
        shader = gpu_ctx.compute_shader((SHADER_DIR / 'clear.comp').read_text())
        assert shader is not None

    def test_reduce_max(self, gpu_ctx):
        from flame_sheep.renderer import SHADER_DIR
        shader = gpu_ctx.compute_shader((SHADER_DIR / 'reduce_max.comp').read_text())
        assert shader is not None

    def test_tonemap(self, gpu_ctx):
        from flame_sheep.renderer import SHADER_DIR
        prog = gpu_ctx.program(
            vertex_shader=(SHADER_DIR / 'tonemap.vert').read_text(),
            fragment_shader=(SHADER_DIR / 'tonemap.frag').read_text(),
        )
        assert prog is not None


class TestCnnShaderCompile:
    """CNN training/inference shaders must compile (Vulkan SPIR-V not testable here)."""
    pass  # CNN shaders are Vulkan GLSL, compiled by glslangValidator, not moderngl
