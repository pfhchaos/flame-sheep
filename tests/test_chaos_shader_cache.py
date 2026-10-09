"""Regression tests for the trimmed-chaos-shader cache bound.

History: `_chaos_shader_cache` was an unbounded dict. Over a 35-day run it
accumulated ~15k distinct variation-set programs at ~0.9MB of Mesa-compiled
code each -> ~13GB of (swapped-out) heap in the renderer, ~8GB in the
precompile worker. The cache is now LRU-capped at `_CHAOS_SHADER_CACHE_MAX`
with `release()` on eviction; the worker drops its copy outright.

These tests exercise the cap logic without a GL context: the renderer is
built via __new__ and `gpu.compile_compute_shader` is mocked, so no real
shader is compiled. They guard against the cache silently going unbounded
again.
"""
from __future__ import annotations

from collections import OrderedDict
from unittest.mock import MagicMock

import pytest

from flame_sheep.rendering.renderer import (
    FlameRenderer,
    _CHAOS_SHADER_CACHE_MAX,
)


class _FakeProg:
    """Stand-in for a moderngl.ComputeShader: just tracks release()."""

    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True


def _make_renderer(monkeypatch) -> FlameRenderer:
    """A FlameRenderer with only the attrs _get_chaos_shader_for_keep_vars
    touches — no GL context, no real compile."""
    r = FlameRenderer.__new__(FlameRenderer)
    r._chaos_shader_cache = OrderedDict()
    r._flame_defines = {}
    r.gpu = MagicMock()
    r.gpu.compile_compute_shader.side_effect = lambda *a, **k: _FakeProg()
    # _sync_chaos_program_uniforms needs a live GL program; stub it out.
    monkeypatch.setattr(r, "_sync_chaos_program_uniforms", lambda prog: None,
                        raising=False)
    # A distinct active + universal program, neither initially in the cache.
    r.compute_shader = _FakeProg()
    r._chaos_universal_shader = r.compute_shader
    # Don't write real warm-marker files into ~/.cache during the test.
    import flame_sheep.rendering.vk.pipeline_warm as pw
    monkeypatch.setattr(pw, "mark_warm", lambda *a, **k: None)
    return r


def test_cache_is_bounded_and_evicts_lru(monkeypatch):
    r = _make_renderer(monkeypatch)
    overflow = 100
    n = _CHAOS_SHADER_CACHE_MAX + overflow
    progs = [r._get_chaos_shader_for_keep_vars(frozenset({i})) for i in range(n)]

    assert len(r._chaos_shader_cache) == _CHAOS_SHADER_CACHE_MAX
    # The first `overflow` keys are the LRU victims — their programs freed.
    assert all(p.released for p in progs[:overflow])
    # Everything still resident must NOT have been released.
    assert not any(p.released for p in progs[overflow:])


def test_cache_hit_reuses_program_and_marks_mru(monkeypatch):
    r = _make_renderer(monkeypatch)
    key = frozenset({1, 2, 3})
    first = r._get_chaos_shader_for_keep_vars(key)

    # Push it away from the MRU slot (a new compile), then baseline the
    # count so the re-fetch below is the only thing that could recompile.
    r._get_chaos_shader_for_keep_vars(frozenset({4}))
    compiles = r.gpu.compile_compute_shader.call_count
    again = r._get_chaos_shader_for_keep_vars(key)

    assert again is first  # served from cache, same object
    assert r.gpu.compile_compute_shader.call_count == compiles  # no recompile
    assert next(reversed(r._chaos_shader_cache)) == key  # now MRU


def test_active_program_is_never_released_on_eviction(monkeypatch):
    r = _make_renderer(monkeypatch)
    active_key = frozenset({-999})
    # Make a cached program the live one, then never touch it again while
    # flooding the cache so it becomes the LRU victim.
    r.compute_shader = r._get_chaos_shader_for_keep_vars(active_key)
    active = r.compute_shader
    for i in range(_CHAOS_SHADER_CACHE_MAX):
        r._get_chaos_shader_for_keep_vars(frozenset({i}))

    assert active_key not in r._chaos_shader_cache  # evicted from the dict
    assert active.released is False  # but NOT freed — it's in use
