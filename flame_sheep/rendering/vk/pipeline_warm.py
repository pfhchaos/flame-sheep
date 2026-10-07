"""Cross-process marker for "this pipeline cache key has been compiled
at least once" — lets the wallpaper avoid blocking on cold compiles.

The marker is just a zero-byte file at
  ~/.cache/flame-sheep/warm-keys/<sha1-of-key>.warm

Written by ChaosGame._get_chaos_pipe after a successful
vkCreateComputePipelines, by both the wallpaper process AND the
precompile_worker subprocess. Read by wallpaper_vk before a genome
swap to decide whether the swap is cheap (warm — drives a Mesa
on-disk shader cache hit, ~3ms) or expensive (cold — 200-400ms).

Why a marker rather than introspecting Mesa's cache directly:
Mesa's SPIR-V→ISA cache key is private and version-dependent.
Maintaining our own marker is a small abstraction that lasts across
Mesa upgrades. The cost is that a Mesa-cache-cleared-but-marker-still-
present situation will produce an unexpected cold compile; rare in
practice, recovered automatically next session.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

WARM_DIR = Path(os.environ.get('XDG_CACHE_HOME',
                                  Path.home() / '.cache')) \
            / 'flame-sheep' / 'warm-keys'


def key_hash(n_transforms: int, has_final_xform: bool,
              keep_vars) -> str:
    """Deterministic short hash of the pipeline cache key. Same input
    must produce the same hash across processes + sessions."""
    sorted_vars = ','.join(str(int(v)) for v in sorted(keep_vars))
    s = f'{int(n_transforms)}|{int(bool(has_final_xform))}|{sorted_vars}'
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def _marker_path(h: str) -> Path:
    return WARM_DIR / f'{h}.warm'


def is_warm(n_transforms: int, has_final_xform: bool,
             keep_vars) -> bool:
    return _marker_path(
        key_hash(n_transforms, has_final_xform, keep_vars)).exists()


def mark_warm(n_transforms: int, has_final_xform: bool,
               keep_vars) -> None:
    """Atomic-ish touch of the warm marker. Failures are silent — we
    don't want a marker-write hiccup to bubble up out of a successful
    compile path."""
    try:
        WARM_DIR.mkdir(parents=True, exist_ok=True)
        _marker_path(
            key_hash(n_transforms, has_final_xform, keep_vars)).touch()
    except OSError:
        pass
