"""Persistent storage for genomes, loops, and ratings.

SQLite database at ~/.local/share/flame-sheep/library.db

Tables:
  genomes    — individual genomes with aesthetic scores and serialized parameters
  loops      — ordered sequences of genomes that form coherent visual cycles
  loop_items — join table: (loop_id, position, genome_id, motion_field)
  ratings    — user like/dislike history for genomes and loops

This package re-exports the public storage API. External code should
import from `flame_sheep.storage`, not from the submodules directly —
the submodule layout is an implementation detail of Stage 2 of
docs/reorg_plan.md.
"""

from .schema import (
    BLOB_COLS,
    NORMALIZATION_VERSION,
    STATS_COLS,
    _connect,
    _db_path,
    _ensure_schema,
)
from .serialization import (
    _genome_from_json,
    _genome_to_json,
    _transform_from_dict,
    _transform_to_dict,
)
from .library import Library

# Motion field functions moved to flame_sheep.genome.motion_field in
# Stage 4b. Re-export here so existing `from flame_sheep.storage import
# compute_motion_field` imports keep working.
from ..genome.motion_field import (
    MOTION_GRID,
    compute_motion_field,
    motion_field_coherence,
    motion_field_from_blob,
    motion_field_to_blob,
)

# score_palette moved to flame_sheep.palette.scoring. Re-export for
# backward compatibility with any consumer that still expects it here.
from ..palette.scoring import score_palette


# score_loop moved to flame_sheep.loops.scoring in Stage 4c. We re-export
# here, BUT use a lazy attribute access to avoid the import cycle —
# loops.compose imports `..storage`, so we can't have storage's __init__
# eagerly import from loops at module load time.
def __getattr__(name):
    if name == 'score_loop':
        from ..loops.scoring import score_loop
        return score_loop
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

__all__ = [
    'Library',
    'BLOB_COLS',
    'STATS_COLS',
    'NORMALIZATION_VERSION',
    'MOTION_GRID',
    '_db_path',
    '_connect',
    '_ensure_schema',
    '_genome_to_json',
    '_genome_from_json',
    '_transform_to_dict',
    '_transform_from_dict',
    'score_loop',
    'score_palette',
    'compute_motion_field',
    'motion_field_coherence',
    'motion_field_to_blob',
    'motion_field_from_blob',
]
