"""Genome / Transform JSON serialization for the SQLite library.

These are THIN WRAPPERS over the single core serializer — ``Genome.to_dict`` /
``Genome.from_dict`` and the ``Transform`` pair they delegate to
(``flame_sheep/genome/``). There is exactly ONE genome/transform serialization
in the codebase; this module just owns the catalog-specific framing (the JSON
string encoding, and the fact that ``db_id`` is the row key handled by
``library.py``, never a field inside the blob).

The leading-underscore names are historical — they predate the storage package
split and a number of tools and tests import them by name, so they stay as the
catalog's public entry points.

On-disk format note: the core serializer now also persists ``color_speed`` (an
earlier version of this module dropped it — a real bug: a genuinely-varying,
GPU-uploaded field was silently reset to 0.5 on every catalog load) and the
flam3 tone-map params. Old rows that predate those keys load via field defaults
(``Transform.color_speed`` -> 0.5, ``flam3_*`` -> 4.0), so existing catalogs
keep loading unchanged. There is no in-blob version marker — versioning lives in
separate ``genomes`` columns (``score_version`` / ``framing_version``) — and
backward-compat is by field default, exactly as before.
"""

from __future__ import annotations

import json

from ..genome import Genome, Transform


def _transform_to_dict(tr: Transform) -> dict:
    """Serialize a Transform to a JSON-compatible dict (core serializer)."""
    return tr.to_dict()


def _transform_from_dict(td: dict) -> Transform:
    """Deserialize a Transform from a JSON dict (core serializer).

    Short-variation backward-compat for old catalog rows lives in
    ``Transform.from_dict``."""
    return Transform.from_dict(td)


def _genome_to_json(g: Genome) -> str:
    """Serialize a Genome to a JSON string for the catalog ``params`` column.

    ``db_id`` is intentionally absent from the blob — it is the row key, stamped
    back onto the genome by ``library.load_genome``."""
    return json.dumps(g.to_dict(), separators=(',', ':'))


def _genome_from_json(s: str) -> Genome:
    """Deserialize a Genome from a catalog ``params`` JSON string.

    The returned genome's ``db_id`` is ``None``; ``library.load_genome`` sets it
    from the row id."""
    return Genome.from_dict(json.loads(s))
