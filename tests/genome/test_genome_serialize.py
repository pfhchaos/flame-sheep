"""Shared component (shader-warming.md map #12): Genome serialize/deserialize
round-trip.

BEHAVIOR SPEC. ``Genome.to_dict`` / ``Genome.from_dict`` (and the Transform
pair they delegate to) are the SINGLE core serializer — used by both the catalog
DB (``storage.serialization``) and the PLAN protocol — and an EXACT, lossless,
JSON-safe round-trip of every INTRINSIC genome field:
``Genome.from_dict(json.loads(json.dumps(g.to_dict())))`` reproduces transforms
(affine, post_affine, variations, pre_variations, color, color_speed, weight,
var_params), final_xform, palette, zoom, rotation, center, and the flam3
tone-map params.

``db_id`` is DELIBERATELY NOT serialized: identity is the catalog row key, not
genome data. The storage layer stamps it from the row id; the PLAN protocol
carries it as a separate wire field (``test_plan_protocol`` covers that). So a
genome round-tripped through the core comes back with ``db_id is None`` — the
assertion below pins that.

``color_speed`` IS serialized — an earlier DB-only serializer dropped it (a real
bug: a GPU-uploaded, genuinely-varying field reset to 0.5 on catalog load). The
catalog round-trip guard for that lives in ``tests/storage/test_storage.py``.

Decomposition: pure unit test — no Library, no GL, no RNG dependence on the
result. Mutation-provable: drop any field from ``to_dict`` (or stop casting the
arrays back to float32) and a specific assert below goes red.
"""
from __future__ import annotations

import json

import numpy as np

from flame_sheep.genome import Genome, Transform


def _mk_rich_genome() -> Genome:
    """A genome exercising every serialized field, including the optional ones
    (post_affine, pre_variations, final_xform, var_params)."""
    rng = np.random.default_rng(7)
    t0 = Transform.random(rng)                 # may or may not have post_affine
    t1 = Transform.random(rng)
    t1.post_affine = np.array([1, 0.1, 0, 0, 1, -0.2], dtype=np.float32)
    t1.pre_variations = np.zeros(t1.variations.shape[0], dtype=np.float32)
    t1.pre_variations[3] = 0.4
    t1.var_params = {'julian_power': 2.0, 'splits_x': -0.5}
    t1.color_speed = 0.37
    g = Genome(transforms=[t0, t1], final_xform=Transform.random(rng))
    g.palette = rng.uniform(0, 1, (256, 3)).astype(np.float32)
    g.zoom = 0.6125
    g.rotation = 1.2345
    g.center = np.array([0.11, -0.22], dtype=np.float32)
    g.db_id = 4242
    g.flam3_brightness = 3.75
    g.flam3_gamma = 2.125
    return g


def _assert_transform_equal(a: Transform, b: Transform) -> None:
    assert np.array_equal(a.affine, b.affine)
    assert np.array_equal(a.variations, b.variations)
    assert a.color == b.color
    assert a.color_speed == b.color_speed          # DB serializer DROPS this
    assert a.weight == b.weight
    assert a.var_params == b.var_params
    if a.post_affine is None:
        assert b.post_affine is None
    else:
        assert np.array_equal(a.post_affine, b.post_affine)
    if a.pre_variations is None:
        assert b.pre_variations is None
    else:
        assert np.array_equal(a.pre_variations, b.pre_variations)


def test_genome_roundtrip_exact_through_json():
    g = _mk_rich_genome()
    # Full JSON trip — proves the dict is actually JSON-serializable, not just
    # round-trippable in-process.
    g2 = Genome.from_dict(json.loads(json.dumps(g.to_dict())))

    assert len(g2.transforms) == len(g.transforms)
    for a, b in zip(g.transforms, g2.transforms):
        _assert_transform_equal(a, b)
    assert g2.final_xform is not None
    _assert_transform_equal(g.final_xform, g2.final_xform)

    assert np.array_equal(g2.palette, g.palette)
    assert g2.zoom == g.zoom
    assert g2.rotation == g.rotation
    assert np.array_equal(g2.center, g.center)
    # db_id is NOT genome data — the core serializer excludes it, so it must not
    # appear in the blob and must come back None regardless of the source value.
    assert 'db_id' not in g.to_dict()
    assert g2.db_id is None
    assert g2.flam3_brightness == g.flam3_brightness
    assert g2.flam3_gamma == g.flam3_gamma
    # Arrays come back at the original dtype (value-preserving round-trip).
    assert g2.palette.dtype == np.float32
    assert g2.center.dtype == np.float32
    assert g2.transforms[0].affine.dtype == np.float32


def test_genome_roundtrip_minimal_no_optionals():
    """A bare genome (no final_xform, no post_affine/pre_variations, db_id None)
    must round-trip too — the optional keys are absent, not null-crashing."""
    g = Genome(transforms=[Transform()])
    g.db_id = None
    g2 = Genome.from_dict(json.loads(json.dumps(g.to_dict())))
    assert g2.final_xform is None
    assert g2.db_id is None
    assert g2.transforms[0].post_affine is None
    assert g2.transforms[0].pre_variations is None
    _assert_transform_equal(g.transforms[0], g2.transforms[0])
