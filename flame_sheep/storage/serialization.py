"""Genome / Transform JSON serialization for the SQLite library.

Symmetric `_to_dict` / `_from_dict` helpers. The leading underscore is
historical — they predate the storage package split and a number of
tools and tests already import them by name.
"""

from __future__ import annotations

import json

import numpy as np

from ..genome import Genome, Transform, NUM_VARIATIONS


def _transform_to_dict(tr: Transform) -> dict:
    """Serialize a Transform to a JSON-compatible dict."""
    d = {
        'affine': tr.affine.tolist(),
        'variations': tr.variations.tolist(),
        'color': tr.color,
        'weight': tr.weight,
        'var_params': tr.var_params,
    }
    if tr.post_affine is not None:
        d['post_affine'] = tr.post_affine.tolist()
    if tr.pre_variations is not None:
        d['pre_variations'] = tr.pre_variations.tolist()
    return d


def _transform_from_dict(td: dict) -> Transform:
    """Deserialize a Transform from a JSON dict."""
    tr = Transform()
    tr.affine = np.array(td['affine'], dtype=np.float32)
    # Backwards compat: old genomes have shorter variation arrays
    raw_vars = np.array(td['variations'], dtype=np.float32)
    if len(raw_vars) < NUM_VARIATIONS:
        tr.variations = np.zeros(NUM_VARIATIONS, dtype=np.float32)
        tr.variations[:len(raw_vars)] = raw_vars
    else:
        tr.variations = raw_vars
    tr.color = td['color']
    tr.weight = td['weight']
    tr.var_params = td.get('var_params', {})
    if 'post_affine' in td:
        tr.post_affine = np.array(td['post_affine'], dtype=np.float32)
    if 'pre_variations' in td:
        raw_pre = np.array(td['pre_variations'], dtype=np.float32)
        if len(raw_pre) < NUM_VARIATIONS:
            tr.pre_variations = np.zeros(NUM_VARIATIONS, dtype=np.float32)
            tr.pre_variations[:len(raw_pre)] = raw_pre
        else:
            tr.pre_variations = raw_pre
    return tr


def _genome_to_json(g: Genome) -> str:
    """Serialize a Genome to a JSON string."""
    data = {
        'transforms': [_transform_to_dict(tr) for tr in g.transforms],
        'palette': g.palette.tolist(),
        'zoom': g.zoom,
        'rotation': g.rotation,
        'center': g.center.tolist(),
    }
    if g.final_xform is not None:
        data['final_xform'] = _transform_to_dict(g.final_xform)
    return json.dumps(data, separators=(',', ':'))


def _genome_from_json(s: str) -> Genome:
    """Deserialize a Genome from a JSON string."""
    data = json.loads(s)
    g = Genome()
    g.transforms = [_transform_from_dict(td) for td in data['transforms']]
    if 'final_xform' in data:
        g.final_xform = _transform_from_dict(data['final_xform'])
    g.palette = np.array(data['palette'], dtype=np.float32)
    g.zoom = data['zoom']
    g.rotation = data['rotation']
    g.center = np.array(data['center'], dtype=np.float32)
    return g
