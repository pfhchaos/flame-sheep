"""Parse Electric Sheep flam3 XML genomes into flame-sheep Genome objects.

Handles both newer (gen 242+) and older (gen 165-198) XML formats.
Silently drops unknown variations (move, split_shift) to match flam3's
rendering behavior — those sheep were rated based on renders without
those variations.
"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET

import numpy as np

from .genome import Genome, Transform, NUM_VARIATIONS
from .variations._registry import Variation, VAR_PARAMS_SPEC

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XML attribute name → Variation index mapping
# ---------------------------------------------------------------------------

def _build_var_name_map() -> dict[str, int]:
    """Build mapping from flam3 XML attribute names to variation indices."""
    name_map: dict[str, int] = {}
    for attr_name in dir(Variation):
        if attr_name.startswith('_'):
            continue
        val = getattr(Variation, attr_name)
        if not isinstance(val, int):
            continue
        # Default: lowercase enum name
        name_map[attr_name.lower()] = val

    # Aliases where XML name differs from enum name
    name_map.update({
        'disc':    Variation.DISK,
        'exp':     Variation.EXP_FUNC,
        'log':     Variation.LOG_FUNC,
        'sin':     Variation.SIN_FUNC,
        'cos':     Variation.COS_FUNC,
        'tan':     Variation.TAN_FUNC,
        'sec':     Variation.SEC_FUNC,
        'csc':     Variation.CSC_FUNC,
        'cot':     Variation.COT_FUNC,
        'sinh':    Variation.SINH_FUNC,
        'cosh':    Variation.COSH_FUNC,
        'tanh':    Variation.TANH_FUNC,
        'sech':    Variation.SECH_FUNC,
        'csch':    Variation.CSCH_FUNC,
        'coth':    Variation.COTH_FUNC,
        'secant':  Variation.SECANT_FUNC,
        'modulus': Variation.MODULUS_FUNC,
        'polar2':  Variation.POLAR2,
    })
    return name_map


VAR_NAME_MAP = _build_var_name_map()

# Build reverse param map: xml_param_name → (variation_index, param_position)
# e.g. 'curl_c1' → we know it belongs to CURL variation
_PARAM_TO_VAR: dict[str, tuple[int, str]] = {}
for _var_idx, _param_names in VAR_PARAMS_SPEC.items():
    for _pname in _param_names:
        _PARAM_TO_VAR[_pname] = (_var_idx, _pname)

# Attributes on <xform> that are NOT variations or params
_XFORM_META = {
    'weight', 'color', 'color_speed', 'animate', 'opacity',
    'coefs', 'post', 'symmetry', 'chaos', 'plotmode', 'var_color',
}

# Variations to silently ignore (unknown plugins, flam3 dropped them too)
_IGNORED_VARIATIONS = {'move', 'split_shift'}

# Param prefixes for ignored variations
_IGNORED_PARAMS = {'move_x', 'move_y', 'split_shift'}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_affine(coefs_str: str) -> np.ndarray:
    """Parse flam3 'v0 v1 v2 v3 v4 v5' affine coefficient string.

    flam3 stores affine coefficients as column-major pairs:
        c[0][0] c[0][1]  c[1][0] c[1][1]  c[2][0] c[2][1]
        (v0     v1       v2      v3        v4      v5)

    and applies them as:
        tx = c[0][0]*x + c[1][0]*y + c[2][0]   = v0*x + v2*y + v4
        ty = c[0][1]*x + c[1][1]*y + c[2][1]   = v1*x + v3*y + v5

    Our convention is row-major [a, b, c, d, e, f]:
        nx = a*x + b*y + c
        ny = d*x + e*y + f

    So we reorder: a=v0, b=v2, c=v4, d=v1, e=v3, f=v5.
    """
    vals = [float(v) for v in coefs_str.split()]
    if len(vals) != 6:
        raise ValueError(f'Expected 6 affine coefficients, got {len(vals)}: {coefs_str}')
    v0, v1, v2, v3, v4, v5 = vals
    reordered = [v0, v2, v4, v1, v3, v5]
    arr = np.array(reordered, dtype=np.float64)
    np.clip(arr, -1e30, 1e30, out=arr)
    return arr.astype(np.float32)


def _parse_palette_colors(flame_elem: ET.Element) -> np.ndarray:
    """Parse palette from <color index="N" rgb="R G B"/> elements."""
    palette = np.zeros((256, 3), dtype=np.float32)
    for color_elem in flame_elem.findall('color'):
        idx = int(color_elem.get('index', '0'))
        rgb = color_elem.get('rgb', '0 0 0').split()
        if idx < 256 and len(rgb) >= 3:
            palette[idx] = [float(rgb[0]) / 255.0,
                           float(rgb[1]) / 255.0,
                           float(rgb[2]) / 255.0]
    return palette


def _parse_xform(elem: ET.Element) -> Transform:
    """Parse an <xform> or <finalxform> element into a Transform."""
    tr = Transform()

    # Affine coefficients
    coefs = elem.get('coefs')
    if coefs:
        tr.affine = _parse_affine(coefs)

    # Post-affine
    post = elem.get('post')
    if post:
        tr.post_affine = _parse_affine(post)

    # Color and weight (older formats may have 'color' as "value speed")
    color_str = elem.get('color', '0')
    tr.color = float(color_str.split()[0])
    tr.color_speed = float(elem.get('color_speed', '0.5'))
    tr.weight = float(elem.get('weight', '1'))

    # Parse all remaining attributes as variations or params
    variations = np.zeros(NUM_VARIATIONS, dtype=np.float32)
    var_params: dict[str, float] = {}
    pre_variations = None

    for attr_name, attr_val in elem.attrib.items():
        if attr_name in _XFORM_META:
            continue

        # Check if it's a known variation
        if attr_name in VAR_NAME_MAP:
            var_idx = VAR_NAME_MAP[attr_name]
            variations[var_idx] = float(attr_val)
            continue

        # Check if it's pre_blur (blur in pre-variation stage)
        if attr_name == 'pre_blur':
            if pre_variations is None:
                pre_variations = np.zeros(NUM_VARIATIONS, dtype=np.float32)
            pre_variations[Variation.BLUR] = float(attr_val)
            continue

        # Check if it's a known param
        if attr_name in _PARAM_TO_VAR:
            _, param_name = _PARAM_TO_VAR[attr_name]
            var_params[param_name] = float(attr_val)
            continue

        # Check param names that match our internal naming with prefix mapping
        # e.g. XML 'oscilloscope_separation' → our 'osc_separation'
        mapped = _map_param_name(attr_name)
        if mapped:
            var_params[mapped] = float(attr_val)
            continue

        # Silently ignored variations/params
        if attr_name in _IGNORED_VARIATIONS or attr_name in _IGNORED_PARAMS:
            continue

        # Unknown attribute — could be a param for an unknown variation
        log.debug('Unknown xform attribute: %s="%s"', attr_name, attr_val)

    tr.variations = variations
    tr.var_params = var_params
    if pre_variations is not None:
        tr.pre_variations = pre_variations

    return tr


def _map_param_name(xml_name: str) -> str | None:
    """Map flam3 XML param names to our internal param names.

    Handles cases where our naming convention differs from flam3's,
    e.g. flam3 uses 'oscilloscope_separation' but we use 'osc_separation'.
    """
    # Direct lookup first (covers most cases)
    if xml_name in _PARAM_TO_VAR:
        return _PARAM_TO_VAR[xml_name][1]

    # Mapping for name differences between flam3 XML and our param names
    _RENAMES = {
        'oscilloscope_separation': 'osc_separation',
        'oscilloscope_frequency': 'osc_frequency',
        'oscilloscope_amplitude': 'osc_amplitude',
        'oscilloscope_damping': 'osc_damping',
        # disc2 uses 'disc2_rot' in XML but we store 'disc2_twist' + 'disc2_cosadd' + 'disc2_sinadd'
        # The XML 'disc2_rot' is actually the 'disc2_twist' param
        'disc2_rot': 'disc2_twist',
        # juliascope params use 'juliascope_' prefix in XML but share
        # 'julian_power'/'julian_dist' internally with julian
        'juliascope_power': 'julian_power',
        'juliascope_dist': 'julian_dist',
        # rectangles params
        'rectangles_x': 'rect_x',
        'rectangles_y': 'rect_y',
    }
    return _RENAMES.get(xml_name)


def parse_genome_xml(xml_str: str) -> Genome | None:
    """Parse a flam3 XML string into a Genome object.

    Returns None if the XML is malformed or unparseable.
    """
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        # Try wrapping in a root element in case it's a fragment
        try:
            root = ET.fromstring(f'<root>{xml_str}</root>')
            root = root.find('flame')
            if root is None:
                return None
        except ET.ParseError:
            return None

    if root.tag != 'flame':
        root = root.find('flame')
        if root is None:
            return None

    g = Genome()

    # Parse global attributes
    #
    # Coordinate system mapping (flam3 → ours):
    # flam3: pixel = rotate(point - center, -rot) * scale + (w/2, h/2)
    # ours:  pixel = ((rotate(point, rot) - center) * zoom + 1) * size/2
    #
    # zoom = flam3_scale / (flam3_width / 2)   — resolution-independent
    # rotation = -flam3_rotate (flam3 rotates opposite direction)
    # center = rotate(flam3_center, our_rotation) (our shader rotates before centering)
    size_parts = root.get('size', '800 600').split()
    width = float(size_parts[0])
    flam3_center = [float(v) for v in root.get('center', '0 0').split()]
    g.zoom = float(root.get('scale', '100')) / (width * 0.5)
    g.rotation = float(root.get('rotate', '0')) * np.pi / 180.0  # degrees → radians
    g.flam3_brightness = float(root.get('brightness', '4'))
    g.flam3_gamma = float(root.get('gamma', '4'))
    # Rotate flam3 center into our coordinate system
    cos_r = np.cos(g.rotation)
    sin_r = np.sin(g.rotation)
    g.center = np.array([
        cos_r * flam3_center[0] - sin_r * flam3_center[1],
        sin_r * flam3_center[0] + cos_r * flam3_center[1],
    ], dtype=np.float32)

    # Parse palette
    g.palette = _parse_palette_colors(root)

    # Parse transforms
    g.transforms = []
    for xform_elem in root.findall('xform'):
        tr = _parse_xform(xform_elem)
        g.transforms.append(tr)

    # Parse final xform
    final_elem = root.find('finalxform')
    if final_elem is not None:
        g.final_xform = _parse_xform(final_elem)

    if not g.transforms:
        log.warning('Genome has no transforms')
        return None

    return g


def load_esheep_genomes(db_path: str,
                        min_rating: int = 0,
                        generation: int | None = None,
                        ) -> list[tuple[Genome, int, int, int]]:
    """Load and parse Electric Sheep genomes from the scrape database.

    Returns list of (genome, rating, generation, sheep_id) tuples.
    Skips genomes that fail to parse.
    """
    import sqlite3

    db = sqlite3.connect(db_path)
    query = 'SELECT generation, sheep_id, rating, genome_xml FROM sheep WHERE genome_xml IS NOT NULL'
    params: list = []

    if min_rating > 0:
        query += ' AND rating >= ?'
        params.append(min_rating)
    if generation is not None:
        query += ' AND generation = ?'
        params.append(generation)

    query += ' ORDER BY rating DESC'

    results = []
    failed = 0
    for gen, sid, rating, xml in db.execute(query, params):
        genome = parse_genome_xml(xml)
        if genome is None:
            failed += 1
            continue
        results.append((genome, rating or 0, gen, sid))

    if failed:
        log.warning('Failed to parse %d/%d genomes', failed, failed + len(results))

    db.close()
    return results
