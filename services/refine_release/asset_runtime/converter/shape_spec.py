"""Strict body-shape boundary; no bpy and no request-selected filesystem paths.

The operator approves authoring sources as **data**, not code: the registry file
named by ``CONVERTER_SHAPE_REGISTRY`` (``shape-bake.v2``) lists each approved
controller version once under ``sources`` with absolute paths and SHA-256 of the
.blend and ``standin_body.py``. Character recipes only reference a source by its
``rules`` name. A new body version therefore needs a new registry release (built
and hash-verified by ``scripts/build_shape_release.py``), never a code change.

Trust chain: the registry itself is shipped inside the immutable, manifest-pinned
body release; this module re-hashes every source file, checks that the
controller declares the same ``VERSION`` as its ``rules`` and that every control's
default lies on its own slider grid, and fails closed on anything else.
"""
from __future__ import annotations
import hashlib, json, math, os, re
from pathlib import Path

REGISTRY_SCHEMA = 'shape-bake.v2'
SHAPE_SCHEMA = 'body-shape.v1'
RULES_RE = re.compile(r'^[a-z0-9][a-z0-9.-]{0,63}$')
SHA_RE = re.compile(r'^[0-9a-f]{64}$')
KEY_RE = re.compile(r'^[a-z][a-z0-9_]{0,39}$')
# The authoring controller's version literal, read as text. The module is never
# imported here; the Blender job loads it only after these checks pass.
VERSION_RE = re.compile(r"^VERSION\s*=\s*['\"]([^'\"]+)['\"]\s*$", re.M)
GRID_TOLERANCE = 1e-5


class ShapeError(ValueError):
    pass


def digest_shape(character_id, asset_sha, controls):
    # Match JSON.stringify's integer representation for the finite slider domain.
    rows = [[k, int(v) if float(v).is_integer() else v] for k, v in sorted(controls.items())]
    return hashlib.sha256(json.dumps([SHAPE_SCHEMA, character_id, asset_sha, rows], ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def on_grid(value, control):
    n = (value - control['min']) / control['step']
    return abs(n - round(n)) <= GRID_TOLERANCE


def _number(v):
    return not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)


def validate_source(rules, source, verify_files=True):
    """One approved authoring source: absolute files whose bytes and VERSION match."""
    if not isinstance(rules, str) or not RULES_RE.fullmatch(rules):
        raise ShapeError('invalid shape rules')
    if not isinstance(source, dict) or set(source) != {'blend', 'controller', 'blend_sha256', 'controller_sha256'}:
        raise ShapeError('invalid shape source')
    for key in ('blend', 'controller'):
        expected = source[key + '_sha256']
        if not isinstance(expected, str) or not SHA_RE.fullmatch(expected):
            raise ShapeError('invalid shape source checksum')
        path = Path(source[key]) if isinstance(source[key], str) else None
        if path is None or not path.is_absolute():
            raise ShapeError('shape source must be absolute')
        if verify_files:
            if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ShapeError('shape source checksum mismatch')
    if verify_files:
        declared = VERSION_RE.findall(Path(source['controller']).read_text(encoding='utf-8'))
        if declared != [rules]:
            raise ShapeError('controller VERSION does not match shape rules')
    return source


def validate_recipe(recipe):
    """Character recipe without source paths: preset values and the slider grid."""
    if not isinstance(recipe, dict):
        raise ShapeError('invalid shape recipe')
    required = {'rules', 'preset', 'asset', 'asset_sha256', 'defaults', 'controls'}
    if set(recipe) != required:
        raise ShapeError('invalid shape recipe fields')
    if not isinstance(recipe['asset_sha256'], str) or not SHA_RE.fullmatch(recipe['asset_sha256']):
        raise ShapeError('invalid shape asset checksum')
    defaults, controls = recipe['defaults'], recipe['controls']
    if not isinstance(defaults, dict) or not defaults or not all(_number(v) for v in defaults.values()):
        raise ShapeError('invalid shape defaults')
    if not isinstance(controls, list) or not 1 <= len(controls) <= 64:
        raise ShapeError('invalid shape controls')
    seen = set()
    for c in controls:
        if not isinstance(c, dict) or not KEY_RE.fullmatch(str(c.get('key', ''))) or c['key'] in seen:
            raise ShapeError('invalid shape control key')
        seen.add(c['key'])
        if not all(_number(c.get(k)) for k in ('min', 'max', 'step', 'default')):
            raise ShapeError('invalid shape control range')
        if not (c['min'] < c['max'] and c['step'] > 0 and c['min'] <= c['default'] <= c['max']):
            raise ShapeError('invalid shape control range')
        # A preset default off its own grid would make every request for this body fail.
        if not on_grid(c['default'], c):
            raise ShapeError('shape control default off step')
        if controller_key(c['key']) not in defaults:
            raise ShapeError('shape control has no controller field')
    return recipe


def controller_key(key):
    return 'height_cm' if key == 'height' else key


def load_registry(path=None, verify_files=True):
    path = path or os.getenv('CONVERTER_SHAPE_REGISTRY')
    if not path:
        return None
    try:
        registry = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ShapeError('shape registry unreadable') from exc
    if not isinstance(registry, dict) or registry.get('schema') != REGISTRY_SCHEMA:
        raise ShapeError('invalid shape registry')
    sources, characters = registry.get('sources'), registry.get('characters')
    if not isinstance(sources, dict) or not sources or not isinstance(characters, dict) or not characters:
        raise ShapeError('invalid shape registry')
    for rules, source in sources.items():
        validate_source(rules, source, verify_files)
    for recipe in characters.values():
        validate_recipe(recipe)
        if recipe['rules'] not in sources:
            raise ShapeError('shape recipe references an unapproved source')
    return registry


def resolve_shape(character_id, asset_sha, raw=None, expected_source=None):
    """Recipe for one job, or None when the request has no shape.

    ``expected_source`` is what the BFF's catalog promises for this body
    (``{rules, blendSha256, controllerSha256}``); a different approved source is
    rejected before any Blender work starts.
    """
    registry = load_registry()
    if registry is None:
        if raw is not None:
            raise ShapeError('shape baking unavailable')
        return None
    recipe = registry['characters'].get(character_id)
    if recipe is None:
        if raw is not None:
            raise ShapeError('character does not support shape')
        return None
    if recipe['asset_sha256'] != asset_sha:
        raise ShapeError('shape character checksum mismatch')
    source = registry['sources'][recipe['rules']]
    if expected_source is not None:
        if (not isinstance(expected_source, dict) or set(expected_source) != {'rules', 'blendSha256', 'controllerSha256'}):
            raise ShapeError('invalid expected shape source')
        if (expected_source['rules'] != recipe['rules'] or expected_source['blendSha256'] != source['blend_sha256']
                or expected_source['controllerSha256'] != source['controller_sha256']):
            raise ShapeError('shape source mismatch')
    controls = {c['key']: c['default'] for c in recipe['controls']}
    shape_sha = None
    if raw is not None:
        if (not isinstance(raw, dict) or set(raw) != {'schemaVersion', 'controls', 'shapeSha256'} or raw['schemaVersion'] != SHAPE_SCHEMA or not isinstance(raw['controls'], dict)):
            raise ShapeError('invalid shape envelope')
        specs = {c['key']: c for c in recipe['controls']}
        if set(raw['controls']) - set(specs):
            raise ShapeError('unknown shape control')
        controls.update(raw['controls'])
        for k, v in controls.items():
            s = specs[k]
            if not _number(v) or not s['min'] <= v <= s['max']:
                raise ShapeError('shape control out of range')
            if not on_grid(v, s):
                raise ShapeError('shape control off step')
        shape_sha = digest_shape(character_id, asset_sha, controls)
        if raw['shapeSha256'] != shape_sha:
            raise ShapeError('shape digest mismatch')
    full = dict(recipe['defaults'])
    for k, v in controls.items():
        full[controller_key(k)] = v
    return {**recipe, **source, 'values': full, 'shape_sha256': shape_sha}
