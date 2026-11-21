"""Immutable, operator-built neutral preset bodies; no runtime cache writes."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from pathlib import Path
import sys

from converter.protocol import EXPECTED_BLENDER_VERSION, EXPECTED_BLENDER_BUILD_HASH
from converter.shape_spec import controller_key, validate_source
from converter.final_artifact import valid_receipt

SCHEMA = 'prepared-neutral-body.v1'
MAX_FBX_BYTES = 64 * 1024 * 1024
MAX_RECEIPT_BYTES = 256 * 1024


def preset_values(recipe):
    values = dict(recipe['defaults'])
    for control in recipe['controls']:
        values[controller_key(control['key'])] = control['default']
    return values


def identity(recipe):
    """Explicit default-valued shape envelopes may share the preset prototype."""
    values = recipe['values']
    if values != preset_values(recipe) or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in values.values()
    ):
        return None
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ('prepared_body.py', 'shape_bake.py', 'shape_spec.py', 'retarget.py', 'bone_map.py', 'final_artifact.py'):
        digest.update(name.encode())
        digest.update((root / name).read_bytes().replace(b'\r\n', b'\n'))
    return {
        'schema': SCHEMA, 'rules': recipe['rules'], 'preset': recipe['preset'],
        'asset': recipe['asset'], 'asset_sha256': recipe['asset_sha256'],
        'blend_sha256': recipe['blend_sha256'],
        'controller_sha256': recipe['controller_sha256'],
        'values': {k: int(v) if float(v).is_integer() else v for k, v in sorted(values.items())},
        'runtime': [EXPECTED_BLENDER_VERSION, EXPECTED_BLENDER_BUILD_HASH,
                    sys.platform, platform.machine(), digest.hexdigest()],
    }


def body_key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _read(path, limit):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError('invalid prepared file')
    # Reject symlinked parent components too. The bundle must be immutable data.
    if path.absolute() != path.resolve():
        raise ValueError('indirect prepared file')
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    if not 0 < len(data) <= limit:
        raise ValueError('prepared file size')
    return data


def _rest_frames(value, bones):
    if not isinstance(value, dict) or type(bones) is not int or len(value) != bones or not 1 <= bones <= 256:
        raise ValueError('prepared rest bones')
    for name, matrix in value.items():
        if not isinstance(name, str) or not 1 <= len(name) <= 128:
            raise ValueError('prepared bone name')
        if (not isinstance(matrix, list) or len(matrix) != 4
                or any(not isinstance(row, list) or len(row) != 4 for row in matrix)
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) for row in matrix for v in row)
                or matrix[3] != [0, 0, 0, 1]):
            raise ValueError('prepared bone frame')
        a, b, c = matrix[:3]
        determinant = a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
        if abs(determinant) < 1e-18:
            raise ValueError('degenerate prepared rest frame')
    return value


def read_prepared(recipe, entry, *, check_runtime=True):
    expected = identity(recipe)
    if expected is None:
        raise ValueError('custom shapes are not prepared presets')
    validate_source(recipe['rules'], {k: recipe[k] for k in
        ('blend', 'controller', 'blend_sha256', 'controller_sha256')})
    raw = _read(entry['path'], MAX_RECEIPT_BYTES)
    if hashlib.sha256(raw).hexdigest() != entry['sha256']:
        raise ValueError('prepared receipt checksum')
    receipt = json.loads(raw)
    if not check_runtime:
        # A Linux-built bundle may be packaged on a different host; runtime use
        # still requires an exact platform match. Code/build lineage never changes.
        runtime = receipt['identity'].get('runtime', [])
        if len(runtime) != 5 or not all(isinstance(v, str) for v in runtime):
            raise ValueError('prepared runtime')
        expected['runtime'][2:4] = runtime[2:4]
    if receipt['identity'] != expected or receipt['key'] != body_key(expected):
        raise ValueError('prepared identity')
    source = Path(entry['path']).parent.parent / 'fbx' / ('prepared-' + receipt['key'] + '.fbx')
    data = _read(source, MAX_FBX_BYTES)
    digest = hashlib.sha256(data).hexdigest()
    qa = receipt['neutral_validation']
    if (not data.startswith(b'Kaydara FBX Binary  \x00\x1a\x00')
            or digest != receipt['fbx_sha256'] or len(data) != receipt['fbx_bytes']
            or not valid_receipt(qa, digest, 'full')):
        raise ValueError('prepared artifact integrity')
    rest = _rest_frames(receipt['rest_frames'], qa['bones'])
    return {'data': data, 'fbx_sha256': digest, 'rest_frames': rest, 'key': receipt['key']}


def restore(recipe, destination):
    """Return a private neutral FBX/rest snapshot, or safely use legacy generation.

    The API cannot provide this path. `resolve_shape` obtains it exclusively
    from the already pinned operator shape registry. No custom shape is stored.
    """
    if os.getenv('CONVERTER_PREPARED_PRESETS_ENABLED', '1') != '1':
        return None, 'disabled'
    expected = identity(recipe)
    if expected is None:
        return None, 'custom_shape'
    entry = recipe.get('prepared_neutral')
    if not isinstance(entry, dict) or set(entry) != {'path', 'sha256'}:
        return None, 'not_prepared'
    try:
        prototype = read_prepared(recipe, entry)
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        target = destination / 'model.fbx'
        target.write_bytes(prototype.pop('data'))  # never mutate a shared prototype
        return {**prototype, 'path': target}, 'prepared'
    except (OSError, ValueError, TypeError, KeyError):
        # Unusable optimization inputs never substitute another body or weaken
        # the original source/shape/pose/final-FBX validation path.
        return None, 'invalid_prepared'
