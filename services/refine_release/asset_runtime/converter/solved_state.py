"""Immutable, content-addressed display state; no Blender or HTTP authority.

Canonical form is UTF-8 sorted JSON with every float encoded as its IEEE754
big-endian binary64 hex string. Hash the canonical bytes, not a JS re-stringify.
Paths, clocks, display orbit and download readiness are deliberately absent.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
import struct

STATE_SCHEMA = 'solved-pose-state-v1'
MODEL_SCHEMA = 'interactive-pose-model-v1'
RECEIPT_SCHEMA = 'interactive-fbx-receipt-v1'
MAX_BYTES = 64 * 1024 * 1024
SHA = re.compile(r'^[0-9a-f]{64}$')
IDENTITY_FIELDS = {'ownerBinding', 'selection', 'source', 'body', 'shape', 'hands',
                   'scope', 'mirror', 'camera', 'solver'}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _typed(value):
    if value is None or type(value) in (str, bool):
        return value
    if type(value) is int:
        if abs(value) > 2**53 - 1:
            raise ValueError('integer exceeds cross-runtime exact range')
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('nonfinite state value')
        return {'$f64': struct.pack('>d', value).hex()}
    if type(value) in (list, tuple):
        return [_typed(v) for v in value]
    if type(value) is dict and all(type(k) is str and k.isascii() for k in value):
        if '$f64' in value:
            raise ValueError('reserved canonical key')
        return {k: _typed(v) for k, v in sorted(value.items())}
    raise ValueError('unsupported canonical state value')


def canonical(value) -> bytes:
    return json.dumps(_typed(value), ensure_ascii=True, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('ascii')


def _untyped(value):
    if isinstance(value, dict):
        if set(value) == {'$f64'}:
            raw = value['$f64']
            if not isinstance(raw, str) or not re.fullmatch('[0-9a-f]{16}', raw):
                raise ValueError('invalid canonical float')
            result = struct.unpack('>d', bytes.fromhex(raw))[0]
            if not math.isfinite(result):
                raise ValueError('nonfinite canonical float')
            return result
        return {k: _untyped(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_untyped(v) for v in value]
    return value


def decode(data: bytes):
    if not 0 < len(data) <= MAX_BYTES:
        raise ValueError('state size exceeded')
    result = _untyped(json.loads(data))
    if canonical(result) != data:
        raise ValueError('state is not canonical')
    return result


def sha(value, field):
    if type(value) is not str or not SHA.fullmatch(value):
        raise ValueError(f'invalid SHA256: {field}')


def _matrix(value):
    if not isinstance(value, list) or len(value) != 16 or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in value):
        raise ValueError('expected finite row-major matrix4')


def validate_identity(identity):
    if type(identity) is not dict or set(identity) != IDENTITY_FIELDS:
        raise ValueError('identity fields mismatch')
    from converter.framing import validate_scope
    validate_scope(identity['scope'])
    if type(identity['mirror']) is not bool:
        raise ValueError('mirror must be boolean')
    owner = identity['ownerBinding']
    sha(owner, 'ownerBinding')  # BFF HMAC/opaque ownership binding, never a raw install ID.
    selection = identity['selection']
    if (type(selection) is not dict or set(selection) != {'operation', 'revision', 'person', 'candidate'}
            or type(selection['revision']) is not int or selection['revision'] < 1
            or any(type(selection[k]) is not str or not selection[k] for k in ('operation', 'person', 'candidate'))):
        raise ValueError('invalid selection binding')
    expected = {'source': {'originalBvhSha256', 'refinedBvhSha256', 'refineVersion'},
                'body': {'characterId', 'characterSha256', 'rigProfile'},
                'shape': {'sha256', 'rulesVersion', 'sourceSha256', 'controlsSha256'},
                'hands': {'left', 'right', 'librarySha256', 'rulesVersion'},
                'solver': {'version', 'manifestSha256', 'adapterVersion', 'runtimeSha256'}}
    for field, keys in expected.items():
        item = identity[field]
        if type(item) is not dict or set(item) != keys:
            raise ValueError(f'invalid identity {field}')
        for key, value in item.items():
            if key.endswith('Sha256') or key == 'sha256':
                if value is None and (key == 'refinedBvhSha256' or field == 'shape'):
                    continue
                sha(value, field + '.' + key)
            elif field == 'hands' and key in ('left', 'right') and value is None:
                continue
            elif type(value) is not str or not value:
                raise ValueError(f'invalid identity {field}.{key}')
    camera = identity['camera']
    if (type(camera) is not dict or set(camera) != {'projection', 'viewMatrix', 'projectionMatrix',
            'distanceTorso', 'torsoReferenceSha256', 'baked'}
            or camera['projection'] not in ('orthographic', 'perspective') or type(camera['baked']) is not bool):
        raise ValueError('invalid camera contract')
    _matrix(camera['viewMatrix']); _matrix(camera['projectionMatrix'])
    if type(camera['distanceTorso']) not in (int, float) or not math.isfinite(camera['distanceTorso']) or camera['distanceTorso'] <= 0:
        raise ValueError('invalid torso distance')
    sha(camera['torsoReferenceSha256'], 'torso reference')


def validate_model(model):
    if type(model) is not dict or set(model) != {'schema', 'coordinates', 'bones', 'meshes', 'capabilities'} or model['schema'] != MODEL_SCHEMA:
        raise ValueError('invalid interactive model')
    coords = model['coordinates']
    if (type(coords) is not dict or set(coords) != {'space', 'originBlender', 'matrixLayout', 'baked'}
            or coords['space'] != 'right-handed-y-up-metres-hip-origin'
            or coords['matrixLayout'] != 'row-major' or coords['baked'] != 'rigged_rest'
            or len(coords['originBlender']) != 3 or any(not math.isfinite(v) for v in coords['originBlender'])):
        raise ValueError('invalid model coordinates')
    if model['capabilities'] != {'localHands': False}:
        raise ValueError('capability requires a separately validated variant table')
    bones = model['bones']
    if not isinstance(bones, list) or not 1 <= len(bones) <= 256:
        raise ValueError('invalid bone count')
    names = set()
    for i, bone in enumerate(bones):
        if set(bone) != {'name', 'parent', 'restLocal', 'poseLocal', 'inverseBind'} or type(bone['name']) is not str or not bone['name'] or bone['name'] in names:
            raise ValueError('invalid bone record')
        names.add(bone['name'])
        if type(bone['parent']) is not int or not -1 <= bone['parent'] < i:
            raise ValueError('bone parent must precede child')
        for key in ('restLocal', 'poseLocal', 'inverseBind'):
            _matrix(bone[key])
    meshes = model['meshes']
    if not isinstance(meshes, list) or not 1 <= len(meshes) <= 32:
        raise ValueError('invalid mesh count')
    total = 0
    mesh_names = set()
    for mesh in meshes:
        if set(mesh) != {'name', 'positions', 'normals', 'normalTriangles', 'triangles', 'weights'} or mesh['name'] in mesh_names:
            raise ValueError('invalid mesh fields')
        mesh_names.add(mesh['name'])
        positions = mesh['positions']; total += len(positions)
        if (not 1 <= len(positions) <= 500_000 or total > 500_000 or not 1 <= len(mesh['normals']) <= 3_000_000
                or len(mesh['normalTriangles']) != len(mesh['triangles']) or len(mesh['weights']) != len(positions)):
            raise ValueError('invalid mesh bounds')
        for point in positions + mesh['normals']:
            if len(point) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) for v in point):
                raise ValueError('nonfinite surface')
        if not 1 <= len(mesh['triangles']) <= 1_000_000:
            raise ValueError('invalid triangle count')
        for face in mesh['triangles']:
            if len(face) != 3 or any(type(v) is not int or not 0 <= v < len(positions) for v in face):
                raise ValueError('invalid surface triangle')
        for face in mesh['normalTriangles']:
            if len(face) != 3 or any(type(v) is not int or not 0 <= v < len(mesh['normals']) for v in face):
                raise ValueError('invalid normal triangle')
        for weights in mesh['weights']:
            if not 1 <= len(weights) <= 32:
                raise ValueError('unbound or excessive vertex influences')
            seen = set()
            for bone, weight in weights:
                if type(bone) is not int or not 0 <= bone < len(bones) or bone in seen or type(weight) not in (int, float) or not math.isfinite(weight) or not 0 < weight <= 1:
                    raise ValueError('invalid skin weight')
                seen.add(bone)
            if abs(sum(w for _, w in weights) - 1) > 1e-4:
                raise ValueError('skin weights are not normalized')


@dataclass(frozen=True)
class SolvedPoseState:
    """Store only immutable canonical bytes; getters always return fresh values."""
    canonical_bytes: bytes

    def __post_init__(self):
        if type(self.canonical_bytes) is not bytes:
            raise ValueError('state storage must be immutable bytes')
        payload = decode(self.canonical_bytes)
        if set(payload) != {'schema', 'identity', 'model', 'safety', 'dependencies'} or payload['schema'] != STATE_SCHEMA:
            raise ValueError('invalid solved state envelope')
        validate_identity(payload['identity']); validate_model(payload['model'])
        if payload['safety'] != {'bodyAccepted': True, 'handChecksPassed': True,
                                 'surface': 'existing-adapter-checks-only', 'finalFileVerified': False}:
            raise ValueError('state must not certify final export or full collision safety')
        if payload['dependencies'] != {'initialFbxRequired': True, 'boundary': 'post-frozen-body-pre-final-export-v1'}:
            raise ValueError('unsupported solve boundary')
        # Cache only immutable scalars. Packet construction must not decode a
        # multi-megabyte surface merely to read its operation/revision.
        selection = payload['identity']['selection']
        object.__setattr__(self, '_operation', selection['operation'])
        object.__setattr__(self, '_revision', selection['revision'])
        object.__setattr__(self, '_state_hash', digest(self.canonical_bytes))

    @classmethod
    def create(cls, identity, model):
        return cls(canonical({'schema': STATE_SCHEMA, 'identity': identity, 'model': model,
            'safety': {'bodyAccepted': True, 'handChecksPassed': True,
                       'surface': 'existing-adapter-checks-only', 'finalFileVerified': False},
            'dependencies': {'initialFbxRequired': True, 'boundary': 'post-frozen-body-pre-final-export-v1'}}))

    @property
    def state_hash(self):
        return self._state_hash

    @property
    def payload(self):
        return decode(self.canonical_bytes)

    def packet(self):
        return {'schema': STATE_SCHEMA, 'operation': self._operation, 'revision': self._revision,
                'stateHash': self.state_hash, 'previewState': 'authoritative', 'exportState': 'not-requested',
                'canonicalEncoding': 'typed-f64be-json-v1', 'canonicalState': self.canonical_bytes.decode('ascii')}
