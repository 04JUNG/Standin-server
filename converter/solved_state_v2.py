"""Binary canonical encoding of the interactive solved pose state (v2).

Same logical state as ``typed-f64be-json-v1`` (``solved_state.py``), stored as
little-endian IEEE754 binary64 / uint32 / uint16 arrays instead of one
``{"$f64": hex}`` JSON object per float. Values are lossless: decoding a v2
state yields exactly the v1 payload (same Python floats and ints).

Layout (all integers little-endian, every section 8-byte aligned, padding is
zero and part of the canonical form, no trailing bytes)::

    "STNDSPS2" | u32 S | state header (S bytes, v1 canonical JSON) | pad
    model:  "STNDMDL2" | u32 M | model header (M bytes, v1 canonical JSON) | pad
            bones   f64[B*3*16]   restLocal, poseLocal, inverseBind per bone
            per mesh, in header order:
              positions f64[V*3] | normals f64[N*3] | triangles u32[T*3]
              normalTriangles u32[T*3] | weightOffsets u32[V+1]
              weightBones u16[W] | weightValues f64[W]

stateHash = SHA256(all bytes). modelSha256 = SHA256(bytes from the model magic
to the end), i.e. the contiguous model suffix, so readers hash a view.
Validation is the same contract as ``solved_state.validate_model`` plus strict
canonical form (sorted strictly increasing bone ids per vertex, zero padding).
"""
from __future__ import annotations

import copy
import hashlib
import struct

import numpy as np

ENCODING = 'standin-binary-f64le-v2'
MEDIA_TYPE = 'application/vnd.standin.solved-pose-state-v2'
STATE_MAGIC = b'STNDSPS2'
MODEL_MAGIC = b'STNDMDL2'
MAX_HEADER = 4 * 1024 * 1024
STATE_KEYS = {'encoding', 'schema', 'identity', 'safety', 'dependencies'}
MODEL_KEYS = {'schema', 'coordinates', 'capabilities', 'bones', 'meshes'}
MESH_KEYS = {'name', 'vertices', 'normals', 'triangles', 'influences'}
MAX_VERTICES = 500_000
MAX_NORMALS = 3_000_000
MAX_TRIANGLES = 1_000_000
MAX_INFLUENCES = 32
WEIGHT_SUM_TOLERANCE = 1e-4


def _align(offset):
    return (offset + 7) & ~7


def _padding(offset):
    return b'\0' * (_align(offset) - offset)


def is_v2(data) -> bool:
    return bytes(data[:8]) == STATE_MAGIC


def _int(value, low, high, what):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'invalid {what}')
    return value


def _check_meta(meta):
    from converter.solved_state import MODEL_SCHEMA
    if type(meta) is not dict or set(meta) != MODEL_KEYS or meta['schema'] != MODEL_SCHEMA:
        raise ValueError('invalid binary model header')
    coords = meta['coordinates']
    if (type(coords) is not dict or set(coords) != {'space', 'originBlender', 'matrixLayout', 'baked'}
            or coords['space'] != 'right-handed-y-up-metres-hip-origin'
            or coords['matrixLayout'] != 'row-major' or coords['baked'] != 'rigged_rest'
            or type(coords['originBlender']) is not list or len(coords['originBlender']) != 3
            or any(type(v) is not float or not np.isfinite(v) for v in coords['originBlender'])):
        raise ValueError('invalid model coordinates')
    if meta['capabilities'] != {'localHands': False}:
        raise ValueError('capability requires a separately validated variant table')
    bones = meta['bones']
    if type(bones) is not list or not 1 <= len(bones) <= 256:
        raise ValueError('invalid bone count')
    names = set()
    for i, bone in enumerate(bones):
        if (type(bone) is not dict or set(bone) != {'name', 'parent'} or type(bone['name']) is not str
                or not bone['name'] or bone['name'] in names):
            raise ValueError('invalid bone record')
        names.add(bone['name'])
        _int(bone['parent'], -1, i - 1, 'bone parent')
    meshes = meta['meshes']
    if type(meshes) is not list or not 1 <= len(meshes) <= 32:
        raise ValueError('invalid mesh count')
    total = 0; mesh_names = set()
    for mesh in meshes:
        if (type(mesh) is not dict or set(mesh) != MESH_KEYS or type(mesh['name']) is not str
                or not mesh['name'] or mesh['name'] in mesh_names):
            raise ValueError('invalid mesh fields')
        mesh_names.add(mesh['name'])
        vertices = _int(mesh['vertices'], 1, MAX_VERTICES, 'vertex count'); total += vertices
        _int(mesh['normals'], 1, MAX_NORMALS, 'normal count')
        _int(mesh['triangles'], 1, MAX_TRIANGLES, 'triangle count')
        _int(mesh['influences'], vertices, vertices * MAX_INFLUENCES, 'influence count')
        if total > MAX_VERTICES:
            raise ValueError('invalid mesh bounds')


class _Reader:
    def __init__(self, data, offset):
        self.data = data; self.offset = offset

    def array(self, dtype, count):
        dtype = np.dtype(dtype)
        size = dtype.itemsize * count
        end = self.offset + size
        if end > len(self.data):
            raise ValueError('binary state truncated')
        value = np.frombuffer(self.data, dtype=dtype, count=count, offset=self.offset)
        aligned = _align(end)
        if aligned > len(self.data) or any(self.data[end:aligned]):
            raise ValueError('binary state padding is not canonical')
        self.offset = aligned
        return value


def _header(data, offset, magic):
    if bytes(data[offset:offset + 8]) != magic:
        raise ValueError('binary state magic mismatch')
    if offset + 12 > len(data):
        raise ValueError('binary state truncated')
    (size,) = struct.unpack_from('<I', data, offset + 8)
    if not 0 < size <= MAX_HEADER or offset + 12 + size > len(data):
        raise ValueError('invalid binary header size')
    from converter.solved_state import decode
    raw = bytes(data[offset + 12:offset + 12 + size])
    value = decode(raw)  # strict v1 canonical JSON, including typed floats
    end = offset + 12 + size
    aligned = _align(end)
    if aligned > len(data) or any(data[end:aligned]):
        raise ValueError('binary header padding is not canonical')
    return value, aligned


class ModelView:
    """Validated read-only arrays of one binary model. ``to_dict`` gives v1 form."""

    def __init__(self, meta, bone_matrices, meshes):
        self.meta = meta; self.bone_matrices = bone_matrices; self.meshes = meshes

    def bones(self):
        names = ('restLocal', 'poseLocal', 'inverseBind')
        values = self.bone_matrices.tolist()
        return [dict(name=b['name'], parent=b['parent'],
                     **{names[k]: values[i][k] for k in range(3)}) for i, b in enumerate(self.meta['bones'])]

    def to_dict(self):
        meshes = []
        for header, mesh in zip(self.meta['meshes'], self.meshes):
            offsets = mesh['weightOffsets'].tolist(); bones = mesh['weightBones'].tolist()
            values = mesh['weightValues'].tolist()
            weights = [[[bones[j], values[j]] for j in range(offsets[i], offsets[i + 1])]
                       for i in range(header['vertices'])]
            meshes.append(dict(name=header['name'], positions=mesh['positions'].tolist(),
                normals=mesh['normals'].tolist(), normalTriangles=mesh['normalTriangles'].tolist(),
                triangles=mesh['triangles'].tolist(), weights=weights))
        # Fresh containers on every call: callers may mutate the returned model.
        return dict(schema=self.meta['schema'], coordinates=copy.deepcopy(self.meta['coordinates']),
                    bones=self.bones(), meshes=meshes, capabilities=copy.deepcopy(self.meta['capabilities']))


def decode_model(data, offset=0):
    """Validate one model container; it must extend exactly to the end of data."""
    meta, start = _header(data, offset, MODEL_MAGIC)
    _check_meta(meta)
    reader = _Reader(data, start)
    bone_count = len(meta['bones'])
    matrices = reader.array('<f8', bone_count * 48)
    if not np.isfinite(matrices).all():
        raise ValueError('expected finite row-major matrix4')
    meshes = []
    for header in meta['meshes']:
        v, n, t, w = header['vertices'], header['normals'], header['triangles'], header['influences']
        positions = reader.array('<f8', v * 3); normals = reader.array('<f8', n * 3)
        triangles = reader.array('<u4', t * 3); normal_triangles = reader.array('<u4', t * 3)
        offsets = reader.array('<u4', v + 1); bone_ids = reader.array('<u2', w)
        values = reader.array('<f8', w)
        if not (np.isfinite(positions).all() and np.isfinite(normals).all()):
            raise ValueError('nonfinite surface')
        if int(triangles.max()) >= v:
            raise ValueError('invalid surface triangle')
        if int(normal_triangles.max()) >= n:
            raise ValueError('invalid normal triangle')
        if offsets[0] != 0 or offsets[-1] != w:
            raise ValueError('invalid skin weight offsets')
        counts = np.diff(offsets.astype(np.int64))
        if counts.min() < 1 or counts.max() > MAX_INFLUENCES:
            raise ValueError('unbound or excessive vertex influences')
        if int(bone_ids.max()) >= bone_count:
            raise ValueError('invalid skin weight')
        same_vertex = np.ones(w - 1, dtype=bool) if w > 1 else np.zeros(0, dtype=bool)
        if w > 1:
            same_vertex[offsets[1:-1].astype(np.int64) - 1] = False
            if not (bone_ids[1:][same_vertex] > bone_ids[:-1][same_vertex]).all():
                raise ValueError('skin joint order is not canonical')
        if not (np.isfinite(values).all() and (values > 0).all() and (values <= 1).all()):
            raise ValueError('invalid skin weight')
        sums = np.add.reduceat(values, offsets[:-1].astype(np.int64))
        if (np.abs(sums - 1) > WEIGHT_SUM_TOLERANCE).any():
            raise ValueError('skin weights are not normalized')
        meshes.append(dict(positions=positions.reshape(v, 3), normals=normals.reshape(n, 3),
            triangles=triangles.reshape(t, 3), normalTriangles=normal_triangles.reshape(t, 3),
            weightOffsets=offsets, weightBones=bone_ids, weightValues=values))
    if reader.offset != len(data):
        raise ValueError('binary model has trailing bytes')
    return ModelView(meta, matrices.reshape(bone_count, 3, 16), meshes)


def _matrices(bones):
    rows = []
    for bone in bones:
        for key in ('restLocal', 'poseLocal', 'inverseBind'):
            value = bone[key]
            if not isinstance(value, list) or len(value) != 16:
                raise ValueError('expected finite row-major matrix4')
            rows.append(value)
    return np.asarray(rows, dtype='<f8')


def _vectors(values, what):
    array = np.asarray(values, dtype='<f8')
    if array.ndim != 2 or array.shape[1] != 3 or not len(array):
        raise ValueError(f'invalid {what}')
    return array


def _indices(values, what):
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != 3 or not len(array) or array.dtype.kind not in 'iu':
        raise ValueError(f'invalid {what}')
    if array.min() < 0 or array.max() > 0xFFFFFFFF:
        raise ValueError(f'invalid {what}')
    return array.astype('<u4')


def encode_model(model) -> bytes:
    """Encode a v1-shaped model dict; the result is validated by decode_model."""
    from converter.solved_state import canonical
    if type(model) is not dict or set(model) != MODEL_KEYS:
        raise ValueError('invalid interactive model')
    sections = [_matrices(model['bones'])]
    meshes = []
    for mesh in model['meshes']:
        if type(mesh) is not dict or set(mesh) != {'name', 'positions', 'normals', 'normalTriangles', 'triangles', 'weights'}:
            raise ValueError('invalid mesh fields')
        weights = mesh['weights']
        counts = [len(entry) for entry in weights]
        bone_ids = [bone for entry in weights for bone, _ in entry]
        values = [weight for entry in weights for _, weight in entry]
        if any(type(b) is not int for b in bone_ids) or any(type(v) not in (int, float) for v in values):
            raise ValueError('invalid skin weight')
        if bone_ids and (min(bone_ids) < 0 or max(bone_ids) > 0xFFFF):
            raise ValueError('invalid skin weight')
        offsets = np.zeros(len(counts) + 1, dtype='<u4'); offsets[1:] = np.cumsum(counts, dtype=np.int64)
        positions = _vectors(mesh['positions'], 'positions'); normals = _vectors(mesh['normals'], 'normals')
        triangles = _indices(mesh['triangles'], 'triangles')
        normal_triangles = _indices(mesh['normalTriangles'], 'normal triangles')
        sections += [positions, normals, triangles, normal_triangles, offsets,
                     np.asarray(bone_ids, dtype='<u2'), np.asarray(values, dtype='<f8')]
        meshes.append(dict(name=mesh['name'], vertices=len(positions), normals=len(normals),
                           triangles=len(triangles), influences=len(values)))
    meta = dict(schema=model['schema'], coordinates=model['coordinates'], capabilities=model['capabilities'],
                bones=[dict(name=b['name'], parent=b['parent']) for b in model['bones']], meshes=meshes)
    header = canonical(meta)
    parts = [MODEL_MAGIC, struct.pack('<I', len(header)), header, _padding(12 + len(header))]
    for section in sections:
        raw = np.ascontiguousarray(section).tobytes()
        parts += [raw, _padding(len(raw))]
    data = b''.join(parts)
    decode_model(data)
    return data


def encode_state(head, model_bytes) -> bytes:
    """head = {schema, identity, safety, dependencies}; model_bytes from encode_model."""
    from converter.solved_state import canonical
    header = canonical({**head, 'encoding': ENCODING})
    data = b''.join([STATE_MAGIC, struct.pack('<I', len(header)), header,
                     _padding(12 + len(header)), model_bytes])
    return data


def decode_state(data):
    """Return (head without encoding, model offset, ModelView). Bytes are not copied."""
    head, model_offset = _header(data, 0, STATE_MAGIC)
    if type(head) is not dict or set(head) != STATE_KEYS or head['encoding'] != ENCODING:
        raise ValueError('invalid binary state header')
    view = decode_model(data, model_offset)
    return {k: v for k, v in head.items() if k != 'encoding'}, model_offset, view


def model_digest(data, model_offset=0) -> str:
    return hashlib.sha256(memoryview(data)[model_offset:]).hexdigest()
