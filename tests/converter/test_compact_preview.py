"""Dense registered surfaces fit the static transport without changing triangles."""
from array import array
import json
import struct

import pytest

from converter.preview_model import MAX_MODEL_BYTES, encode_surface, partition_surface


def document(data):
    size = struct.unpack_from('<I', data, 12)[0]
    return json.loads(data[20:20 + size])


def test_parts_preserve_split_normals_and_winding():
    positions = [0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.]
    normals = [0., 0., 1.] * 3 + [0., 0., -1.]
    indices = [0, 1, 2, 3, 2, 1]
    parts = partition_surface(positions, normals, indices, max_vertices=3)
    assert len(parts) == 2
    actual = [(tuple(p[3*i:3*i+3]), tuple(n[3*i:3*i+3]))
              for p, n, faces, _ in parts for i in faces]
    expected = [(tuple(positions[3*i:3*i+3]), tuple(normals[3*i:3*i+3])) for i in indices]
    assert actual == expected
    assert all(component == 5123 for _, _, _, component in parts)


def test_dense_child_surface_fits_original_byte_and_accessor_limits():
    rows, cols = 300, 600
    positions = array('f', (v for y in range(rows + 1) for x in range(cols + 1)
                            for v in (float(x), float(y), 0.)))
    normals = array('f', [0., 0., 1.]) * ((rows + 1) * (cols + 1))
    indices = array('I')
    for y in range(rows):
        for x in range(cols):
            a = y * (cols + 1) + x
            indices.extend((a, a + 1, a + cols + 1, a + 1, a + cols + 2, a + cols + 1))
    data = encode_surface(positions, normals, indices, {'source': 'dense-fixture'})
    doc = document(data)
    parts = doc['meshes'][0]['primitives']
    assert len(data) < MAX_MODEL_BYTES == 8 * 1024 * 1024
    assert 1 < len(parts) <= 8
    assert sum(doc['accessors'][p['indices']]['count'] for p in parts) == len(indices)
    assert all(doc['accessors'][p['indices']]['componentType'] == 5123 for p in parts)
    assert all(a['count'] <= 500_000 for a in doc['accessors'])
    assert all(v['byteOffset'] % 4 == 0 for v in doc['bufferViews'])
    assert all(v['byteOffset'] + v['byteLength'] <= doc['buffers'][0]['byteLength']
               for v in doc['bufferViews'])


def test_surface_still_rejects_more_than_bounded_parts():
    positions = list(range(27 * 3))
    with pytest.raises(ValueError, match='too many parts'):
        partition_surface(positions, positions, list(range(27)), max_vertices=3)
