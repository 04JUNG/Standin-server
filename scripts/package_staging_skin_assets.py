"""Package exact per-model neutral UVs for the staging anatomy release (Blender).

Existing reviewed skin assets are preserved. New anatomy models receive the
solid flesh skin; guide artwork needs its own anatomical authoring review.
Run after downloading the pinned body release, with an existing skin asset root.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from converter.outline_worker import import_source, tree
from converter.outline_fbx import one
from converter.skin_spec import weights_hash


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--existing-assets', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
    manifest_path = ROOT / 'converter/skin_assets.v1.json'
    manifest = json.loads(manifest_path.read_bytes())
    release = json.loads((ROOT / 'config/staging-anatomy-release.json').read_bytes())
    registry = json.loads((a.models / 'characters.json').read_bytes())['characters']
    a.out.mkdir(parents=True, exist_ok=True)
    for row in manifest['models'].values():
        for entry in [row['uv'], *row['textures'].values(), *row['preview_textures'].values()]:
            data = (a.existing_assets / entry['path']).read_bytes()
            assert len(data) == entry['bytes'] and digest(data) == entry['sha256']
            (a.out / entry['path']).write_bytes(data)
    flesh = next(iter(manifest['models'].values()))
    manifest['revision'] = 'staging-anatomy-flesh-20261112-v1'
    manifest['authoring_sha256'][Path(__file__).name] = digest(Path(__file__).read_bytes())
    for cid in release['characterIds']:
        source = a.models / 'fbx' / (cid + '.fbx')
        source_sha = digest(source.read_bytes())
        assert source_sha == registry[cid]['sha256'], cid
        root, _ = tree(source)
        geometry = [g for g in one(root, b'Objects').elems if g.id == b'Geometry' and g.props[2] == b'Mesh']
        assert len(geometry) == 1
        indices = one(geometry[0], b'PolygonVertexIndex').props[0]
        arm, meshes = import_source(source)
        assert len(meshes) == 1
        obj = meshes[0]
        points = [tuple(v.co) for v in obj.data.vertices]
        weights = weights_hash(obj)
        assert not obj.data.uv_layers, 'never replace an existing UV layer'
        faces, face = [], []
        for i in indices:
            face.append(-i-1 if i < 0 else i)
            if i < 0:
                faces.append(face)
                face = []
        assert len(faces) == len(obj.data.polygons)
        # Solid flesh is spatially uniform. Exact corner-center UVs avoid
        # unnecessary unwrap seams and remain valid under every supported shape.
        values = [[.5, .5]] * len(indices)
        for expected, poly in zip(faces, obj.data.polygons):
            assert expected == list(poly.vertices)
        obj.data.calc_loop_triangles()
        triangles = [list(t.vertices) for t in obj.data.loop_triangles]
        assert all(len(f) == 3 for f in faces) and faces == triangles
        data = gzip.compress(json.dumps(dict(uv=values, triangles=triangles, source_sha256=source_sha), separators=(',', ':'), allow_nan=False).encode(), mtime=0)
        name = cid + '-uv.json.gz'
        (a.out / name).write_bytes(data)
        manifest['models'][cid] = dict(
            character_sha256=source_sha, vertices=len(obj.data.vertices), bones=len(arm.data.bones),
            polygon_indices_sha256=digest(indices.tobytes()), weights_sha256=weights,
            triangles_sha256=digest(json.dumps(triangles, separators=(',', ':')).encode()),
            uv=dict(path=name, bytes=len(data), sha256=digest(data)),
            textures={'flesh': flesh['textures']['flesh']},
            preview_textures={'flesh': flesh['preview_textures']['flesh']},
            preview_mesh='mesh_000', preview_resolution=2048)
        print('PACKAGED', cid, len(obj.data.vertices), len(arm.data.bones), flush=True)
    raw = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
    manifest_path.write_bytes(raw)
    (a.out / 'manifest.json').write_bytes(raw)
    print('MANIFEST', digest(raw), flush=True)


if __name__ == '__main__':
    main()
