"""Independent, versioned final-FBX skin choice; default is a legacy no-op."""
import hashlib
import json
import os
from pathlib import Path

VERSION='fbx-skin-v1'
EXPRESSION='neutral-uv-embedded-basecolor-v1'
SKINS=('default','gray-guide','flesh','flesh-guide','muscle')
MAX_FBX_BYTES=30*1024*1024
MANIFEST=Path(__file__).with_name('skin_assets.v1.json')
# Actual single-FBX neutral16 imports + representative male/female posed imports.
# Acceptance binds the exact asset manifest as well as expression/model/scope/ID.
CSP_CERTIFIED=frozenset((EXPRESSION,
    '1fb0fa382d4fa7d27233b6139cba9c5f1ec575c6b8f86b7deac2a3dc4d447386',
    sha,'full',skin) for sha in (
        '085dd9c65a3a66035f7e0ce9e0a9171b9abb0e33dada92194676d92e00c8b026',
        '712ac75e58efd4d6ccded0b1e931590900015de023327f1320cf6d7d94026ffc')
    for skin in SKINS[1:])


def normalize(value=None):
    if value is None:return dict(version=VERSION,id='default')
    if type(value) is not dict or set(value)!={'version','id'} or value['version']!=VERSION or value['id'] not in SKINS:
        raise ValueError('INVALID_SKIN')
    return dict(version=VERSION,id=value['id'])


def skin_hash(value=None):
    return hashlib.sha256(json.dumps(normalize(value),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def manifest():
    return json.loads(MANIFEST.read_bytes())


def manifest_hash():return hashlib.sha256(MANIFEST.read_bytes()).hexdigest()


def runtime_revision():
    root=Path(__file__).parent
    return hashlib.sha256(b''.join((root/name).read_bytes().replace(b'\r\n',b'\n') for name in
        ('skin_spec.py','skin_fbx.py','skin_worker.py','skin_assets.v1.json'))).hexdigest()


def certified(character_sha,scope,skin):
    return (EXPRESSION,manifest_hash(),character_sha,scope,normalize(skin)['id']) in CSP_CERTIFIED


def checked_asset(character_id,character_sha,skin):
    settings=normalize(skin)
    row=manifest()['models'].get(character_id)
    if not row or row['character_sha256']!=character_sha:raise ValueError('SKIN_UNSUPPORTED')
    root_value=os.getenv('CONVERTER_SKIN_ASSET_ROOT','')
    if not root_value:raise ValueError('SKIN_ASSETS_UNAVAILABLE')
    root=Path(root_value).resolve(strict=True)
    def read(entry):
        path=root/entry['path']
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):raise ValueError('SKIN_ASSETS_UNAVAILABLE')
        data=path.read_bytes()
        if len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('SKIN_ASSETS_UNAVAILABLE')
        return data
    uv=read(row['uv']);texture=read(row['textures'][settings['id']])
    import gzip
    return row,json.loads(gzip.decompress(uv)),texture


def weights_hash(mesh):
    # Blender importer returns the same float32 vertex weights on neutral,
    # posed and shaped fixed FBXs. Preserve vertex-index correspondence exactly.
    rows=[sorted((mesh.vertex_groups[g.group].name,float(g.weight)) for g in v.groups if g.weight>0) for v in mesh.data.vertices]
    return hashlib.sha256(json.dumps(rows,separators=(',',':'),allow_nan=False).encode()).hexdigest()
