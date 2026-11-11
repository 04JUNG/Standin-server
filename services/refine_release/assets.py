"""Content-addressed original assets, prepared on demand without a pose allowlist."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
VERSION = 'original-bvh-assets-v1'
MAX_BVH_BYTES = 16 * 1024 * 1024

def digest(data): return hashlib.sha256(data).hexdigest()
def sha(path): return digest(Path(path).read_bytes())

def verify_builder():
    manifest = json.loads((ROOT/'asset-runtime-manifest.json').read_text())
    actual = {str(p.relative_to(ROOT/'asset_runtime')) for p in (ROOT/'asset_runtime').rglob('*') if p.suffix in {'.py','.json'}}
    if actual != set(manifest['files']): raise ValueError('asset converter inventory changed')
    for rel, expected in manifest['files'].items():
        p = (ROOT/'asset_runtime'/rel).resolve()
        if not p.is_relative_to((ROOT/'asset_runtime').resolve()) or sha(p) != expected:
            raise ValueError('asset converter source changed')
    return digest((ROOT/'asset-runtime-manifest.json').read_bytes() + (ROOT/'prepare_blender.py').read_bytes())

def configuration():
    body = Path(os.environ['REFINE_RELEASE_BODY']).resolve(strict=True)
    if sha(body) != os.environ['REFINE_RELEASE_BODY_SHA256']:
        raise ValueError('source body changed')
    return body, verify_builder()

def cache_key(bvh_sha, body_sha, builder_sha, blender_version):
    return digest(json.dumps([VERSION,bvh_sha,body_sha,builder_sha,blender_version],separators=(',',':')).encode())

def verified_cache(path, identity):
    try:
        record = json.loads((path/'cache.json').read_text())
        if record['identity'] != identity: return False
        if set(record['files']) != {'source.bvh','source.fbx','rig.json','mesh.npz'}: return False
        return all(sha(path/name)==expected for name,expected in record['files'].items()) and sha(path/'source.bvh')==identity['bvh_sha256']
    except (OSError,ValueError,KeyError): return False

def prepare(pose_id, bvh_bytes, image, destination, cfg):
    if not pose_id or len(pose_id)>512: raise ValueError('invalid library pose identity')
    if not 0 < len(bvh_bytes) <= MAX_BVH_BYTES: raise ValueError('source BVH exceeds limit')
    if not bvh_bytes.lstrip().startswith(b'HIERARCHY') or b'MOTION' not in bvh_bytes:
        raise ValueError('invalid source BVH')
    body, builder = configuration()
    blender_version = subprocess.check_output([cfg['blender'],'--version'],text=True).splitlines()[0]
    identity = dict(version=VERSION,bvh_sha256=digest(bvh_bytes),body_sha256=sha(body),
                    builder_sha256=builder,blender_version=blender_version)
    key = cache_key(identity['bvh_sha256'],identity['body_sha256'],builder,blender_version)
    root = Path(cfg['REFINE_RELEASE_STATE'])/'original-cache'; root.mkdir(parents=True,exist_ok=True)
    cache = root/key
    hit = verified_cache(cache,identity)
    if not hit:
        # Caller holds the worker lock. Publish only a fully validated directory.
        with tempfile.TemporaryDirectory(prefix=key+'.',dir=root) as tmp:
            temporary = Path(tmp); (temporary/'source.bvh').write_bytes(bvh_bytes)
            with (temporary/'prepare.log').open('w') as log:
                subprocess.run([cfg['blender'],'--background','--factory-startup','--python',str(ROOT/'prepare_blender.py'),'--',str(temporary/'source.bvh'),str(body),str(temporary)],
                               check=True,timeout=120,stdout=log,stderr=subprocess.STDOUT)
            record = dict(identity=identity,files={name:sha(temporary/name) for name in ['source.bvh','source.fbx','rig.json','mesh.npz']})
            (temporary/'cache.json').write_text(json.dumps(record))
            if not verified_cache(temporary,identity): raise ValueError('prepared original integrity failed')
            if cache.exists(): shutil.rmtree(cache)
            temporary.replace(cache)
    destination = Path(destination)
    copies = {'source.bvh':'input/original/source.bvh','source.fbx':'input/p1-native/source/source.fbx',
              'rig.json':'input/p1-native/source-source.json','mesh.npz':'base-mesh/source.npz'}
    for name, rel in copies.items():
        out=destination/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(cache/name,out)
    rough=destination/'input/rough/source.png';rough.parent.mkdir(parents=True,exist_ok=True);rough.write_bytes(image)
    row=dict(case_id='source',status='original_library_bvh_confirmed',target_identity='single_depicted_target',
             person_index=0,representation='unspecified',rough='rough/source.png',rough_sha256=digest(image),
             selection=dict(selected_pose_id=pose_id,verified_selected_bvh='original/source.bvh',selected_bvh_sha256=identity['bvh_sha256']))
    (destination/'input/manifest.json').write_text(json.dumps({'rows':[row]}))
    (destination/'original-provenance.json').write_text(json.dumps(dict(identity=identity,cache_key=key,cache_hit=hit,pose_id=pose_id,zero_manual_edits=True),indent=2))
    return 'source'
