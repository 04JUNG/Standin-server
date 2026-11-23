"""Derive new skin pins only after exact neutral topology/weight/UV correspondence.

No inherited CSP certification is granted. This creates review data, not an
active skin manifest replacement or approval alias.
"""
import argparse,copy,gzip,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from converter.shape_bake import import_full_character
from converter import retarget as rt
from converter.skin_spec import weights_hash
from converter.outline_worker import tree
from converter.outline_fbx import one

def sha(data):return hashlib.sha256(data).hexdigest()

ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--reference-root',type=Path,required=True)
a=ap.parse_args(sys.argv[sys.argv.index('--')+1:])
manifest=json.loads((ROOT/'converter/skin_assets.v1.json').read_bytes())
reference=manifest['models']['female-base-v0-4-0'];root=a.reference_root
uv_bytes=(root/reference['uv']['path']).read_bytes()
assert sha(uv_bytes)==reference['uv']['sha256']
uv=json.loads(gzip.decompress(uv_bytes));triangles=np.asarray(uv['triangles'])
out=a.candidate/'skin';out.mkdir(exist_ok=False)
result={'schema':'skin-assets.v1','revision':'female040-breast7-review-v1','models':{},
        'reference_manifest_sha256':sha((ROOT/'converter/skin_assets.v1.json').read_bytes()),
        'csp_review_status':'pending_for_this_candidate','authoring_sha256':{'package_skin040.py':sha(Path(__file__).read_bytes())}}
for kind in ['textures','preview_textures']:
 for entry in reference[kind].values():
  raw=(root/entry['path']).read_bytes();assert len(raw)==entry['bytes'] and sha(raw)==entry['sha256'];shutil.copy2(root/entry['path'],out/entry['path'])
proof=[]
for fbx in sorted((a.candidate/'release/fbx').glob('*.fbx')):
 rt.reset_scene();arm,meshes=import_full_character(fbx);assert len(meshes)==1
 mesh=meshes[0];faces=[list(p.vertices) for p in mesh.data.polygons]
 assert np.array_equal(triangles,np.asarray(faces)), 'UV face order differs'
 weights=weights_hash(mesh);assert weights==reference['weights_sha256'],'weights differ'
 assert len(mesh.data.vertices)==reference['vertices'] and len(arm.data.bones)==reference['bones']
 raw_tree,_=tree(fbx);geom=next(e for e in one(raw_tree,b'Objects').elems if e.id==b'Geometry' and e.props[2]==b'Mesh')
 indices=one(geom,b'PolygonVertexIndex').props[0]
 assert sha(indices.tobytes())==reference['polygon_indices_sha256']
 source_sha=sha(fbx.read_bytes());values={**uv,'source_sha256':source_sha}
 raw=gzip.compress(json.dumps(values,separators=(',',':')).encode(),mtime=0)
 name=fbx.stem+'-uv.json.gz';(out/name).write_bytes(raw)
 row={k:copy.deepcopy(reference[k]) for k in ('bones','vertices','preview_mesh','preview_resolution','textures','preview_textures')}
 row.update(character_sha256=source_sha,weights_sha256=weights,
   polygon_indices_sha256=sha(indices.tobytes()),triangles_sha256=sha(json.dumps(faces,separators=(',',':')).encode()),
   uv={'path':name,'bytes':len(raw),'sha256':sha(raw)})
 result['models'][fbx.stem]=row
 proof.append({'character':fbx.stem,'character_sha256':source_sha,'triangles_exact':True,'weights_exact':True,'uv_corners_exact':True,'csp_certification_copied':False})
 print('SKIN_PINNED',fbx.stem,flush=True)
(out/'skin-assets.json').write_text(json.dumps(result,indent=2)+'\n')
(out/'correspondence.json').write_text(json.dumps(proof,indent=2)+'\n')
