"""Blender-only one-time packaging of current neutral UV/texture assets."""
from pathlib import Path
import sys,json,gzip,hashlib
import bpy,numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from converter.outline_worker import tree
from converter.outline_fbx import one
from converter.skin_spec import weights_hash
assets=ROOT/'out/skin-fbx-csp/assets';release=ROOT/'out/skin-fbx-csp/release';release.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=dict(authoring_sha256={p.name:sha(p) for p in (ROOT/'scripts/skin_fbx').glob('*') if p.suffix in ('.py','.json')},schema='skin-assets.v1',revision='current-neutral-uv-20261106-v1',guide_revision='0.1.3',muscle_revision='anatomy-v2-neutral-rebind-v1',models={})
for gender in ('male','female'):
 info=json.loads((assets/f'{gender}-source.json').read_text());source=Path(info['source']);assert sha(source)==info['source_sha256']
 root,_=tree(source);geometry=next(g for g in one(root,b'Objects').elems if g.id==b'Geometry' and g.props[2]==b'Mesh');raw_indices=one(geometry,b'PolygonVertexIndex').props[0]
 bpy.ops.wm.open_mainfile(filepath=str(assets/f'{gender}.blend'));obj=next(o for o in bpy.context.scene.objects if o.type=='MESH');uv=obj.data.uv_layers['StandinSkinUV_v1']
 arm=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE');protected=hashlib.sha256()
 for v in obj.data.vertices:
  protected.update(np.asarray(v.co,dtype='<f4').tobytes());protected.update(np.array([(g.group,g.weight) for g in v.groups],dtype='<f8').tobytes())
 for f in obj.data.polygons:protected.update(np.asarray(f.vertices,dtype='<i4').tobytes())
 for b in arm.data.bones:protected.update(b.name.encode());protected.update(np.array(b.matrix_local,dtype='<f4').tobytes())
 assert protected.hexdigest()==info['protected_before']==info['protected_after'],'prepared neutral protected data changed'
 faces=[];face=[]
 for i in raw_indices:
  face.append(-i-1 if i<0 else i)
  if i<0:faces.append(face);face=[]
 assert len(faces)==len(obj.data.polygons)
 values=[]
 for expected,poly in zip(faces,obj.data.polygons):
  lookup={obj.data.loops[i].vertex_index:list(uv.data[i].uv) for i in poly.loop_indices}
  assert len(lookup)==len(expected) and set(lookup)==set(expected)
  values.extend(lookup[v] for v in expected)
 obj.data.calc_loop_triangles();triangles=[list(t.vertices) for t in obj.data.loop_triangles]
 assert all(len(face)==3 for face in faces) and faces==triangles,'preview requires exact triangular corner correspondence'
 payload=dict(uv=values,triangles=triangles,source_sha256=info['source_sha256'])
 data=gzip.compress(json.dumps(payload,separators=(',',':'),allow_nan=False).encode(),mtime=0);uvpath=release/f'{gender}-uv.json.gz';uvpath.write_bytes(data)
 row=dict(preview_resolution=2048,preview_textures={},preview_mesh='mesh_000',triangles_sha256=hashlib.sha256(json.dumps(triangles,separators=(',',':')).encode()).hexdigest(),character_sha256=info['source_sha256'],vertices=info['vertices'],bones=info['bones'],polygon_indices_sha256=hashlib.sha256(raw_indices.tobytes()).hexdigest(),weights_sha256=weights_hash(obj),uv=dict(path=uvpath.name,bytes=len(data),sha256=sha(uvpath)),textures={})
 for skin in ('gray-guide','flesh','flesh-guide','muscle'):
  dst=release/f'{gender}-{skin}.png';dst.write_bytes((assets/dst.name).read_bytes());row['textures'][skin]=dict(path=dst.name,bytes=dst.stat().st_size,sha256=sha(dst))
  preview=release/f'{gender}-{skin}-preview.png';row['preview_textures'][skin]=dict(path=preview.name,bytes=preview.stat().st_size,sha256=sha(preview))
 cid='male-default83-v0-7-6-p1' if gender=='male' else 'female-default80-v0-3-9-p1';manifest['models'][cid]=row
(ROOT/'converter/skin_assets.v1.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n');(release/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
print('PACKAGED',len(manifest['models']),'current neutral asset sets',sha(ROOT/'converter/skin_assets.v1.json'))
