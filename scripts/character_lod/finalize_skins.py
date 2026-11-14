"""Recreate every candidate skin identity from its actual neutral FBX and UVs."""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
import bpy,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from converter.skin_spec import weights_hash
from converter.outline_worker import tree
from converter.outline_fbx import one
from converter.shape_bake import import_full_character
from converter import retarget as rt
p=argparse.ArgumentParser();p.add_argument('--package',required=True);p.add_argument('--male',required=True);p.add_argument('--female',required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);out=Path(a.package);manifest=json.loads((out/'skin-assets.json').read_text());jobs=json.loads((out/'jobs.json').read_text());report={}
for job in jobs:
 if job['variant']!='candidate':continue
 cid=job['character'];folder=Path(getattr(a,job['gender']));source=out/'fbx'/(cid+'.fbx');rt.reset_scene();arm,meshes=import_full_character(source);obj=meshes[0];faces=[list(p.vertices) for p in obj.data.polygons];uv=np.load(folder/'skin-uv.npy');assert uv.shape==(len(faces),3,2)
 root,_=tree(source);geom=next(e for e in one(root,b'Objects').elems if e.id==b'Geometry' and e.props[2]==b'Mesh');indices=one(geom,b'PolygonVertexIndex').props[0];row=manifest['models'][cid];row['character_sha256']=hashlib.sha256(source.read_bytes()).hexdigest();row.update(vertices=len(obj.data.vertices),bones=len(arm.data.bones),weights_sha256=weights_hash(obj),polygon_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),triangles_sha256=hashlib.sha256(json.dumps(faces,separators=(',',':')).encode()).hexdigest())
 data=gzip.compress(json.dumps(dict(uv=uv.reshape(-1,2).tolist(),triangles=faces,source_sha256=row['character_sha256']),separators=(',',':')).encode(),mtime=0);name=cid+'-uv.json.gz';(out/'skins'/name).write_bytes(data);row['uv']=dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest());report[cid]={k:row[k] for k in ('character_sha256','vertices','bones','weights_sha256','polygon_indices_sha256','triangles_sha256','uv')}
(out/'skin-assets.json').write_text(json.dumps(manifest,indent=2)+'\n');(out/'skin-identity-report.json').write_text(json.dumps(report,indent=2)+'\n');print('SKINS_REPINNED_LOCALLY',len(report),flush=True)
