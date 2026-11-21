"""Keep measured support points while simplifying the rest; no relaxed gates.

Kept points need exact identity, not a frozen incident-face graph. Hands and
other protected regions retain the stronger graph/connectivity constraint.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.character_lod.region_policy import REGIONS
p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--quality',required=True);p.add_argument('--reference',required=True);p.add_argument('--out',required=True);p.add_argument('--previous');p.add_argument('--source-co');p.add_argument('--spacing-mm',type=float,default=0.);p.add_argument('--threshold-mm',type=float,default=.45);a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);folder=Path(a.candidate);quality=Path(a.quality);reference=Path(a.reference);out=Path(a.out);q=json.loads((quality/'quality.json').read_text());cases=q['candidates'][folder.name];labels=np.load(folder/'source-regions.npy');body=np.flatnonzero(np.isin(labels,[REGIONS.index(x) for x in ('torso','arms','legs')]));keep=set(np.load(a.previous).tolist()) if a.previous else set();report=[];scores={}
for case,record in cases.items():
 for pose in record['poses']:
  actual=np.load(quality/(folder.name+'-'+case+'-'+pose+'.npz'));expected=np.load(reference/(case+'-'+pose+'.npz'));co=actual['co'];faces=actual['faces'];reference_co=expected['co'];tree=BVHTree.FromPolygons(co.tolist(),faces.tolist(),all_triangles=True);d=np.array([tree.find_nearest(Vector(reference_co[i]))[3] for i in body])*1000;bad=body[d>a.threshold_mm]
  for sid,error in zip(body,d):
   if error>a.threshold_mm:scores[int(sid)]=max(scores.get(int(sid),0.),float(error))
  row={'case':case,'pose':pose,'body_sample_count':len(body),'body_surface_p99_mm':float(np.percentile(d,99)),'body_surface_max_mm':float(d.max()),'required_points':len(bad)};report.append(row);print('SURFACE_SUPPORT',row,flush=True)
if a.spacing_mm:
 if not a.source_co:raise ValueError('spaced support requires original neutral coordinates')
 source=np.load(a.source_co);tree=KDTree(len(source))
 for i,point in enumerate(source):tree.insert(point,i)
 tree.balance()
 for sid in sorted(scores,key=lambda i:(-scores[i],i)):
  if not any(other in keep for _,other,_ in tree.find_range(Vector(source[sid]),a.spacing_mm/1000)):keep.add(sid)
else:keep.update(scores)
np.save(out,np.array(sorted(keep),np.int32));Path(str(out)+'.json').write_text(json.dumps({'threshold_mm':a.threshold_mm,'spacing_mm':a.spacing_mm,'bad_source_vertices':len(scores),'required_source_vertices':len(keep),'cases':report},indent=2)+'\n');print('REQUIRED_SURFACE_SUPPORT',len(keep),flush=True)
