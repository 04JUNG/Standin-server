"""Expand protection at actual failed morph triangles; never relax safety gates."""
import argparse,importlib.util,json,sys
from pathlib import Path
import bpy,numpy as np
from mathutils import Vector
from mathutils.kdtree import KDTree
p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--registry',required=True);p.add_argument('--source-co',required=True);p.add_argument('--gender',required=True);p.add_argument('--out',required=True);p.add_argument('--previous');a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);folder=Path(a.candidate)
bpy.ops.wm.open_mainfile(filepath=str(folder/'model.blend'));spec=importlib.util.spec_from_file_location('repair',folder/'controller.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);c.register();m,arm=c.objects();s=c.source();faces=s['faces'];ids=np.load(folder/'source-indices.npy');registry=json.loads(Path(a.registry).read_text());cases={k:v['defaults'] for k,v in registry['characters'].items() if k.startswith(a.gender+'-') and v['rules']==('standin-body-0.8.0' if a.gender=='male' else 'standin-female-body-0.4.0')};base=next(v for k,v in cases.items() if '-base-' in k);cases['custom-large']={**base,'height_cm':190.,'heads':7.,'physique':1.2,'muscle':1.8,'shoulders':.6,'leg_length':.55,'arm_length':.3};failed=set();report={}
for name,controls in cases.items():
 out,_,ref=c.compute(c.normalize(controls));r=np.cross(ref[faces[:,1]]-ref[faces[:,0]],ref[faces[:,2]]-ref[faces[:,0]]);n=np.cross(out[faces[:,1]]-out[faces[:,0]],out[faces[:,2]]-out[faces[:,0]]);bad=(np.einsum('ij,ij->i',r,n)<=0)|(np.linalg.norm(n,axis=1)<2e-12);failed.update(ids[faces[bad]].ravel().tolist());report[name]={'bad_triangles':int(bad.sum()),'geometry':c.geometry_metrics(out,ref,faces)};print(name,report[name],flush=True)
if not Path(a.source_co).exists():
 stem='standin-body-0-8-0' if a.gender=='male' else 'standin-female-body-0-4-0'
 bpy.ops.wm.open_mainfile(filepath=str(folder.parents[1]/'originals'/(stem+'.blend')))
 obj=next(o for o in bpy.context.scene.objects if o.get('standin_studio_mesh'));co=c.arr(obj.data.shape_keys.key_blocks['Basis'].data);np.save(a.source_co,co)
else:co=np.load(a.source_co)
tree=KDTree(len(co))
for i,v in enumerate(co):tree.insert(Vector(v),i)
tree.balance();protected=set(np.load(a.previous).tolist()) if a.previous else set()
for i in failed:
 protected.update(x[1] for x in tree.find_range(Vector(co[i]),.04))
np.save(a.out,np.array(sorted(protected),dtype=np.int32));Path(a.out+'.json').write_text(json.dumps({'cases':report,'failed_source_vertices':len(failed),'expanded_protection_vertices':len(protected),'failed_bounds':{'min':co[list(failed)].min(0).tolist(),'max':co[list(failed)].max(0).tolist()} if failed else None},indent=2));print('REPAIR',len(failed),len(protected),flush=True)
