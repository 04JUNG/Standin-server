"""Compare actual authoring and independently reloaded fixed FBXs by frozen IDs.

Coordinates at retained source points are a correspondence error, not Hausdorff
error. Bidirectional point-to-surface checks are reported separately. Existing
controller validity and export round-trip gates run unchanged.
"""
import argparse,hashlib,importlib.util,json,sys,time
from pathlib import Path
import bpy,numpy as np
from mathutils import Matrix,Vector
from mathutils.bvhtree import BVHTree
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from converter.shape_bake import import_full_character
from converter import retarget as rt

def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def load(blend,controller):
 bpy.ops.wm.open_mainfile(filepath=str(blend));spec=importlib.util.spec_from_file_location('qa_controller',controller);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);c.register();return c

def pose(c,name):
 m,a=c.objects()
 for b in a.pose.bones:b.matrix_basis=Matrix.Identity(4)
 c.apply_pose({'neutral':'T','raised':'RAISE','elbow':'ELBOW','deep':'DEEP','sit':'SIT90','cross':'T','hands':'T'}[name])
 if name=='cross':
  for s,sign in [('Left',1),('Right',-1)]:c.rotate(a,s+'Arm',(0,0,1),sign*35);c.rotate(a,s+'ForeArm',(0,0,1),sign*120)
 from scripts.character_lod.pose_controls import finger_test_rotations
 for bone,axis,degrees in finger_test_rotations(name):c.rotate(a,bone,axis,degrees)
 a.update_tag();bpy.context.view_layer.update();c.correct_pose(check_intersections=True,contacts=True)
 return c.evaluated(m)

def dist(points,co,faces):
 tree=BVHTree.FromPolygons(co.tolist(),faces.tolist(),all_triangles=True)
 # Fixed stride includes every protected vertex separately in correspondence tests.
 idx=np.unique(np.r_[np.arange(0,len(points),max(1,len(points)//12000)),len(points)-1])
 d=np.array([tree.find_nearest(Vector(points[i]))[3] for i in idx])*1000
 return {'sample_count':len(d),'max_mm':float(d.max()),'p99_mm':float(np.percentile(d,99)),'mean_mm':float(d.mean())}

def main(a):
 out=Path(a.out);out.mkdir(parents=True,exist_ok=False);registry=json.loads(Path(a.registry).read_text());gender=a.gender
 rows={k:v for k,v in registry['characters'].items() if k.startswith(gender+'-') and v['rules']==('standin-body-0.8.0' if gender=='male' else 'standin-female-body-0.4.0')}
 base=rows[gender+'-base-v'+('0-8-0' if gender=='male' else '0-4-0')]['defaults']
 cases={k:dict(v['defaults']) for k,v in rows.items()}
 cases['custom-large']={**base,'height_cm':190.,'heads':7.,'physique':1.2,'muscle':1.8,'shoulders':.6,'leg_length':.55,'arm_length':.3}
 originals=Path(a.originals);stem='standin-body-0-8-0' if gender=='male' else 'standin-female-body-0-4-0'
 report={'gender':gender,'source_cases':{},'candidates':{},'limits':{'retained_p99_mm':.5,'protected_max_mm':.05,'surface_p99_mm':.5},'limitations':['No automatic penetration-free or CSP certificate is implied by finite geometry or round-trip success.']}
 if a.reference:report['source_cases']=json.loads((Path(a.reference)/'quality.json').read_text())['source_cases']
 # Source references are produced once, including every registered preset.
 for cid,values in ([] if a.reference else cases.items()):
  c=load(originals/(stem+'.blend'),originals/(stem+'.py'));m,arm=c.objects();values=c.normalize(values);c.set_controls(values,correct=False)
  faces=np.array([p.vertices[:] for p in m.data.polygons],np.int32)
  poses=['neutral','raised','sit'] if cid!='custom-large' and '-base-' not in cid else ['neutral','raised','elbow','deep','sit','cross','hands']
  row={'controls':values,'geometry':c.validate_geometry(values),'poses':{}}
  for name in poses:
   points=pose(c,name);np.savez_compressed(out/(cid+'-'+name+'.npz'),co=points,faces=faces)
   row['poses'][name]={'finite':bool(np.isfinite(points).all()),'psd':{k.name:float(k.value) for k in m.data.shape_keys.key_blocks if k.name.startswith('PSD_') and k.value>1e-5}}
  pose(c,'neutral');export=c.export_fixed(str(out/'original-fbx'),posed=False);row['neutral_export']=str(export/'model.fbx');row['roundtrip']=json.loads((export/'manifest.json').read_text())
  report['source_cases'][cid]=row;write(out/'quality.json',report);print('SOURCE',cid,flush=True)
 for folder in [Path(p) for p in a.candidate]:
  tag=folder.name;ids=np.load(folder/'source-indices.npy');protected=set(np.load(folder/'protected-source.npy'));pidx=np.array([i for i,x in enumerate(ids) if x in protected]);cr={}
  for cid,values in cases.items():
   row={'poses':{}}
   try:
    c=load(folder/'model.blend',folder/'controller.py');m,arm=c.objects();c.set_controls(values,correct=False);faces=np.array([p.vertices[:] for p in m.data.polygons],np.int32)
    row['geometry']=c.validate_geometry(values)
    for name in report['source_cases'][cid]['poses']:
     co=pose(c,name);ref=np.load((Path(a.reference) if a.reference else out)/(cid+'-'+name+'.npz'));d=np.linalg.norm(co-ref['co'][ids],axis=1)*1000
     metrics={'finite':bool(np.isfinite(co).all()),'retained_max_mm':float(d.max()),'retained_p99_mm':float(np.percentile(d,99)),'protected_max_mm':float(d[pidx].max()),'source_to_candidate':dist(ref['co'],co,faces),'candidate_to_source':dist(co,ref['co'],ref['faces'])}
     metrics['within_numeric_screen']=metrics['finite'] and metrics['retained_p99_mm']<=.5 and metrics['protected_max_mm']<=.05 and metrics['source_to_candidate']['p99_mm']<=.5
     row['poses'][name]=metrics;np.savez_compressed(out/(tag+'-'+cid+'-'+name+'.npz'),co=co,faces=faces)
    pose(c,'neutral');export=c.export_fixed(str(out/tag/'neutral-fbx'));row['neutral_export']=str(export/'model.fbx');row['roundtrip']=json.loads((export/'manifest.json').read_text())
    # Independently reload real final neutral file and prove subset weights/rig.
    rt.reset_scene();arm,meshes=import_full_character(export/'model.fbx');mesh=meshes[0];names={g.index:g.name for g in mesh.vertex_groups}
    candidate_weights=[{names[g.group]:float(g.weight) for g in v.groups if g.weight>1e-7} for v in mesh.data.vertices];candidate_rig={b.name:np.array(arm.matrix_world@b.matrix_local) for b in arm.data.bones}
    rt.reset_scene();arm,meshes=import_full_character(report['source_cases'][cid]['neutral_export']);mesh=meshes[0];names={g.index:g.name for g in mesh.vertex_groups}
    original_weights=[{names[g.group]:float(g.weight) for g in mesh.data.vertices[int(i)].groups if g.weight>1e-7} for i in ids]
    row['reload_weight_max_delta']=max(abs(v.get(k,0)-w.get(k,0)) for v,w in zip(original_weights,candidate_weights) for k in v.keys()|w.keys());row['reload_rig_max_delta']=max(float(np.max(abs(candidate_rig[b.name]-np.array(arm.matrix_world@b.matrix_local)))) for b in arm.data.bones)
    row['numeric_screen_pass']=all(p['within_numeric_screen'] for p in row['poses'].values()) and row['reload_weight_max_delta']<1e-5 and row['reload_rig_max_delta']<1e-5
   except Exception as e:row['error']=str(e);row['numeric_screen_pass']=False
   cr[cid]=row;report['candidates'][tag]=cr;write(out/'quality.json',report);print('CANDIDATE',tag,cid,row.get('numeric_screen_pass'),row.get('error'),flush=True)
 write(out/'quality.json',report)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--reference');p.add_argument('--originals',required=True);p.add_argument('--registry',required=True);p.add_argument('--gender',required=True);p.add_argument('--candidate',action='append',required=True);p.add_argument('--out',required=True);main(p.parse_args(sys.argv[sys.argv.index('--')+1:]))
