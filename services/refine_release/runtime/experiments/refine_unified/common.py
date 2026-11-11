import hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'refine_new'))
from core import prepare,frame,targets,skin,deformation,project,render_projected

def read(p):return json.loads(Path(p).read_text())
def write(p,d):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def copy_checkpoint(src,dst):
 dst.mkdir(parents=True,exist_ok=False)
 shutil.copy2(src/'fixture.json',dst/'fixture.json');shutil.copytree(src/'fit',dst/'fit')
 for p in (dst/'fit').iterdir():
  if p.suffix in ['.fbx','.blend','.blend1'] or p.name=='export-check.json':p.unlink()
def camera(pr,w,ix):
 return dict(camera=pr['camera'],origin=pr['origin'],anchor_camera_xyz=((w[ix['Hips'],:3,3]-pr['origin'])@pr['camera']).tolist(),projection='orthographic',inverse_distance=0,pixel_scale=pr['pixel_scale'],offset=pr['offset'],image_size=[535,687],embedded_in_fbx=False)
def source_checkpoint(assets,c,obs,out):
 # Only a manifest-bound original selected rig can seed this entry point.
 row=next(r for r in read(assets/'input/manifest.json')['rows'] if r['case_id']==c)
 if row.get('status')!='original_library_bvh_confirmed' or row.get('target_identity')!='single_depicted_target':raise ValueError('No unambiguous original selected pose')
 rigpath=assets/f'input/p1-native/{c}-source.json';rig=read(rigpath)
 if rig['source_bvh_sha256']!=row['selection']['selected_bvh_sha256']:raise ValueError('Not the selected original BVH')
 paths=dict(rig=str(rigpath.relative_to(assets)),mesh=f'base-mesh/{c}.npz',rough='input/'+row['rough'],source='input/'+rig['source'],bvh='input/'+row['selection']['verified_selected_bvh'])
 fx=dict(case_id=c,image_size=obs['image_size'],files={k:dict(path=v,sha256=sha(assets/v)) for k,v in paths.items()})
 if fx['files']['rough']['sha256']!=obs['image_sha256'] or fx['files']['rough']['sha256']!=row['rough_sha256'] or fx['files']['source']['sha256']!=rig['source_sha256'] or fx['files']['bvh']['sha256']!=rig['source_bvh_sha256']:raise ValueError('Source/image/BVH binding mismatch')
 names,ix,_,_,base=prepare(rig);_,origin,_,basis=frame(rig['bones']);goal,*_=targets(obs,fx['image_size'])
 if len(goal)<2:raise ValueError('Insufficient joints to frame original pose')
 q=((base[[ix[n] for n in goal],:3,3]-origin)@basis)[:,:2]*[1,-1];g=np.array(list(goal.values()));qc=q-q.mean(0);gc=g-g.mean(0);scale=float(np.clip(np.sum(qc*gc)/max(np.sum(qc*qc),1e-8),100,1600));off=g.mean(0)-q.mean(0)*scale
 pr=dict(edits={},source=str((assets/paths['source']).resolve()),camera=basis.tolist(),origin=origin.tolist(),pixel_scale=scale,offset=off.tolist())
 write(out/'fixture.json',fx);write(out/'observations.json',obs);write(out/'fit/adjustment.json',pr);np.savez_compressed(out/'fit/expected-world.npz',names=names,world=base);write(out/'fit/camera.json',camera(pr,base,ix))
 error=float(np.sqrt(np.mean((q*scale+off-g)**2)));write(out/'fit/metrics.json',dict(case_id=c,files=fx['files'],rmse_px=error,old_rmse=error,geometry_accepted=False,rejection_reasons=[],bound_hit_bones=[],regional_acceptance=[],orientation_stage={},lens=dict(selected='orthographic',inverse_distance=0,pixel_scale=scale,offset=off.tolist()),manual_targets=False,manual_conditions_used=False,ground_truth_used=False))
 write(out/'provenance.json',dict(original_selected_pose_id=row['selection']['selected_pose_id'],original_selected_bvh_sha256=rig['source_bvh_sha256'],original_world_sha256=sha(out/'fit/expected-world.npz'),zero_edits=True,source_rig=str(rigpath)))
 return fx
