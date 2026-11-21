"""Terminal Shadow/Preserve selector: measured gates, constrained repair, fail closed.
Never writes input runs. Only passed, round-tripped FBX enters release/.
"""
import argparse,sys,json,subprocess,shutil,hashlib,copy,time
TIMINGS=[]
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'refine_unified'))
from common import read,write,prepare,targets,skin,project,render_projected
from core import fk
from policy import LIMITS,MeshPolicy,articulation,structural,relationship_errors,region,rotation,unit,angle,safety_margin_penalty
from relations import validate
from release_policy import contact_issue,unresolved_hand_disagreement

class Case:
 def __init__(self,c,run,assets,out,relations_dir):
  self.c=c;self.assets=assets;self.out=out/c;self.out.mkdir(parents=True);self.raw=run/'shadow'/c/'final'/c/'fit';self.preserve=run/'preserve'/c/'final'/c/'fit'
  self.seed_sources=[(label,path) for label,path in [('shadow',self.raw),('preserve',self.preserve)] if all((path/name).is_file() for name in ['adjustment.json','camera.json','expected-world.npz'])]
  if not self.seed_sources:raise ValueError('No complete candidate branch')
  reference_label,reference_path=self.seed_sources[0];self.seed_errors=[]
  fx=read(run/reference_label/c/'source'/c/'fixture.json');self.fx=fx
  for item in fx['files'].values():
   if hashlib.sha256((assets/item['path']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Asset hash mismatch')
  self.rig=read(assets/fx['files']['rig']['path']);self.names,self.ix,self.parents,self.local,self.base=prepare(self.rig);self.mesh=np.load(assets/fx['files']['mesh']['path']);self.v,self.f,self.weights=[self.mesh[n] for n in ['vertices','faces','weights']];self.scale=np.linalg.norm(self.base[self.ix['Spine2'],:3,3]-self.base[self.ix['Hips'],:3,3]);self.meshpolicy=MeshPolicy(self.v,self.f,self.weights,self.names,self.scale,joint_positions=self.base[:,:3,3])
  self.obs=read(run/reference_label/c/'observations'/f'{c}.json');self.goal,*_=targets(self.obs,fx['image_size']);self.cue=read(run/reference_label/c/'camera-cues'/f'{c}.json')['raw'];self.body=read(run/reference_label/c/'body-cues'/f'{c}.json')['raw']
  if self.obs['image_sha256']!=fx['files']['rough']['sha256']:raise ValueError('Observation binding mismatch')
  self.reference=read(reference_path/'adjustment.json');self.refcam=read(reference_path/'camera.json');self.refworld=np.load(reference_path/'expected-world.npz')['world'];self.refxy=self.xy(self.refworld,self.refcam);self.calibration=None;self.relations=[];self.relation_available=False
  if relations_dir and (relations_dir/f'{c}.json').exists():self.relations=validate(read(relations_dir/f'{c}.json'),fx['files']['rough']['sha256']);self.relation_available=True
  self.pool=[]
 def xy(self,w,cam):
  cv=(w[:,:3,3]-cam['origin'])@cam['camera'];return project(cv,np.array(cam['anchor_camera_xyz']),cam['inverse_distance'],cam['pixel_scale'],cam['offset'])
 def world(self,ed):return fk(self.local,self.parents,[self.ix[n] for n in ed],np.radians(list(ed.values()))) if ed else self.base.copy()
 def add(self,label,ed,pr=None,cam=None):
  pr=copy.deepcopy(pr or self.reference);cam=copy.deepcopy(cam or self.refcam);pr['edits']=copy.deepcopy(ed);w=self.world(ed);dest=self.out/'candidates'/label;dest.mkdir(parents=True,exist_ok=False);write(dest/'adjustment.json',pr);write(dest/'camera.json',cam);np.savez_compressed(dest/'expected-world.npz',names=self.names,world=w)
  row=dict(id=label,world=w,pr=pr,cam=cam,path=dest);self.pool.append(row);return row
 def seed(self):
  for label,path in self.seed_sources:
   try:
    pr=read(path/'adjustment.json');cam=read(path/'camera.json');saved=np.load(path/'expected-world.npz')['world'];world=self.world(pr['edits'])
    if not np.allclose(saved,world,atol=1e-5,rtol=0):raise ValueError('Saved pose differs from edits/FK')
    self.add(label,pr['edits'],pr,cam)
   except Exception as exc:self.seed_errors.append(dict(candidate=label,error=str(exc)))
  if not self.pool:raise ValueError('No valid candidate branch')
 def job(self,candidates):return dict(case=self.c,rig=str(self.assets/self.fx['files']['rig']['path']),mesh=str(self.assets/self.fx['files']['mesh']['path']),source=str(self.assets/self.fx['files']['source']['path']),source_sha256=self.fx['files']['source']['sha256'],calibration_output=str(self.out/'calibration.json'),candidates=[dict(id=r['id'],world=str(r['path']/'expected-world.npz')) for r in candidates])
 def load_calibration(self):
  d=read(self.out/'calibration.json')
  if d['source_sha256']!=self.fx['files']['source']['sha256'] or d['names']!=self.names:raise ValueError('Invalid rig calibration')
  self.calibration=np.array(d['rest_world'])
 def diagnose(self,row,collision):
  w=row['world'];cam=row['cam'];xy=self.xy(w,cam);issues=[dict(kind=k) for k in structural(w,self.base,self.parents)]
  art=articulation(w,self.base,self.parents,self.ix,neutral=self.calibration);issues+=art['issues'];vertices=skin(self.v,self.weights,w,self.base);surface=self.meshpolicy.measure(vertices,w@np.linalg.inv(self.base));issues+=surface['issues']
  if self.calibration is None:issues.append(dict(kind='missing_neutral_calibration'))
  if collision.get('missing_required_groups'):issues.append(dict(kind='incomplete_collision_coverage',groups=collision['missing_required_groups']))
  for pair,count in collision['intersections'].items():
   finding=contact_issue(pair,count,collision,self.scale)
   if finding:issues.append(finding)
  for buried in collision.get('burial',[]):
   depth=np.array(buried['vertex_depths'])/self.scale;area=np.array(buried['vertex_area_fractions']);fraction=float(area[depth>LIMITS.burial_depth_torso].sum())
   if fraction>LIMITS.burial_area_fraction:issues.append(dict(kind='limb_burial',region=buried['moving_region'],target=buried['target_region'],deep_area_fraction=fraction,max_depth_torso=float(depth.max(initial=0))))
  # Ambiguity is a ranking diagnostic, not a reason to reject safe alternatives.
  issues+=relationship_errors(w,self.ix,np.array(cam['camera']),self.scale,self.relations)
  # Don't replace a behind-head gesture with an unrelated, collision-free hand.
  for side in ['Left','Right']:
   key=side+'_hand/head';key2='head/'+side+'_hand';origin_id=row['pr'].get('local_repair_origin',row['id']);origin_row=next((r for r in self.pool if r['id']==origin_id),row);rawcol=origin_row.get('diagnostic',{}).get('collision',collision)
   if rawcol['intersections'].get(key,0)+rawcol['intersections'].get(key2,0)>0:
    supported=any(set((r['near'],r['far']))=={side+'Hand','Head'} for r in self.relations)
    if not supported and 'local_repair_origin' in row['pr']:
     anchor=next(r for r in self.pool if r['id']==row['pr']['local_repair_origin'])
     hi,he=self.ix[side+'Hand'],self.ix['Head'];before=((anchor['world'][hi,:3,3]-anchor['world'][he,:3,3])@anchor['cam']['camera'])[2];after=((w[hi,:3,3]-w[he,:3,3])@cam['camera'])[2]
     if before*after<0 and abs(after-before)>LIMITS.depth_margin_torso*self.scale:issues.append(dict(kind='hand_head_depth_unresolved',region=side+'_arm'))
  # Preserve whole-body intent while permitting small local defect corrections.
  origin=row['pr'].get('local_repair_origin',row['id'] if row['id'] in ('shadow','preserve') else 'shadow')
  anchor=next(r for r in self.pool if r['id']==origin);reference_xy=self.xy(anchor['world'],anchor['cam'])
  ids=[self.ix[n] for n in self.goal];motion=np.linalg.norm(xy[ids]-reference_xy[ids],axis=1);shift=float(motion.max(initial=0))
  if 'local_repair_origin' in row['pr']:
   world_motion=float(np.linalg.norm(w[:,:3,3]-anchor['world'][:,:3,3],axis=1).max()/self.scale)
   budget=LIMITS.expanded_repair_motion_torso if row['pr'].get('bounded_repair') else LIMITS.repair_motion_torso
   if world_motion>budget:issues.append(dict(kind='local_motion_budget',value=world_motion,limit=budget))
   if row['pr'].get('bounded_repair') and shift>LIMITS.landmark_error_increase_px:issues.append(dict(kind='bounded_repair_projection_shift',value=shift,limit=LIMITS.landmark_error_increase_px))
   if not np.allclose(cam['camera'],anchor['cam']['camera'],atol=0,rtol=0):issues.append(dict(kind='repair_camera_changed'))
  target=np.array(list(self.goal.values()));before_error=np.linalg.norm(reference_xy[ids]-target,axis=1);after_error=np.linalg.norm(xy[ids]-target,axis=1);damaged=(motion>LIMITS.landmark_budget_px)&(after_error>before_error+LIMITS.landmark_error_increase_px)
  # Every repair preserves its own origin; raw Preserve is not judged against Shadow.
  if damaged.any():issues.append(dict(kind='intent_landmark_shift',max_px=float(motion[damaged].max()),joints=[n for n,bad in zip(self.goal,damaged) if bad]))
  # A wrong torso view cannot be made safe merely by distorting distal limbs.
  shoulder=(w[self.ix['LeftArm'],:3,3]-w[self.ix['RightArm'],:3,3])@cam['camera'];yaw=abs(float(np.degrees(np.arctan2(shoulder[2],shoulder[0]))));yaw=min(yaw,180-yaw)
  if self.cue.get('torso_view')=='profile' and yaw<LIMITS.profile_min_yaw_deg:issues.append(dict(kind='torso_view_conflict',yaw_deg=yaw))
  torso_cue=self.body.get('torso') or {}
  if torso_cue.get('confidence')=='clear' and torso_cue.get('view') in ['front','back']:
   across=w[self.ix['LeftArm'],:3,3]-w[self.ix['RightArm'],:3,3];up=w[self.ix['Spine2'],:3,3]-w[self.ix['Hips'],:3,3];facing=float((unit(np.cross(across,up))@cam['camera'])[2]);sign=1 if torso_cue['view']=='front' else -1
   if sign*facing<-.2:issues.append(dict(kind='torso_front_back_conflict',expected=torso_cue['view'],camera_facing=facing))
  head_quality=0.
  h=self.body.get('head') or {}
  if self.calibration is not None and h.get('confidence')=='clear' and h.get('direction') in ('screen_left','screen_right'):
   n=self.calibration;right=unit(n[self.ix['LeftArm'],:3,3]-n[self.ix['RightArm'],:3,3]);up=unit(n[self.ix['Spine2'],:3,3]-n[self.ix['Hips'],:3,3]);forward=unit(np.cross(right,up));axis=np.linalg.solve(n[self.ix['Head'],:3,:3],forward);facing=(w[self.ix['Head'],:3,:3]@axis)@cam['camera'];sign=1 if h['direction']=='screen_right' else -1;head_quality=max(0,.15-sign*facing[0])*20
  rmse=float(np.sqrt(np.mean([(xy[self.ix[n]]-g)**2 for n,g in self.goal.items()])))
  if rmse>LIMITS.maximum_projection_rmse_viewport*np.hypot(535,687):issues.append(dict(kind='insufficient_pose_fidelity',rmse_px=rmse,limit_px=LIMITS.maximum_projection_rmse_viewport*np.hypot(535,687)))
  result=dict(safety_margin_penalty=safety_margin_penalty(art),origin=origin,issues=issues,passed=not issues,articulation=art,surface=surface,collision=collision,rmse_px=rmse,maximum_landmark_shift_px=shift,head_direction_quality_penalty=head_quality,candidate_depth_ambiguity=unresolved_hand_disagreement(self),relations_available=self.relation_available,relations=self.relations,mesh_safety_certified=False,fidelity_fallback=bool(row['id']=='preserve' and damaged.any()))
  row['diagnostic']=result
  if row['pr'].get('automatic_arm_shell'):
   from shell_policy import accept
   shell=accept(self,anchor,row)
   result['legacy_gate']=dict(passed=result['passed'],issues=copy.deepcopy(result['issues']))
   shell['passed']=not shell['issues'];result.update(issues=shell['issues'],passed=shell['passed'],shell_gate=shell)
  write(row['path']/'gate.json',result);return result
 def generate_repairs(self):
  from local_repair import propose as existing_local_repair
  from arm_shell import propose as arm_shell_repair
  # Retain the previously validated local repair branch; shell repair is additive.
  for seed in list(self.pool[:2]):
   if not seed['diagnostic']['passed']:
    t=time.monotonic();existing_local_repair(self,seed);TIMINGS.append(dict(stage="existing_local_repair",case=self.c,origin=seed["id"],seconds=time.monotonic()-t))
    t=time.monotonic();arm_shell_repair(self,seed);TIMINGS.append(dict(stage="arm_shell_repair",case=self.c,origin=seed["id"],seconds=time.monotonic()-t))

def collision_batch(cases,pool,out,blender,label):
 t=time.monotonic()
 job=out/(label+'-collision-job.json');result=out/(label+'-collisions.json');write(job,dict(output=str(result),cases=[c.job(pool[c.c]) for c in cases]))
 with (out/(label+'-blender.log')).open('w') as log:r=subprocess.run([blender,'-b','--python-exit-code','1','--python',str(HERE/'collision_blender.py'),'--',str(job)],stdout=log,stderr=subprocess.STDOUT)
 if r.returncode or not result.exists():raise RuntimeError('Collision/calibration worker failed; no release permitted')
 TIMINGS.append(dict(stage='collision_'+label,seconds=time.monotonic()-t))
 return {(r['case'],r['candidate']):r for r in read(result)}

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cases',nargs='+',required=True);p.add_argument('--relations-dir',type=Path);p.add_argument('--repair',action='store_true');p.add_argument('--export',action='store_true');p.add_argument('--blender',default=__import__('os').environ.get('REFINE_BLENDER','blender'));a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False);cases=[];rows=[]
 from policy import policy_fingerprint
 write(a.out/'config.json',dict(policy_sha256=policy_fingerprint(),version='New Refine Release RC4',cases=a.cases,limits=LIMITS.__dict__,repair=a.repair,export=a.export,production_deployed=False,source_run=str(a.run),new_vlm_calls=0))
 for c in a.cases:
  try:case=Case(c,a.run,a.assets,a.out,a.relations_dir);case.seed();cases.append(case)
  except Exception as e:
   failed=dict(case=c,status='review_required',error=str(e),release_eligible=False,selected=None,reason='no_valid_candidate');rows.append(failed);write(a.out/c/'decision.json',failed)
 try:measured=collision_batch(cases,{c.c:c.pool for c in cases},a.out,a.blender,'initial') if cases else {}
 except Exception as e:
  for c in cases:
   failed=dict(case=c.c,status='review_required',release_eligible=False,selected=None,error=str(e),reason='collision_measurement_failed');rows.append(failed);write(c.out/'decision.json',failed)
  write(a.out/'summary.json',rows);return
 for c in cases:
  c.load_calibration()
  for row in c.pool:c.diagnose(row,measured[(c.c,row['id'])])
 if a.repair:
  repair_cases=[];pools={}
  for c in cases:
   old=len(c.pool)
   try:c.generate_repairs()
   except Exception as e:
    write(c.out/'repair-error.json',dict(error=str(e)));c.pool=c.pool[:old];continue
   if len(c.pool)>old:repair_cases.append(c);pools[c.c]=c.pool[old:]
  try:measured=collision_batch(repair_cases,pools,a.out,a.blender,'repairs') if repair_cases else {}
  except Exception as e:
   write(a.out/'repair-worker-error.json',dict(error=str(e)))
   for c in repair_cases:c.pool=[r for r in c.pool if 'diagnostic' in r]
   repair_cases=[];measured={}
  for c in repair_cases:
   for row in pools[c.c]:c.diagnose(row,measured[(c.c,row['id'])])
 # Retry when nothing is usable OR a substantially better fit has only local defects.
 # Expanded angular motion still has exact surface/depth and screen-motion guards.
 if a.repair:
  retry_cases=[];retry_pools={}
  from local_repair import propose
  for c in cases:
   safe=[r for r in c.pool if r['diagnostic']['passed']]
   best_safe_error=min((r['diagnostic']['rmse_px'] for r in safe),default=float('inf'))
   repairable={'surface_damage','joint_axial_twist','hinge_plane_reversal','joint_overfold','joint_surface_fold','cervical_overbend'}
   seeds=[r for r in c.pool[:2] if not safe or (not r['diagnostic']['passed'] and r['diagnostic']['rmse_px']+5<best_safe_error and all(i['kind'] in repairable for i in r['diagnostic']['issues']))]
   if not seeds:continue
   old=len(c.pool)
   for seed in seeds:
    start=next((r for r in c.pool if r['id']=='local-'+seed['id']),seed)
    try:propose(c,seed,startrow=start,suffix='-bounded',rotation_budget=LIMITS.expanded_repair_rotation_deg)
    except Exception as e:write(c.out/('bounded-'+seed['id']+'-error.json'),dict(error=str(e)))
   if len(c.pool)>old:retry_cases.append(c);retry_pools[c.c]=c.pool[old:]
  if retry_cases:
   try:
    measured=collision_batch(retry_cases,retry_pools,a.out,a.blender,'bounded-repairs')
    for c in retry_cases:
     for row in retry_pools[c.c]:c.diagnose(row,measured[(c.c,row['id'])])
   except Exception as e:
    write(a.out/'bounded-repair-worker-error.json',dict(error=str(e)))
    for c in retry_cases:c.pool=[r for r in c.pool if 'diagnostic' in r]
 for c in cases:
  passed=[x for x in c.pool if x['diagnostic']['passed']];selected=min(passed,key=lambda r:(r['diagnostic']['rmse_px']+r['diagnostic'].get('safety_margin_penalty',0)+r['diagnostic']['head_direction_quality_penalty']+LIMITS.selection_shift_weight*r['diagnostic']['maximum_landmark_shift_px'],r['id']!='shadow')) if passed else None
  record=dict(branch_errors=c.seed_errors,available_branches=[r['id'] for r in c.pool if r['id'] in ('shadow','preserve')],case=c.c,status='measured_candidate' if selected else 'review_required',release_eligible=False,selected=selected['id'] if selected else None,candidates=[dict(id=r['id'],**r['diagnostic']) for r in c.pool],semantic_relations_verified_from_image=c.relation_available)
  # No unverified outputs masquerade as deliverable FBX.
  if selected and a.export:
   dest=c.out/'export-pending';shutil.copytree(selected['path'],dest);write(dest/'metrics.json',dict(case_id=c.c,files=c.fx['files'],manual_targets=False,mesh_path=str(c.assets/c.fx['files']['mesh']['path']),rig_path=str(c.assets/c.fx['files']['rig']['path'])));cmd=[a.blender,'-b','--python-exit-code','1','--python',str(HERE/'export_blender.py'),'--',str(dest)]
   t=time.monotonic()
   with (c.out/'export.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
   TIMINGS.append(dict(stage='final_fbx_export_reimport',case=c.c,seconds=time.monotonic()-t))
   if r.returncode==0 and (dest/'export-check.json').exists():
    record['export']=read(dest/'export-check.json');record['status']='accepted_under_measured_policy';record['release_eligible']=True;shutil.move(dest,c.out/'release')
   else:record['status']='review_required';record['export_failed']=True
  write(c.out/'decision.json',record);rows.append(record)
 write(a.out/'timings.json',TIMINGS);write(a.out/'summary.json',rows);print([(r['case'],r['status'],r.get('selected')) for r in rows])
if __name__=='__main__':main()
