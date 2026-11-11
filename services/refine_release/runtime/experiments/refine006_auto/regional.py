"""Baseline-anchored regional depth refinement; no per-case rules or manual targets."""
import argparse,json,copy,sys,hashlib,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'refine_new'))
from core import prepare,frame,fk,targets,skin,deformation,render_projected
import sys,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'refine_unified'))
from validity import shadow, check_world


import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'refine_unified'))
from solve_budget import solve as budget_solve, eligible
def least_squares(fun,x0,*args,**kwargs):
 return budget_solve(fun,x0,*args,stage='depth',**kwargs)

def bend(points):
 a,b,c=np.asarray(points);u=b-a;v=c-b
 return float(np.degrees(np.arccos(np.clip(np.dot(u,v)/max(1e-12,np.linalg.norm(u)*np.linalg.norm(v)),-1,1))))

def regions(goal):
 return [(s,kind,[s+n for n in chain]) for s in ['Left','Right'] for kind,chain in [('arm',['Arm','ForeArm','Hand']),('leg',['UpLeg','Leg','Foot'])] if all(s+n in goal for n in chain)]

def solve(assets,fixture,obs,sem,baseline,out,accepted_prior=None):
 started=time.perf_counter();out.mkdir(parents=True,exist_ok=False)
 assert obs['image_sha256']==sem['image_sha256']==fixture['files']['rough']['sha256']
 assert obs['producer']=='develop HumanArt M multiview consensus'
 for f in fixture['files'].values():assert hashlib.sha256((assets/f['path']).read_bytes()).hexdigest()==f['sha256']
 facts=sem['raw'];assert facts['person_count']==1
 rig=json.loads((assets/fixture['files']['rig']['path']).read_text());names,ix,parents,local,base=prepare(rig);_,_,torso,_=frame(rig['bones']);goal,kp,valid=targets(obs,fixture['image_size'])
 seed=baseline
 if accepted_prior and (accepted_prior/'metrics.json').exists():
  prior_meta=json.loads((accepted_prior/'metrics.json').read_text())
  if prior_meta.get('geometry_accepted') and prior_meta.get('files')==fixture['files']:seed=accepted_prior
 pr=json.loads((seed/'adjustment.json').read_text());original=copy.deepcopy(pr);ed=copy.deepcopy(pr['edits']);cam=np.array(pr['camera']);origin=np.array(pr['origin']);scale=pr['pixel_scale'];off=np.array(pr['offset']);ids=[ix[n] for n in goal];tar=np.array(list(goal.values()))
 with np.load(seed/'expected-world.npz') as d:oldw=d['world'].copy()
 with np.load(assets/fixture['files']['mesh']['path']) as d:v,f,weights=d['vertices'],d['faces'],d['weights']
 def world(e):return fk(local,parents,[ix[n] for n in e],np.radians(list(e.values())).ravel())
 def xy(w):return ((w[:,:3,3]-origin)@cam)[:,:2]*[scale,-scale]+off
 olderror=float(np.sqrt(np.mean((xy(oldw)[ids]-tar)**2)));olddef=deformation(skin(v,weights,oldw,base),v,f)
 # A camera's orientation is not changed to fit limbs. Framing is retained as well.
 side_order=sorted(['Left','Right'],key=lambda s:goal.get(s+'Arm',[kp[5 if s=='Left' else 6,0]])[0]);mapping=dict(zip(['screen_left','screen_right'],side_order));sweeps={mapping[a['shoulder_position']]:a['sweep'] for a in facts.get('arms',[]) if a.get('shoulder_position') in mapping and a.get('confidence')=='clear' and a.get('sweep') in ['away','toward','lateral']}
 leg=facts.get('raised_leg',{});raised=None
 feet=[s for s in ['Left','Right'] if s+'Foot' in goal]
 if len(feet)==2:
  ordered=sorted(feet,key=lambda s:goal[s+'Foot'][0]);raised=dict(zip(['screen_left','screen_right'],ordered)).get(leg.get('ankle_position'))
 elif len(feet)==1 and all(s+'UpLeg' in goal for s in ['Left','Right']):
  s=feet[0];mid=(goal['LeftUpLeg'][0]+goal['RightUpLeg'][0])/2
  pos='screen_left' if goal[s+'Foot'][0]<mid else 'screen_right'
  if pos==leg.get('ankle_position'):raised=s
 edges=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0);lengths=np.linalg.norm(v[edges[:,0]]-v[edges[:,1]],axis=1);edges=edges[lengths>1e-7];lengths=lengths[lengths>1e-7]
 rows=[];w=oldw.copy();selected=[]
 for side,kind,chain in regions(goal):
  key=side+'_'+kind;dest=out/'regions'/key;dest.mkdir(parents=True)
  moving=[side+n for n in (['Shoulder','Arm','ForeArm'] if kind=='arm' else ['UpLeg','Leg'])];jointids=[ix[n] for n in chain];target=np.array([goal[n] for n in chain]);start=np.radians([ed.get(n,[0,0,0]) for n in moving]).ravel();start_ed=copy.deepcopy(ed);before=w.copy()
  limb=np.linalg.norm(before[jointids[1],:3,3]-before[jointids[0],:3,3])+np.linalg.norm(before[jointids[2],:3,3]-before[jointids[1],:3,3]);target_bend=bend(target);straight=kind=='arm' and target_bend<22
  sweep=sweeps.get(side) if kind=='arm' else None
  depthcenter=({'away':-.2,'toward':.2,'lateral':-.12}.get(sweep)) if kind=='arm' else None
  legdepth=kind=='leg' and side==raised and leg.get('bend')=='backward'
  # Only sample edges affected by this chain; use the same absolute strain gate at selection.
  mass=weights[:,[ix[n] for n in moving]].sum(1);mask=(mass[edges].max(1)>.05)&(mass[edges].min(1)<.999)
  ee=edges[mask];ll=lengths[mask];bv=skin(v,weights,before,base);rat=np.linalg.norm(bv[ee[:,0]]-bv[ee[:,1]],axis=1)/ll
  rank=np.argsort(np.maximum(rat-1,1/np.maximum(rat,.01)-1));keep=np.unique(np.r_[rank[-700:],np.arange(0,len(ee),max(1,len(ee)//500))]);ee=ee[keep];ll=ll[keep]
  inds,inverse=np.unique(ee,return_inverse=True);se=inverse.reshape(-1,2);sv=v[inds];sw=weights[inds]
  baseline_ratios=np.linalg.norm(bv[ee[:,0]]-bv[ee[:,1]],axis=1)/ll
  safe_hi=np.maximum(1.8,baseline_ratios);safe_lo=np.minimum(.6,baseline_ratios)
  def candidate(x):
   e=copy.deepcopy(start_ed);e.update({n:np.degrees(x[3*i:3*i+3]).tolist() for i,n in enumerate(moving)});return e,world(e)
  def quality(ww):
   zz=(ww[:,:3,3]-origin)@cam;points=ww[jointids,:3,3];depth=(zz[jointids[2],2]-zz[jointids[0],2])/limb
   dr=0 if depthcenter is None else max(0,.08-depth) if sweep=='toward' else max(0,depth+.08)
   if legdepth:
    shin=np.linalg.norm(points[2]-points[1]);dr=abs((zz[jointids[2],2]-zz[jointids[1],2])/shin+.65)
   return dict(rmse=float(np.sqrt(np.mean((xy(ww)[jointids]-target)**2))),bend=bend(points),depth=float(depth),depth_error=float(dr))
  beforeq=quality(before);beforedef=deformation(bv,v,f)
  bend_cap=min(max(8,target_bend+3),beforeq['bend']+3) if straight else 180
  def loss(x,depth_weight):
   _,ww=candidate(x);zz=(ww[:,:3,3]-origin)@cam;vv=skin(sv,sw,ww,base);ratio=np.linalg.norm(vv[se[:,0]]-vv[se[:,1]],axis=1)/ll
   # Continuous penalty starts before the selector's 0.5..2 boundary.
   strain=np.r_[np.maximum(ratio-safe_hi,0),np.maximum(safe_lo-ratio,0)]
   res=[(xy(ww)[jointids]-target).ravel()/7,(x-start)*.6,strain*8]
   if straight:res.append([max(0,quality(ww)['bend']-bend_cap)*.7])
   if depthcenter is not None:res.append([quality(ww)['depth_error']*depth_weight])
   if legdepth:
    shin=np.linalg.norm(ww[jointids[2],:3,3]-ww[jointids[1],:3,3]);res.append([((zz[jointids[2],2]-zz[jointids[1],2])/shin+.65)*depth_weight])
   return np.concatenate([np.asarray(r).ravel() for r in res])
  # Continuation keeps the optimizer near the good baseline; each region has 6 or 9 variables.
  lo=start-np.radians(100);hi=start+np.radians(100)
  if kind=='arm':lo[:3]=start[:3]-np.radians(12);hi[:3]=start[:3]+np.radians(12)
  x=start.copy();stages=[]
  if straight:
   seed_ed=copy.deepcopy(start_ed);direction=np.r_[target[2]-target[0],0.];direction[1]*=-1;direction[:2]/=scale;source_depth=float(((base[jointids[2],:3,3]-base[jointids[0],:3,3])@cam)[2]);depth_sign=1 if sweep=='toward' else -1 if sweep in ('away','lateral') else float(np.sign(source_depth));direction[2]=depth_sign*np.sqrt(max((.08*limb)**2,limb**2-np.dot(direction[:2],direction[:2])));direction=cam@(direction/max(1e-9,np.linalg.norm(direction)))
   for bone,child in [(chain[0],chain[1]),(chain[1],chain[2])]:
    ww=world(seed_ed);u=ww[ix[child],:3,3]-ww[ix[bone],:3,3];u/=np.linalg.norm(u);axis=np.cross(u,direction);angle=np.arctan2(np.linalg.norm(axis),np.dot(u,direction));swing=Rotation.from_rotvec(axis/max(1e-9,np.linalg.norm(axis))*angle).as_matrix();oldrot=Rotation.from_rotvec(np.radians(seed_ed.get(bone,[0,0,0]))).as_matrix();seed_ed[bone]=np.degrees(Rotation.from_matrix(oldrot@np.linalg.solve(ww[ix[bone],:3,:3],swing@ww[ix[bone],:3,:3])).as_rotvec()).tolist()
   seed_x=np.radians([seed_ed.get(n,[0,0,0]) for n in moving]).ravel();x=np.clip(seed_x,lo+1e-8,hi-1e-8)
  for dw in [3,7,14]:
   result=least_squares(lambda z:loss(z,dw),x,motion=lambda z:xy(candidate(z)[1]),bounds=(lo,hi),max_nfev=180,ftol=1e-5,xtol=1e-5,gtol=1e-5,x_scale='jac');x=result.x;stages.append(dict(depth_weight=dw,converged=bool(result.success),evaluations=result.nfev,optimality=float(result.optimality)))
  pe,pw=candidate(x);q=quality(pw);pd=deformation(skin(v,weights,pw,base),v,f)
  reasons=[]
  if not eligible(result):reasons.append('solver_not_converged')
  if q['rmse']>beforeq['rmse']+1:reasons.append('regional_2d_regression')
  if pd['extreme_edges']>beforedef['extreme_edges']:reasons.append('skin_count_regression')
  if pd['ratio_max']>max(2,beforedef['ratio_max'])*1.05:reasons.append('skin_peak_regression')
  if straight and q['bend']>bend_cap+3:reasons.append('straight_arm_bent')
  gain=beforeq['rmse']-q['rmse']>1 or beforeq['depth_error']-q['depth_error']>.04 or (straight and beforeq['bend']-q['bend']>5)
  if not gain:reasons.append('no_material_gain')
  if sweep=='lateral':
   if q['depth']>max(.01,beforeq['depth']+.01):reasons.append('lateral_forward_regression')
  elif (depthcenter is not None or legdepth) and q['depth_error']>beforeq['depth_error']+.03:reasons.append('depth_regression')
  check_world(pw,base,parents)
  accepted=shadow() or not reasons
  candidate_pr=copy.deepcopy(pr);candidate_pr['edits']=pe;(dest/'adjustment.json').write_text(json.dumps(candidate_pr,indent=2));np.savez_compressed(dest/'expected-world.npz',names=names,world=pw)
  diagnostic=dict(region=key,accepted=accepted,would_reject=bool(reasons),reasons=reasons,straight_arm=straight,rough_bend_deg=target_bend,sweep=sweep,leg_depth=legdepth,before=beforeq,after=q,before_deformation=beforedef,after_deformation=pd,stages=stages)
  (dest/'diagnostic.json').write_text(json.dumps(diagnostic,indent=2));rows.append(diagnostic)
  if accepted:ed,w=pe,pw;selected.append(key)
  print(fixture['case_id'],key,'accepted' if accepted else reasons,flush=True)
 pr['edits']=ed;pr['targets']={n:x.tolist() for n,x in goal.items()};fit=out/'fit';fit.mkdir()
 (fit/'adjustment.json').write_text(json.dumps(pr,indent=2));np.savez_compressed(fit/'expected-world.npz',names=names,world=w)
 if not selected:
  import shutil
  for name in ['adjustment.json','expected-world.npz']:shutil.copy2(seed/name,fit/name)
 camera=dict(camera=pr['camera'],origin=pr['origin'],anchor_camera_xyz=((w[ix['Hips'],:3,3]-origin)@cam).tolist(),projection='orthographic',inverse_distance=0,pixel_scale=scale,offset=off.tolist(),embedded_in_fbx=False)
 (fit/'camera.json').write_text(json.dumps(camera,indent=2));length=max(abs(np.linalg.norm(w[i,:3,3]-w[j,:3,3])-np.linalg.norm(base[i,:3,3]-base[j,:3,3])) for i,j in enumerate(parents) if j>=0)
 met=dict(case_id=fixture['case_id'],files=fixture['files'],rmse_px=float(np.sqrt(np.mean((xy(w)[ids]-tar)**2))),old_rmse=olderror,old_deformation=olddef,deformation=deformation(skin(v,weights,w,base),v,f),max_bone_length_error=float(length),geometry_accepted=bool(selected) or seed!=baseline,seed_source=str(seed),seed_is_accepted_prior=seed!=baseline,regional_acceptance=selected,regions=rows,missing_regions=[s+'_'+k for s in ['Left','Right'] for k in ['arm','leg'] if s+'_'+k not in [r['region'] for r in rows]],bound_hit_bones=[],rejection_reasons=[] if selected else ['no_safe_regional_gain'],manual_targets=False,manual_conditions_used=False,reference_used_for_fit=False,production_enabled=False,camera_rotation_delta_deg=0,camera_framing_preserved=True,torso_preserved=bool(np.allclose(w[[ix[n] for n in ['Hips','Spine','Spine1','Spine2']]],oldw[[ix[n] for n in ['Hips','Spine','Spine1','Spine2']]],atol=1e-7)),lens=dict(selected='orthographic',inverse_distance=0,pixel_scale=scale,offset=off.tolist()),orientation_stage={},elapsed_seconds=time.perf_counter()-started)
 (fit/'metrics.json').write_text(json.dumps(met,indent=2));return met

def main():
 p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--inputs',type=Path);p.add_argument('--observations',type=Path);p.add_argument('--semantics',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--cases',nargs='+',required=True);p.add_argument('--accepted-prior',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);rows=[]
 for c in a.cases:
  fixture=json.loads((a.baseline/c/'fixture.json').read_text());obs=json.loads(((a.observations or a.inputs/'observations')/f'{c}.json').read_text());sp=a.semantics/f'{c}.json' if a.semantics else next(p for p in [a.inputs/'semantics-controls'/f'{c}.json',a.inputs/'semantics-new'/f'{c}.json'] if p.exists());sem=json.loads(sp.read_text());m=solve(a.assets,fixture,obs,sem,a.baseline/c/'fit',a.out/c,a.accepted_prior/c/'fit' if a.accepted_prior else None);rows.append(dict(case_id=c,status='fit',geometry_accepted=m['geometry_accepted'],projection='orthographic'));(a.out/'results.json').write_text(json.dumps(rows,indent=2))
 (a.out/'status.json').write_text(json.dumps(dict(status='provisional_requires_mesh_audit')))
if __name__=='__main__':main()
