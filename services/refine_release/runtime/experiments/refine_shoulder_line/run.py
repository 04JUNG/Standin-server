"""Fit observed shoulder line before deciding whether the upper body is an anchor."""
import argparse,copy,json,sys,hashlib,shutil
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'refine_new'))
from core import prepare,fk,targets,skin,deformation
import sys,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'refine_unified'))
from validity import shadow, check_world


import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'refine_unified'))
from solve_budget import solve as budget_solve, eligible
def least_squares(fun,x0,*args,**kwargs):
 return budget_solve(fun,x0,*args,stage='shoulder',**kwargs)

def line_metrics(points,goal):
 delta=points[0]-points[1];target=goal[0]-goal[1]
 return dict(endpoint_errors=np.linalg.norm(points-goal,axis=1).tolist(),rmse=float(np.sqrt(np.mean((points-goal)**2))),width=float(np.linalg.norm(delta)),target_width=float(np.linalg.norm(target)),vertical_difference=float(delta[1]),target_vertical_difference=float(target[1]),center_error=float(np.linalg.norm(points.mean(0)-goal.mean(0))),vector_error=float(np.linalg.norm(delta-target)))

def solve(assets,baseline,observation,out,case,torso):
 dest=out/case;dest.mkdir(parents=True,exist_ok=False);fixture=json.loads((baseline/case/'fixture.json').read_text());obs=json.loads(observation.read_text());assert obs['image_sha256']==fixture['files']['rough']['sha256'];shutil.copy2(baseline/case/'fixture.json',dest/'fixture.json');prior=baseline/case/'fit';pr=json.loads((prior/'adjustment.json').read_text());cam=np.array(pr['camera']);origin=np.array(pr['origin']);scale=pr['pixel_scale'];off=np.array(pr['offset']);camera_record=json.loads((prior/'camera.json').read_text()) if (prior/'camera.json').exists() else dict(camera=pr['camera'],origin=pr['origin'],pixel_scale=scale,offset=off.tolist(),projection='orthographic',inverse_distance=pr.get('inverse_distance',0));assert camera_record.get('inverse_distance',0)==0,'Perspective requires its original projector'
 names,ix,parents,local,base=prepare(json.loads((assets/fixture['files']['rig']['path']).read_text()));old=np.load(prior/'expected-world.npz')['world'];goal,*_=targets(obs,fixture['image_size']);shoulders=['LeftArm','RightArm'];assert all(n in goal for n in shoulders),'Both shoulder observations required'
 active=(['Spine2'] if torso else [])+[s+b for s in ['Left','Right'] for b in ['Shoulder','Arm','ForeArm']];ed=copy.deepcopy(pr['edits']);start=np.zeros(len(active)*3);limit=np.repeat(np.radians(([20] if torso else [])+[35,45,45]*2),3)
 def world(e):return fk(local,parents,[ix[n] for n in e],np.radians(list(e.values())).ravel())
 def orient(e,n,desired):
  w=world(e);rot=Rotation.from_rotvec(np.radians(e.get(n,[0,0,0]))).as_matrix();e[n]=np.degrees(Rotation.from_matrix(rot@np.linalg.solve(w[ix[n],:3,:3],desired)).as_rotvec()).tolist()
 def candidate(x):
  e=copy.deepcopy(ed)
  for j,n in enumerate(active):e[n]=np.degrees((Rotation.from_rotvec(np.radians(ed.get(n,[0,0,0])))*Rotation.from_rotvec(x[j*3:j*3+3])).as_rotvec()).tolist()
  for n in ['Neck','LeftHand','RightHand']:orient(e,n,old[ix[n],:3,:3])
  return e,world(e)
 def xy(w):return ((w[:,:3,3]-origin)@cam)[:,:2]*[scale,-scale]+off
 assert np.max(abs(candidate(start)[1]-old))<1e-5
 sid=[ix[n] for n in shoulders];st=np.array([goal[n] for n in shoulders]);other=[n for n in ['LeftForeArm','RightForeArm','LeftHand','RightHand'] if n in goal];oid=[ix[n] for n in other];ot=np.array([goal[n] for n in other]);before=line_metrics(xy(old)[sid],st);other_before=np.linalg.norm(xy(old)[oid]-ot,axis=1)
 with np.load(assets/fixture['files']['mesh']['path']) as m:v,f,weights=m['vertices'],m['faces'],m['weights']
 oldv=skin(v,weights,old,base);olddef=deformation(oldv,v,f);edges=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0);lengths=np.linalg.norm(v[edges[:,0]]-v[edges[:,1]],axis=1);mask=lengths>1e-7;edges=edges[mask];lengths=lengths[mask];rat=np.linalg.norm(oldv[edges[:,0]]-oldv[edges[:,1]],axis=1)/lengths
 # Include affected boundaries; unchanged lower-body vertices cannot alter the loss.
 mass=weights[:,[ix[n] for n in active]].sum(1);mask=mass[edges].max(1)>.01;edges=edges[mask];lengths=lengths[mask];rat=rat[mask];indices,inverse=np.unique(edges,return_inverse=True);ee=inverse.reshape(-1,2);sv=v[indices];sw=weights[indices]
 def residual(x):
  _,w=candidate(x);points=xy(w);ss=points[sid];vv=skin(sv,sw,w,base);r=np.linalg.norm(vv[ee[:,0]]-vv[ee[:,1]],axis=1)/lengths
  return np.r_[(ss.mean(0)-st.mean(0))/2,((ss[0]-ss[1])-(st[0]-st[1]))/2,(points[oid]-ot).ravel()/5,np.maximum(np.linalg.norm(points[oid]-ot,axis=1)-other_before-1,0),x*2,np.maximum(r-np.maximum(rat,1.7),0)*150/np.sqrt(len(r)),np.maximum(np.minimum(rat,.65)-r,0)*150/np.sqrt(len(r))]
 r=budget_solve(residual,start,stage='shoulder',motion=lambda z:xy(candidate(z)[1]),cache_inputs=[prior/'adjustment.json',prior/'expected-world.npz',observation,assets/fixture['files']['rig']['path'],assets/fixture['files']['mesh']['path'],Path(__file__)],bounds=(-limit,limit),max_nfev=180,ftol=1e-6,xtol=1e-6,gtol=1e-6);trials=[];selected=None
 for alpha in [1.,.75,.5,.25,.125]:
  e,w=candidate(r.x*alpha);after=line_metrics(xy(w)[sid],st);df=deformation(skin(v,weights,w,base),v,f);reasons=[]
  if not eligible(r):reasons.append('not_converged')
  if after['rmse']>before['rmse']-max(.2,before['rmse']*.05):reasons.append('no_shoulder_improvement')
  if any(a>b+.5 for a,b in zip(after['endpoint_errors'],before['endpoint_errors'])):reasons.append('one_shoulder_regressed')
  if after['vector_error']>before['vector_error']+.5:reasons.append('line_regressed')
  if np.any(np.linalg.norm(xy(w)[oid]-ot,axis=1)>other_before+2):reasons.append('arm_projection_regressed')
  if df['extreme_edges']>olddef['extreme_edges']:reasons.append('skin_count_regression')
  if df['ratio_max']>max(2,olddef['ratio_max'])*1.05:reasons.append('skin_peak_regression')
  protected=[ix[n] for n in names if n=='Hips' or any(n.startswith(s+b) for s in ['Left','Right'] for b in ['UpLeg','Leg','Foot','Toe'])];assert np.max(abs(w[protected]-old[protected]))<1e-7
  head_error=float(np.max(abs(w[ix['Head'],:3,:3]-old[ix['Head'],:3,:3])));assert head_error<1e-5
  record=dict(alpha=alpha,before=before,after=after,before_deformation=olddef,after_deformation=df,reasons=reasons,accepted=not reasons,head_orientation_error=head_error,unchanged_lower_body=True);trials.append(record)
  if not reasons:selected=(e,w,record);break
 proposal=dest/'proposal';proposal.mkdir();pe,pw=candidate(r.x);pp=copy.deepcopy(pr);pp['edits']=pe;(proposal/'adjustment.json').write_text(json.dumps(pp,indent=2));np.savez_compressed(proposal/'expected-world.npz',names=names,world=pw)
 gated_alpha=selected[2]["alpha"] if selected else None
 if shadow():
  check_world(pw,base,parents);selected=(pe,pw,trials[0])
 chosen=selected if selected else (ed,old,trials[0]);ce,cw,diag=chosen;fit=dest/'fit';fit.mkdir();cp=copy.deepcopy(pr);cp['edits']=ce;(fit/'adjustment.json').write_text(json.dumps(cp,indent=2));np.savez_compressed(fit/'expected-world.npz',names=names,world=cw);(fit/'camera.json').write_text(json.dumps(camera_record,indent=2))
 if not selected:
  for n in ['adjustment.json','expected-world.npz']:shutil.copy2(prior/n,fit/n)
 allnames=[n for n in goal if n in ix];aid=[ix[n] for n in allnames];at=np.array([goal[n] for n in allnames]);oldrmse=float(np.sqrt(np.mean((xy(old)[aid]-at)**2)));newrmse=float(np.sqrt(np.mean((xy(cw)[aid]-at)**2)))
 met=dict(case_id=case,files=fixture['files'],manual_targets=False,manual_conditions_used=False,reference_used_for_fit=False,geometry_accepted=bool(selected),old_rmse=oldrmse,rmse_px=newrmse,old_deformation=olddef,rejection_reasons=[] if selected else diag['reasons'],bound_hit_bones=[],regional_acceptance=['shoulder_line'] if selected else [],lens=dict(selected='orthographic',inverse_distance=0,pixel_scale=scale,offset=off.tolist()),orientation_stage={},diagnostics=dict(shadow=shadow(),gated_alpha=gated_alpha,mode='bounded_chest' if torso else 'clavicle_only',before=before,after=diag['after'] if selected else before,proposal_after=trials[0]['after'],trials=trials,converged=bool(r.success),evaluations=r.nfev,budget_stop=getattr(r,'budget_reason',None),camera_unchanged=True,head_orientation_preserved=True,lower_body_unchanged=True,observation_sha256=hashlib.sha256(observation.read_bytes()).hexdigest()))
 (fit/'metrics.json').write_text(json.dumps(met,indent=2));return dict(case_id=case,status='fit',geometry_accepted=bool(selected))

def main():
 p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--observations',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--case',required=True);p.add_argument('--torso',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);result=solve(a.assets,a.baseline,a.observations,a.out,a.case,a.torso);(a.out/'results.json').write_text(json.dumps([result],indent=2));print(result)
if __name__=='__main__':main()
