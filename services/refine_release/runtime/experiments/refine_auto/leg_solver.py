"""Automatic leg solve consuming measured upper-body anchor policy."""
import argparse,json,sys,copy,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from policy import upper_action
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'refine_new'))
from core import prepare,frame,fk,targets,skin,deformation,render_projected
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'refine006_auto'))
from regional import bend
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'refine015_relative'))
from overlap_constraints import local_points,partial_visibility_ok,screen_supported_relations
from review_v3 import raster_foot_visibility
import sys,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'refine_unified'))
from validity import shadow, check_world


import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'refine_unified'))
from solve_budget import solve as budget_solve, eligible
def least_squares(fun,x0,*args,**kwargs):
 return budget_solve(fun,x0,*args,stage='leg',**kwargs)

def overlap_points(near,far):
 """Depth comparison at closest projected points of the two shin segments."""
 u=near[1]-near[0];v=far[1]-far[0];a=np.column_stack([u[:2],-v[:2]]);b=(far[0]-near[0])[:2];st=np.clip(np.linalg.lstsq(a,b,rcond=None)[0],0,1)
 for _ in range(3):
  st[0]=np.clip(np.dot((far[0]+st[1]*v-near[0])[:2],u[:2])/max(np.dot(u[:2],u[:2]),1e-12),0,1)
  st[1]=np.clip(np.dot((near[0]+st[0]*u-far[0])[:2],v[:2])/max(np.dot(v[:2],v[:2]),1e-12),0,1)
 pn=near[0]+st[0]*u;pf=far[0]+st[1]*v
 return pn,pf,st

def segment_distance(a,b):
 u=a[1]-a[0];v=b[1]-b[0];st=np.clip(np.linalg.lstsq(np.column_stack([u,-v]),b[0]-a[0],rcond=None)[0],0,1)
 for _ in range(5):
  st[0]=np.clip(np.dot(b[0]+st[1]*v-a[0],u)/max(np.dot(u,u),1e-12),0,1)
  st[1]=np.clip(np.dot(a[0]+st[0]*u-b[0],v)/max(np.dot(v,v),1e-12),0,1)
 return float(np.linalg.norm(a[0]+st[0]*u-b[0]-st[1]*v))

def skin_edge_mask(weights,edges,active):
 mass=weights[:,active].sum(1)
 mixed=weights[edges].max(axis=2).min(axis=1)<.999
 different=weights[edges[:,0]].argmax(axis=1)!=weights[edges[:,1]].argmax(axis=1)
 return (mass[edges].max(1)>.05)&(mixed|different)

def limited_rotation(vector,limit_degrees):
 vector=np.asarray(vector,float);length=np.linalg.norm(vector)
 return vector*min(1.,np.radians(limit_degrees)/max(length,1e-12))

def main():
 p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--observations',type=Path,required=True);p.add_argument('--conditions',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--case',required=True);p.add_argument('--coupled-pelvis',action='store_true');p.add_argument('--skin-weight',type=float,default=50.);p.add_argument('--depth-seed',action='store_true');p.add_argument('--fit-framing',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);c=a.case
 fixture=json.loads((a.baseline/c/'fixture.json').read_text());obs=json.loads(a.observations.read_text());cond=json.loads(a.conditions.read_text());assert cond['image_sha256']==obs['image_sha256']==fixture['files']['rough']['sha256'];assert cond['observation_sha256']==hashlib.sha256(a.observations.read_bytes()).hexdigest();leg_solve=True
 rig=json.loads((a.assets/fixture['files']['rig']['path']).read_text());names,ix,parents,local,base=prepare(rig);_,_,torso,_=frame(rig['bones']);goal,_,_=targets(obs,fixture['image_size']);cond,rejected_relations=screen_supported_relations(cond,goal);order_enabled=cond.get('relative_order_enabled',True);pr=json.loads((a.baseline/c/'fit/adjustment.json').read_text());cam=np.array(pr['camera']);origin=np.array(pr['origin']);scale=pr['pixel_scale'];off=np.array(pr['offset']);ed=pr['edits'];baseline_pr=copy.deepcopy(pr)
 with np.load(a.baseline/c/'fit/expected-world.npz') as d:old=d['world']
 upper_policy=json.loads((a.baseline/c/'fit/metrics.json').read_text()).get('anchor_policy',{})
 upper_decision=upper_action(upper_policy,goal)
 upper_free=upper_decision=='bounded_joint_repair'
 a.coupled_pelvis=upper_free
 original_framing=dict(pixel_scale=scale,offset=off.tolist())
 if a.fit_framing and order_enabled:
  anchors=[n for n in ['LeftArm','RightArm','LeftUpLeg','RightUpLeg','Neck'] if n in goal];q=((old[[ix[n] for n in anchors],:3,3]-origin)@cam)[:,:2]*[1,-1];target=np.array([goal[n] for n in anchors]);design=np.zeros((2*len(q),3));design[0::2,0]=q[:,0];design[1::2,0]=q[:,1];design[0::2,1]=1;design[1::2,2]=1;fit=np.linalg.lstsq(design,target.ravel(),rcond=None)[0];scale=float(np.clip(fit[0],scale*.65,scale*1.35));off=(target-q*scale).mean(0);pr['pixel_scale']=scale;pr['offset']=off.tolist()
 with np.load(a.assets/fixture['files']['mesh']['path']) as d:v,f,weights=d['vertices'],d['faces'],d['weights']
 moving=[s+n for s in ['Left','Right'] for n in ['UpLeg','Leg']];active=[ix[n] for n in moving]+[ix[s+'Foot'] for s in ['Left','Right']]+([ix['Hips'],ix['Spine'],ix['Spine2'],ix['LeftShoulder'],ix['RightShoulder']] if a.coupled_pelvis else []);start=np.radians([ed.get(n,[0,0,0]) for n in moving]).ravel();jointnames=[s+n for s in ['Left','Right'] for n in ['UpLeg','Leg','Foot'] if s+n in goal and s+n not in cond.get('unobserved_joints',[])];ids=[ix[n] for n in jointnames];tar=np.array([goal[n] for n in jointnames]);jointweights=np.array([.2 if n in cond.get('unobserved_joints',[]) else 1 for n in jointnames])
 # Eight directional variables, with minimal-swing rotations rather than free axial twists.
 start=[]
 for side in ['Left','Right']:
  for bone,child in [(side+'UpLeg',side+'Leg'),(side+'Leg',side+'Foot')]:
   direction=(old[ix[child],:3,3]-old[ix[bone],:3,3])@cam
   start.extend([np.arctan2(direction[0],-direction[1]),np.arctan2(direction[2],np.linalg.norm(direction[:2]))])
 start=np.r_[np.array(start),np.zeros(16 if upper_free else 10)]
 large_leg_correction={s:any(n in goal and np.linalg.norm((((old[ix[n],:3,3]-origin)@cam)[:2]*[scale,-scale]+off)-goal[n])>20 for n in [s+'Leg',s+'Foot'] if n not in cond.get('unobserved_joints',[])) for s in ['Left','Right']}
 def world(e):return fk(local,parents,[ix[n] for n in e],np.radians(list(e.values())).ravel())
 def candidate(x):
  e=copy.deepcopy(ed)
  if a.coupled_pelvis:
   e['Hips']=np.degrees((Rotation.from_rotvec(np.radians(ed.get('Hips',[0,0,0])))*Rotation.from_rotvec(x[18:21])).as_rotvec()).tolist()
   temp=world(e);old_spine=Rotation.from_rotvec(np.radians(ed.get('Spine',[0,0,0]))).as_matrix();e['Spine']=np.degrees(Rotation.from_matrix(old_spine@np.linalg.solve(temp[ix['Spine'],:3,:3],old[ix['Spine'],:3,:3])).as_rotvec()).tolist()
  # Preserve the accepted local twist; direction-only reconstruction from zero
  # otherwise changes the mesh even at the nominal starting parameters.
  for j,(bone,child) in enumerate([(s+n,s+c) for s in ['Left','Right'] for n,c in [('UpLeg','Leg'),('Leg','Foot')]]):
   theta,phi=x[2*j:2*j+2];direction=cam@np.array([np.sin(theta)*np.cos(phi),-np.cos(theta)*np.cos(phi),np.sin(phi)])
   w=world(e);u=w[ix[child],:3,3]-w[ix[bone],:3,3];u/=np.linalg.norm(u);axis=np.cross(u,direction);angle=np.arctan2(np.linalg.norm(axis),np.dot(u,direction))
   if np.linalg.norm(axis)<1e-8 and np.dot(u,direction)<0:axis=np.cross(u,np.eye(3)[np.argmin(abs(u))])
   swing=Rotation.from_rotvec(direction*x[8+j]).as_matrix()@Rotation.from_rotvec(axis/max(1e-12,np.linalg.norm(axis))*angle).as_matrix();oldrot=Rotation.from_rotvec(np.radians(e.get(bone,[0,0,0]))).as_matrix();e[bone]=np.degrees(Rotation.from_matrix(oldrot@np.linalg.solve(w[ix[bone],:3,:3],swing@w[ix[bone],:3,:3])).as_rotvec()).tolist()
  if not leg_solve:e=copy.deepcopy(ed)
  # Keep each ankle's original world orientation as the reference. An inherited
  # shin swing must not arbitrarily tilt the shoe. Only bounded world corrections.
  for j,side in enumerate(['Left','Right']):
   n=side+'Foot';w=world(e);limit=45 if cond.get('foot_evidence',{}).get(side,{}).get('foot_visibility')=='fully_occluded' else 25;reference=w[ix[n],:3,:3] if large_leg_correction[side] else old[ix[n],:3,:3];desired=Rotation.from_rotvec(limited_rotation(x[12+j*3:15+j*3],limit)).as_matrix()@reference
   oldrot=Rotation.from_rotvec(np.radians(e.get(n,[0,0,0]))).as_matrix();e[n]=np.degrees(Rotation.from_matrix(oldrot@np.linalg.solve(w[ix[n],:3,:3],desired)).as_rotvec()).tolist()
  if upper_free:
   e['Spine2']=np.degrees((Rotation.from_rotvec(np.radians(ed.get('Spine2',[0,0,0])))*Rotation.from_rotvec(limited_rotation(x[21:24],15))).as_rotvec()).tolist()
   for n in ['Neck','LeftHand','RightHand']:
    ww=world(e);rot=Rotation.from_rotvec(np.radians(e.get(n,[0,0,0]))).as_matrix();e[n]=np.degrees(Rotation.from_matrix(rot@np.linalg.solve(ww[ix[n],:3,:3],old[ix[n],:3,:3])).as_rotvec()).tolist()
  return e,world(e)
 _identity_ed,_identity_world=candidate(start)
 identity_error=float(np.max(np.abs(_identity_world-old)))
 assert identity_error<1e-5, ('parameterization_changes_baseline',identity_error)
 def xy(w):return ((w[:,:3,3]-origin)@cam)[:,:2]*[scale,-scale]+off
 edges=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0);ll=np.linalg.norm(v[edges[:,0]]-v[edges[:,1]],axis=1);mass=weights[:,active].sum(1);mask=(ll>1e-7)&skin_edge_mask(weights,edges,active);edges=edges[mask];ll=ll[mask];oldv=skin(v,weights,old,base);rat=np.linalg.norm(oldv[edges[:,0]]-oldv[edges[:,1]],axis=1)/ll;rank=np.argsort(abs(np.log(np.maximum(rat,.001))));keep=np.arange(len(edges));edges=edges[keep];ll=ll[keep];rat=rat[keep];indices,inverse=np.unique(edges,return_inverse=True);ee=inverse.reshape(-1,2);sv=v[indices];sw=weights[indices]
 def facts(w):
  z=(w[:,:3,3]-origin)@cam;near=cond.get('near') or 'Right';far=cond.get('far') or 'Left';pn,pf,st=overlap_points(z[[ix[near+'Leg'],ix[near+'Foot']]],z[[ix[far+'Leg'],ix[far+'Foot']]])
  for relation in cond.get('local_occlusions',[]):
   segments={'thigh':('UpLeg','Leg'),'shin':('Leg','Foot'),'foot':('Foot','ToeBase')}
   if relation.get('near')==near and relation.get('far')==far and relation.get('front_segment') in segments and relation.get('back_segment') in segments:
    aa=[ix[near+b] for b in segments[relation['front_segment']]];bb=[ix[far+b] for b in segments[relation['back_segment']]];pn,pf,st=local_points(z[aa],z[bb],relation.get('along_front'),relation.get('along_back'));break
  return dict(relative_depth=float((pn[2]-pf[2])/torso),projected_gap_px=float(np.linalg.norm(pn[:2]-pf[:2])*scale),sample_parameters=st.tolist(),knee_bend={s:bend(w[[ix[s+n] for n in ['UpLeg','Leg','Foot']],:3,3]) for s in ['Left','Right']},rmse=float(np.sqrt(np.mean((xy(w)[ids]-tar)**2))))
 before=facts(old);olddef=deformation(oldv,v,f)
 radii={}
 for side in ['Left','Right']:
  for bone,child in [('UpLeg','Leg'),('Leg','Foot')]:
   b0,b1=base[ix[side+bone],:3,3],base[ix[side+child],:3,3];direction=b1-b0;points=v[weights[:,ix[side+bone]]>.6];t=np.clip((points-b0)@direction/np.dot(direction,direction),0,1);rad=np.linalg.norm(points-b0-t[:,None]*direction,axis=1);radii[side+bone]=float(np.quantile(rad,.95))
 required_depth=max(.08,(radii['LeftLeg']+radii['RightLeg'])*1.1/torso)
 pairs=[('LeftUpLeg','RightUpLeg'),('LeftLeg','RightLeg'),('LeftLeg','RightUpLeg'),('LeftUpLeg','RightLeg')]
 def collision_residual(w):
  capsules={}
  for side in ['Left','Right']:
   for bone,child in [('UpLeg','Leg'),('Leg','Foot')]:
    a0,a1=w[ix[side+bone],:3,3],w[ix[side+child],:3,3];capsules[side+bone]=np.array([a0*.88+a1*.12,a0*.12+a1*.88])
  return np.array([max(0,radii[a]+radii[b]+.005-segment_distance(capsules[a],capsules[b]))/torso*80 for a,b in pairs])
 foot_samples={};shin_samples={};thigh_samples={}
 for side in ['Left','Right']:
  wi=[i for i,n in enumerate(names) if n.startswith(side+'Foot') or n.startswith(side+'Toe')];vi=np.flatnonzero(weights[:,wi].sum(1)>.75);foot_samples[side]=vi;shin_samples[side]=np.flatnonzero(weights[:,ix[side+'Leg']]>.5);thigh_samples[side]=np.flatnonzero(weights[:,ix[side+'UpLeg']]>.5)
 def foot_residual(w):
  res=[];z=(w[:,:3,3]-origin)@cam
  for side,cue in cond.get('foot_evidence',{}).items():
   n=side+'Foot';child=side+'ToeBase'
   if n not in ix or child not in ix:continue
   if cue.get('foot_visibility')=='visible' and cue.get('confidence')=='clear':
    dirs={'screen_down':[0,-1],'screen_down_left':[-.707,-.707],'screen_down_right':[.707,-.707],'screen_left':[-1,0],'screen_right':[1,0],'screen_up':[0,1]};t=dirs.get(cue.get('toe_direction'))
    if t:
     d=z[ix[child]]-z[ix[n]];d=d[:2]/max(1e-6,np.linalg.norm(d[:2]));res.extend((d-np.array(t))*5)
   visibility=cue.get('foot_visibility');occluder=cue.get('foot_occluder')
   supported_partial=visibility=='partly_occluded' and occluder in ['other_shin','other_thigh','other_foot']
   supported_full=visibility=='fully_occluded' and ((order_enabled and side==cond['far'] and occluder=='clothing') or occluder in ['other_shin','other_thigh','other_foot'])
   if supported_partial or supported_full:
    front='Right' if side=='Left' else 'Left';bone='UpLeg' if occluder=='other_thigh' else 'Leg';child='Leg' if bone=='UpLeg' else 'Foot';vi=foot_samples[side];points=(skin(v[vi],weights[vi],w,base)-origin)@cam
    si=thigh_samples[front] if occluder=='other_thigh' else foot_samples[front] if occluder=='other_foot' else shin_samples[front];surface=(skin(v[si],weights[si],w,base)-origin)@cam;planes=ConvexHull(surface[:,:2]).equations;outside=np.max(points[:,:2]@planes[:,:2].T+planes[:,2],axis=1)*scale
    a0,a1=z[ix[front+bone]],z[ix[front+child]];d=a1-a0;t=np.clip((points[:,:2]-a0[:2])@d[:2]/max(1e-9,np.dot(d[:2],d[:2])),0,1);closest=a0+t[:,None]*d
    if supported_partial:
     # A range preserves a visible portion; never collapse partial into total hiding.
     order=np.argsort(outside);count=len(order);hidden=order[:max(1,int(count*.2))];visible=order[max(1,int(count*.7)):]
     res.extend(np.maximum(0,outside[hidden]+1)/max(1,np.sqrt(len(hidden))));res.extend(np.maximum(0,1-outside[visible])/max(1,np.sqrt(len(visible))))
     res.extend(np.maximum(0,points[hidden,2]-surface[:,2].min()+.02)*scale/max(1,np.sqrt(len(hidden))))
    else:
     res.extend(np.maximum(0,outside+3)/2);res.extend(np.maximum(0,points[:,2]-closest[:,2]+radii[front+bone]*.85)*scale/5)
  return np.array(res)
 def residual(x,weight):
  e,w=candidate(x);fa=facts(w);vv=skin(sv,sw,w,base);r=np.linalg.norm(vv[ee[:,0]]-vv[ee[:,1]],axis=1)/ll;res=[((xy(w)[ids]-tar)*jointweights[:,None]).ravel()/8,(x-start)*.5,np.maximum(r-np.maximum(1.8,rat),0)*a.skin_weight/np.sqrt(len(r)),np.maximum(np.minimum(.6,rat)-r,0)*a.skin_weight/np.sqrt(len(r)),([max(0,required_depth-fa['relative_depth'])*weight*8,fa['projected_gap_px']] if order_enabled else []),np.maximum(collision_residual(w)-collision_residual(old),0)*3,foot_residual(w),x[12:18]*3,x[18:]*3]
  if a.coupled_pelvis:
   upper=[n for n in goal if n not in moving and not n.endswith('Foot')];res.append((xy(w)[[ix[n] for n in upper]]-np.array([goal[n] for n in upper])).ravel()/4)
   ss=xy(w)[[ix['LeftArm'],ix['RightArm']]];tt=np.array([goal['LeftArm'],goal['RightArm']]);res.append(((ss[0]-ss[1])-(tt[0]-tt[1]))/2)
  if leg_solve:
   for side in ['Left','Right']:
    if side+'Leg' in goal:res.append((xy(w)[ix[side+'Leg']]-goal[side+'Leg'])/2)
  for side in ([s for s in cond.get('bent_knees',[]) if cond.get('knee_confidence',{}).get(s)=='clear' if cond.get('knee_confidence',{}).get(s)=='clear'] if leg_solve else []):res.append([max(0,(15 if cond.get('knee_confidence',{}).get(side)=='weak' else 30)-fa['knee_bend'][side])*.2*weight/8])
  for side in (cond.get('straight_knees',[]) if leg_solve else []):res.append([max(0,fa['knee_bend'][side]-15)*5*weight/8])
  for side in ['Left','Right']:res.append([max(0,fa['knee_bend'][side]-150)*.5])
  return np.concatenate([np.asarray(q).ravel() for q in res])
 records=[];lo=start-np.radians(110);hi=start+np.radians(110)
 lo[1:8:2]=-np.pi/2+.001;hi[1:8:2]=np.pi/2-.001
 lo[8:12]=-np.pi;hi[8:12]=np.pi
 for j,side in enumerate(['Left','Right']):
  if side+'Leg' not in goal:lo[4*j:4*j+2]=start[4*j:4*j+2]-1e-8;hi[4*j:4*j+2]=start[4*j:4*j+2]+1e-8
 lo[12:]=-np.radians(35);hi[12:]=np.radians(35)
 if not leg_solve:lo[:12]=start[:12]-1e-8;hi[:12]=start[:12]+1e-8
 seed0=start.copy()
 # Deterministic geometric initialization for a badly mismatched visible chain.
 # Preserve existing depth sign except where the VLM supplied a supported order.
 for j,(side,bone,child) in enumerate([(s,b,ch) for s in ['Left','Right'] for b,ch in [('UpLeg','Leg'),('Leg','Foot')]]):
  n0,n1=side+bone,side+child
  if n0 in goal and n1 in goal and n1 not in cond.get('unobserved_joints',[]):
   delta=(goal[n1]-goal[n0])*[1,-1]/scale;length=np.linalg.norm(old[ix[n1],:3,3]-old[ix[n0],:3,3]);zold=((old[ix[n1],:3,3]-old[ix[n0],:3,3])@cam)[2]
   if np.linalg.norm(xy(old)[ix[n1]]-goal[n1])>15:
    sign=(1 if side==cond['near'] else -1) if order_enabled else (1 if zold>=0 else -1)
    z=sign*np.sqrt(max(0,length*length-np.dot(delta,delta)));theta=np.arctan2(delta[0],-delta[1]);theta=start[2*j]+np.arctan2(np.sin(theta-start[2*j]),np.cos(theta-start[2*j]));seed0[2*j:2*j+2]=[theta,np.arctan2(z,np.linalg.norm(delta))]
 if a.depth_seed and order_enabled:
  for j,side in enumerate(['Left','Right']):seed0[4*j+1]=0;seed0[4*j+3]=-.6 if side==cond['far'] else .05
 for number,seed in enumerate([np.clip(seed0,lo+1e-8,hi-1e-8)]):
  x=seed.copy();stages=[]
  for strength in [3,8,16]:
   r=least_squares(lambda z:residual(z,strength),x,motion=lambda z:xy(candidate(z)[1]),bounds=(lo,hi),max_nfev=250,x_scale='jac',ftol=1e-5,xtol=1e-5,gtol=1e-5);x=r.x;stages.append(dict(converged=bool(r.success),evaluations=r.nfev))
  # Axial polish holds joint directions fixed; choose by mesh deformation, not XY alone.
  unpolished=x.copy();polish=[];polish_diagnostics=[]
  for twist_seed in [x[8:12].copy(),np.zeros(4),np.array([np.pi-.001,0,np.pi-.001,0])]:
   def twist_res(t):
    xx=x.copy();xx[8:12]=t;_,ww=candidate(xx);vv=skin(sv,sw,ww,base);rr=np.linalg.norm(vv[ee[:,0]]-vv[ee[:,1]],axis=1)/ll
    return np.r_[np.maximum(rr-np.maximum(1.5,rat),0)*30/np.sqrt(len(rr)),np.maximum(np.minimum(.7,rat)-rr,0)*30/np.sqrt(len(rr)),foot_residual(ww)*.05,t*.005]
   pp=least_squares(twist_res,np.clip(twist_seed,-np.pi+1e-6,np.pi-1e-6),bounds=(-np.pi,np.pi),max_nfev=100)
   xx=x.copy();xx[8:12]=pp.x;_,ww=candidate(xx);dd=deformation(skin(v,weights,ww,base),v,f);polish.append((dd['extreme_edges'],dd['ratio_max'],xx));polish_diagnostics.append(dict(converged=bool(pp.success),evaluations=pp.nfev,deformation=dd))
  x=unpolished if shadow() else min(polish,key=lambda t:t[:2])[2]
  # Backtrack only along the solved update; preserve any feasible local improvement.
  # Every step must pass the same semantic, skin and actual visibility checks.
  solved=x.copy();step_fraction=1.
  for alpha in [1.,.75,.5,.25,.125,.0625]:
   xx=start+alpha*(solved-start);_,ww=candidate(xx);ff=facts(ww);vv=skin(v,weights,ww,base);dd=deformation(vv,v,f)
   if ff['rmse']>before['rmse']+1 or dd['extreme_edges']>olddef['extreme_edges'] or dd['ratio_max']>max(2,olddef['ratio_max'])*1.05:continue
   if order_enabled and (ff['projected_gap_px']>10 or ff['relative_depth']<required_depth*.9):continue
   if any(ff['knee_bend'][ss]>25 for ss in cond.get('straight_knees',[])):continue
   if any(ff['knee_bend'][ss]<25 for ss in cond.get('bent_knees',[]) if cond.get('knee_confidence',{}).get(ss)=='clear'):continue
   if np.linalg.norm(foot_residual(ww))>np.linalg.norm(foot_residual(old))+1e-6:continue
   visible_ok=True
   for ss,cc in cond.get('foot_evidence',{}).items():
    if cc.get('foot_visibility')=='partly_occluded' and cc.get('foot_occluder') in ['other_shin','other_thigh','other_foot']:
     fi=[i for i,n in enumerate(names) if n.startswith(ss+'Foot') or n.startswith(ss+'Toe')];c0=(oldv-origin)@cam;c1=(vv-origin)@cam
     vb=raster_foot_visibility(c0,f,c0[:,:2]*[scale,-scale]+off,weights,fi);va=raster_foot_visibility(c1,f,c1[:,:2]*[scale,-scale]+off,weights,fi)
     visible_ok &= partial_visibility_ok(vb,va)
   if visible_ok:x=xx;step_fraction=alpha;break
  gated_step_fraction=step_fraction
  if shadow():x=solved;step_fraction=1.
  e,w=candidate(x);check_world(w,base,parents);after=facts(w);df=deformation(skin(v,weights,w,base),v,f);reasons=[]
  visibility={}
  for side,cue in cond.get('foot_evidence',{}).items():
   if cue.get('foot_visibility')=='partly_occluded' and cue.get('foot_occluder') in ['other_shin','other_thigh','other_foot']:
    ff=[i for i,n in enumerate(names) if n.startswith(side+'Foot') or n.startswith(side+'Toe')];vv0=(oldv-origin)@cam;vv1=(skin(v,weights,w,base)-origin)@cam
    vb=raster_foot_visibility(vv0,f,vv0[:,:2]*[scale,-scale]+off,weights,ff);va=raster_foot_visibility(vv1,f,vv1[:,:2]*[scale,-scale]+off,weights,ff);visibility[side]=dict(before=vb,after=va)
    if not partial_visibility_ok(vb,va):reasons.append('rendered_partial_visibility_unsatisfied_'+side)
  if a.coupled_pelvis:
   upper=[n for n in goal if n not in moving and not n.endswith('Foot')];ui=[ix[n] for n in upper];ut=np.array([goal[n] for n in upper]);ub=float(np.sqrt(np.mean((xy(old)[ui]-ut)**2)));ua=float(np.sqrt(np.mean((xy(w)[ui]-ut)**2)))
   if ua>ub+2:reasons.append('upper_projection_regression')
   for nn in ['LeftArm','RightArm']:
    if np.linalg.norm(xy(w)[ix[nn]]-goal[nn])>np.linalg.norm(xy(old)[ix[nn]]-goal[nn])+.5:reasons.append('shoulder_regression_'+nn)
  if not eligible(r):reasons.append('not_converged')
  if np.linalg.norm(foot_residual(w))>np.linalg.norm(foot_residual(old))+1e-6:reasons.append('foot_evidence_regression')
  if order_enabled and after['projected_gap_px']>10:reasons.append('projected_overlap_missing')
  if order_enabled and after['relative_depth']<required_depth*.9:reasons.append('relative_order_unsatisfied')
  if leg_solve and any(after['knee_bend'][s]<(10 if cond.get('knee_confidence',{}).get(s)=='weak' else 25) for s in cond.get('bent_knees',[]) if cond.get('knee_confidence',{}).get(s)=='clear'):reasons.append('knee_flexion_unsatisfied')
  if leg_solve and any(after['knee_bend'][s]>25 for s in cond.get('straight_knees',[])):reasons.append('straight_knee_bent')
  if after['rmse']>before['rmse']+1:reasons.append('projection_regression')
  if df['extreme_edges']>olddef['extreme_edges']:reasons.append('skin_count_regression')
  if df['ratio_max']>max(2,olddef['ratio_max'])*1.05:reasons.append('skin_peak_regression')
  folder=a.out/c/'candidates'/str(number);folder.mkdir(parents=True);proposal=copy.deepcopy(pr);proposal['edits']=e;(folder/'adjustment.json').write_text(json.dumps(proposal,indent=2));np.savez_compressed(folder/'expected-world.npz',names=names,world=w);record=dict(candidate=number,shadow=shadow(),gated_step_fraction_on_tested_update=gated_step_fraction,axial_polish=polish_diagnostics,upper_anchor_decision=upper_decision,upper_policy_consumed=upper_policy,required_depth=required_depth,step_fraction=step_fraction,coupled_pelvis=a.coupled_pelvis,rejected_relations=rejected_relations,parameters=x.tolist(),identity_error=identity_error,rendered_visibility=visibility,foot_follows_leg=large_leg_correction,relative_order_enabled=order_enabled,partial_occlusion_sides=[s for s,q in cond.get('foot_evidence',{}).items() if q.get('foot_visibility')=='partly_occluded' and q.get('foot_occluder') in ['other_shin','other_thigh','other_foot']],foot_rotation_degrees={s:float(np.degrees(Rotation.from_matrix(w[ix[s+'Foot'],:3,:3]@old[ix[s+'Foot'],:3,:3].T).magnitude())) for s in ['Left','Right']},leg_solve=leg_solve,foot_residual_before=float(np.linalg.norm(foot_residual(old))),foot_residual_after=float(np.linalg.norm(foot_residual(w))),capsule_radii=radii,capsule_residual=collision_residual(w).tolist(),before=before,after=after,before_deformation=olddef,after_deformation=df,accepted=not reasons,reasons=reasons,stages=stages);(folder/'diagnostic.json').write_text(json.dumps(record,indent=2));records.append(record)
  advance=shadow() or not reasons
  chosen_w=w if advance else old;chosen_pr=proposal if advance else baseline_pr;fit=a.out/c/'fit';fit.mkdir();(fit/'adjustment.json').write_text(json.dumps(chosen_pr,indent=2));np.savez_compressed(fit/'expected-world.npz',names=names,world=chosen_w)
  if not advance:
   import shutil
   for name in ['adjustment.json','expected-world.npz']:shutil.copy2(a.baseline/c/'fit'/name,fit/name)
  protected=[i for i,n in enumerate(names) if not any(n.startswith(s+n2) for s in ['Left','Right'] for n2 in ['UpLeg','Leg','Foot','Toe'])];assert a.coupled_pelvis or np.allclose(chosen_w[protected],old[protected],atol=1e-7)
  anchor=((chosen_w[ix['Hips'],:3,3]-origin)@cam).tolist();camera=dict(camera=pr['camera'],origin=pr['origin'],anchor_camera_xyz=anchor,projection='orthographic',inverse_distance=0,pixel_scale=chosen_pr['pixel_scale'],offset=chosen_pr['offset']);(fit/'camera.json').write_text(json.dumps(camera,indent=2));met=dict(case_id=c,files=fixture['files'],manual_targets=False,manual_conditions_used=cond['source']=='user_review',reference_used_for_fit=False,conditions=cond,framing_before=original_framing,framing_after=dict(pixel_scale=chosen_pr['pixel_scale'],offset=chosen_pr['offset']),camera_rotation_unchanged=True,geometry_accepted=advance,rmse_px=after['rmse'] if advance else before['rmse'],old_rmse=before['rmse'],old_deformation=olddef,rejection_reasons=reasons,bound_hit_bones=[],regional_acceptance=['paired_legs_and_feet' if leg_solve else 'feet_only'] if advance else [],lens=dict(selected='orthographic',inverse_distance=0,pixel_scale=chosen_pr['pixel_scale'],offset=chosen_pr['offset']),orientation_stage={},diagnostics=record);(fit/'metrics.json').write_text(json.dumps(met,indent=2));(a.out/'results.json').write_text(json.dumps([dict(case_id=c,status='fit',geometry_accepted=advance,projection='orthographic')],indent=2));print(record)
if __name__=='__main__':main()
