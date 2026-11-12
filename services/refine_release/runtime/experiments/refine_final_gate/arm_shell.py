"""Geometry-generated arm-chain repair. No image IDs or manual pose references."""
import copy,itertools
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from common import skin,write,project
from policy import unit

def propose(c,seed):
 if any(seed.get('pr',{}).get(k) for k in ['manual_targets','manual_conditions_used','ground_truth_used','manual_experiment']):raise ValueError('Manual seed cannot enter automatic arm repair')
 contacts=seed['diagnostic']['collision']['intersections'];affected=[]
 for s in ['Left','Right']:
  if any(any(s+t in pair for t in ['_hand','_forearm','_upperarm']) and any(b in pair for b in ['pelvis','thigh','chest']) for pair in contacts):affected.append(s)
 if not affected:return []
 # This path repairs existing rearward arm chains only. Other contacts keep their baseline.
 w0=seed['world'];cam=np.array(seed['cam']['camera']);origin=np.array(seed['cam']['origin']);points=(w0[:,:3,3]-origin)@cam;verts=(skin(c.v,c.weights,w0,c.base)-origin)@cam
 center=points[c.ix['Spine']];signs={};source_points=(c.base[:,:3,3]-origin)@cam
 for s in affected:
  # Use image-bound shoulder identities, never the already damaged candidate,
  # to map the VLM's screen-side arm evidence to anatomical sides.
  if all(n in c.goal for n in ['LeftArm','RightArm']):
   left=min(['Left','Right'],key=lambda side:c.goal[side+'Arm'][0]);position='screen_left' if s==left else 'screen_right'
   if any(a.get('shoulder_position')==position and a.get('confidence')=='clear' and a.get('sweep')=='toward' for a in c.body.get('arms',[])):continue
  # A collapsed candidate is not evidence for an originally hidden arm.
  source_rear=source_points[c.ix[s+'Hand'],2]-source_points[c.ix['Spine'],2]<-.04*c.scale
  image_rear=any(r['far']==s+'Hand' and r['near'] in ['Hips','Spine','Head','Neck'] for r in c.relations)
  if not source_rear and not image_rear:continue
  depth=np.mean(points[[c.ix[s+'Arm'],c.ix[s+'ForeArm']],2])-center[2]
  if depth>-.04*c.scale:continue
  # Don't reinterpret a clear existing forward-hand relation as behind-body.
  if any(r['near']==s+'Hand' and r['far'] in ['Hips','Spine','Head','Neck'] for r in c.relations):continue
  signs[s]=-1
 if not signs:return []
 # Include both chains when both have a measured contact, not merely by symmetry.
 body_names=['Hips','Spine','Spine1','Spine2','LeftUpLeg','RightUpLeg'];mass=c.weights[:,[c.ix[n] for n in body_names]].sum(1);body=verts[mass>.55]
 base_ed=copy.deepcopy(seed['pr']['edits']);results=[];targets_cache=[]
 for height_fraction,clearance_factor in itertools.product([0.,.5,1.],[1.,1.6]):
  ed=copy.deepcopy(base_ed);targets={}
  for s in signs:
   arm=points[c.ix[s+'Arm']];elbow=points[c.ix[s+'ForeArm']];wrist=points[c.ix[s+'Hand']];lateral=1 if arm[0]>=center[0] else -1
   hi=[i for i,n in enumerate(c.names) if n.startswith(s+'Hand')];hand=verts[c.weights[:,hi].sum(1)>.6];length=np.linalg.norm(w0[c.ix[s+'HandMiddle3'],:3,3]-w0[c.ix[s+'Hand'],:3,3]);radius=max(.03*c.scale,np.percentile(np.linalg.norm(hand[:,:2]-wrist[:2],axis=1),70)*.35)
   # Estimate local back surface at elbow and wrist bands, using the actual source mesh.
   wy=(1-height_fraction)*wrist[1]+height_fraction*points[c.ix['Hips'],1]
   ey=elbow[1]
   def rear(y):
    band=body[np.abs(body[:,1]-y)<.25*c.scale]
    return float(np.min((band if len(band) else body)[:,2]))
   # Separate relaxed hands by palm width, while keeping them inside shoulder span.
   width=min(abs(arm[0]-center[0])*.65,max(radius*1.6,length*.35))
   wt=np.array([center[0]+lateral*width,wy,rear(wy)-radius*clearance_factor])
   et=np.array([arm[0],ey,rear(ey)-radius*.7*clearance_factor])
   ns=[s+'Arm',s+'ForeArm'];x0=np.array([ed.get(n,[0,0,0]) for n in ns]).ravel();ids=[c.ix[s+'ForeArm'],c.ix[s+'Hand']];target=np.array([et,wt])
   def fun(x):
    e=copy.deepcopy(ed);e.update({n:v.tolist() for n,v in zip(ns,x.reshape(-1,3))});w=c.world(e);p=(w[ids,:3,3]-origin)@cam
    return np.r_[((p-target)/c.scale*30).ravel(),(x-x0)*.004]
   sol=least_squares(fun,x0,max_nfev=140,ftol=1e-9,xtol=1e-9,gtol=1e-9);ed.update({n:v.tolist() for n,v in zip(ns,sol.x.reshape(-1,3))});w=c.world(ed)
   # Relax fingers along torso-down, preserving twist by shortest-arc swing.
   hand_i=c.ix[s+'Hand'];tip=c.ix[s+'HandMiddle3'];direction=unit(w[tip,:3,3]-w[hand_i,:3,3]);down=-unit(w0[c.ix['Spine2'],:3,3]-w0[c.ix['Hips'],:3,3]);swing,_=Rotation.align_vectors([down],[direction]);wanted=swing.as_matrix()@w[hand_i,:3,:3];parent=c.parents[hand_i];m=np.linalg.solve(c.local[hand_i,:3,:3],np.linalg.solve(w[parent,:3,:3],wanted));u,_,vh=np.linalg.svd(m);ed[s+'Hand']=np.degrees(Rotation.from_matrix(u@vh).as_rotvec()).tolist()
   targets[s]=dict(elbow=et.tolist(),wrist=wt.tolist(),radius=radius,chain_depth=float(np.mean(points[[c.ix[s+'Arm'],c.ix[s+'ForeArm']],2])-center[2]),converged=bool(sol.success))
  pr=copy.deepcopy(seed['pr']);pr.update(local_repair_origin=seed['id'],automatic_arm_shell=True,manual_targets=False,manual_conditions_used=False);label=f'shell-{seed["id"]}-{len(results):02d}';row=c.add(label,ed,pr,seed['cam']);write(row['path']/'repair.json',dict(algorithm='body_surface_arm_chain',targets_from_geometry=targets,height_fraction=height_fraction,clearance_factor=clearance_factor,manual_coordinates=False,source_world_only=True));results.append(row)
 return results
