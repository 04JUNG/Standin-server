"""Profile camera fit using torso-only anchors; shoulder ownership may abstain.
No image IDs, face landmarks, distal limbs, reference poses or mesh candidates.
"""
import copy
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ANCHORS=('LeftArm','RightArm','LeftUpLeg','RightUpLeg')

def fit_profile(world,pr,basis,origin,ix,goal,cue):
 out=copy.deepcopy(pr);diag=dict(changed=False,reason='profile_not_applicable')
 if cue.get('person_count')!=1 or cue.get('torso_view')!='profile' or cue.get('confidence') not in ('clear','weak'):return out,diag
 anchors=[n for n in ANCHORS if n in goal and n in ix]
 if len(anchors)<2:return out,dict(diag,reason='insufficient_torso_anchors',profile_evidence_retained=True)
 ids=[ix[n] for n in anchors];target=np.asarray([goal[n] for n in anchors]);origin=np.asarray(origin)
 # Orthogonal anatomical torso frame; shoulder height is not mistaken for yaw.
 up=world[ix['Spine2'],:3,3]-world[ix['Hips'],:3,3];up/=max(np.linalg.norm(up),1e-10)
 right=world[ix['LeftArm'],:3,3]-world[ix['RightArm'],:3,3];right-=up*np.dot(up,right)
 if np.linalg.norm(right)<1e-8:return out,dict(diag,reason='degenerate_torso_frame')
 right/=np.linalg.norm(right);frame=np.column_stack([right,up,np.cross(right,up)])
 facing=cue.get('torso_facing','uncertain');near=cue.get('nearer_shoulder','uncertain');sign=None;source='unsigned_profile'
 # Chest-facing evidence is independent from head direction. Only use explicit,
 # confident new-schema evidence; free-form evidence is never parsed as a command.
 if facing in ('screen_left','screen_right') and cue.get('torso_facing_confidence')=='clear':
  sign=1 if facing=='screen_left' else -1;source='chest_facing'
 elif near in ('screen_left','screen_right') and all(n in goal for n in ('LeftArm','RightArm')):
  separation=goal['LeftArm'][0]-goal['RightArm'][0]
  if abs(separation)>5:
   left_near=(near=='screen_right')==(separation>0);sign=1 if left_near else -1;source='separated_shoulder_ownership'
 signs=[sign] if sign else [-1,1]
 elevation=cue.get('elevation','uncertain');pitch=np.radians({'above':[-30,-5],'level':[-8,8],'below':[5,30]}.get(elevation,[-34,34]))
 trials=[]
 for side in signs:
  yaw=np.radians([65,88] if side>0 else [-88,-65]);lo=np.r_[yaw[0],pitch[0],-.55,np.log(100),[-500,-500]];hi=np.r_[yaw[1],pitch[1],.55,np.log(1600),[1100,1400]]
  def calc(x):
   cam=frame@Rotation.from_euler('yxz',x[:3]).as_matrix();scale=np.exp(x[3]);off=x[4:];xy=((world[:,:3,3]-origin)@cam)[:,:2]*[scale,-scale]+off
   return cam,scale,off,xy
  def residual(x):
   _,_,_,xy=calc(x)
   return np.r_[(xy[ids]-target).ravel()/25,(x[0]-np.mean(yaw))*.4,x[1]*.3,x[2]*.3]
  x=np.r_[np.mean(yaw),np.mean(pitch),0,np.log(pr['pixel_scale']),pr['offset']];initial=calc(np.clip(x,lo,hi));x[4:]+=target.mean(0)-initial[-1][ids].mean(0)
  result=least_squares(residual,np.clip(x,lo+1e-7,hi-1e-7),bounds=(lo,hi),max_nfev=200)
  cam,scale,off,xy=calc(result.x);rmse=float(np.sqrt(np.mean((xy[ids]-target)**2)))
  angle=float(np.degrees(np.arctan2((right@cam)[2],(right@cam)[0])))
  trials.append(dict(sign=side,converged=bool(result.success),rmse=rmse,shoulder_yaw_deg=angle,camera=cam,pixel_scale=float(scale),offset=off,evaluations=result.nfev))
 usable=[t for t in trials if t['converged'] and np.isfinite(t['rmse'])]
 if not usable:return out,dict(diag,reason='profile_solver_failed',profile_evidence_retained=True)
 best=min(usable,key=lambda t:(t['rmse'],Rotation.from_matrix(np.asarray(pr['camera']).T@t['camera']).magnitude()))
 out.update(camera=best['camera'].tolist(),origin=origin.tolist(),pixel_scale=best['pixel_scale'],offset=best['offset'].tolist())
 public=[{k:v for k,v in t.items() if k not in ('camera','offset','pixel_scale')} for t in trials]
 return out,dict(changed=True,reason='profile_torso_fit',torso_only=True,torso_world_unchanged=True,anchors=anchors,sign_source=source,sign_ambiguous=sign is None,selected_sign=best['sign'],shoulder_yaw_deg=best['shoulder_yaw_deg'],rmse=best['rmse'],converged=True,trials=public)
