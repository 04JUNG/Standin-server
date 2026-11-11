"""Stage acceptance contract for an established profile; never splice bone arrays."""
import numpy as np
from scipy.spatial.transform import Rotation
from common import read,write,copy_checkpoint,camera,prepare,targets

def rotation(m):
 u,_,v=np.linalg.svd(m);r=u@v
 if np.linalg.det(r)<0:u[:,-1]*=-1;r=u@v
 return r

def drift(before,after,a,b,ix):
 ca=np.array(a['camera']);cb=np.array(b['camera'])
 result={'camera':float(np.degrees(Rotation.from_matrix(ca.T@cb).magnitude()))}
 for n in ['Hips','Spine2']:
  ra=rotation(ca.T@before[ix[n],:3,:3]);rb=rotation(cb.T@after[ix[n],:3,:3]);result[n]=float(np.degrees(Rotation.from_matrix(ra.T@rb).magnitude()))
 for n in ['Neck','Head']:
  parent='Spine2'
  ra=rotation(np.linalg.solve(before[ix[parent],:3,:3],before[ix[n],:3,:3]));rb=rotation(np.linalg.solve(after[ix[parent],:3,:3],after[ix[n],:3,:3]));result[n+'_relative']=float(np.degrees(Rotation.from_matrix(ra.T@rb).magnitude()))
 return result

def enforce(anchor,prior,candidate,fx,assets,record):
 _,ix,_,_,_=prepare(read(assets/fx['files']['rig']['path']))
 a=read(anchor/'fit/adjustment.json');b=read(candidate/'fit/adjustment.json')
 before=np.load(anchor/'fit/expected-world.npz')['world'];after=np.load(candidate/'fit/expected-world.npz')['world'];d=drift(before,after,a,b,ix)
 accepted=d['camera']<=3 and d['Hips']<=5 and d['Spine2']<=5 and d['Neck_relative']<=45 and d['Head_relative']<=60
 write(record,dict(accepted=accepted,drift_degrees=d,limits_degrees=dict(camera=3,Hips=5,Spine2=5,Neck_relative=45,Head_relative=60),reason='within_profile_contract' if accepted else 'profile_contract_changed',fallback='previous_valid_stage',candidate_preserved=True))
 return candidate if accepted else prior

def camera_only(source,dest,fitted,fx,obs,assets):
 copy_checkpoint(source,dest);p=read(dest/'fit/adjustment.json')
 for k in ['camera','origin','pixel_scale','offset']:p[k]=fitted[k]
 p['edits']={};write(dest/'fit/adjustment.json',p)
 _,ix,_,_,_=prepare(read(assets/fx['files']['rig']['path']));w=np.load(dest/'fit/expected-world.npz')['world'];write(dest/'fit/camera.json',camera(p,w,ix))
 goal,*_=targets(obs,fx['image_size']);xy=((w[:,:3,3]-p['origin'])@p['camera'])[:,:2]*[p['pixel_scale'],-p['pixel_scale']]+p['offset'];m=read(dest/'fit/metrics.json');m.update(geometry_accepted=False,rejection_reasons=['pose_rejected_camera_retained'],rmse_px=float(np.sqrt(np.mean([(xy[ix[n]]-g)**2 for n,g in goal.items()]))));m['lens'].update(pixel_scale=p['pixel_scale'],offset=p['offset']);write(dest/'fit/metrics.json',m)
