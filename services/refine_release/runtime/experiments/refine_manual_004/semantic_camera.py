"""Offline semantic camera intervals followed by a single frozen-camera pose fit."""
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from PIL import Image,ImageOps,ImageDraw
from geometry import prepare,fk,frame,render_projected
from auto_targets import map_targets
import sys,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'refine_unified'))
from validity import shadow, check_world

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--assets',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cases',nargs='+',required=True)
p.add_argument('--cues',type=Path,required=True);p.add_argument('--fixed-pose',type=Path)
a=p.parse_args();

a.out.mkdir(parents=True,exist_ok=False)
records=[]
for cid in a.cases:
 old=a.baseline/cid;folder=a.out/cid/'fit';folder.mkdir(parents=True)
 fixture=json.loads((old/'fixture.json').read_text());obs=json.loads((old/'observations.json').read_text())
 for item in fixture['files'].values():
  assert hashlib.sha256((a.assets/item['path']).read_bytes()).hexdigest()==item['sha256']
 rig=json.loads((a.assets/fixture['files']['rig']['path']).read_text());names,ix,parents,local,base=prepare(rig)
 _,origin,_,basis=frame(rig['bones']);tar,hip,diag=map_targets(obs,fixture['files']['rough']['sha256'],fixture['image_size'])
 width,height=fixture['image_size'];rs=min(535/width,687/height);shift=np.array([(535-width*rs)/2,(687-height*rs)/2])
 kp=np.array(obs['keypoints'])*rs+shift;tar={n:np.array(v)*rs+shift for n,v in tar.items()};hip=hip*rs+shift
 doc=json.loads((a.cues/f'{cid}.json').read_text())
 assert doc['image_sha256']==fixture['files']['rough']['sha256']
 cue=doc['raw']
 ranges={'frontal':(0,10),'slight_oblique':(15,35),'three_quarter':(30,60),'profile':(65,85)}
 if cue['person_count']!=1 or cue['confidence']!='clear' or cue['torso_view'] not in ranges or cue['nearer_shoulder'] not in ['screen_left','screen_right','equal']:
  shutil.copytree(old/'fit',folder,dirs_exist_ok=True)
  records.append(dict(case_id=cid,status='fit',mode='baseline_preserved_uncertain_cues'))
  (a.out/'results.json').write_text(json.dumps(records,indent=2))
  continue
 strength=25 if cue['confidence']=='clear' else 8
 low,high=ranges[cue['torso_view']]
 if cue['nearer_shoulder']=='equal' and cue['torso_view']!='frontal':raise ValueError('ambiguous shoulder sign')
 left_is_screen_right=kp[5,0]>kp[6,0]
 left_near=(cue['nearer_shoulder']=='screen_right')==left_is_screen_right
 sign=1 if left_near else -1
 yaw_interval=np.radians([low,high] if sign>0 else [-high,-low])
 pitch_interval=np.radians({'above':[-30,-5],'level':[-8,8],'below':[5,30],'uncertain':[-34,34]}[cue['elevation']])
 def interval_error(value,interval):return max(interval[0]-value,0)+min(interval[1]-value,0)
 torso=['Spine','Spine1','Spine2'];ti=[ix[n] for n in torso]
 # Stage 1: fit only torso keypoints. Distal limbs cannot rotate the camera.
 torso_kp=np.array([5,6,11,12]);rawkp=np.array(obs['keypoints'])[torso_kp]
 valid=(rawkp[:,0]>width*.03)&(rawkp[:,0]<width*.97)&(rawkp[:,1]>height*.03)&(rawkp[:,1]<height*.95)&np.asarray(obs.get('joint_mask',np.asarray(obs['scores'])>=.3))[torso_kp]
 if np.count_nonzero(valid)<2:raise ValueError('Insufficient interior torso anchors')
 tid=np.array([ix[n] for n in ['LeftArm','RightArm','LeftUpLeg','RightUpLeg']])[valid];tgoal=kp[torso_kp[valid]]
 frozen=None
 if a.fixed_pose:
  with np.load(a.fixed_pose/'expected-world.npz') as d:frozen=d['world'].copy()
 def stage1_calc(x):
  w=frozen.copy() if frozen is not None else fk(local,parents,ti,x[:9]);cam=basis@Rotation.from_rotvec(x[9:12]).as_matrix();scale=np.exp(x[12]);off=x[13:15]
  xy=((w[:,:3,3]-origin)@cam)[:,:2]*[scale,-scale]+off
  return w,cam,scale,off,xy
 def torso_res(x):
  w,cam,scale,off,xy=stage1_calc(x)
  shoulder=(w[ix['LeftArm'],:3,3]-w[ix['RightArm'],:3,3])@cam
  angle=np.arctan2(shoulder[2],shoulder[0])
  return np.r_[(xy[tid]-tgoal).ravel()/25,x[:9]*2,x[9:12]*.5,interval_error(angle,yaw_interval)*strength,interval_error(x[9],pitch_interval)*strength,(angle-np.mean(yaw_interval))*2,(x[9]-np.mean(pitch_interval))*2]
 lower=np.r_[np.repeat(-np.radians(8),9),[-.6,-1.1,-.55],np.log(200),[0,100]]
 upper=np.r_[np.repeat(np.radians(8),9),[.6,1.1,.55],np.log(1100),[535,750]]
 runs=[]
 for yaw in [float(np.mean(yaw_interval))]:
  x=np.r_[np.zeros(9),[0,yaw,0],np.log(600),[270,560]]
  r=least_squares(torso_res,np.clip(x,lower,upper),bounds=(lower,upper),max_nfev=500);runs.append(r)
 r1=min(runs,key=lambda r:r.cost)
 if not r1.success and not shadow():raise RuntimeError(f"camera not converged: {r1.message}; nfev={r1.nfev}")
 torso_world,cam,scale,off,_=stage1_calc(r1.x)
 # Stage 2: keep torso and camera fixed, fit limbs and an independent head goal.
 moving=['RightArm','RightForeArm','LeftArm','LeftForeArm','LeftUpLeg','Neck','Head'];active=torso+moving
 lim=np.repeat(np.radians([85,100,85,100,95,35,55]),3)
 pts=np.array([ix[n] for n in tar]);goal=np.array(list(tar.values()))
 # Camera-space head direction proxy from nose-to-ear-center offset.
 # COCO face predictions on featureless drawings remain uncertain, not ground truth.
 eyes=kp[1]-kp[2];ears=kp[3]-kp[4];center=(kp[3]+kp[4])/2
 yaw=np.arcsin(np.clip(2*(kp[0,0]-center[0])/max(10,abs(ears[0])),-.8,.8))
 roll=np.arctan2(-eyes[1],eyes[0])
 desired=Rotation.from_euler('zy',[roll,yaw]).as_matrix()
 head_valid=bool(np.min(np.array(obs['scores'])[:5])>=.3 and eyes[0]>5 and ears[0]>10)
 def calc(x):
  w=frozen.copy() if frozen is not None else fk(local,parents,[ix[n] for n in active],np.r_[r1.x[:9],x]);xy=((w[:,:3,3]-origin)@cam)[:,:2]*[scale,-scale]+off
  return w,xy
 def residual(x):
  w,xy=calc(x)
  hr=Rotation.from_matrix(desired.T@cam.T@w[ix['Head'],:3,:3]).as_rotvec() if head_valid else Rotation.from_matrix(w[ix['Head'],:3,:3]@np.linalg.inv(base[ix['Head'],:3,:3])).as_rotvec()
  return np.r_[(xy[pts]-goal).ravel()/40,x*.6,hr*5]
 if frozen is None:
  r2=least_squares(residual,np.zeros(len(lim)),bounds=(-lim,lim),max_nfev=500)
  if not r2.success and not shadow():raise RuntimeError(f"pose not converged: {r2.message}; nfev={r2.nfev}")
 else:
  from types import SimpleNamespace
  r2=SimpleNamespace(x=np.zeros(len(lim)),success=True)
 world,xy=calc(r2.x);edits={n:np.rad2deg(d).tolist() for n,d in zip(active,np.r_[r1.x[:9],r2.x].reshape(-1,3))}
 if frozen is not None:
  world=frozen.copy();xy=((world[:,:3,3]-origin)@cam)[:,:2]*[scale,-scale]+off
  edits=json.loads((a.fixed_pose/'adjustment.json').read_text())['edits']
  assert np.array_equal(world,frozen)
 check_world(world,base,parents)
 params=dict(edits=edits,source=str((a.assets/'input'/rig['source']).resolve()),camera=cam.tolist(),origin=origin.tolist(),pixel_scale=scale,offset=off.tolist(),targets={k:v.tolist() for k,v in tar.items()})
 (folder/'adjustment.json').write_text(json.dumps(params,indent=2));np.savez_compressed(folder/'expected-world.npz',names=names,world=world)
 lengths=max(abs(np.linalg.norm(world[i,:3,3]-world[j,:3,3])-np.linalg.norm(base[i,:3,3]-base[j,:3,3])) for i,j in enumerate(parents) if j>=0)
 bound=[n for i,n in enumerate(moving) if np.any(np.abs(np.abs(r2.x.reshape(-1,3)[i])-lim.reshape(-1,3)[i])<1e-4)]
 metric=dict(case_id=cid,files=fixture['files'],manual_targets=False,ground_truth_used=False,production_enabled=False,solver_success=bool(r1.success and r2.success),convergence=dict(camera=dict(converged=bool(r1.success),evaluations=r1.nfev,message=r1.message),pose=dict(converged=bool(r2.success),evaluations=getattr(r2,"nfev",0),message=getattr(r2,"message","fixed"))),rmse_px=float(np.sqrt(np.mean((xy[pts]-goal)**2))),max_bone_length_error=float(lengths),bound_hit_bones=bound,head_goal_valid=head_valid,head_goal_yaw_deg=float(np.degrees(yaw)),head_goal_roll_deg=float(np.degrees(roll)),head_orientation_error_deg=float(np.degrees(np.linalg.norm(Rotation.from_matrix(desired.T@cam.T@world[ix['Head'],:3,:3]).as_rotvec()))),torso_rmse_px=float(np.sqrt(np.mean((stage1_calc(r1.x)[-1][tid]-tgoal)**2))),camera_cues=cue,torso_anchor_coco_ids=torso_kp[valid].tolist(),shoulder_yaw_interval_deg=np.degrees(yaw_interval).tolist(),camera_rotvec_x_interval_deg=np.degrees(pitch_interval).tolist(),reference_used_for_fit=False,fixed_pose_source=str(a.fixed_pose) if a.fixed_pose else None,pose_unchanged=frozen is not None,camera_frozen=True,camera_stage2_delta_deg=0,head_direction_source='automatic_COCO_nose_ears_eyes_proxy',face_estimate_not_visual_ground_truth=True)
 (folder/'metrics.json').write_text(json.dumps(metric,indent=2))
 with np.load(a.assets/fixture['files']['mesh']['path']) as m:v,f,weights=m['vertices'],m['faces'],m['weights']
 oldparams=json.loads((old/'fit/adjustment.json').read_text())
 with np.load(old/'fit/expected-world.npz') as data:oldworld=data['world']
 ims=[ImageOps.pad(Image.open(a.assets/fixture['files']['rough']['path']).convert('RGB'),(535,687),color='white')]
 for w,camera,sc,offset in [(oldworld,np.array(oldparams['camera']),oldparams['pixel_scale'],oldparams['offset']),(world,cam,scale,off)]:
  vv=np.einsum('vn,nvi->vi',weights,np.einsum('nij,vj->nvi',w@np.linalg.inv(base),np.c_[v,np.ones(len(v))])[:,:,:3]);cv=(vv-origin)@camera;uv=cv[:,:2]*[sc,-sc]+offset
  ims.append(render_projected(cv,f,uv,(535,687)))
 canvas=Image.new('RGB',(1605,722),'white');draw=ImageDraw.Draw(canvas)
 for i,(im,label) in enumerate(zip(ims,['ROUGH','BASELINE JOINT CAMERA','AUTOMATIC VIEW CUES / FIXED CAMERA'])):canvas.paste(im,(535*i,35));draw.text((535*i+8,10),cid+' '+label,fill='black')
 canvas.save(folder/'comparison.png');records.append(dict(case_id=cid,status='fit',rmse_px=metric['rmse_px'],bound_hit_bones=bound));print(cid,metric,flush=True)
 (a.out/'results.json').write_text(json.dumps(records,indent=2))
