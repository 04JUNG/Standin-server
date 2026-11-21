"""Available-chain 004/010 fallback; missing hips do not disable visible arms."""
import argparse
from common import *
from core import camera_stage,pose_stage
p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--cues',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--case',required=True);a=p.parse_args();c=a.case;old=a.baseline/c;fx=read(old/'fixture.json');obs=read(old/'observations.json');rig=read(a.assets/fx['files']['rig']['path']);names,ix,parents,local,base=prepare(rig);_,origin,torso,basis=frame(rig['bones']);goal,kp,valid=targets(obs,fx['image_size']);pr=read(old/'fit/adjustment.json');w=np.load(old/'fit/expected-world.npz')['world']
# Even uncertain view semantics retain measured torso framing. Distal limbs must
# not shrink the torso to compensate for an initially wrong source limb pose.
anchors=[n for n in ['LeftArm','RightArm','LeftUpLeg','RightUpLeg'] if n in goal]
if len(anchors)>=2:
 q=((w[[ix[n] for n in anchors],:3,3]-pr['origin'])@pr['camera'])[:,:2]*[1,-1];g=np.array([goal[n] for n in anchors]);qc=q-q.mean(0);gc=g-g.mean(0);scale=float(np.clip(np.sum(qc*gc)/max(np.sum(qc*qc),1e-8),100,1600));pr.update(pixel_scale=scale,offset=(g.mean(0)-q.mean(0)*scale).tolist())
pr,cd=camera_stage(w,pr,basis,origin,ix,goal,read(a.cues/f'{c}.json'),obs['image_sha256']);ww,ed,pd=pose_stage(local,parents,base,ix,w,pr,goal,kp,valid,torso,[],cd['changed']);pr['edits']=ed;copy_checkpoint(old,a.out/c);fit=a.out/c/'fit';write(fit/'adjustment.json',pr);np.savez_compressed(fit/'expected-world.npz',names=names,world=ww);write(fit/'camera.json',camera(pr,ww,ix));m=read(fit/'metrics.json');m.update(camera_stage=cd,pose_stage=pd,partial_observation_path=True);write(fit/'metrics.json',m);write(a.out/'results.json',[dict(case_id=c,status='fit')])
