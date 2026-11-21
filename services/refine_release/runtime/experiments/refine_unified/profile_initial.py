"""Signed profile camera only; preserve original head/limbs until later stages."""
import argparse
from common import *
from torso_view import fit_profile
p=argparse.ArgumentParser()
for n in ['assets','baseline','cues','direction','out']:p.add_argument('--'+n,type=Path,required=True)
p.add_argument('--case',required=True);a=p.parse_args();c=a.case;source=a.baseline/c;fx=read(source/'fixture.json');obs=read(source/'observations.json');rig=read(a.assets/fx['files']['rig']['path']);names,ix,_,_,base=prepare(rig);goal,*_=targets(obs,fx['image_size']);_,origin,_,basis=frame(rig['bones']);pr=read(source/'fit/adjustment.json')
from profile_direction import select
cue,reason=select(read(a.cues/f'{c}.json'),read(a.direction))
if cue is None:raise ValueError(reason)
pr,d=fit_profile(base,pr,basis,origin,ix,goal,cue)
if not d['changed']:raise ValueError(d['reason'])
copy_checkpoint(source,a.out/c);fit=a.out/c/'fit';write(fit/'adjustment.json',pr);write(fit/'camera.json',camera(pr,base,ix));m=read(fit/'metrics.json');m.update(camera_stage=d,profile_camera_only=True);write(fit/'metrics.json',m);write(a.out/'results.json',[dict(case_id=c,status='fit')])
