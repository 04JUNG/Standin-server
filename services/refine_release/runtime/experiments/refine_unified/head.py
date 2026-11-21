"""Independent categorical head direction; protected limbs/camera stay unchanged."""
from common import *
from core import orientation_stage

def apply_head(source,dest,fx,obs,body,assets,shadow=False):
 copy_checkpoint(source,dest);rig=read(assets/fx['files']['rig']['path']);names,ix,parents,local,base=prepare(rig);old=np.load(source/'fit/expected-world.npz')['world'];pr=read(source/'fit/adjustment.json');goal,*_=targets(obs,fx['image_size']);w,p,diagnostic=orientation_stage(local,parents,ix,old,pr,goal,{'orientation':{'person_count':body.get('person_count'),'head':body.get('head') or {},'feet':[]}})
 with np.load(assets/fx['files']['mesh']['path']) as m:
  before=deformation(skin(m['vertices'],m['weights'],old,base),m['vertices'],m['faces']);after=deformation(skin(m['vertices'],m['weights'],w,base),m['vertices'],m['faces'])
 safe=after['extreme_edges']<=before['extreme_edges'] and after['ratio_max']<=max(2,before['ratio_max'])*1.05
 met=read(dest/'fit/metrics.json');met['head_stage']=dict(diagnostic,accepted=bool(diagnostic.get('applied') and (safe or shadow)),would_reject=bool(diagnostic.get('applied') and not safe),before_deformation=before,after_deformation=after)
 if diagnostic.get('applied') and (safe or shadow):
  protected=[i for n,i in ix.items() if n not in ['Neck','Head'] and not n.startswith('Head')]
  if np.max(abs(w[protected]-old[protected]))>1e-6:raise ValueError('Head orientation moved protected joints')
  write(dest/'fit/adjustment.json',p);np.savez_compressed(dest/'fit/expected-world.npz',names=names,world=w);met['geometry_accepted']=True
 write(dest/'fit/metrics.json',met);return met['head_stage']
