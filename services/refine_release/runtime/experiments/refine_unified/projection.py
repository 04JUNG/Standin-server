"""A single terminal projection contract; all geometry stages run orthographically."""
from common import *
from lens import select_lens

def choose_projection(fit,fx,obs,assets,conditions,shadow=False):
 pr=read(fit/'adjustment.json');names,ix,_,_,base=prepare(read(assets/fx['files']['rig']['path']));w=np.load(fit/'expected-world.npz')['world'];_,_,torso,_=frame(read(assets/fx['files']['rig']['path'])['bones']);goal,*_=targets(obs,fx['image_size'])
 with np.load(assets/fx['files']['mesh']['path']) as mesh:cv=(skin(mesh['vertices'],mesh['weights'],w,base)-pr['origin'])@pr['camera']
 cj=(w[:,:3,3]-pr['origin'])@pr['camera'];ids=[ix[n] for n in goal];before=sha(fit/'expected-world.npz');lens=select_lens(cj,cv,cj[ix['Hips']],ids,list(goal.values()),torso,pr['pixel_scale'],pr['offset'])
 # Current partial-foot visibility guard is orthographic. Never invalidate it
 # by silently accepting an unchecked perspective silhouette afterwards.
 sensitive=bool(conditions.get('relative_order_enabled') or conditions.get('local_occlusions') or any(v.get('foot_visibility') in ['partly_occluded','fully_occluded'] for v in conditions.get('foot_evidence',{}).values()))
 if sensitive and lens['selected']=='perspective':lens['occlusion_guard_would_override']=True
 if sensitive and lens['selected']=='perspective' and not shadow:lens.update(selected='orthographic',reason='preserve_audited_occlusion_projection',inverse_distance=0,pixel_scale=pr['pixel_scale'],offset=pr['offset'])
 cam=camera(pr,w,ix);cam.update(projection=lens['selected'],inverse_distance=lens['inverse_distance'],pixel_scale=lens['pixel_scale'],offset=lens['offset']);write(fit/'camera.json',cam)
 meta=read(fit/'metrics.json');meta['lens']=lens;meta['projection_contract']=dict(geometry_stages='orthographic',lens_stage='terminal_frozen_pose',world_sha256=before,occlusion_sensitive=sensitive);write(fit/'metrics.json',meta)
 assert sha(fit/'expected-world.npz')==before
 return lens

def render(fit,fx,assets,override=None):
 pr=read(fit/'adjustment.json');_,_,_,_,base=prepare(read(assets/fx['files']['rig']['path']));w=np.load(fit/'expected-world.npz')['world'];cam=override or read(fit/'camera.json')
 with np.load(assets/fx['files']['mesh']['path']) as m:
  cv=(skin(m['vertices'],m['weights'],w,base)-cam['origin'])@cam['camera'];xy=project(cv,np.array(cam['anchor_camera_xyz']),cam['inverse_distance'],cam['pixel_scale'],cam['offset']);return render_projected(cv,m['faces'],xy,(535,687))
