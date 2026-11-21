"""Local rotation attenuation before the unchanged strict crossing audit.
No new collision allowance; preserve all unmodified local rotations exactly.
"""
import sys,json,ast,shutil
from pathlib import Path
import numpy as np
from mathutils.bvhtree import BVHTree
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from skin_policy import SurfaceGate,region
# Share precisely the audit's tested patch definition without executing its exporter.
tree=ast.parse((HERE.parent/'refine_shoulder_line/audit_blender.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='crossings')
exec(compile(ast.Module(body=[node],type_ignores=[]),'<shared-crossings>','exec'))
root,baseline,assets=map(Path,sys.argv[sys.argv.index('--')+1:])
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2))
for row in read(root/'results.json'):
 c=row['case_id'];dest=root/c/'fit';prior=baseline/c/'fit';fx=read(root/c/'fixture.json');rig=read(assets/fx['files']['rig']['path']);bones=rig['bones'];names=[b['name'] for b in bones];ix={n:i for i,n in enumerate(names)};base=np.array([b['matrix'] for b in bones]);parents=[ix[b['parent']] if b['parent'] else -1 for b in bones];local=np.array([np.linalg.solve(base[p],base[i]) if p>=0 else base[i] for i,p in enumerate(parents)])
 m=np.load(assets/fx['files']['mesh']['path']);v,f,weights=m['vertices'],m['faces'],m['weights'];inv=np.linalg.inv(base);vh=np.c_[v,np.ones(len(v))]
 def skin(w):return np.einsum('vn,nvi->vi',weights,np.einsum('nij,vj->nvi',w@inv,vh)[:,:,:3])
 def world(e):
  w=[]
  for i,n in enumerate(names):
   a=local[i].copy()
   if n in e:
    vec=np.radians(e[n]);theta=np.linalg.norm(vec)
    if theta>1e-12:
     x,y,z=vec/theta;k=np.array([[0,-z,y],[z,0,-x],[-y,x,0]]);a[:3,:3]=a[:3,:3]@(np.eye(3)+np.sin(theta)*k+(1-np.cos(theta))*(k@k))
   w.append(w[parents[i]]@a if parents[i]>=0 else a)
  return np.array(w)
 pr=read(dest/'adjustment.json');old=read(prior/'adjustment.json')['edits'];proposal=pr['edits'];edits=dict(proposal);w=np.load(dest/'expected-world.npz')['world'];bv=skin(np.load(prior/'expected-world.npz')['world']);original_cross=crossings(bv,f,weights,names);gate=SurfaceGate(v,f,weights,names,np.linalg.norm(base[ix['Spine2'],:3,3]-base[ix['Hips'],:3,3]));original_skin=gate.measure(bv)
 def evaluate(e):
  nw=world(e);nv=skin(nw);cr=crossings(nv,f,weights,names);inc={k:cr[k]-original_cross[k] for k in cr if cr[k]>original_cross[k]};damage=gate.failures(original_skin,gate.measure(nv));cost=sum(float(np.sum((np.array(e.get(n,[0,0,0]))-proposal[n])**2)) for n in proposal)
  return nw,cr,inc,damage,cost
 nw,cr,inc,damage,cost=evaluate(edits);initial=cr;steps=[]
 for iteration in range(12):
  if not inc:break
  options=[]
  # Single-bone first: avoid throwing away the whole arm for a shoulder issue.
  for n in sorted(proposal):
   if np.allclose(edits[n],old.get(n,[0,0,0])):continue
   for fraction in (.9,.75,.5,.25,0):
    e=dict(edits);e[n]=(np.array(old.get(n,[0,0,0]))+fraction*(np.array(edits[n])-old.get(n,[0,0,0]))).tolist();ww,cc,ii,dd,cost=evaluate(e)
    if not dd and sum(ii.values())<sum(inc.values()):options.append((sum(ii.values()),cost,n,fraction,e,ww,cc,ii))
  if not options:break
  _,cost,n,fraction,edits,nw,cr,inc=min(options,key=lambda q:(q[0],q[1]));steps.append(dict(bone=n,step_retained_fraction=fraction,remaining_increases=inc))
 if not inc and not damage:
  # Keep before/after evidence, including when no repair is necessary.
  saved=root/c/'before-local-collision-repair';saved.mkdir(exist_ok=True)
  for name in ['adjustment.json','expected-world.npz','camera.json']:shutil.copy2(dest/name,saved/name)
  pr['edits']=edits;write(dest/'adjustment.json',pr);np.savez_compressed(dest/'expected-world.npz',names=names,world=nw)
  cam=read(dest/'camera.json');cam['anchor_camera_xyz']=((nw[ix['Hips'],:3,3]-pr['origin'])@pr['camera']).tolist();write(dest/'camera.json',cam)
 selected=not inc and not damage
 met=read(dest/'metrics.json');met['local_collision_repair']=dict(before=original_cross,proposal=initial,selected=cr if selected else initial,steps=steps,accepted=selected,remaining_increases=inc,changed_bones=[n for n in proposal if not np.allclose(edits[n],proposal[n])],scope='tested weighted interior patches only');write(dest/'metrics.json',met)
 print(c,'LOCAL COLLISION',selected,'steps',len(steps),'remaining',inc,flush=True)
