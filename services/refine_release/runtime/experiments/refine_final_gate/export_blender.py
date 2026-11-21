"""Shared export plus full transform round-trip; axial flips cannot hide in endpoints."""
import sys,runpy,json
from pathlib import Path
import numpy as np
import bpy
sys.path.insert(0,str(Path(__file__).resolve().parent))
from policy import LIMITS
out=Path(sys.argv[sys.argv.index('--')+1]).resolve()
runpy.run_path(str(Path(__file__).resolve().parents[1]/'refine_manual_004/export_blender.py'),run_name='__main__')
arm=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE');by={b.name.split(':')[-1]:b for b in arm.pose.bones}
z=np.load(out/'expected-world.npz');actual=np.array([np.asarray(arm.matrix_world@by[str(n)].matrix) for n in z['names']]);expected=z['world'];error=float(abs(actual-expected).max())
# FBX uses float32 matrices (often with unit scale 100). Check angle and
# relative scale separately rather than a unit-dependent matrix entry epsilon.
def proper(m):
 u,_,v=np.linalg.svd(m);return u@v
angles=[];scales=[]
for a,b in zip(actual,expected):
 r=proper(a[:3,:3])@proper(b[:3,:3]).T
 angles.append(float(np.degrees(np.arccos(np.clip((np.trace(r)-1)/2,-1,1)))))
 scales.append(float(np.max(abs(np.linalg.svd(a[:3,:3],compute_uv=False)/np.linalg.svd(b[:3,:3],compute_uv=False)-1))))
position=float(abs(actual[:,:3,3]-expected[:,:3,3]).max())
metrics=json.loads((out/'metrics.json').read_text())
rig=json.loads(Path(metrics['rig_path']).read_text());base=np.array([b['matrix'] for b in rig['bones']]);names=[b['name'] for b in rig['bones']]
if names!=list(z['names']):raise ValueError('Mesh/rig binding mismatch')
mesh=np.load(metrics['mesh_path']);vertices=mesh['vertices'];weights=mesh['weights'];hom=np.c_[vertices,np.ones(len(vertices))]
# Compare the actual reimported skin deformation, not unit-dependent entries.
delta=(actual-expected)@np.linalg.inv(base)
movement=np.einsum('vn,nij,vj->vi',weights,delta,hom,optimize=True)[:,:3]
torso=np.linalg.norm(base[names.index('Spine2'),:3,3]-base[names.index('Hips'),:3,3])
max_skin=float(np.linalg.norm(movement,axis=1).max()/torso)
passed=bool(np.isfinite(actual).all() and np.isfinite(max_skin) and max(angles)<=LIMITS.export_rotation_deg and max(scales)<=LIMITS.export_relative_scale and position<=max(1e-6,LIMITS.export_position_torso*torso) and max_skin<=LIMITS.export_skin_torso)
report=json.loads((out/'export-check.json').read_text());report.update(max_full_matrix_roundtrip_error=error,max_rotation_error_deg=max(angles),max_relative_scale_error=max(scales),max_world_position_error=position,max_skin_displacement_torso=max_skin,full_matrix_roundtrip_passed=passed)
(out/'export-check.json').write_text(json.dumps(report,indent=2))
if not passed:raise ValueError('Full transform export mismatch')
