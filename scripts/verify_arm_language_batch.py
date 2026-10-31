"""Blender: independent saved-FBX admission including inferred orientation."""
import sys,json,hashlib,shutil,math
from pathlib import Path
import numpy as np
import bpy
from mathutils import Vector,Matrix
repo,root,run=map(lambda p:Path(p).resolve(),sys.argv[sys.argv.index('--')+1:]);sys.path.insert(0,str(repo));sys.path.insert(0,str(root/'pose-complement-1000-20261015'))
from converter import retarget as rt
from self_collision import inspect_self
from src.arm_refine.blender_adapter import snapshot
from src.arm_refine.contact_parts import ContactPart,admission_reasons,contact_axis
from src.arm_refine.contact_adapter import bones,hand_samples,target_frame,wrist_guard,unit,elbow_rule_margin
from src.arm_refine.relational_adapter import measure_arm,reasons_for,new_skin
from src.arm_refine.language_adapter import contact_directions

def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
profile=load(run/'rig-profile.json');rows=[]
for req in load(run/'requests.json'):
 folder=run/req['id'];receipt=load(folder/'contact-validation.json');plan=load(folder/'plan.json');reasons=[];members=[]
 assert sha(folder/'before.fbx')==req['source_sha256']
 rt.reset_scene();a,meshes=rt.import_character(str(folder/'before.fbx'));before=snapshot(a);baseline=inspect_self(a);b=bones(a)
 surface_before=None;surface_mask=None
 if req.get('canonical_surface_rebuild'):
  assert len(meshes)==1
  obj=meshes[0];names={g.index:g.name.split(':')[-1] for g in obj.vertex_groups};titles=[s.title() for s in req['selection']]
  surface_mask=np.array([sum(g.weight for g in v.groups if any(names[g.group] in [s+'Arm',s+'ForeArm'] or names[g.group].startswith(s+'Hand') for s in titles))==0 for v in obj.data.vertices])
  surface_before=np.array([obj.matrix_world@v.co for v in obj.data.vertices])
 lengths={};source_wrists={}
 for side in req['selection']:
  s,e,w=[a.matrix_world@b[side.title()+t].head for t in ['Arm','ForeArm','Hand']];lengths[side]=((e-s).length,(w-e).length)
  source_wrists[side]=list(w)
 rt.reset_scene();a,meshes=rt.import_character(str(folder/'after.fbx'));after=snapshot(a);skin=inspect_self(a);b=bones(a)
 def mutable(name,member):
  s=member['side'].title();n=name.split(':')[-1]
  if member.get('part',{}).get('id')=='fist_preserve_arm':return n.startswith(s+'Hand') and n!=s+'Hand'
  return n in [s+t for t in ['Arm','ForeArm','Hand']] or n.startswith(s+'Hand')
 allowed={n for n in before if any(m['active'] and mutable(n,m) for m in plan['members'])}
 protected=max(float(np.max(abs(after[n]-v))) for n,v in before.items() if n not in allowed);new=new_skin(skin,baseline)
 surface_delta=None
 if surface_before is not None:
  assert len(meshes)==1 and len(meshes[0].data.vertices)==len(surface_before)
  surface_after=np.array([meshes[0].matrix_world@v.co for v in meshes[0].data.vertices])
  surface_delta=float(np.max(abs(surface_after[surface_mask]-surface_before[surface_mask])))
  if surface_delta>=2e-5:reasons.append('protected_surface_changed')
 if receipt['accepted']:
  for member in receipt['members']:
   side=member['side'];title=side.title();semantic=next(m for m in plan['members'] if m['side']==side)
   if receipt['engine']=='relational':
    metrics=measure_arm(a,meshes,title,member['part'],profile,lengths[side],member['wrist_goal_world'],member['contact'],source_wrist=source_wrists[side]);mr=reasons_for(metrics);direction=[]
   else:
    kwargs={k:v for k,v in member['part'].items() if k in ContactPart.__dataclass_fields__};part=ContactPart(**kwargs)
    target,anchor,normal,_,_,axes=target_frame(a,meshes,title,part);hf,_,patch=hand_samples(a,meshes,title,part)
    s,e,w=[a.matrix_world@b[title+t].head for t in ['Arm','ForeArm','Hand']];pts=patch@np.array(hf).T+np.array(w);ds=[target.find_nearest(Vector(p))[3] for p in pts];swing,twist=wrist_guard(a,title,profile)
    metrics={'patch_median_m':float(np.median(ds)),'patch_p90_m':float(np.quantile(ds,.9)),'patch_samples':len(ds),
      'normal_error_deg':math.degrees(hf.col[contact_axis(part)].angle(-normal)),
      'hand_frame_error_deg':math.degrees((hf.transposed()@Matrix(member['hand_frame_world'])).to_quaternion().angle),
      'wrist_axis_deg':math.degrees(unit(w-e).angle(hf.col[0])),'wrist_swing_deg':swing,'wrist_twist_deg':twist,
      'elbow_flexion_deg':math.degrees(unit(e-s).angle(unit(w-e))),
      'elbow_rule_margin_m':elbow_rule_margin(part,e,w,s,lengths[side],axes),
      'bone_length_error_m':max(abs((e-s).length-lengths[side][0]),abs((w-e).length-lengths[side][1])),
      'wrist_goal_error_m':(w-Vector(member['wrist_goal_world'])).length,'protected_matrix_delta':protected,'new_skin':new}
    mr=admission_reasons(metrics,part);direction=contact_directions(a,meshes,semantic,part)
    if not all(c['passed'] for c in direction):mr.append('inferred_direction_failed')
   members.append({'side':side,'metrics':metrics,'direction_checks':direction,'reasons':mr});reasons += [side+': '+r for r in mr]
  if new:reasons.append('new_skin')
  if protected>=2e-5:reasons.append('protected_matrix_delta')
 else:
  assert sha(folder/'after.fbx')==sha(folder/'before.fbx');reasons=receipt['reasons']
 applied=receipt['accepted'] and not reasons
 row={'id':req['id'],'authored_accepted':receipt['accepted'],'final_applied':applied,'status':'applied' if applied else 'held' if receipt['status']=='held' else 'not_applied',
  'reasons':reasons,'members':members,'protected_matrix_delta':protected,'new_skin':new,'actual_skin':skin}
 if surface_delta is not None:row['nonselected_surface_max_delta_m']=surface_delta
 if receipt['accepted'] and not applied:
  shutil.copy2(folder/'after.fbx',folder/'export-rejected.fbx');shutil.copy2(folder/'after.blend',folder/'export-rejected.blend')
  shutil.copy2(folder/'before.fbx',folder/'after.fbx');rt.reset_scene();a,_=rt.import_character(str(folder/'before.fbx'));bpy.ops.wm.save_as_mainfile(filepath=str(folder/'after.blend'))
  row.update(protected_matrix_delta=0.,new_skin=[],actual_skin=baseline,export_rollback=True)
 if not applied:row['byte_exact_fallback']=sha(folder/'before.fbx')==sha(folder/'after.fbx')
 row['final_fbx_sha256']=sha(folder/'after.fbx');save(folder/'export-verification.json',row);rows.append(row);save(run/'verified-summary.json',rows)
 print(json.dumps({k:v for k,v in row.items() if k not in ['actual_skin','members','new_skin']},ensure_ascii=False),flush=True)
