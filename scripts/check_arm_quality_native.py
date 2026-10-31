"""Saved-surface and direction regressions for current-pose quality corrections."""
import sys,json,math,copy
from pathlib import Path
import numpy as np
import bpy
from mathutils import Matrix
repo,root,run=map(lambda p:Path(p).resolve(),sys.argv[sys.argv.index('--')+1:])
sys.path.insert(0,str(repo));sys.path.insert(0,str(root/'pose-complement-1000-20261015'))
from converter import retarget as rt
from self_collision import inspect_self
from src.arm_refine.canonical_surface import capture_canonical,rebuild_selected_surface
from src.arm_refine.contact_adapter import bones,hand_samples
from src.arm_refine.contact_parts import ContactPart
from src.arm_refine.language_adapter import apply_language_plan,contact_directions
from src.arm_refine.relational_adapter import measure_arm,reasons_for
from src.arm_refine.blender_adapter import snapshot
def load(p):return json.loads(p.read_text())
def stage_input(t):
 working=run/t/'prepared-fist.blend' if (run/t/'prepared-fist.blend').exists() else run/t/'before.fbx'
 if working.suffix=='.blend':
  bpy.ops.wm.open_mainfile(filepath=str(working));a=next(o for o in bpy.data.objects if o.type=='ARMATURE');m=[o for o in bpy.data.objects if o.type=='MESH' and o.vertex_groups]
 else:rt.reset_scene();a,m=rt.import_character(str(working))
 bpy.context.view_layer.update()
 return a,m,working
profile=load(run/'rig-profile.json');checks=[]
rt.reset_scene();a,m=rt.import_character(str(run/'T002/before.fbx'));b=bones(a);hand=a.matrix_world@b['RightHand'].matrix
finger_relative={n:np.array(hand.inverted()@a.matrix_world@p.matrix) for n,p in b.items() if n.startswith('RightHand') and n!='RightHand'}
rt.reset_scene();a,m=rt.import_character(str(run/'T002/after.fbx'));b=bones(a);hand=a.matrix_world@b['RightHand'].matrix
reuse_delta=max(float(np.max(abs(np.array(hand.inverted()@a.matrix_world@b[n].matrix)-matrix))) for n,matrix in finger_relative.items())
assert reuse_delta<2e-5
checks.append({'test':'baked_phone_grip_not_curled_again','hand_relative_matrix_delta':reuse_delta,'passed':True})
rt.reset_scene();a,m=rt.import_character(str(root/load(run/'canonical-source.json')['path']));canonical=capture_canonical(a,m)
before=np.array([v.co for v in m[0].data.vertices]);report=rebuild_selected_surface(a,m,canonical,['left'])
delta=float(np.max(abs(np.array([v.co for v in m[0].data.vertices])-before)))
assert delta<2e-6 and report['nonselected_surface_max_delta_m']==0
checks.append({'test':'canonical_identity_and_nonselected_surface_preservation','max_delta_m':delta,'passed':True})
bad=copy.deepcopy(canonical);bad['faces'][0]=tuple(reversed(bad['faces'][0]));before=np.array([v.co for v in m[0].data.vertices])
try:rebuild_selected_surface(a,m,bad,['left'])
except ValueError:pass
else:raise AssertionError('topology mismatch must fail')
assert np.array_equal(before,np.array([v.co for v in m[0].data.vertices]))
checks.append({'test':'topology_mismatch_rejected_before_surface_mutation','passed':True})
rt.reset_scene();a,m=rt.import_character(str(run/'T002/after.fbx'));member=load(run/'T002/contact-validation.json')['members'][0]
b=bones(a);hand=b['RightHand'];h,_,_=hand_samples(a,m,'Right',type('Shape',(),{'patch':'ulnar_edge'})());pos=a.matrix_world@hand.head
hand.matrix=a.matrix_world.inverted()@Matrix.Translation(pos)@Matrix.Rotation(math.pi,4,h.col[0])@Matrix.Translation(-pos)@a.matrix_world@hand.matrix;bpy.context.view_layer.update()
metrics=measure_arm(a,m,'Right',member['part'],profile,member['lengths'],member['wrist_goal_world'],None)
assert 'palm_up_dot' in reasons_for(metrics)
checks.append({'test':'inverted_phone_support_palm_rejected','passed':True})
rt.reset_scene();a,m=rt.import_character(str(run/'T027/after.fbx'));native=load(run/'T027/contact-validation.json')['members'][0]
part=ContactPart(**{k:v for k,v in native['part'].items() if k in ContactPart.__dataclass_fields__})
semantic=next(v for v in load(run/'T027/plan.json')['members'] if v['active'])
assert all(c['passed'] for c in contact_directions(a,m,semantic,part))
assert load(run/'T027/export-verification.json')['members'][0]['metrics']['wrist_axis_deg']<25
checks.append({'test':'saved_forehead_contact_dorsal_forward_and_straight_wrist','passed':True})
for t in ['T001','T025']:
 a,m,working=stage_input(t);before=snapshot(a);changed=[False]
 def checker(arm):
  if max(float(np.max(abs(snapshot(arm)[n]-v))) for n,v in before.items())>1e-5:
   changed[0]=True
   raise OSError('injected failure after native arm mutation')
  return inspect_self(arm)
 try:apply_language_plan(a,m,load(run/t/'plan.json'),checker,profile)
 except OSError:pass
 else:raise AssertionError('injected fault must surface')
 delta=max(float(np.max(abs(snapshot(a)[n]-v))) for n,v in before.items())
 assert changed[0] and delta==0,{'case':t,'observed_mutation':changed[0],'rollback_delta':delta}
 checks.append({'test':'two_arm_stage_failure_restores_stage_input','case':t,'stage_input':working.name,'matrix_delta':delta,'passed':True})
rt.reset_scene();a,m=rt.import_character(str(run/'T001/before.fbx'));original_skin=inspect_self(a)
a,m,_=stage_input('T001');before=snapshot(a);calls=[0]
old=max((c['max_depth_mm'] for c in original_skin['collisions'] if set(c['regions'])=={'Head','RightHand'}),default=0.)
fake={'regions':['Head','RightHand'],'max_depth_mm':old+5.,'penetrating_vertices':1,'surface_crossing_triangle_pairs':1,'witness_world':[0.,0.,0.]}
def staged_checker(arm):
 calls[0]+=1;result=copy.deepcopy(original_skin)
 if calls[0] in [1,3]:result['collisions'].append(fake)
 return result
receipt=apply_language_plan(a,m,load(run/'T001/plan.json'),staged_checker,profile,baseline_skin=original_skin)
assert calls[0]==3 and not receipt['accepted'] and receipt['members'][0]['deferred_pending_arm_skin']
assert max(float(np.max(abs(snapshot(a)[n]-v))) for n,v in before.items())==0
checks.append({'test':'pending_arm_warning_cannot_escape_final_atomic_skin_check','passed':True})
assert all(v['final_applied'] and v['new_skin']==[] for v in load(run/'verified-summary.json'))
checks.append({'test':'all_six_saved_fbx_pass_without_new_skin_warning','passed':True})
(run/'native-contract-checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks),flush=True)
