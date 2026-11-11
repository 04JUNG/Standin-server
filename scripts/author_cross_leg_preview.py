"""Explicit user-directed native preview; never a production gate bypass.

Blender --background --python this_file -- SOURCE.fbx CAMERA.json NEW_OUTPUT
The camera JSON is an actual camera matrix (local +X = screen right).
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import bpy
from mathutils import Matrix
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from converter import retarget as rt
from src.experimental.part_composition import shortest_arc
from src.refine_v4.pipeline import snapshot
from src.refine_v4.leg_adapter import measure_legs
from src.refine_v4.native_export import bake_selected_rest
from src.refine_v4.native_geometry import leg_name
from src.refine_v4.skin_collision import inspect_self
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('source',type=Path);p.add_argument('camera',type=Path);p.add_argument('output',type=Path)
p.add_argument('--left-knee-deg',type=float,default=60.)
p.add_argument('--right-ankle-left-m',type=float,default=.04)
p.add_argument('--right-toe-left-m',type=float,default=.025)
p.add_argument('--right-knee-left-m',type=float,default=.035)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);a.output.mkdir(parents=True,exist_ok=False)
assert 30<=a.left_knee_deg<=90 and 0<=a.right_ankle_left_m<=.08 and 0<=a.right_toe_left_m<=.05 and 0<=a.right_knee_left_m<=.05
camera=json.loads(a.camera.read_text());screen_left=-np.array(camera['matrix_world'])[:3,0];screen_left/=np.linalg.norm(screen_left)
rt.reset_scene();arm,meshes=rt.import_character(str(a.source));b={p.name.split(':')[-1]:p for p in arm.pose.bones}
before=snapshot(arm);old=measure_legs(arm)
def point(n):return np.array(arm.matrix_world@b[n].head)
def rotate(n,rotation):
    pb=b[n];world=arm.matrix_world@pb.matrix;desired=(Matrix(rotation.tolist())@world.to_3x3()).to_4x4()
    desired.translation=world.translation;pb.matrix=arm.matrix_world.inverted()@desired;bpy.context.view_layer.update()
upper=point('LeftLeg')-point('LeftUpLeg');upper/=np.linalg.norm(upper)
shin=point('LeftFoot')-point('LeftLeg');shin/=np.linalg.norm(shin)
axis=np.cross(shin,upper);axis/=np.linalg.norm(axis)
delta=np.radians(old['knee_flex_deg']['left']-a.left_knee_deg)
foot_orientation=(arm.matrix_world@b['LeftFoot'].matrix).to_3x3().copy()
rotate('LeftLeg',np.array(Matrix.Rotation(delta,3,axis.tolist())))
world=arm.matrix_world@b['LeftFoot'].matrix;desired=foot_orientation.to_4x4();desired.translation=world.translation
b['LeftFoot'].matrix=arm.matrix_world.inverted()@desired;bpy.context.view_layer.update()
toe_before=point('RightToeBase');foot_before=point('RightFoot')
thigh=point('RightLeg')-point('RightUpLeg')
rotate('RightUpLeg',shortest_arc(thigh,thigh+screen_left*a.right_knee_left_m))
shin=point('RightFoot')-point('RightLeg');target=shin+screen_left*a.right_ankle_left_m
rotate('RightLeg',shortest_arc(shin,target))
toe=point('RightToeBase')-point('RightFoot')
rotate('RightFoot',shortest_arc(toe,toe+screen_left*a.right_toe_left_m))
after=snapshot(arm);measured=measure_legs(arm)
protected=max(float(np.max(abs(after[n]-v))) for n,v in before.items() if not leg_name(n))
toe_shift=float((point('RightToeBase')-toe_before)@screen_left)
print('preview measurements',measured,'protected',protected,'toe shift',toe_shift,flush=True)
assert abs(measured['knee_flex_deg']['left']-a.left_knee_deg)<.01 and measured['top_side']=='right' and protected<2e-5 and toe_shift>.025
bake_selected_rest(arm,meshes,[n for n in before if leg_name(n)])
rt.export_fbx(str(a.output/'after.fbx'));rt.reset_scene();arm,meshes=rt.import_character(str(a.output/'after.fbx'))
saved=snapshot(arm);error=max(float(np.max(abs(saved[n]-v))) for n,v in after.items());assert error<2e-5
report={'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),'after_sha256':hashlib.sha256((a.output/'after.fbx').read_bytes()).hexdigest(),
 'parameters':{'left_knee_target_deg':a.left_knee_deg,'right_ankle_screen_left_m':a.right_ankle_left_m,'right_toe_extra_screen_left_m':a.right_toe_left_m,'right_knee_screen_left_m':a.right_knee_left_m},
 'left_knee_before_after_deg':[old['knee_flex_deg']['left'],measure_legs(arm)['knee_flex_deg']['left']],
 'right_toe_screen_left_shift_m':toe_shift,'screen_left_world':screen_left.tolist(),'top_side':measured['top_side'],
 'protected_upper_matrix_error':protected,'saved_matrix_error':error,'skin':inspect_self(arm),
 'preview_only':True,'automatic_apply_allowed':False,'score_status':'changed_user_reference_not_comparable_to_frozen_rough'}
(a.output/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False),flush=True)
