from pathlib import Path
import sys,json
import bpy
from mathutils import Vector
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[1];sys.path.insert(0,str(WORK/'.worktrees/standin-multi-person-fbx'))
from converter import retarget as rt
rt.reset_scene();arm,meshes=rt.import_character(str(WORK/'pose-head-contact-repair-20261015/mesh-review/assets/standin-master-v2.fbx'))
left=arm.matrix_world@arm.data.bones['mixamorig:LeftShoulder'].tail_local;right=arm.matrix_world@arm.data.bones['mixamorig:RightShoulder'].tail_local;r=left-right;r.z=0;r.normalize();up=Vector((0,0,1));forward=r.cross(up).normalized();head=arm.matrix_world@arm.data.bones['mixamorig:Head'].matrix_local
localforward=head.to_3x3().inverted()@forward;localup=head.to_3x3().inverted()@up
record=dict(master_body_right=list(r),master_body_forward=list(forward),head_local_forward=list(localforward.normalized()),head_local_up=list(localup.normalized()),head_matrix=[list(row) for row in head],bone_count=len(arm.data.bones));(ROOT/'master-head-axes.json').write_text(json.dumps(record,indent=2));print(record)
