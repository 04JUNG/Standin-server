"""Create explicit overhead and seated solver probes from a library skeleton."""
import math,sys,tempfile
from pathlib import Path
import bpy
from mathutils import Matrix,Quaternion,Vector
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from converter import retarget as rt
root=Path(sys.argv[sys.argv.index('--')+1]);out=root/'probes';out.mkdir(exist_ok=True)
tempfile.tempdir=str(out)
source='C:/workspaces/Standin-server/data/bvh/Arm Stretching_00054.bvh'
for label in ('overhead','seated'):
    rt.reset_scene();arm=rt.import_bvh(source);arm.animation_data_clear()
    for pb in arm.pose.bones:pb.matrix_basis=Matrix.Identity(4);pb.rotation_mode='QUATERNION'
    def rotate(name,axis,angle):
        b=arm.data.bones['mixamorig:'+name];q=b.matrix_local.to_quaternion()
        arm.pose.bones[b.name].rotation_quaternion=q.inverted()@Quaternion(Vector(axis),math.radians(angle))@q
    for side,sign in [('Left',1),('Right',-1)]:
        rotate(side+'Arm',(0,1,0),sign*(-80 if label=='overhead' else 65))
        if label=='seated':rotate(side+'UpLeg',(-1,0,0),90);rotate(side+'Leg',(1,0,0),90)
    bpy.context.view_layer.update()
    # BVH is Y-up; convert the entire object before serializing its channels.
    arm.rotation_euler=(-math.pi/2,0,0)
    bpy.context.view_layer.objects.active=arm;arm.select_set(True)
    bpy.ops.object.transform_apply(location=False,rotation=True,scale=True)
    for pb in arm.pose.bones:pb.keyframe_insert(data_path='rotation_quaternion',frame=1)
    bpy.context.scene.frame_set(1)
    path=out/(label+'.bvh')
    bpy.ops.export_anim.bvh(filepath=str(path),frame_start=1,frame_end=1,rotate_mode='XYZ',root_transform_only=True)
    rt.reset_scene();arm=rt.import_bvh(str(path))
    vertical=[float((arm.pose.bones['mixamorig:'+s+'Arm'].tail-arm.pose.bones['mixamorig:'+s+'Arm'].head).normalized().z) for s in ('Left','Right')]
    if label=='overhead' and min(vertical)<.9:raise ValueError(('overhead fixture is not overhead',vertical))
    print('PROBE',label,vertical,flush=True)
