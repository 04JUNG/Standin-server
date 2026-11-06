"""Run with Blender --background --factory-startup --python SCRIPT -- ROOT.
Reads native baked geometry, never guesses visibility or certifies skin safety.
"""
import bpy, sys, json, hashlib
from pathlib import Path
from mathutils import Vector
root=Path(sys.argv[sys.argv.index('--')+1]); records=[]
for path in sorted((root/'정답_3d인형').glob('*.fbx')):
    try:
        bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.fbx(filepath=str(path))
        rigs=[o for o in bpy.context.scene.objects if o.type=='ARMATURE'];meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
        if len(rigs)!=1: raise ValueError('expected exactly one rig')
        arm=rigs[0]; bones=[]
        for b in arm.pose.bones:
            bones.append(dict(name=b.name,parent=b.parent.name if b.parent else None,
              head=list(arm.matrix_world@b.head),tail=list(arm.matrix_world@b.tail),matrix=[list(r) for r in arm.matrix_world@b.matrix]))
        records.append(dict(case_id=path.stem,status='imported',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bones=bones,
          vertices=sum(len(m.data.vertices) for m in meshes),polygons=sum(len(m.data.polygons) for m in meshes),
          cameras=sum(o.type=='CAMERA' for o in bpy.context.scene.objects),
          nonidentity_pose_bones=sum((b.matrix_basis.to_translation().length>1e-5 or b.matrix_basis.to_quaternion().angle>1e-4) for b in arm.pose.bones),
          mesh_checked=False,visibility='unreviewed'))
    except Exception as e: records.append(dict(case_id=path.stem,status='error',reason=type(e).__name__))
(root/'native_audit.json').write_text(json.dumps(records,indent=2))
print('V5_AUDIT',json.dumps(dict(total=len(records),imported=sum(r['status']=='imported' for r in records))))
