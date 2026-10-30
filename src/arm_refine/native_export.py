"""Bake only selected arm rest bones; preserve unrelated native rest matrices."""
import bpy
from mathutils import Matrix


def bake_arm_rest(arm,meshes,*,sides=("left","right")):
    if not sides or not set(sides)<={"left","right"}:raise ValueError("explicit arm sides required")
    prefixes=[s.title() for s in sides]
    if arm.animation_data:raise ValueError('bake animation before exporting arm part')
    for p in arm.pose.bones:
        n=p.name.split(':')[-1]
        moving=any(n in [s+t for t in ['Arm','ForeArm','Hand']] or n.startswith(s+'Hand') for s in prefixes)
        if not moving and max(abs(p.matrix_basis[r][c]-Matrix.Identity(4)[r][c]) for r in range(4) for c in range(4))>1e-6:
            raise ValueError('bake non-arm stages before arm-only rest export')
    baked=[]
    for mesh in meshes:
        if mesh.data.shape_keys:raise ValueError('shape-key arm rest export unsupported')
        bpy.ops.object.select_all(action='DESELECT');bpy.context.view_layer.objects.active=mesh;mesh.select_set(True)
        for mod in list(mesh.modifiers):
            if mod.type=='ARMATURE':
                if mod.object!=arm:raise ValueError('unexpected modifier armature')
                baked.append((mesh,mod.name));bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.ops.object.select_all(action='DESELECT');bpy.context.view_layer.objects.active=arm;arm.select_set(True)
    bpy.ops.object.mode_set(mode='POSE');bpy.ops.pose.select_all(action='DESELECT')
    for bone in arm.pose.bones:
        n=bone.name.split(':')[-1]
        bone.select=any(n in [s+t for t in ['Arm','ForeArm','Hand']] or n.startswith(s+'Hand') for s in prefixes)
    bpy.ops.pose.armature_apply(selected=True);bpy.ops.object.mode_set(mode='OBJECT')
    for mesh,name in baked:
        mod=mesh.modifiers.new(name=name,type='ARMATURE');mod.object=arm;mod.use_vertex_groups=True
