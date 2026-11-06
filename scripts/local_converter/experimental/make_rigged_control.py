"""Generate a private, independent skinned FBX control and sampled pose."""
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix
import numpy as np


def build(folder, local_patch):
    folder.mkdir(exist_ok=False)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sides, levels = 16, 18
    vertices = []
    faces = []
    for row in range(levels):
        height = row / (levels-1)
        radius = .20 + .03 * math.cos(2*math.pi*height)
        for side in range(sides):
            angle = 2*math.pi*side/sides
            vertices.append((radius*math.cos(angle),radius*math.sin(angle),height))
    for row in range(levels-1):
        for side in range(sides):
            a = row*sides+side
            b = row*sides+(side+1)%sides
            faces.extend(((a,b,b+sides),(a,b+sides,a+sides)))
    for row in (0,levels-1):
        center = len(vertices)
        vertices.append((0.,0.,row/(levels-1)))
        for side in range(sides):
            a,b = row*sides+side,row*sides+(side+1)%sides
            faces.append((center,b,a) if row==0 else (center,a,b))
    mesh = bpy.data.meshes.new("Control surface")
    mesh.from_pydata(vertices,[],faces)
    mesh.update()
    body = bpy.data.objects.new("Control body",mesh)
    bpy.context.collection.objects.link(body)
    rig = bpy.data.armatures.new("Control skeleton")
    arm = bpy.data.objects.new("Control rig",rig)
    bpy.context.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    root = rig.edit_bones.new("Proximal")
    root.head,root.tail = (0,0,0),(0,0,.5)
    tip = rig.edit_bones.new("Distal")
    tip.head,tip.tail = (0,0,.58),(0,0,1)
    bpy.ops.object.mode_set(mode="OBJECT")
    body.parent = arm
    modifier = body.modifiers.new("Skin","ARMATURE")
    modifier.object = arm
    lower = body.vertex_groups.new(name="Proximal")
    upper = body.vertex_groups.new(name="Distal")
    for index,point in enumerate(vertices):
        moves = point[2] >= .5
        if local_patch:
            moves = point[0] > .17 and .35 < point[2] < .65
        group = upper if moves else lower
        group.add([index],1.,"REPLACE")
    file = folder/"control.fbx"
    bpy.ops.export_scene.fbx(filepath=str(file),add_leaf_bones=False,bake_anim=False)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(file))
    body = next(o for o in bpy.context.scene.objects if o.type=="MESH")
    arm = next(o for o in bpy.context.scene.objects if o.type=="ARMATURE")

    def sample():
        bpy.context.view_layer.update()
        obj=body.evaluated_get(bpy.context.evaluated_depsgraph_get())
        return np.asarray([obj.matrix_world@v.co for v in obj.data.vertices])

    rest=sample()
    tip = arm.pose.bones["Distal"]
    shift = (.18,0,0) if local_patch else (0,0,.18)
    tip.matrix = (arm.matrix_world.inverted() @ Matrix.Translation(shift)
                  @ arm.matrix_world @ tip.matrix)
    posed=sample()
    body.data.calc_loop_triangles()
    triangles=np.array([tuple(t.vertices) for t in body.data.loop_triangles])
    motion = float(np.linalg.norm(posed-rest,axis=1).max())
    assert motion>.15, (motion, tip.bone.use_connect, tip.location[:],
                         list(body.vertex_groups.keys()),list(body.modifiers.keys()))
    np.savez_compressed(folder/"stages-control.npz",rest=rest,corrected=posed,
                        triangles=triangles,anchors=np.zeros(len(rest),bool))


if __name__=="__main__":
    build(Path(sys.argv[-1]), "--local-patch" in sys.argv)
