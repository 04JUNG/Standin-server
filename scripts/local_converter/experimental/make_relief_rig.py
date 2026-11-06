"""Generate two independent FBXs differing at one injected weight assignment.

A clean animated three-bone sheet provides the expected evaluated surface.
Its geometry is visible only to the verification step, never to the solver.
"""
import math
import sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.local_converter.make_csp_controls import export_existing_rig
from converter import retarget
from converter.local import surface
from scripts.local_converter.experimental.read_registered_weights import mesh_weights


def write_asset(path, anomaly, split):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    nx,ny=29,13
    vertices=[]
    faces=[]
    for row in range(ny):
        for col in range(nx):
            x,y=(col/(nx-1)-.5)*1.4,(row/(ny-1)-.5)*.5
            z=.018*math.sin(13*x)+.012*math.cos(19*y)
            vertices.append((x,y,z))
    for row in range(ny-1):
        for col in range(nx-1):
            a=row*nx+col
            faces.extend([(a,a+1,a+nx),(a+nx,a+1,a+nx+1)])
    if split:
        copies={}
        column=nx//2
        for row in range(ny):
            old=row*nx+column
            copies[old]=len(vertices)
            vertices.append(vertices[old])
        faces=[tuple(copies.get(i,i) for i in face) if max(vertices[i][0] for i in face)>0
               else face for face in faces]
    mesh=bpy.data.meshes.new("Relief")
    mesh.from_pydata(vertices,[],faces)
    mesh.update()
    for name,color in (("Fabric",(.34,.54,.70,1.)),("Trim",(.64,.32,.18,1.))):
        material=bpy.data.materials.new(name)
        material.diffuse_color=color
        mesh.materials.append(material)
    for polygon in mesh.polygons:
        polygon.material_index=int(polygon.index%8==0)
        polygon.use_smooth=polygon.index%9!=0
    body=bpy.data.objects.new("Relief",mesh)
    bpy.context.collection.objects.link(body)
    rig=bpy.data.armatures.new("Sheet rig")
    arm=bpy.data.objects.new("Sheet rig",rig)
    bpy.context.collection.objects.link(arm)
    bpy.context.view_layer.objects.active=arm
    bpy.ops.object.mode_set(mode="EDIT")
    for name,x in (("Left",-.45),("Middle",0.),("Right",.45)):
        bone=rig.edit_bones.new(name)
        bone.head,bone.tail=(x,-.1,0),(x,.1,0)
    bpy.ops.object.mode_set(mode="OBJECT")
    body.parent=arm
    mod=body.modifiers.new("Armature","ARMATURE")
    mod.object=arm
    groups=[body.vertex_groups.new(name=n) for n in ("Left","Middle","Right")]
    for i,(x,y,z) in enumerate(vertices):
        if anomaly and i==6*nx+5:
            groups[0].add([i],.35,"REPLACE")
            groups[2].add([i],.65,"REPLACE")
        else:
            # Smooth transitions separate the rigid interiors of the three regions.
            weights=np.array([max(0.,1-abs(x-c)/.35) for c in (-.5,0.,.5)])
            weights/=weights.sum()
            for group,weight in zip(groups,weights):
                if weight>0:
                    group.add([i],weight,"REPLACE")
    bpy.ops.export_scene.fbx(filepath=str(path),add_leaf_bones=False,bake_anim=False)


def sample(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(path))
    body=next(o for o in bpy.context.scene.objects if o.type=="MESH")
    arm=next(o for o in bpy.context.scene.objects if o.type=="ARMATURE")
    def evaluated():
        bpy.context.view_layer.update()
        obj=body.evaluated_get(bpy.context.evaluated_depsgraph_get())
        return np.array([obj.matrix_world@v.co for v in obj.data.vertices],dtype=float)
    rest=evaluated()
    for name,angle,shift in (("Left",.18,(.01,.02,.03)),("Middle",-.22,(0,0,.01)),
                             ("Right",.72,(0,.05,.22))):
        bone=arm.pose.bones[name]
        delta=Matrix.Translation(shift)@Matrix.Rotation(angle,4,"Y")
        bone.matrix=arm.matrix_world.inverted()@delta@arm.matrix_world@bone.matrix
    return arm,body,rest,evaluated()


def make(out,split):
    out.mkdir(exist_ok=False)
    write_asset(out/"clean-source.fbx",False,split)
    _,clean_body,clean_rest,oracle=sample(out/"clean-source.fbx")
    clean_body.data.calc_loop_triangles()
    clean_faces=np.array([tuple(t.vertices) for t in clean_body.data.loop_triangles])
    clean_ids,clean_weights=mesh_weights(clean_body)
    np.savez_compressed(out/"clean-control.npz",rest=clean_rest,corrected=oracle,
                        triangles=clean_faces,anchors=surface.prepare_reference(clean_body.data).anchors,
                        boneIds=clean_ids,boneWeights=clean_weights)
    write_asset(out/"input.fbx",True,split)
    arm,body,rest,posed=sample(out/"input.fbx")
    np.testing.assert_allclose(clean_rest,rest,rtol=0,atol=1e-7)
    body.data.calc_loop_triangles()
    faces=np.array([tuple(t.vertices) for t in body.data.loop_triangles])
    ids,weights=mesh_weights(body)
    np.savez_compressed(out/"stages-0-0.npz",rest=rest,corrected=posed,triangles=faces,
                        anchors=surface.prepare_reference(body.data).anchors,boneIds=ids,boneWeights=weights)
    np.savez_compressed(out/"oracle.npz",points=oracle)
    retarget.apply_output_mode(arm,[body],"rigged_rest")
    export_existing_rig(out/"stage-0-0-before-surface.fbx")


if __name__=="__main__":
    make(Path(sys.argv[-1]),"--split" in sys.argv)
