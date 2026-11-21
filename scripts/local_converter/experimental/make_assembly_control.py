"""Independent two-mesh FBX controls with an accessory in or away from the repair path."""
import sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.local_converter.experimental.make_relief_rig import sample
from scripts.local_converter.experimental.read_registered_weights import mesh_weights
from converter.local import surface


def evaluate_mesh(obj):
    bpy.context.view_layer.update()
    evaluated=obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return np.array([evaluated.matrix_world@v.co for v in evaluated.data.vertices],float)


def build(base,out,mode):
    out.mkdir(exist_ok=False)
    arm,body,rest,posed=sample(base/"input.fbx")
    reference=np.load(base/"oracle.npz")["points"]
    difference=posed-reference
    spike=np.linalg.norm(difference,axis=1).argmax()
    axis=difference[spike]/np.linalg.norm(difference[spike])
    perpendicular=np.eye(3)[np.argmin(abs(axis))]
    u=np.cross(axis,perpendicular)
    u/=np.linalg.norm(u)
    v=np.cross(axis,u)
    center=(posed[spike]+reference[spike])/2
    if mode=="clear":
        center+=2*u
    elif mode=="touch":
        center=posed[spike]
    elif mode=="embedded":
        incident=next(p for p in body.data.polygons if spike in p.vertices[:])
        others=[idx for idx in incident.vertices if idx!=spike]
        center=.85*posed[spike]+.15*posed[others].mean(0)
    desired=center+.012*np.stack([u, -.5*u+.866*v,-.5*u-.866*v])
    bone=arm.pose.bones["Left"]
    transform=np.asarray(arm.matrix_world@bone.matrix@bone.bone.matrix_local.inverted()@arm.matrix_world.inverted())
    homogeneous=np.column_stack([desired,np.ones(3)])@np.linalg.inv(transform).T
    mesh=bpy.data.meshes.new("Accessory plate")
    mesh.from_pydata(homogeneous[:,:3].tolist(),[],[(0,1,2)])
    plate=bpy.data.objects.new("Accessory plate",mesh)
    bpy.context.collection.objects.link(plate)
    plate.parent=arm
    plate.matrix_parent_inverse=arm.matrix_world.inverted()
    modifier=plate.modifiers.new("Skin","ARMATURE")
    modifier.object=arm
    plate.vertex_groups.new(name="Left").add([0,1,2],1.,"REPLACE")
    np.testing.assert_allclose(evaluate_mesh(plate),desired,atol=2e-5,rtol=0)
    matrices={bone.name:bone.matrix_basis.copy() for bone in arm.pose.bones}
    for bone in arm.pose.bones:
        bone.matrix_basis=Matrix.Identity(4)
    bpy.context.view_layer.update()
    file=out/"assembly.fbx"
    bpy.ops.export_scene.fbx(filepath=str(file),add_leaf_bones=False,bake_anim=False)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(file),ignore_leaf_bones=False)
    rig=next(o for o in bpy.context.scene.objects if o.type=="ARMATURE")
    meshes=sorted((o for o in bpy.context.scene.objects if o.type=="MESH"),key=lambda x:len(x.data.vertices))
    neutral=[evaluate_mesh(o) for o in meshes]
    for bone in rig.pose.bones:
        bone.matrix_basis=matrices[bone.name]
    recovered=[evaluate_mesh(o) for o in meshes]
    np.testing.assert_allclose(recovered[0],desired,atol=2e-5,rtol=0)
    np.testing.assert_allclose(recovered[1],posed,atol=2e-5,rtol=0)
    for index,obj in enumerate(meshes):
        ref=surface.prepare_reference(obj.data)
        ids,weights=mesh_weights(obj)
        np.savez_compressed(out/f"mesh-{index}.npz",rest=neutral[index],corrected=recovered[index],
                            triangles=ref.triangles,anchors=ref.anchors,boneIds=ids,boneWeights=weights)


if __name__=="__main__":
    base,out,mode=sys.argv[sys.argv.index("--")+1:]
    if mode not in ("obstacle","clear","touch","embedded"):
        raise ValueError("Unknown control")
    build(Path(base),Path(out),mode)
