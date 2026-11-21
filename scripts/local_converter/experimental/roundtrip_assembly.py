"""Verify a posed two-object diagnostic FBX against its actual skinned input."""
import json
import sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from converter import retarget
from converter.local import surface,inspection,materials
from scripts.local_converter.experimental.make_relief_rig import apply_relief_pose
from scripts.local_converter.make_csp_controls import export_existing_rig


def inventory(arm,objects):
    return dict(bones={b.name:(b.parent.name if b.parent else None) for b in arm.data.bones},
                objects={obj.name:dict(faces=[tuple(p.vertices) for p in obj.data.polygons],
                weights=[sorted((obj.vertex_groups[g.group].name,g.weight) for g in v.groups)
                         for v in obj.data.vertices],
                materials=[slot.name for slot in obj.material_slots],
                assignments=[p.material_index for p in obj.data.polygons]) for obj in objects})


def verify(folder):
    source=folder/"assembly.fbx"
    output=folder/"evaluated.fbx"
    bpy.ops.wm.read_factory_settings(use_empty=True)
    arm,meshes=inspection.import_fbx(source)
    before=inventory(arm,meshes)
    apply_relief_pose(arm)
    bpy.context.view_layer.update()
    retarget.apply_output_mode(arm,meshes,"rigged_rest")
    expected={}
    for i,obj in enumerate(sorted(meshes,key=lambda o:len(o.data.vertices))):
        raw=np.load(folder/f"mesh-{i}.npz")
        baseline=raw["corrected"]
        points=np.load(folder/f"result-{i}.npy")
        observed=np.asarray(inspection.surface([obj])[obj.name])
        np.testing.assert_allclose(observed,baseline,atol=2e-5,rtol=0)
        moved=np.any(points!=baseline,axis=1)
        policy=surface.normal_policy(obj.data)
        normals=np.array([n.vector[:] for n in obj.data.corner_normals])
        inverse=obj.matrix_world.inverted()
        for v in np.flatnonzero(moved):
            obj.data.vertices[int(v)].co=inverse@Vector(points[v])
        if moved.any():
            obj.data.update()
            displacement=np.linalg.norm(points-baseline,axis=1)/abs(obj.matrix_world.to_scale().x)
            surface.update_shading(obj.data,policy,moved,displacement=displacement,posed_normals=normals)
        expected[obj.name]=points
    export_existing_rig(output)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    rig,objects=inspection.import_fbx(output)
    assert before==inventory(rig,objects)
    sampled=inspection.surface(objects)
    errors={name:float(np.linalg.norm(np.asarray(sampled[name])-points,axis=1).max())
            for name,points in expected.items()}
    assert max(errors.values())<2e-5
    report=dict(objects=len(objects),boneCount=len(rig.data.bones),exactWeightTopologyAndHierarchy=True,
                maxErrorsM=errors,solidMaterials=materials.compare_materials(
                    inspection.fbx_materials(source),inspection.fbx_materials(output)))
    (folder/"roundtrip.json").write_text(json.dumps(report,indent=2))
    print(report)


if __name__=="__main__":
    for path in sys.argv[sys.argv.index("--")+1:]:
        verify(Path(path))
