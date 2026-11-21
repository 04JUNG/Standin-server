"""Blender export/reimport and matched diagnostic views for adaptive result."""
import argparse
import json
import sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from converter.local import inspection, surface, materials as material_checks
from scripts.local_converter.make_csp_controls import export_existing_rig
from converter import retarget
from scripts.local_converter.render_followup import render

parser = argparse.ArgumentParser()
parser.add_argument("base",type=Path)
parser.add_argument("candidate",type=Path)
parser.add_argument("out",type=Path)
parser.add_argument("--authored-shading",action="store_true")
parser.add_argument("--original-rig",type=Path)
parser.add_argument("--reference",type=Path,help="Posed FBX corresponding to the staged baseline")
parser.add_argument("--model",type=int,default=1)
parser.add_argument("--pose",type=int,default=1)
args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
base,candidate,out=args.base,args.candidate,args.out
case=f"{args.model}-{args.pose}"
out.mkdir(parents=True,exist_ok=True)
raw=np.load(base/f"stages-{case}.npz")
change=np.load(candidate)
posed=raw["corrected"]
corrected=posed.copy()
corrected[change["ids"]]+=change["delta"]
reference=args.reference or base/f"stage-{case}-before-surface.fbx"
center=posed[change["ids"]].mean(0)
original=None
if args.original_rig:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    source_arm,source_meshes=inspection.import_fbx(args.original_rig)
    assert len(source_meshes)==1
    source=source_meshes[0]
    original=dict(weights=[sorted((source.vertex_groups[g.group].name,g.weight) for g in v.groups)
                           for v in source.data.vertices],
                  polygons=[tuple(p.vertices) for p in source.data.polygons],
                  hierarchy={b.name:b.parent.name if b.parent else None for b in source_arm.data.bones},
                  materials=[slot.name for slot in source.material_slots],
                  assignments=[p.material_index for p in source.data.polygons])
rows=[]
for label,points in (("baseline",posed),("guarded",corrected)):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    arm,meshes=inspection.import_fbx(reference)
    obj=meshes[0]
    assert len(obj.data.vertices)==len(points)
    weights=[[ (obj.vertex_groups[g.group].name,g.weight) for g in v.groups] for v in obj.data.vertices]
    polygons=[tuple(p.vertices) for p in obj.data.polygons]
    hierarchy={b.name:b.parent.name if b.parent else None for b in arm.data.bones}
    materials=[slot.name for slot in obj.material_slots]
    assignments=[p.material_index for p in obj.data.polygons]
    original_normals=np.array([n.vector[:] for n in obj.data.corner_normals])
    original_points=np.asarray(inspection.surface(meshes)[obj.name])
    moved=np.any(points!=posed,axis=1)
    if args.authored_shading:
        assert float(np.linalg.norm(original_points-posed,axis=1).max())<2e-5
        policy=surface.normal_policy(obj.data)
    inverse=obj.matrix_world.inverted()
    indices=np.flatnonzero(moved) if args.authored_shading else range(len(points))
    for index in indices:
        obj.data.vertices[int(index)].co=inverse@Vector(points[index])
    obj.data.update()
    obj.data.calc_loop_triangles()
    faces=np.array([t.vertices[:] for t in obj.data.loop_triangles])
    if args.authored_shading:
        displacements=np.linalg.norm(points-posed,axis=1)/abs(obj.matrix_world.to_scale().x)
        if moved.any():
            surface.update_shading(obj.data,policy,moved,displacement=displacements,
                                   posed_normals=original_normals)
        neighbors=np.unique(faces[moved[faces].any(1)])
        healthy=~np.isin(np.array([loop.vertex_index for loop in obj.data.loops]),neighbors)
    else:
        normals=surface.geometric_normals(np.array([v.co[:] for v in obj.data.vertices]),faces)
        obj.data.normals_split_custom_set_from_vertices(normals.tolist())
    path=out/f"{label}.fbx"
    if args.original_rig:
        export_existing_rig(path)
    else:
        retarget.export_fbx(str(path),embed_textures=False)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    check,meshes=inspection.import_fbx(path)
    obj=meshes[0]
    recovered=np.asarray(inspection.surface(meshes)[obj.name])
    error=float(np.linalg.norm(recovered-points,axis=1).max())
    assert error<2e-5
    assert polygons==[tuple(p.vertices) for p in obj.data.polygons]
    assert weights==[[(obj.vertex_groups[g.group].name,g.weight) for g in v.groups] for v in obj.data.vertices]
    assert all((check.data.bones[name].parent.name if check.data.bones[name].parent else None)==parent for name,parent in hierarchy.items())
    row=dict(label=label,maxPositionErrorM=error,weightsPolygonsHierarchyPreserved=True)
    if original is not None:
        assert original["weights"]==[sorted((obj.vertex_groups[g.group].name,g.weight)
                                     for g in v.groups) for v in obj.data.vertices]
        assert original["polygons"]==[tuple(p.vertices) for p in obj.data.polygons]
        assert all((check.data.bones[n].parent.name if check.data.bones[n].parent else None)==parent
                   for n,parent in original["hierarchy"].items())
        assert original["materials"]==[slot.name for slot in obj.material_slots]
        assert original["assignments"]==[p.material_index for p in obj.data.polygons]
        assert set(check.data.bones.keys())==set(original["hierarchy"])
        row["originalSolidMaterials"]=material_checks.compare_materials(
            inspection.fbx_materials(args.original_rig),inspection.fbx_materials(path))
        row.update(originalSourceWeightTopologyHierarchyMaterialsVerified=True,
                   addedBoneCount=len(set(check.data.bones.keys())-set(original["hierarchy"])))
    if args.authored_shading:
        assert materials==[slot.name for slot in obj.material_slots]
        assert assignments==[p.material_index for p in obj.data.polygons]
        current=np.array([n.vector[:] for n in obj.data.corner_normals])
        cosine=(current*original_normals).sum(1)/(np.linalg.norm(current,axis=1)*np.linalg.norm(original_normals,axis=1))
        degrees=np.degrees(np.arccos(np.clip(cosine,-1,1)))
        row.update(materialAssignmentsPreserved=True,healthyCorners=int(healthy.sum()),
                   healthyMaxNormalErrorDegrees=float(degrees[healthy].max()),
                   allCornersMaxNormalChangeDegrees=float(degrees.max()))
        assert row["healthyMaxNormalErrorDegrees"]<.15
    rows.append(row)
    for back in (False,True):
        render(path,out/f"{label}-{back}.png",center,.45,back)
(out/"verification.json").write_text(json.dumps(dict(rows=rows,
    shading=("Source normals with localized updates" if args.authored_shading else
             "Identical recomputed geometric normals, diagnostic views"),
    cspVerified=False),indent=2))
