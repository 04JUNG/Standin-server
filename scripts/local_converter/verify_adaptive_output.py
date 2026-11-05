"""Blender export/reimport and matched diagnostic views for adaptive result."""
import json
import sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from converter.local import inspection, surface
from converter import retarget
from scripts.local_converter.render_followup import render

base,candidate,out=map(Path,sys.argv[sys.argv.index("--")+1:])
out.mkdir(parents=True,exist_ok=True)
raw=np.load(base/"stages-1-1.npz")
change=np.load(candidate)
posed=raw["corrected"]
corrected=posed.copy()
corrected[change["ids"]]+=change["delta"]
reference=base/"stage-1-1-before-surface.fbx"
center=posed[change["ids"]].mean(0)
rows=[]
for label,points in (("baseline",posed),("guarded",corrected)):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    arm,meshes=inspection.import_fbx(reference)
    obj=meshes[0]
    assert len(obj.data.vertices)==len(points)
    weights=[[ (obj.vertex_groups[g.group].name,g.weight) for g in v.groups] for v in obj.data.vertices]
    polygons=[tuple(p.vertices) for p in obj.data.polygons]
    hierarchy={b.name:b.parent.name if b.parent else None for b in arm.data.bones}
    inverse=obj.matrix_world.inverted()
    for vertex,point in zip(obj.data.vertices,points):
        vertex.co=inverse@Vector(point)
    obj.data.update()
    obj.data.calc_loop_triangles()
    faces=np.array([t.vertices[:] for t in obj.data.loop_triangles])
    normals=surface.geometric_normals(np.array([v.co[:] for v in obj.data.vertices]),faces)
    obj.data.normals_split_custom_set_from_vertices(normals.tolist())
    path=out/f"{label}.fbx"
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
    rows.append(dict(label=label,maxPositionErrorM=error,weightsPolygonsHierarchyPreserved=True))
    for back in (False,True):
        render(path,out/f"{label}-{back}.png",center,.45,back)
(out/"verification.json").write_text(json.dumps(dict(rows=rows,
    shading="Identical recomputed geometric normals, diagnostic views",
    cspVerified=False),indent=2))
