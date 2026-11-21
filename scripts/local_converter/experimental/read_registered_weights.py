"""Extract verified weight rows from the approved registered QA FBXs."""
import json
import sys
import hashlib
from pathlib import Path
import bpy
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from converter.local import inspection


def mesh_weights(obj):
    width=max(len(v.groups) for v in obj.data.vertices)
    ids=np.full((len(obj.data.vertices),width),-1,int)
    weights=np.zeros_like(ids,dtype=float)
    for v in obj.data.vertices:
        for i,g in enumerate(v.groups):
            ids[v.index,i]=g.group
            weights[v.index,i]=g.weight
    return ids,weights


def extract(source,stage,destination):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    _,meshes=inspection.import_fbx(source)
    if len(meshes)!=1:
        raise ValueError("Expected one registered mesh for staged comparison")
    obj=meshes[0]
    raw=np.load(stage)
    rest=raw["rest"]
    points=np.asarray(inspection.surface(meshes)[obj.name])
    if points.shape!=rest.shape or np.linalg.norm(points-rest,axis=1).max()>2e-5:
        raise ValueError("Registered vertex correspondence is unverified")
    ids,weights=mesh_weights(obj)
    obj.data.calc_loop_triangles()
    faces=np.asarray([t.vertices[:] for t in obj.data.loop_triangles],dtype=int)
    if not np.array_equal(faces,raw["triangles"]):
        raise ValueError("Registered triangle correspondence is unverified")
    np.savez_compressed(destination,boneIds=ids,boneWeights=weights,
                        restSHA=hashlib.sha256(rest.astype("<f8").tobytes()).hexdigest(),
                        facesSHA=hashlib.sha256(faces.astype("<i8").tobytes()).hexdigest(),
                        registeredSha256=hashlib.sha256(source.read_bytes()).hexdigest())


if __name__=="__main__":
    base,out=map(Path,sys.argv[sys.argv.index("--")+1:])
    out.mkdir(exist_ok=False)
    manifest={}
    for source in sorted(base.glob("model-*-registered.fbx")):
        number=source.stem.split("-")[1]
        weight_file=out/f"weights-{number}.npz"
        extract(source,base/f"stages-{number}-0.npz",weight_file)
        for stage in base.glob(f"stages-{number}-*.npz"):
            manifest[stage.name]=weight_file.name
    (out/"weights.json").write_text(json.dumps(manifest,indent=2))
