"""Sample a constructed neutral-to-target skeletal path for diagnosis."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix


def trace(runtime, model, pose, stages, output, pairs):
    sys.path.insert(0, str(runtime))
    from converter import retarget
    from converter.local import inspection, surface

    bpy.ops.wm.read_factory_settings(use_empty=True)
    arm, meshes = inspection.import_fbx(model)
    profile, _ = inspection.inspect(arm, meshes)
    assert len(meshes) == 1
    mesh = meshes[0]
    reference = surface.prepare_reference(mesh.data)
    rest = np.asarray(inspection.surface(meshes)[mesh.name])
    retarget._apply_object_transform(arm)
    initial = {b.name:b.matrix_basis.copy() for b in arm.pose.bones}
    src = retarget.import_bvh(str(pose), frame=0)
    result = retarget.retarget(src,arm,dst_profile=profile["profile"],apply_root_translation=False)
    assert result.ok
    target = {b.name:b.matrix_basis.copy() for b in arm.pose.bones}
    data = np.load(stages)
    graph = surface.Neighborhood(reference.triangles, rest)
    lengths = surface.lengths(rest[graph.a]-rest[graph.b])
    ratio = surface.lengths(data["solver"][graph.a]-data["solver"][graph.b])/lengths
    edge = int(np.argmax(ratio))
    contact_ids = np.array(pairs,dtype=int).reshape(-1,2)
    frames = []
    contacts = []
    first_damage = None
    for amount in (0.,.001,.005,.01,.02,.05,.1,.25,.5,.75,1.):
        for b in arm.pose.bones:
            loc, rot, scale = initial[b.name].decompose()
            loc1, rot1, scale1 = target[b.name].decompose()
            b.matrix_basis = Matrix.LocRotScale(loc.lerp(loc1,amount),rot.slerp(rot1,amount),
                                                scale.lerp(scale1,amount))
        bpy.context.view_layer.update()
        points = np.asarray(inspection.surface(meshes)[mesh.name])
        if amount in (0.,1.):
            expected = rest if amount == 0 else data["solver"]
            error = float(surface.lengths(points-expected).max())
            assert error < 2e-5, (amount,error)
        ratios = surface.lengths(points[graph.a]-points[graph.b])/lengths
        if first_damage is None and ratios.max()>2.:
            first_damage = points.copy()
        frames.append(dict(fraction=amount,targetRatio=float(ratios[edge]),
                           maxRatio=float(ratios.max()),aboveTwo=int((ratios>2).sum())))
        contacts.append(points[reference.triangles[contact_ids]])
    output.mkdir(parents=True,exist_ok=True)
    if first_damage is not None:
        np.savez_compressed(output/"stages-first-damage.npz",rest=rest,corrected=first_damage,
                            triangles=reference.triangles,anchors=reference.anchors)
    np.savez_compressed(output/"diagnostic-pairs.npz",triangles=np.array(contacts))
    report=dict(modelSHA=hashlib.sha256(model.read_bytes()).hexdigest(),
                poseSHA=hashlib.sha256(pose.read_bytes()).hexdigest(),
                edge=[int(graph.a[edge]),int(graph.b[edge])],pairs=contact_ids.tolist(),
                frames=frames,path="constructed local-bone rotation interpolation, not captured motion")
    (output/"trace.json").write_text(json.dumps(report,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("runtime","model","pose","stages","output"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--pairs",nargs="*",type=int,default=[])
    trace(**vars(parser.parse_args(sys.argv[sys.argv.index("--")+1:])))
