"""Measure the range of raw-skinning peak relief from compensated bone twists."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.local_converter.experimental.probe_joint_rotation import capture_pose


def probe(runtime,model,pose,stage,output,palette=None):
    arm,_,guard,bones,transforms,weights,homogeneous,error = capture_pose(runtime,model,pose,stage)
    if palette:
        indices={bone.name:i for i,bone in enumerate(bones)}
        heads=np.array([arm.matrix_world @ bone.head for bone in bones])
        parents=[-1 if bone.parent is None else indices[bone.parent.name] for bone in bones]
        np.savez_compressed(palette,maps=transforms,weights=weights,heads=heads,parents=parents,
                            names=[bone.name for bone in bones])
    index = guard.reference["edge"].argmax()
    a,b = guard.graph.a[index],guard.graph.b[index]
    edge = guard.baseline[b]-guard.baseline[a]
    fields = []
    for i,bone in enumerate(bones):
        if not weights[[a,b],i].any() or len(bone.children)!=1:
            continue
        head = np.asarray(arm.matrix_world @ bone.head)
        child = np.asarray(arm.matrix_world @ bone.children[0].head)
        axis = child-head
        if np.linalg.norm(axis)<1e-8: continue
        axis /= np.linalg.norm(axis)
        points = homogeneous @ transforms[i,:3].T - head
        radial = points - (points@axis)[:,None]*axis
        cosine = weights[:,i,None]*radial
        sine = np.cross(axis,cosine)
        fields.append((bone.name,cosine,sine))
    if len(fields)!=2:
        raise ValueError("This two-angle experiment requires exactly two active single-child bones")
    count=1024
    angles=np.arange(count)*2*np.pi/count
    rotations=[]
    radii=[]
    for name,cosine,sine in fields:
        radial = cosine[b]-cosine[a]
        tangent = sine[b]-sine[a]
        radii.append(float(np.linalg.norm(radial)))
        rotations.append((np.cos(angles)-1)[:,None]*radial + np.sin(angles)[:,None]*tangent)
    norms=np.linalg.norm(edge+rotations[0][:,None]+rotations[1][None,:],axis=2)
    i,j=np.unravel_index(norms.argmin(),norms.shape)
    length=guard.lengths[index]
    # A continuous angle is within half a grid step of a sampled angle.
    # Its omitted radial displacement is bounded by the corresponding chord.
    bound=max(0.,norms[i,j]-2*np.sin(np.pi/(2*count))*sum(radii))/length
    q=guard.baseline.copy()
    for theta,(_,cosine,sine) in zip(angles[[i,j]],fields):
        q+=(np.cos(theta)-1)*cosine+np.sin(theta)*sine
    guard.upper=np.maximum(1.,guard.reference["stretch"])
    guard.edge_upper=np.maximum(1.,guard.reference["edge"])
    guard.crease_limit=np.maximum(1e-7,abs(guard.angle_difference(guard.reference["bend"],guard.rest_bend)))
    ratios=guard.measure(q)["edge"]
    report=dict(stage="raw_skinning",before=float(guard.reference["edge"][index]),
                bestSample=float(ratios[index]),lowerBound=float(bound),globalPeak=float(ratios.max()),
                axes=[f[0] for f in fields],degrees=((np.degrees(angles[[i,j]])+180)%360-180).tolist(),
                newlySevere=int(((guard.reference["edge"]<=2)&(ratios>2+1e-7)).sum()),
                violations=guard.violations(q),reconstructionErrorM=error)
    output.write_text(json.dumps(report,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    for name in ("runtime","model","pose","stage","output"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--palette",type=Path)
    probe(**vars(parser.parse_args(sys.argv[sys.argv.index("--")+1:])))
