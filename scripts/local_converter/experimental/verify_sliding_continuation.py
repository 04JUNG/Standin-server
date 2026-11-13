"""Retest a saved trial on its source mesh under sliding contact constraints."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.local_converter.experimental.automatic_deformation import solve_patch
from scripts.local_converter.experimental.contact import baseline_contact_pins,contact_guard
from scripts.local_converter.experimental.deformation_guard import DeformationGuard


def verify(source, previous, destination):
    raw=np.load(source)
    rest,posed,faces=(raw[k] for k in ("rest","corrected","triangles"))
    changes=np.load(previous)
    guard=DeformationGuard(rest,posed,faces,raw["anchors"])
    guard.upper=np.maximum(1.,guard.reference["stretch"])
    guard.edge_upper=np.maximum(1.,guard.reference["edge"])
    guard.crease_limit=np.maximum(1e-7,np.abs(guard.angle_difference(
        guard.reference["bend"],guard.rest_bend)))
    graph=guard.graph
    worst=int(guard.reference["edge"].argmax())
    a,b=graph.a[worst],graph.b[worst]
    distance=np.linalg.norm(rest-(rest[a]+rest[b])/2,axis=1)
    _,inverse,counts=np.unique(rest,axis=0,return_inverse=True,return_counts=True)
    protected=guard.fixed|graph.fixed|(counts[inverse.ravel()]>1)
    free=distance<=.018*guard.scale
    free[changes["ids"]]=True
    free &= ~protected
    warm=posed.copy()
    warm[changes["ids"]]+=changes["delta"]
    warm_contact=contact_guard(posed,warm,faces,sliding=True)
    if (guard.violations(warm) or not warm_contact["passCheck"] or
            np.any(warm[protected]!=posed[protected]) or not guard.trajectory_safe(posed,warm)):
        raise ValueError("Input candidate does not meet the baseline constraints")
    features=[]
    pins,info=baseline_contact_pins(posed,faces,free,sliding_features=features)
    free &= ~pins
    target=warm.copy()
    unit=(posed[a]-posed[b])/np.linalg.norm(posed[a]-posed[b])
    change=.5*(np.linalg.norm(posed[a]-posed[b])-guard.lengths[worst])*unit
    target[a]-=change
    target[b]+=change
    trial,report=solve_patch(rest,posed,faces,guard.fixed,free,target,iterations=12,
                             deadline=time.monotonic()+30,sliding_features=features,
                             initial_points=warm)
    result=dict(sourceSHA=hashlib.sha256(source.read_bytes()).hexdigest(),
                previousSHA=hashlib.sha256(previous.read_bytes()).hexdigest(),
                warmContact=warm_contact,pins=info,activePairs=len(features),
                before=float(guard.reference["edge"][worst]),
                warm=float(guard.measure(warm)["edge"][worst]),
                after=float(guard.measure(trial)["edge"][worst]),shape=guard.violations(trial),
                contact=contact_guard(posed,trial,faces,sliding=True),
                trajectorySafe=guard.trajectory_safe(posed,trial),history=report["history"])
    destination.write_text(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("source","previous","destination"):
        parser.add_argument(name,type=Path)
    verify(**vars(parser.parse_args()))
