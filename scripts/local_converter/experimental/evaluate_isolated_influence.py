"""Compare a pure isolated-influence proposal to its original posed baseline."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.local_converter.experimental.deformation_guard import DeformationGuard
from scripts.local_converter.experimental.isolated_influence import propose


def evaluate(source, weights, destination):
    stages=np.load(source)
    skin=np.load(weights)
    rest,posed,faces=(np.asarray(stages[k],dtype=int if k=="triangles" else float)
                     for k in ("rest","corrected","triangles"))
    if Path(source).resolve()!=Path(weights).resolve():
        for key, data in (("restSHA",rest.astype("<f8")),("facesSHA",faces.astype("<i8"))):
            if key not in skin or str(skin[key])!=hashlib.sha256(data.tobytes()).hexdigest():
                raise ValueError("Separate weight map requires a matching registered mesh")
    start=time.monotonic()
    guard=DeformationGuard(rest,posed,faces,stages["anchors"])
    guard.upper=np.maximum(1.,guard.reference["stretch"])
    guard.edge_upper=np.maximum(1.,guard.reference["edge"])
    guard.crease_limit=np.maximum(1e-7,abs(guard.angle_difference(guard.reference["bend"],guard.rest_bend)))
    _,inverse,counts=np.unique(rest,axis=0,return_inverse=True,return_counts=True)
    fixed=guard.fixed|guard.graph.fixed|(counts[inverse.ravel()]>1)
    result,report=propose(rest,posed,faces,fixed,guard,skin["boneIds"],skin["boneWeights"])
    report["elapsedSeconds"]=time.monotonic()-start
    ids=np.flatnonzero(np.any(result!=posed,axis=1))
    np.savez_compressed(destination.with_suffix(".npz"),ids=ids,delta=result[ids]-posed[ids])
    destination.write_text(json.dumps(report,indent=2))
    print(source.name,report,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("source","weights","destination"):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    evaluate(args.source,args.weights,args.destination)
