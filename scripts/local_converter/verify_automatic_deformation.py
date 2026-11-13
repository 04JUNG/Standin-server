"""Run the same automatic policy on the six approved staged geometries."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.local_converter.experimental.automatic_deformation import correct
from scripts.local_converter.experimental.deformation_guard import DeformationGuard

parser=argparse.ArgumentParser()
parser.add_argument("base",type=Path)
parser.add_argument("out",type=Path)
parser.add_argument("--weight-manifest",type=Path)
parser.add_argument("--sliding",action="store_true")
args=parser.parse_args()
base,out=args.base,args.out
out.mkdir(parents=True,exist_ok=True)
weights_by_source=json.loads(args.weight_manifest.read_text()) if args.weight_manifest else {}
rows=[]
sources = sorted(base.glob("stages-*.npz"))
if not sources:
    raise ValueError("No staged meshes found in the verification input")
for source in sources:
    data=np.load(source)
    rest,posed,faces=data["rest"],data["corrected"],data["triangles"]
    skin=data
    if args.weight_manifest:
        skin=np.load(args.weight_manifest.parent/weights_by_source[source.name])
        for key,array,dtype in (("restSHA",rest,"<f8"),("facesSHA",faces,"<i8")):
            if str(skin[key])!=hashlib.sha256(array.astype(dtype).tobytes()).hexdigest():
                raise ValueError("Registered weights belong to different geometry")
    started=time.monotonic()
    points,report=correct(rest,posed,faces,data["anchors"],iterations=12,expansions=2,sliding=args.sliding,
                          bone_ids=skin.get("boneIds"),bone_weights=skin.get("boneWeights"))
    report["elapsedSeconds"]=time.monotonic()-started
    guard=DeformationGuard(rest,posed,faces,data["anchors"])
    guard.upper=np.maximum(1.,guard.reference["stretch"])
    guard.edge_upper=np.maximum(1.,guard.reference["edge"])
    guard.crease_limit=np.maximum(1e-7,np.abs(guard.angle_difference(guard.reference["bend"],guard.rest_bend)))
    assert not guard.violations(points)
    ids=np.flatnonzero(np.any(points!=posed,axis=1))
    np.savez_compressed(out/source.name,ids=ids,delta=points[ids]-posed[ids])
    rows.append(dict(source=source.name,report=report))
    (out/"report.json").write_text(json.dumps(rows,indent=2))
    print(source.name,{key:value for key,value in report.items() if key!="history"},flush=True)
