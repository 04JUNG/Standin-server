"""Run the same automatic policy on the six approved staged geometries."""
import json
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.local_converter.experimental.automatic_deformation import correct
from scripts.local_converter.experimental.deformation_guard import DeformationGuard

base,out=map(Path,sys.argv[1:])
out.mkdir(parents=True,exist_ok=True)
rows=[]
sources = sorted(base.glob("stages-*.npz"))
if not sources:
    raise ValueError("No staged meshes found in the verification input")
for source in sources:
    data=np.load(source)
    rest,posed,faces=data["rest"],data["corrected"],data["triangles"]
    started=time.monotonic()
    points,report=correct(rest,posed,faces,data["anchors"],iterations=12,expansions=2)
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
