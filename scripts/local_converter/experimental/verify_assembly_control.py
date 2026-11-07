"""Compare single-object and file-level decisions for an independent rig."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.local_converter.experimental.assembly import correct_assembly
from scripts.local_converter.experimental.automatic_deformation import correct
from scripts.local_converter.experimental.audit_pose_trace import inspect_pair


def run(source):
    parts=[np.load(p) for p in sorted(source.glob("mesh-*.npz"))]
    if len(parts)!=2:
        raise ValueError("Expected accessory and body")
    plate,body=parts
    trial,alone=correct(body["rest"],body["corrected"],body["triangles"],body["anchors"],
                        bone_ids=body["boneIds"],bone_weights=body["boneWeights"],regions=0)
    repaired,report=correct_assembly(parts,regions=0)
    for i,points in enumerate(repaired):
        np.save(source/f"result-{i}.npy",points)
    original=body["corrected"]
    extent=float(np.ptp(original,axis=0).max())
    counts=[]
    accepted_counts=[]
    for fraction in (0,.25,.5,.75,1):
        points=original+fraction*(trial-original)
        hits=0
        for face in body["triangles"][np.any((trial!=original).any(1)[body["triangles"]],axis=1)]:
            pair=inspect_pair(points[face],plate["corrected"][plate["triangles"][0]],extent)
            hits+=pair["strictInteriorHits"]==2
        counts.append(dict(fraction=fraction,strictCrossings=int(hits)))
        accepted=original+fraction*(repaired[1]-original)
        accepted_plate=plate["corrected"]+fraction*(repaired[0]-plate["corrected"])
        affected_faces=body["triangles"][np.any((trial!=original).any(1)[body["triangles"]],axis=1)]
        hits=sum(inspect_pair(accepted[face],accepted_plate[plate["triangles"][0]],extent)["strictInteriorHits"]==2
                 for face in affected_faces)
        accepted_counts.append(dict(fraction=fraction,strictCrossings=int(hits)))
    result=dict(sourceSha256=hashlib.sha256((source/"assembly.fbx").read_bytes()).hexdigest(),
                bodyAlone=alone,assembly=report,samples=counts,acceptedSamples=accepted_counts,
                maxOriginalDisplacementM=max(float(np.linalg.norm(q-m["corrected"],axis=1).max())
                                             for q,m in zip(repaired,parts)))
    (source/"verification.json").write_text(json.dumps(result,indent=2))
    print(source.name,report["applied"],report["contact"],counts)


if __name__=="__main__":
    run(Path(sys.argv[1]))
