"""Compare split boundaries in staged poses and optionally inject a gap control."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.local_converter.experimental.seams import split_boundaries,gap_changes,guard_gap_step
from scripts.local_converter.experimental.isolated_influence import valid_weights


def audit(source, weight_file=None, inject_gap=False):
    raw=np.load(source)
    rest=np.asarray(raw["rest"],float)
    faces=np.asarray(raw["triangles"],int)
    skin=np.load(weight_file) if weight_file else raw
    if weight_file:
        for name,array,dtype in (("restSHA",rest,"<f8"),("facesSHA",faces,"<i8")):
            if str(skin[name]) != hashlib.sha256(array.astype(dtype).tobytes()).hexdigest():
                raise ValueError("Rest or topology does not match the weight source")
    ids,weights=skin["boneIds"],skin["boneWeights"]
    if not valid_weights(rest,ids,weights):
        raise ValueError("Invalid seam influences")
    pairs,inventory=split_boundaries(rest,faces,ids,weights)
    result=dict(source=source.name,sourceSHA=hashlib.sha256(source.read_bytes()).hexdigest(),
                inventory=inventory,stages={})
    previous=rest
    for stage in ("solver","arm","baked","corrected"):
        if stage in raw:
            current=raw[stage]
            result["stages"][stage]=gap_changes(previous,current,pairs)
            previous=current
    if inject_gap:
        if not len(pairs):
            raise ValueError("Gap control requires a matched boundary")
        candidate=previous.copy()
        candidate[pairs[:,1],2]+=.002
        recovered,report=guard_gap_step(previous,candidate,pairs)
        result["injectedControl"]=dict(report=report,restoredExactly=bool(np.array_equal(recovered,previous)))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--weight-file",type=Path)
    parser.add_argument("--inject-gap",action="store_true")
    args=parser.parse_args()
    args.output.write_text(json.dumps(audit(args.source,args.weight_file,args.inject_gap),indent=2))
