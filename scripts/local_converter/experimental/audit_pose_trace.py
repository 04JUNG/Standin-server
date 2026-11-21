"""Report observed contacts in a constructed pose path and in surface correction."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.local_converter.experimental.contact import static_relation


def strict_crossings(first, second):
    """Edge hits strictly inside the other triangle, excluding near-boundary hits."""
    basis = np.stack((second[1]-second[0],second[2]-second[0]),axis=1)
    normal = np.cross(basis[:,0],basis[:,1])
    hits = []
    for start, end in zip(first, np.roll(first,-1,axis=0)):
        direction = end-start
        denominator = normal @ direction
        if abs(denominator) <= 1e-12*np.linalg.norm(normal)*np.linalg.norm(direction):
            continue
        fraction = normal @ (second[0]-start) / denominator
        if not 1e-7 < fraction < 1-1e-7:
            continue
        point = start+fraction*direction
        u,v = np.linalg.lstsq(basis,point-second[0],rcond=None)[0]
        if min(u,v,1-u-v)>1e-7:
            hits.append(point)
    return hits


def inspect_pair(a,b,extent):
    hits = strict_crossings(a,b) + strict_crossings(b,a)
    return dict(relation=static_relation(a,b,extent),strictInteriorHits=len(hits),
                intersectionSegmentMm=float(1000*np.linalg.norm(hits[0]-hits[1]))
                if len(hits)==2 else None)


def audit(directory, stages):
    report=json.loads((directory/"trace.json").read_text())
    frames=np.load(directory/"diagnostic-pairs.npz")["triangles"]
    raw=np.load(stages)
    extent=float(np.ptp(raw["rest"],axis=0).max())
    for row,pairs in zip(report["frames"],frames,strict=True):
        row["contacts"]=[inspect_pair(a,b,extent) for a,b in pairs]
    triangles=raw["triangles"][np.asarray(report["pairs"],int).reshape(-1,2)]
    report["stageContacts"]={
        stage:[inspect_pair(a,b,extent) for a,b in raw[stage][triangles]]
        for stage in ("rest","solver","arm","baked","corrected")}
    (directory/"audit.json").write_text(json.dumps(report,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("directory",type=Path)
    parser.add_argument("stages",type=Path)
    args=parser.parse_args()
    audit(args.directory,args.stages)
