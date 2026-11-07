"""Bind sampled new intersections to a fixed-axis constraint policy."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.local_converter.experimental.deformation_guard import DeformationGuard
from scripts.local_converter.experimental.separation_witnesses import fixed_axis, interval_planes


def collect(stage,candidate,report,output,previous,divisions=1):
    observed=json.loads(report.read_text())
    for key,path in (("stageSHA",stage),("candidateSHA",candidate)):
        if observed[key]!=hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Intersection witnesses belong to another trial")
    with np.load(stage) as raw, np.load(candidate) as change:
        p,f=raw["corrected"],raw["triangles"]
        guard=DeformationGuard(raw["rest"],p,f,raw["anchors"])
        q=p.copy()
        q[change["ids"]]+=change["delta"]
    policy=json.loads(previous.read_text()) if previous else {}
    entries={tuple(sorted(w[:2])):w for w in policy.get("separate",[])}
    intervals=list(policy.get("intervalSeparators",[]))
    unresolved=[]
    for i,j in observed["pairs"]:
        if divisions>1:
            planes=interval_planes(p,q,f,i,j,guard.scale,divisions)
            if planes: intervals.extend(planes)
            else: unresolved.append([i,j])
            continue
        witness=fixed_axis(p[f[i]],p[f[j]],q[f[i]],q[f[j]],guard.scale)
        if witness is None:
            unresolved.append([i,j])
        else:
            entries.setdefault((i,j),[i,j,witness[0].tolist()])
    policy["separate"]=list(entries.values())
    policy["intervalSeparators"]=intervals
    policy["uncertifiedPairs"]=unresolved
    m=guard.measure(q)
    folded=((abs(guard.rest_bend)<np.pi/6) &
            (abs(guard.reference["bend"])<np.pi/2) & (abs(m["bend"])>=np.pi/2))
    policy["hinges"]=sorted(set(policy.get("hinges",[]))|set(np.flatnonzero(folded).tolist()))
    policy.pop("initial",None)
    output.write_text(json.dumps(policy))
    print(dict(separators=len(entries),ambiguous=len(unresolved),protectedHinges=len(policy["hinges"])))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("stage","candidate","report","output"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--previous",type=Path)
    parser.add_argument("--divisions",type=int,default=1)
    collect(**vars(parser.parse_args()))
