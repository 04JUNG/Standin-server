"""Recompute nonlinear hinge errors after each strain-constrained peak repair."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.local_converter.experimental.deformation_guard import DeformationGuard
from scripts.local_converter.experimental.probe_strain_cones import probe


def run(stage, initial, output, radius, goal, weight, rounds):
    output.mkdir(exist_ok=True,parents=True)
    with np.load(stage) as raw:
        p,f=raw["corrected"],raw["triangles"]
        g=DeformationGuard(raw["rest"],p,f,raw["anchors"])
    with np.load(initial) as data:
        ids,delta=data["ids"],data["delta"]
    original=g.reference
    smooth=abs(g.rest_bend)<np.pi/6
    crease_limit=np.maximum(1e-7,abs(g.angle_difference(original["bend"],g.rest_bend)))
    index=original["edge"].argmax()
    def audit(change):
        points=p.copy()
        points[ids]+=change
        m=g.measure(points)
        bend=np.maximum(abs(m["bend"])-g.bend_limit,0.)
        crease=np.maximum(abs(g.angle_difference(m["bend"],g.rest_bend))-crease_limit,0.)[g.crease]
        return dict(target=float(m["edge"][index]),peak=float(m["edge"].max()),
                    newFold=int((smooth & (abs(original["bend"])<np.pi/2) & (abs(m["bend"])>=np.pi/2)).sum()),
                    newSharp=int((smooth & (abs(original["bend"])<np.pi/3) & (abs(m["bend"])>=np.pi/3)).sum()),
                    newlySevere=int(((original["edge"]<=2+1e-7)&(m["edge"]>2+1e-7)).sum()),
                    creaseAndBendError=float(bend@bend+crease@crease))
    def rank(row):
        return row["newlySevere"],row["newFold"],row["newSharp"],row["creaseAndBendError"]
    best=audit(delta)
    history=[]
    for i in range(rounds):
        current=output/f"current-{i}.npz"
        np.savez_compressed(current,ids=ids,delta=delta)
        path=output/f"trial-{i}.json"
        _,candidate=probe(stage,path,radius,goal,True,bend_weight=weight,bend_reference=current,seconds=160.)
        row=dict(status=candidate["status"],before=best)
        history.append(row)
        if "target" not in candidate: break
        with np.load(path.with_suffix(".npz")) as new:
            if not np.array_equal(new["ids"],ids):
                raise ValueError("Patch changed during hinge restoration")
            update=new["delta"]-delta
        samples=[]
        for fraction in (1.,.5,.25,.125):
            samples.append(dict(fraction=fraction,**audit(delta+fraction*update)))
        row["samples"]=samples
        chosen=min(samples,key=rank)
        if rank(chosen)>=rank(best): break
        delta=delta+chosen["fraction"]*update
        best=chosen
        print(i,chosen,flush=True)
    np.savez_compressed(output/"diagnostic.npz",ids=ids,delta=delta)
    (output/"result.json").write_text(json.dumps(dict(best=best,history=history,qualified=False),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for arg in ("stage","initial","output"):
        parser.add_argument(arg,type=Path)
    parser.add_argument("--radius",type=float,required=True)
    parser.add_argument("--goal",type=float,required=True)
    parser.add_argument("--weight",type=float,default=25.)
    parser.add_argument("--rounds",type=int,default=3)
    run(**vars(parser.parse_args()))
