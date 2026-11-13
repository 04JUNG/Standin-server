"""Keep prior fold witnesses while adding newly folded hinges to the next solve."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.local_converter.experimental.deformation_guard import DeformationGuard
from scripts.local_converter.experimental.probe_strain_cones import probe


def refine(stage,initial,output,radius,goal,rounds,weight,seed_policy):
    output.mkdir(exist_ok=True,parents=True)
    with np.load(stage) as raw:
        p=raw["corrected"]
        g=DeformationGuard(raw["rest"],p,raw["triangles"],raw["anchors"])
    protected=set(json.loads(seed_policy.read_text())["hinges"]) if seed_policy else set()
    history=[]
    source=initial
    smooth=abs(g.rest_bend)<np.pi/6
    for i in range(rounds+1):
        with np.load(source) as current:
            q=p.copy()
            q[current["ids"]]+=current["delta"]
        m=g.measure(q)
        def introduced(angle):
            return np.flatnonzero(smooth & (abs(g.reference["bend"])<angle) & (abs(m["bend"])>=angle))
        folds,sharp=introduced(np.pi/2),introduced(np.pi/3)
        row=dict(candidate=str(source),newFolds=len(folds),newSharp=len(sharp),protected=len(protected),
                 peak=float(m["edge"].max()),newSevere=int(((g.reference["edge"]<=2)&(m["edge"]>2+1e-7)).sum()))
        history.append(row)
        additional=set((folds if len(folds) else sharp).tolist())-protected
        if i==rounds or not additional: break
        protected.update(additional)
        policy=output/f"policy-{i}.json"
        policy.write_text(json.dumps(dict(hinges=sorted(protected))))
        trial=output/f"trial-{i}.json"
        _,result=probe(stage,trial,radius,goal,True,witnesses=policy,bend_weight=weight,seconds=160.)
        row["nextStatus"]=result["status"]
        print(i,row,flush=True)
        if "target" not in result: break
        source=trial.with_suffix(".npz")
    (output/"result.json").write_text(json.dumps(dict(history=history,qualified=False),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("stage","initial","output"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--radius",type=float,required=True)
    parser.add_argument("--goal",type=float,required=True)
    parser.add_argument("--rounds",type=int,default=3)
    parser.add_argument("--weight",type=float,default=1.)
    parser.add_argument("--seed-policy",type=Path)
    refine(**vars(parser.parse_args()))
