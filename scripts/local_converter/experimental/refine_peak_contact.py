"""Refine a measured peak against newly discovered shape and contact witnesses."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.local_converter.experimental.contact import baseline_contact_pins, contact_guard
from scripts.local_converter.experimental.contact_features import measures
from scripts.local_converter.experimental.deformation_guard import DeformationGuard
from scripts.local_converter.experimental.probe_strain_cones import probe
from scripts.local_converter.experimental.separation_witnesses import witnesses


def refine(stage, output, radius, goal, rounds, seed_policy):
    output.mkdir(parents=True,exist_ok=True)
    with np.load(stage) as data:
        p,f=data["corrected"],data["triangles"]
        guard=DeformationGuard(data["rest"],p,f,data["anchors"])
    guard.upper=np.maximum(1.,guard.reference["stretch"])
    guard.edge_upper=np.maximum(1.,guard.reference["edge"])
    guard.crease_limit=np.maximum(1e-7,abs(guard.angle_difference(guard.reference["bend"],guard.rest_bend)))
    policy=json.loads(seed_policy.read_text()) if seed_policy else dict(pairs=[],edges={},separate=[])
    policy.pop("initial",None)
    summary=dict(acceptedStage=False,selected="input",trials=[])
    feature_data=None
    for i in range(rounds):
        policy_path=output/f"policy-{i}.json"
        policy_path.write_text(json.dumps(policy))
        path=output/f"trial-{i}.json"
        _,trial=probe(stage,path,radius,goal,True,shape_steps=20,contacts=True,witnesses=policy_path)
        row=dict(iteration=i,status=trial["status"],pairs=len(policy["pairs"]),separators=len(policy["separate"]))
        summary["trials"].append(row)
        if "target" not in trial:
            break
        with np.load(path.with_suffix(".npz")) as delta:
            q=p.copy()
            ids=delta["ids"]
            q[ids]+=delta["delta"]
        check=contact_guard(p,q,f,sliding=True)
        values=guard.measure(q)
        violations=guard.violations(q)
        row.update(target=trial["target"], peak=float(values["edge"].max()),violations=violations,
                   newSevere=int(((guard.reference["edge"]<=2)&(values["edge"]>2+1e-7)).sum()),
                   trajectory=guard.trajectory_safe(p,q),contact=check)
        if (not violations and row["trajectory"] and check["passCheck"] and not row["newSevere"]
                and row["target"]<=goal+1e-5):
            summary.update(acceptedStage=True,selected=str(path.with_suffix(".npz")))
            break
        previous=json.dumps({k:v for k,v in policy.items() if k!="initial"},sort_keys=True)
        extra=values["edge"]-guard.edge_upper
        for edge in np.flatnonzero(extra>1e-7):
            key=str(edge)
            policy["edges"][key]=policy["edges"].get(key,0.)+max(5e-7,2*extra[edge])
        if feature_data is None:
            features=[]
            moving=np.zeros(len(p),bool)
            moving[ids]=True
            _,scan=baseline_contact_pins(p,f,moving,sliding_features=features)
            if not scan["complete"]: break
            pairs=np.array([[a,b] for a,b,_ in features],dtype=int).reshape(-1,2)
            locations=f[pairs]
            spec=np.asarray([s for _,_,s in features],dtype=int).reshape(-1,2,2)
            feature_data=(pairs,locations,spec,measures(p,locations,spec))
        pairs,locations,spec,base=feature_data
        if len(pairs):
            scores=np.max([(measures(p+t*(q-p),locations,spec)-base).reshape(-1,7).max(1)
                           for t in (.001,.125,.25,.5,.75,1.)],axis=0)
            row["sampledExistingPairsWorsened"]=int((scores>1e-11).sum())
            proper={tuple(sorted(pair)) for pair in pairs.tolist()}
            first=sorted(check.get("faces",[]))
            candidates=([first] if tuple(first) in proper else []) + [
                sorted(pairs[j].tolist()) for j in np.argsort(-scores)[:6] if scores[j]>1e-11]
            for pair in candidates:
                if pair not in policy["pairs"]: policy["pairs"].append(pair)
        separation={tuple(sorted(w[:2])):w for w in policy["separate"]}
        for separator in witnesses(p,q,f,include_swept=True):
            separation.setdefault(tuple(sorted(separator[:2])),separator)
        policy["separate"]=list(separation.values())
        print(i,row["target"],row["violations"],row.get("sampledExistingPairsWorsened"),row["separators"],flush=True)
        updated=json.dumps({k:v for k,v in policy.items() if k!="initial"},sort_keys=True)
        if updated==previous: break
        policy["initial"]=str(path.with_suffix(".npz"))
        (output/"result.json").write_text(json.dumps(summary,indent=2))
    (output/"result.json").write_text(json.dumps(summary,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--radius",type=float,default=.006)
    parser.add_argument("--goal",type=float,required=True)
    parser.add_argument("--rounds",type=int,default=12)
    parser.add_argument("--seed-policy",type=Path)
    refine(**vars(parser.parse_args()))
