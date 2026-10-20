"""Check consistency weights, fixed-label geometry, no-signal fallback and results."""
import hashlib
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiment_consistency_cost import (parse,verify,read,BASE,CONFIDENCE,CONSISTENCY,COMBINED,ARMS,GROUPS,
    load_library,sha256,save_json,valid_query,geometry_hash,normalized_joints)
from standin_eval.perspective_search import Camera,project,normalize_batch
from standin_eval.camera_refinement import project_parameters


def audit(args):
    verify(args);assert not (args.out/'audit.json').exists()
    rows=read(args.out/'rankings.json');assert sha256(args.out/'rankings.json')==read(args.out/'ranked.json')['sha256']
    poses=load_library(args.library_root);assert geometry_hash(poses)==read(args.source/'protocol.json')['hashes']['library_geometry']
    points=normalized_joints(np.stack([p.joints for p in poses]))
    old={r['image']:r for r in read(args.refinement_out/'rankings.json')}
    prior={r['image']:r for r in read(args.confidence_out/'rankings.json')}
    inferred={r['image']:r for r in read(args.rotation_out/'inferred.json')['records']}
    manifest={r['image']:r for r in read(args.source/'manifest.json')}
    checked=failed=enabled=disabled=scalar=0;maxerror=0.
    for row in rows:
        name=row['image'];path=args.source/'predictions'/(Path(name).stem+'.json');assert sha256(path)==row['prediction_sha256']
        assert sha256(args.source/'images'/name)==manifest[name]['sha256']
        pred=read(path);query=valid_query(pred)
        if query is None:
            assert row['error']==old[name]['error'] and all(a not in row for a in ARMS);failed+=1;continue
        xy,mask=query;q=normalize_batch(xy)
        assert hashlib.sha256(xy.tobytes()).hexdigest()==row['shared_xy_sha256'] and mask.tolist()==row['shared_mask']
        path=args.refinement_out/'costs'/(Path(name).stem+'.npz');assert sha256(path)==old[name]['costs_sha256']
        with np.load(path,allow_pickle=False) as d:base,fit,params=[d[k] for k in ('baseline','refined','parameters')]
        path=args.confidence_out/'costs'/(Path(name).stem+'.npz');assert sha256(path)==prior[name]['costs_sha256']
        with np.load(path,allow_pickle=False) as d:conf=d['weighted']
        path=args.rotation_out/'hypotheses'/(Path(name).stem+'.json');assert sha256(path)==inferred[name]['sha256']
        signals=read(path)['agreement']['joints']
        sw=np.ones(17)
        for j in signals:
            if j['support']>=2 and j['disagreement']>=.25:sw[j['index']]=.5
        cw=np.array([.25+.75*min(1.,max(0.,(s-.3)/.4)) for s in pred['scores']])
        on=any(mask[j] and sw[j]<1 for j in range(5,17));assert row['signal_active']==on
        assert row['downweighted_joints']==[j for j in range(5,17) if mask[j] and sw[j]<1]
        path=args.out/'costs'/(Path(name).stem+'.npz');assert sha256(path)==row['costs_sha256']
        with np.load(path,allow_pickle=False) as d:
            costs={BASE:fit,CONFIDENCE:conf,CONSISTENCY:d['consistency'],COMBINED:d['combined']}
            assert np.array_equal(d['weights'],sw)
        features,valid=project_parameters(points,params);assert valid.all()
        means=[];weighted={a:[] for a in (CONFIDENCE,CONSISTENCY,COMBINED)}
        for swap in (False,True):
            pairs=[(j,(j+1 if j%2 else j-1) if swap else j) for j in range(5,17)]
            pairs=[(j,k) for j,k in pairs if mask[k]]
            residual=[np.linalg.norm(features[:,j]-q[k],axis=1) for j,k in pairs]
            means.append(np.stack(residual,axis=1).mean(axis=1))
            for arm,w in ((CONFIDENCE,cw),(CONSISTENCY,sw),(COMBINED,cw*sw)):
                total=np.zeros(len(poses));den=0.
                for r,(j,k) in zip(residual,pairs):total+=r.astype(float)*w[k];den+=w[k]
                weighted[arm].append(total/den)
        labels=np.argmin(abs(np.stack(means)-fit),axis=0);cols=np.arange(len(poses))
        assert np.max(abs(np.stack(means)[labels,cols]-fit))<2e-5
        for arm in (CONFIDENCE,CONSISTENCY,COMBINED):
            expected=np.stack(weighted[arm])[labels,cols]
            if not on and arm==CONSISTENCY:expected=fit
            if not on and arm==COMBINED:expected=conf
            assert np.allclose(costs[arm],expected,atol=1e-12,rtol=1e-12)
        if not on:
            assert np.array_equal(costs[CONSISTENCY],fit) and np.array_equal(costs[COMBINED],conf)
        first=int(np.argmin(base));selected=set(range(0,len(poses),600))
        for arm,cost in costs.items():
            score=.25*base.astype(float)+.75*cost.astype(float)
            ids=np.argsort(score,kind='stable').tolist();ids=[first]+[i for i in ids if i!=first]
            assert row[arm]==[poses[i].pose_id for i in ids[:5]];selected.update(ids[:5])
        assert row[BASE]==prior[name][BASE] and row[CONFIDENCE]==prior[name][CONFIDENCE]
        for i in selected:
            yaw,pitch,inv=map(float,params[i]);px,ok=project(poses[i].joints,Camera(yaw,pitch,None if inv==0 else 1/inv));assert ok
            px=normalize_batch(px)
            for arm,w in ((CONSISTENCY,sw),(COMBINED,cw*sw)):
                num=den=0.
                for j in range(5,17):
                    k=(j+1 if j%2 else j-1) if labels[i] else j
                    if not mask[k]:continue
                    num+=float(np.linalg.norm(px[j]-q[k]))*w[k];den+=w[k]
                delta=abs(num/den-float(costs[arm][i]));assert delta<3e-5
                maxerror=max(maxerror,delta);scalar+=1
        checked+=1;enabled+=int(on);disabled+=int(not on)
    assert checked+failed==len(rows)==360
    answers={r['image']:r for r in read(args.source/'answers.json')};reported=read(args.out/'results.json')
    counts={a:{g:{'top1':0,'top5':0} for g in GROUPS} for a in ARMS}
    changes={a:{b:{g:{'gains':0,'losses':0} for g in GROUPS} for b in ARMS[:2]} for a in ARMS[2:]}
    failures={g:0 for g in GROUPS};active={g:0 for g in GROUPS}
    for row in rows:
        t=answers[row['image']];g,pid=t['group'],t['pose_id'];failures[g]+=bool(row.get('error'));active[g]+=row.get('signal_active',False)
        for arm in ARMS:
            ids=row.get(arm,[]);counts[arm][g]['top1']+=bool(ids and ids[0]==pid);counts[arm][g]['top5']+=pid in ids
        for arm in ARMS[2:]:
            for b in ARMS[:2]:
                before=pid in row.get(b,[]);after=pid in row.get(arm,[])
                changes[arm][b][g]['gains']+=after and not before;changes[arm][b][g]['losses']+=before and not after
    assert counts==reported['counts'] and changes==reported['changes'] and failures==reported['failures'] and active==reported['active']
    choices=[a for a in ARMS[2:] if counts[a]['strong_perspective']['top5']>=counts[BASE]['strong_perspective']['top5']+3
             and counts[a]['strong_perspective']['top5']>counts[CONFIDENCE]['strong_perspective']['top5']
             and all(counts[a][g][k]>=max(counts[b][g][k] for b in ARMS[:2]) for g in GROUPS for k in ('top1','top5'))]
    choices.sort(key=lambda a:(-counts[a]['strong_perspective']['top5'],-sum(counts[a][g]['top5'] for g in GROUPS),a!=CONSISTENCY))
    assert reported['eligible']==choices and reported['selected']==(choices[0] if choices else None)
    verify(args)
    result={'images_verified':360,'all_gallery_costs_and_rankings_checked':checked,'failures_retained':failed,
            'active':enabled,'no_signal_exact_fallback':disabled,'independent_scalar_cost_checks':scalar,
            'max_scalar_cost_error':maxerror,'outcomes_recomputed':True,
            'ranking_sha256':sha256(args.out/'rankings.json'),'auditor_sha256':sha256(Path(__file__))}
    save_json(args.out/'audit.json',result);print(result,flush=True)


if __name__=='__main__':audit(parse())
