"""Recompute hint geometry, rankings, applied subset and development outcome."""
import hashlib
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiment_head_hint import parse,verify,read,BASE,WEIGHTS,GROUPS,load_library,save_json,sha256,valid_query
from scripts.confirm_camera_fit import geometry_hash
from standin_eval.perspective_search import Camera,project,normalize_batch
from standin_eval.camera_refinement import normalized_joints,project_parameters


def audit(args):
    verify(args);rows=read(args.out/'rankings.json')
    assert sha256(args.out/'rankings.json')==read(args.out/'ranked.json')['sha256']
    poses=load_library(args.library_root)
    assert geometry_hash(poses)==read(args.source/'protocol.json')['hashes']['library_geometry']
    points=normalized_joints(np.stack([p.joints for p in poses]))
    old={r['image']:r for r in read(args.refinement_out/'rankings.json')}
    oldblend={r['image']:r for r in read(args.blend_out/'rankings.json')}
    manifest={r['image']:r for r in read(args.source/'manifest.json')}
    checked=failed=enabled=disabled=scalar=0;maxerror=0.
    for row in rows:
        name=row['image'];predpath=args.source/'predictions'/(Path(name).stem+'.json')
        assert sha256(predpath)==row['prediction_sha256']
        assert sha256(args.source/'images'/name)==manifest[name]['sha256']
        pred=read(predpath);query=valid_query(pred)
        if query is None:
            assert row.get('error')==old[name]['error'] and all(a not in row for a in WEIGHTS)
            failed+=1;continue
        xy,mask=query;q=normalize_batch(xy)
        assert hashlib.sha256(xy.tobytes()).hexdigest()==row['shared_xy_sha256']
        assert mask.tolist()==row['shared_mask']
        oldpath=args.refinement_out/'costs'/(Path(name).stem+'.npz')
        assert sha256(oldpath)==row['original_costs_sha256']==old[name]['costs_sha256']
        with np.load(oldpath,allow_pickle=False) as d:base,fit,params=[d[k] for k in ('baseline','refined','parameters')]
        path=args.out/'costs'/(Path(name).stem+'.npz');assert sha256(path)==row['costs_sha256']
        with np.load(path,allow_pickle=False) as d:error,penalty=d['head_error'],d['head_penalty']
        projections,valid=project_parameters(points,params);assert valid.all()
        expected=np.sqrt(((projections[:,0]-q[0])**2).sum(axis=-1))
        assert np.array_equal(expected,error)
        nose=float(pred['scores'][0]);on=np.isfinite(nose) and nose>=.7
        assert row['head_hint_enabled']==on and row['nose_score']==nose
        expected_penalty=np.minimum(1.,np.maximum(0.,expected.astype(float)-.25)) if on else np.zeros(len(poses))
        assert np.array_equal(expected_penalty,penalty)
        enabled+=int(on);disabled+=int(not on)
        score=.25*base.astype(float)+.75*fit.astype(float);first=int(np.argmin(base));selected=set(range(0,len(poses),600))
        for arm,w in ((BASE,0.),*WEIGHTS.items()):
            ids=np.argsort(score+w*penalty,kind='stable').tolist();ids=[first]+[i for i in ids if i!=first]
            assert row[arm]==[poses[i].pose_id for i in ids[:5]]
            assert row[arm][0]==row[BASE][0]
            if not on:assert row[arm]==oldblend[name][BASE]
            selected.update(ids[:5])
        assert row[BASE]==oldblend[name][BASE]
        for i in sorted(selected):
            yaw,pitch,inv=map(float,params[i]);px,v=project(poses[i].joints,Camera(yaw,pitch,None if inv==0 else 1/inv));assert v
            scalar_error=float(np.linalg.norm(normalize_batch(px)[0]-q[0]))
            delta=abs(scalar_error-float(error[i]));assert delta<3e-5
            maxerror=max(maxerror,delta);scalar+=1
        checked+=1
    assert len(rows)==checked+failed==360
    answers={r['image']:r for r in read(args.source/'answers.json')};reported=read(args.out/'results.json')
    counts={a:{g:{'top1':0,'top5':0} for g in GROUPS} for a in (BASE,*WEIGHTS)}
    changes={a:{g:{'gains':0,'losses':0} for g in GROUPS} for a in WEIGHTS}
    applied={g:0 for g in GROUPS};failures={g:0 for g in GROUPS};eligible_misses={g:0 for g in GROUPS}
    for row in rows:
        target=answers[row['image']];g=target['group'];pid=target['pose_id']
        applied[g]+=int(row.get('head_hint_enabled',False));failures[g]+=int(bool(row.get('error')))
        eligible_misses[g]+=int(row.get('head_hint_enabled',False) and pid not in row.get(BASE,[]))
        for arm in counts:
            ids=row.get(arm,[]);counts[arm][g]['top1']+=int(bool(ids) and ids[0]==pid);counts[arm][g]['top5']+=int(pid in ids)
            if arm!=BASE:
                before=pid in row.get(BASE,[]);after=pid in ids
                changes[arm][g]['gains']+=int(after and not before);changes[arm][g]['losses']+=int(before and not after)
    assert counts==reported['counts'] and changes==reported['top5_changes'] and applied==reported['applied'] and failures==reported['failures']
    choices=[a for a in WEIGHTS if counts[a][GROUPS[1]]['top5']>=counts[BASE][GROUPS[1]]['top5']+3 and
             all(counts[a][g][k]>=counts[BASE][g][k] for g in GROUPS for k in ('top1','top5'))]
    choices.sort(key=lambda a:(-counts[a][GROUPS[1]]['top5'],-sum(counts[a][g]['top5'] for g in GROUPS),WEIGHTS[a]))
    assert reported['eligible']==choices and reported['selected']==(choices[0] if choices else None)
    verify(args)
    result={'inputs_verified':360,'rankings_and_all_gallery_head_costs_checked':checked,'failures_retained':failed,
            'enabled':enabled,'disabled_and_exactly_unchanged':disabled,'independent_scalar_projections':scalar,
            'maximum_scalar_error':maxerror,'library_geometry_matches':True,'outcome_recomputed':True,
            'posthoc_enabled_subset_baseline_misses':eligible_misses,'ranking_sha256':sha256(args.out/'rankings.json'),
            'audit_script_sha256':sha256(Path(__file__))}
    save_json(args.out/'audit.json',result);print(result,flush=True)


if __name__=='__main__':audit(parse())
