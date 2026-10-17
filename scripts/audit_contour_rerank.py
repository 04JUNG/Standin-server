"""Audit frozen candidates/ranks and independently check boundary distances."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiment_contour_rerank import verify,BASE,WEIGHTS,GROUPS
from scripts.experiment_z_rendered import sha256,save_json
from scripts.experiment_near_camera import read
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_humanart_z import read_rgb
from scripts.diagnose_dual_camera_coverage import valid_query
from standin_eval.contour_rerank import prepare_query,candidate_evidence


def brute_cost(a,b,torso):
    # All pairs, chunked; no OpenCV distance transform or scoring helper.
    first=np.argwhere(a);second=np.argwhere(b)
    def directional(x,y):
        total=0.
        for start in range(0,len(x),128):
            d=np.sqrt(np.square(x[start:start+128,None].astype(float)-y[None]).sum(2)).min(1)/torso
            total+=np.minimum(d,.5).sum()
        return total/len(x)
    return float((directional(first,second)+directional(second,first))/2)


def main(args):
    verify(args)
    assert sha256(args.out/'rankings.json')==read(args.out/'ranked.json')['sha256']
    rows=read(args.out/'rankings.json');result=read(args.out/'results.json')
    answers={r['image']:r for r in read(args.source/'answers.json')}
    originals={r['image']:r for r in read(args.blend_out/'rankings.json')}
    refined={r['image']:r for r in read(args.refinement_out/'rankings.json')}
    manifest={r['image']:r for r in read(args.source/'manifest.json')}
    assert len(rows)==len(answers)==360 and len({r['image'] for r in rows})==360
    poses=load_library(args.library_root);full=0;fallback=0;scalar=0;max_delta=0.;replays=0;errors={}
    for row in rows:
        name=row['image'];truth=answers[name];old=refined[name]
        predpath=args.source/'predictions'/(Path(name).stem+'.json')
        assert row['prediction_sha256']==sha256(predpath)==old['prediction_sha256']
        assert sha256(args.source/'images'/name)==manifest[name]['sha256']
        query=valid_query(read(predpath))
        if row.get('error'):
            assert query is None and old.get('error');continue
        xy,mask=query
        assert hashlib.sha256(xy.tobytes()).hexdigest()==row['shared_xy_sha256']==old['shared_xy_sha256']
        assert mask.tolist()==row['shared_mask']==old['shared_mask']
        path=args.refinement_out/'costs'/(Path(name).stem+'.npz')
        assert sha256(path)==row['original_costs_sha256']==old['costs_sha256']
        with np.load(path,allow_pickle=False) as data:base,fit,parameters=[data[k] for k in ('baseline','refined','parameters')]
        scores=(1-.75)*base.astype(float)+.75*fit.astype(float)
        first=int(np.argmin(base));ordered=sorted(range(len(poses)),key=lambda i:(scores[i],i))
        ordered=[first]+[i for i in ordered if i!=first]
        assert row['shortlist_indices']==ordered[:20]
        assert row['shortlist']==[poses[i].pose_id for i in ordered[:20]]
        assert row[BASE]==originals[name][BASE]==[poses[i].pose_id for i in ordered[:5]]
        assert row['fallback']==bool(row['candidate_errors'] or row.get('query_error'))
        for error in row['candidate_errors'].values():errors[error]=errors.get(error,0)+1
        assert len(row['contour_costs'])==20 and row['contour_costs'][0]==0
        assert all(0<=c<=.5 for c in row['contour_costs'])
        for arm,w in WEIGHTS.items():
            candidate=list(ordered)
            if not row['fallback']:
                head=sorted(range(1,20),key=lambda k:(scores[ordered[k]]+w*row['contour_costs'][k],k))
                candidate=[first]+[ordered[k] for k in head]+ordered[20:]
            assert candidate[0]==first and candidate[20:]==ordered[20:]
            assert row[arm]==[poses[i].pose_id for i in candidate[:5]]
        fallback+=row['fallback'];full+=1
        maskpath=args.out/'masks'/(Path(name).stem+'.npz')
        if maskpath.exists():
            query=prepare_query(read_rgb(args.source/'images'/name),xy)
            assert query['torso']==row['query_torso'] and query['mask'].sum()==row['query_line_pixels']
            with np.load(maskpath,allow_pickle=False) as masks:
                assert np.array_equal(query['mask'],masks['query'])
                for key in masks.files:
                    if key=='query':continue
                    k=int(key.split('_')[1]);cost=brute_cost(masks['query'],masks[key],row['query_torso'])
                    delta=abs(cost-row['contour_costs'][k]);max_delta=max(max_delta,delta);assert delta<2e-7
                    scalar+=1
                    if k in (1,19):
                        replay,affine,pixels=candidate_evidence(query,poses[ordered[k]].joints,parameters[ordered[k]])
                        assert replay==row['contour_costs'][k] and np.array_equal(pixels,masks[key])
                        assert affine.tolist()==row['affines'][str(k)];replays+=1
    expected={arm:{g:dict(top1=0,top5=0) for g in GROUPS} for arm in (BASE,*WEIGHTS)}
    changes={arm:{g:dict(gains=0,losses=0) for g in GROUPS} for arm in WEIGHTS}
    failures={g:0 for g in GROUPS};fallbacks={g:0 for g in GROUPS};ceiling={g:0 for g in GROUPS}
    for row in rows:
        a=answers[row['image']];g=a['group'];target=a['pose_id']
        failures[g]+=bool(row.get('error'));fallbacks[g]+=bool(row.get('fallback'));ceiling[g]+=target in row.get('shortlist',[])
        for arm in expected:
            ids=row.get(arm,[]);expected[arm][g]['top1']+=bool(ids and ids[0]==target);expected[arm][g]['top5']+=target in ids
            if arm!=BASE:
                before=target in row.get(BASE,[]);after=target in ids
                changes[arm][g]['gains']+=after and not before;changes[arm][g]['losses']+=before and not after
    assert expected==result['counts'] and changes==result['changes']
    assert failures==result['failures'] and fallbacks==result['fallback_images'] and ceiling==result['target_in_original20']
    eligible=[arm for arm in WEIGHTS if expected[arm]['strong_perspective']['top5']>=expected[BASE]['strong_perspective']['top5']+3
              and all(expected[arm][g][k]>=expected[BASE][g][k] for g in GROUPS for k in ('top1','top5'))]
    eligible.sort(key=lambda a:(-expected[a]['strong_perspective']['top5'],-sum(expected[a][g]['top5'] for g in GROUPS),WEIGHTS[a]))
    assert eligible==result['eligible'] and (eligible[0] if eligible else None)==result['selected']
    output={'inputs':len(rows),'full_gallery_baseline_and_shortlist_ranks':full,'fallback_images':fallback,
            'bruteforce_boundary_costs':scalar,'max_boundary_cost_difference':max_delta,'render_replays':replays,
            'candidate_errors':errors,'ranking_sha256':sha256(args.out/'rankings.json'),
            'results_sha256':sha256(args.out/'results.json'),'auditor_sha256':sha256(Path(__file__))}
    save_json(args.out/'audit.json',output);print(json.dumps(output,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','refinement-out','blend-out','out','library-root','model'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
