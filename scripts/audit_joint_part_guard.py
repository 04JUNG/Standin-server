"""Independent region aggregation and guarded-ranking replay; no new experiment."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_z_rendered import sha256,save_json
from standin_eval.camera_refinement import normalized_joints,project_parameters
from standin_eval.perspective_search import normalize_batch
from standin_eval.dual_label_search import LEFT_RIGHT


def read(p):return json.loads(p.read_text())


def audit(a):
    policy=read(a.study/'protocol.json')
    for p,h in policy['inputs'].items():assert sha256(Path(p))==h
    for n,h in policy['code'].items():assert sha256(ROOT/n)==h
    assert read(a.study/'ranked.json')['ranking_sha256']==sha256(a.study/'rankings.json')
    with np.load(a.gallery,allow_pickle=False) as d:ids=d['pose_ids'].tolist();points=normalized_joints(d['joints'])
    source={r['image']:r for r in read(a.source/'rankings.json')};counts={arm:{g:{'top1':0,'top5':0} for g in ('perspective','strong_perspective','orthographic_control')} for arm in ('unlock','guarded')}
    answers={r['image']:r for r in read(a.answers)};checked=0
    for row in read(a.study/'rankings.json'):
        old=source[row['image']]
        if row.get('error'):assert row['error']==old['error']
        else:
            pred=read(a.predictions/(Path(row['image']).stem+'.json'));q=normalize_batch(pred['keypoints2d']).astype(float);mask=np.asarray(pred['scores'],np.float32)>=.3
            with np.load(a.costs/(Path(row['image']).stem+'.npz'),allow_pickle=False) as d:base,fit,params=[d[k] for k in ('baseline','refined','parameters')]
            short=np.asarray(old['shortlist_indices']);order=np.asarray(old['unlock_order20']);head=order[:5]
            xy,valid=project_parameters(points[head],params[head]);assert valid.all();xy=xy.astype(float)
            body=np.flatnonzero(mask[5:])+5;regions=[]
            for p in xy:
                other=p[LEFT_RIGHT]
                direct=np.sqrt(((p[body]-q[body])**2).sum(-1)).mean();swapped=np.sqrt(((other[body]-q[body])**2).sum(-1)).mean()
                aligned=other if swapped<direct else p
                values=[]
                for part in ((5,6,7,8,9,10),(11,12,13,14,15,16),(5,6,11,12)):
                    selected=[j for j in part if mask[j]]
                    values.append(np.sqrt(((aligned[selected]-q[selected])**2).sum(-1)).mean())
                regions.append(values)
            regions=np.asarray(regions);assert np.allclose(regions,row['guard']['regions'],atol=1e-10)
            eligible=[i for i in range(5) if all(regions[i,k]<=regions[0,k]+1e-6 for k in range(3))]
            assert eligible==row['guard']['eligible_top5_positions']
            cost=.25*base.astype(float)+.75*fit.astype(float);cost[short]+=np.zeros(20) if old['contour_errors'] else old['contour_costs']
            additions=np.zeros(len(ids));additions[short]=.15*(np.asarray(old['angle_costs'])+np.asarray(old['cross_costs']))
            winner=min(eligible,key=lambda i:cost[head[i]]+additions[head[i]])
            expected=[int(head[winner])]+[int(j) for i,j in enumerate(head) if i!=winner]+order[5:].tolist()
            assert expected==row['guarded_order20'] and winner==row['guard']['winner_previous_position']
            assert [ids[j] for j in expected[:5]]==row['guarded'] and row['unlock']==old['unlock']
            assert set(row['guarded'])==set(row['unlock']) and np.all(regions[winner]<=regions[0]+1e-6)
            checked+=1
        truth=answers[row['image']];group=truth['group'];target=truth['pose_id']
        for arm in counts:
            hits=row.get(arm,[]);counts[arm][group]['top1']+=bool(hits and hits[0]==target);counts[arm][group]['top5']+=target in hits
    assert counts==read(a.study/'results.json')['counts']
    save_json(a.study/'audit.json',{'ranking_sha256':sha256(a.study/'rankings.json'),'results_sha256':sha256(a.study/'results.json'),
        'valid_queries_replayed':checked,'failures_retained':360-checked,'top5_membership_preserved':True,
        'selected_regions_never_worsen':True,'counts_recomputed':True,'scope':'development data only; no fresh accuracy validation'})
    print('PASS',checked,'region/constraint/ranking replays; denominators360 and source hashes intact',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('study','gallery','source','costs','predictions','answers'):p.add_argument('--'+n,type=Path,required=True)
    audit(p.parse_args())
