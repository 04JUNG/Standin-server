"""Structural follow-up: shape features reorder near-tied existing Top5 only."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_joint_shape import read,GROUPS,verify as verify_shape
from scripts.experiment_z_rendered import sha256,save_json
from scripts.experiment_z_axis_search import load_library
from standin_eval.joint_ties import reorder_near_ties


def rank(a):
    verify_shape(type('Args',(),{'out':a.source})())
    a.out.mkdir(parents=True,exist_ok=False)
    inputs=[a.source/n for n in ('protocol.json','rankings.json','results.json')]
    inputs+=[a.answers]+sorted(a.costs.glob('*.npz'))
    save_json(a.out/'protocol.json',{
        'stage':'structural development follow-up after full feature weighting regressed; same examined120 families',
        'margin':.025,'weights':{'elbow':.15,'crossing':.15},'limit':5,
        'hypothesis':'joint evidence resolves near-tied candidates while keeping existing useful Top5 set',
        'fixed':'original observed XY, fitted cameras, all20 fair contours, all previous computed features',
        'gate':'strong Top1>=unlock+1, other2 Top1>=unlock; all3 Top5 exactly same; no source ID preferences used',
        'inputs':{str(p):sha256(p) for p in inputs},
        'code':{n:sha256(ROOT/n) for n in ('scripts/experiment_joint_ties.py','standin_eval/joint_shape.py','standin_eval/joint_ties.py')}})
    poses=load_library(a.library_root);rows=[]
    for r in read(a.source/'rankings.json'):
        row={'image':r['image']}
        if r.get('error'):row['error']=r['error'];rows.append(row);continue
        path=a.costs/(Path(r['image']).stem+'.npz');assert sha256(path)==r['costs_sha256']
        with np.load(path,allow_pickle=False) as d:base,fit=d['baseline'],d['refined']
        score=.25*base.astype(float)+.75*fit.astype(float); short=np.asarray(r['shortlist_indices'])
        contour=np.zeros(20) if r['contour_errors'] else np.asarray(r['contour_costs'])
        score[short]+=contour
        original=np.asarray(r['unlock_order20']); tail=np.array([j for j in np.argsort(score,kind='stable') if j not in set(short)])
        full=np.concatenate((original,tail)); shape=np.zeros(len(poses));shape[short]=.15*(np.asarray(r['angle_costs'])+np.asarray(r['cross_costs']))
        updated=reorder_near_ties(full,score,shape)
        row['unlock']=r['unlock'];row['joint_ties']=[poses[j].pose_id for j in updated[:5]]
        row['old_order5']=full[:5].tolist();row['new_order5']=updated[:5].tolist()
        rows.append(row)
    save_json(a.out/'rankings.json',rows);save_json(a.out/'ranked.json',{'sha256':sha256(a.out/'rankings.json'),'answers_parsed':False})


def evaluate(a):
    assert not (a.out/'results.json').exists()
    assert read(a.out/'ranked.json')['sha256']==sha256(a.out/'rankings.json')
    answers={r['image']:r for r in read(a.answers)}; counts={arm:{g:dict(top1=0,top5=0) for g in GROUPS} for arm in ('unlock','joint_ties')}
    for r in read(a.out/'rankings.json'):
        answer=answers[r['image']];target=answer['pose_id'];g=answer['group']
        for arm in counts:
            ids=r.get(arm,[]);counts[arm][g]['top1']+=bool(ids and ids[0]==target);counts[arm][g]['top5']+=target in ids
    gate=(counts['joint_ties'][GROUPS[1]]['top1']>=counts['unlock'][GROUPS[1]]['top1']+1
          and all(counts['joint_ties'][g]['top1']>=counts['unlock'][g]['top1'] and counts['joint_ties'][g]['top5']==counts['unlock'][g]['top5'] for g in GROUPS))
    result={'counts':counts,'eligible':gate,'stage':'development only; no fresh verification'}
    save_json(a.out/'results.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('rank','evaluate'))
    for n in ('source','costs','answers','library-root','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();globals()[a.stage](a)
