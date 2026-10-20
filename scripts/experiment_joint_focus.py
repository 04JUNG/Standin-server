"""One fixed positive-cue structural follow-up; no arbitrary weight sweep."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_joint_shape import read,GROUPS,verify as verify_shape
from scripts.experiment_z_rendered import sha256,save_json
from scripts.experiment_z_axis_search import load_library
from standin_eval.joint_focus import focused_costs
from standin_eval.joint_ties import reorder_near_ties
from standin_eval.perspective_search import normalize_batch


def run(a):
    verify_shape(type('Args',(),{'out':a.source})())
    a.out.mkdir(parents=True,exist_ok=False)
    inputs=[a.source/n for n in ('protocol.json','rankings.json','results.json')]+[a.answers]
    inputs+=sorted(a.costs.glob('*.npz'))+sorted(a.predictions.glob('*.json'))
    save_json(a.out/'protocol.json',{'stage':'known-development positive-cue follow-up; previous broad/tie feature policies regressed',
        'fixed_weights':.15,'tie_margin':.025,'limit':5,
        'focus':'only observed crossing=True; right-angle query70..110deg, all three confidence>=.7, both projected bones>=.15torso',
        'gate':'strongTop1>=unlock+1; other2Top1>=unlock; all3Top5 identical',
        'ranking':'input positive image cues, not preferred sourceID; first free among near-tied existingTop5',
        'inputs':{str(p):sha256(p) for p in inputs},
        'code':{p:sha256(ROOT/p) for p in ('scripts/experiment_joint_focus.py','standin_eval/joint_focus.py','standin_eval/joint_ties.py','standin_eval/joint_shape.py')}})
    poses=load_library(a.library_root);rows=[]
    for r in read(a.source/'rankings.json'):
        row={'image':r['image']}
        if r.get('error'):row['error']=r['error'];rows.append(row);continue
        path=a.costs/(Path(r['image']).stem+'.npz');assert sha256(path)==r['costs_sha256']
        with np.load(path,allow_pickle=False) as d:base,fit=d['baseline'],d['refined']
        pred=read(a.predictions/(Path(r['image']).stem+'.json'))
        xy=np.asarray(pred['keypoints2d'],np.float32);scores=np.asarray(pred['scores'],np.float32)
        angle,cross,meta=focused_costs(r['feature_metadata'],normalize_batch(xy),scores)
        score=.25*base.astype(float)+.75*fit.astype(float); short=np.asarray(r['shortlist_indices'])
        contour=np.zeros(20) if r['contour_errors'] else np.asarray(r['contour_costs']);score[short]+=contour
        tail=np.array([j for j in np.argsort(score,kind='stable') if j not in set(short)]);full=np.concatenate((r['unlock_order20'],tail))
        shape=np.zeros(len(poses));shape[short]=.15*(angle+cross)
        updated=reorder_near_ties(full,score,shape)
        row.update(unlock=r['unlock'],focused=[poses[j].pose_id for j in updated[:5]],metadata=meta,angle_costs=angle.tolist(),cross_costs=cross.tolist())
        rows.append(row)
    save_json(a.out/'rankings.json',rows);save_json(a.out/'ranked.json',{'sha256':sha256(a.out/'rankings.json'),'answers_parsed':False})
    # Answer-only evaluation occurs after all ranks are frozen.
    answers={r['image']:r for r in read(a.answers)}; counts={arm:{g:dict(top1=0,top5=0) for g in GROUPS} for arm in ('unlock','focused')}
    for r in rows:
        truth=answers[r['image']];g,target=truth['group'],truth['pose_id']
        for arm in counts:
            ids=r.get(arm,[]);counts[arm][g]['top1']+=bool(ids and ids[0]==target);counts[arm][g]['top5']+=target in ids
    gate=(counts['focused'][GROUPS[1]]['top1']>=counts['unlock'][GROUPS[1]]['top1']+1
        and all(counts['focused'][g]['top1']>=counts['unlock'][g]['top1'] and counts['focused'][g]['top5']==counts['unlock'][g]['top5'] for g in GROUPS))
    result={'counts':counts,'eligible':gate,'stage':'development only','arm_signal':sum(bool(r.get('metadata',{}).get('used_right_angle_sides')) for r in rows),
        'cross_signal':sum(r.get('metadata',{}).get('used_positive_crossing',False) for r in rows)}
    save_json(a.out/'results.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('source','costs','predictions','answers','library-root','out'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
