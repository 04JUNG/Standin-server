"""Known-development region-preserving first-place ablation, cached XY only."""
import argparse
import fcntl
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_z_rendered import save_json,sha256
from standin_eval.camera_refinement import normalized_joints,project_parameters
from standin_eval.perspective_search import normalize_batch
from standin_eval.body_part_evidence import body_part_errors
from standin_eval.joint_part_guard import guarded_first

GROUPS=('perspective','strong_perspective','orthographic_control')


def read(p):return json.loads(p.read_text())


def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    files=[a.gallery,a.source/'protocol.json',a.source/'rankings.json',a.source/'results.json',a.answers]
    files+=sorted(a.costs.glob('*.npz'))+sorted(a.predictions.glob('*.json'))
    policy={'stage':'examined120 families and360 views, structural development follow-up, NOT fresh holdout',
        'hypothesis':'first candidate can change for angle/cross evidence only if arms,legs,torso XY each do not worsen vs fair-contour first',
        'fixed':'same Human-Art XY/mask, historical1981 gallery, camera parameters, fair contour scores and Top5 membership',
        'regions':'single whole-body direct/swap mapping per candidate; each mean observed-joint L2; epsilon1e-6 solely float tolerance',
        'feature_weights':{'elbow':.15,'leg_cross':.15},'shortlist':5,
        'gate':'strong Top1>=unlock+3; other2 Top1>=unlock; all3 Top5 exactly identical; otherwise no adoption or fresh test',
        'failures':'original extraction failures retained in120 denominators; contours reused with original recorded fallback',
        'inputs':{str(p):sha256(p) for p in files},
        'code':{n:sha256(ROOT/n) for n in ('scripts/experiment_joint_part_guard.py','standin_eval/joint_part_guard.py','standin_eval/body_part_evidence.py')},
        'limits':['2D region non-regression is not human relevance or true3D correctness','does not solve input extraction failures or support/furniture contact']}
    save_json(a.out/'protocol.json',policy)
    with np.load(a.gallery,allow_pickle=False) as d:ids=d['pose_ids'].tolist();points=normalized_joints(d['joints'])
    assert len(ids)==1981
    rows=[]
    for r in read(a.source/'rankings.json'):
        row={'image':r['image']}
        if r.get('error'):row['error']=r['error'];rows.append(row);continue
        predpath=a.predictions/(Path(r['image']).stem+'.json');assert sha256(predpath)==r['prediction_sha256']
        pred=read(predpath);xy=np.asarray(pred['keypoints2d'],np.float32);mask=np.asarray(pred['scores'],np.float32)>=.3
        assert hashlib.sha256(xy.tobytes()).hexdigest()==r['shared_xy_sha256'] and mask.tolist()==r['shared_mask']
        cp=a.costs/(Path(r['image']).stem+'.npz');assert sha256(cp)==r['costs_sha256']
        with np.load(cp,allow_pickle=False) as d:base,fit,params=[d[k] for k in ('baseline','refined','parameters')]
        short=np.asarray(r['shortlist_indices']);assert [ids[j] for j in short]==r['shortlist']
        costs=.25*base.astype(float)+.75*fit.astype(float)
        costs[short]+=np.zeros(20) if r['contour_errors'] else np.asarray(r['contour_costs'])
        shape=np.zeros(len(ids));shape[short]=.15*(np.asarray(r['angle_costs'])+np.asarray(r['cross_costs']))
        order=np.asarray(r['unlock_order20']);assert [ids[j] for j in order[:5]]==r['unlock']
        projected,valid=project_parameters(points[order[:5]],params[order[:5]]);assert valid.all()
        regions=[body_part_errors(normalize_batch(xy),mask,p) for p in projected]
        new,meta=guarded_first(order,costs,shape,regions)
        row.update(unlock=r['unlock'],guarded=[ids[j] for j in new[:5]],guard=meta,
                   original_order20=order.tolist(),guarded_order20=new.tolist())
        rows.append(row)
    assert len(rows)==360
    save_json(a.out/'rankings.json',rows)
    save_json(a.out/'ranked.json',{'ranking_sha256':sha256(a.out/'rankings.json'),'answers_parsed':False})
    # Only after every rank is written do answer IDs enter evaluation.
    answers={r['image']:r for r in read(a.answers)}
    counts={arm:{g:{'top1':0,'top5':0} for g in GROUPS} for arm in ('unlock','guarded')}
    for r in rows:
        answer=answers[r['image']];group=answer['group'];target=answer['pose_id']
        for arm in counts:
            hit=r.get(arm,[]);counts[arm][group]['top1']+=bool(hit and hit[0]==target);counts[arm][group]['top5']+=target in hit
    eligible=(counts['guarded'][GROUPS[1]]['top1']>=counts['unlock'][GROUPS[1]]['top1']+3
              and all(counts['guarded'][g]['top1']>=counts['unlock'][g]['top1'] and counts['guarded'][g]['top5']==counts['unlock'][g]['top5'] for g in GROUPS))
    result={'counts':counts,'eligible_for_fresh_confirmation':eligible,'failures':sum(bool(r.get('error')) for r in rows),
            'first_changed':sum(r.get('guard',{}).get('winner_previous_position',0)>0 for r in rows),
            'queries_with_alternative_dominating_region_candidate':sum(len(r.get('guard',{}).get('eligible_top5_positions',[]))>1 for r in rows),
            'ranking_sha256':sha256(a.out/'rankings.json'),'stage':'development only'}
    save_json(a.out/'results.json',result)
    for p,h in policy['inputs'].items():assert sha256(Path(p))==h
    for n,h in policy['code'].items():assert sha256(ROOT/n)==h
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('gallery','source','costs','predictions','answers','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    with (ROOT/'out/.perspective-rough-review.lock').open('a+') as rough_lock,(ROOT/'out/.joint-shape.lock').open('a+') as shape_lock,(ROOT/'out/.first-rank-confirm.lock').open('a+') as fresh_lock:
        for lock in (rough_lock,shape_lock,fresh_lock):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(a)
