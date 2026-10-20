"""Fixed-camera, fixed-body development ablation of uncertain head positions."""
import argparse
import fcntl
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiment_camera_refinement import verify as verify_source,GROUPS
from scripts.experiment_near_camera import read
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import save_json,sha256
from scripts.experiment_refinement_blend import blend_cost
from scripts.diagnose_dual_camera_coverage import valid_query
from standin_eval.perspective_search import normalize_batch
from standin_eval.camera_refinement import normalized_joints
from standin_eval.head_hint import head_hint,NOSE_SCORE_MIN,HEAD_TOLERANCE,PENALTY_CAP

BASE='keepfirst_blend75'
WEIGHTS={'head005':.05,'head010':.1,'head020':.2}
CODE=('scripts/experiment_head_hint.py','standin_eval/head_hint.py',
      'standin_eval/camera_refinement.py','standin_eval/perspective_search.py',
      'scripts/experiment_refinement_blend.py')


def order(base,score):
    first=int(np.argmin(base));ordered=np.argsort(score,kind='stable')
    return np.concatenate(([first],ordered[ordered!=first]))


def select_policy(counts):
    eligible=[a for a in WEIGHTS if counts[a]['strong_perspective']['top5']>=counts[BASE]['strong_perspective']['top5']+3
              and all(counts[a][g][k]>=counts[BASE][g][k] for g in GROUPS for k in ('top1','top5'))]
    return sorted(eligible,key=lambda a:(-counts[a]['strong_perspective']['top5'],
                  -sum(counts[a][g]['top5'] for g in GROUPS),WEIGHTS[a]))


def verify(args):
    verify_source(SimpleNamespace(source=args.source,out=args.refinement_out,library_root=args.library_root,model=args.model))
    policy=read(args.out/'protocol.json')
    assert all(sha256(ROOT/p)==h for p,h in policy['frozen_sources'].items())
    assert all(sha256(Path(p))==h for p,h in policy['input_hashes'].items())


def rank(args):
    verify_source(SimpleNamespace(source=args.source,out=args.refinement_out,library_root=args.library_root,model=args.model))
    assert read(args.refinement_out/'audit.json')['ranking_sha256']==sha256(args.refinement_out/'rankings.json')
    assert read(args.blend_out/'audit.json')['ranking_sha256']==sha256(args.blend_out/'rankings.json')
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'costs').mkdir()
    paths=[args.source/'answers.json',args.source/'manifest.json',args.refinement_out/'rankings.json',
           args.refinement_out/'audit.json',args.blend_out/'rankings.json',args.blend_out/'audit.json',
           ROOT/'out/head-observation-diagnostic-20261013/results.json',ROOT/'out/head-observation-diagnostic-20261013/audit.json']
    paths+=sorted((args.source/'predictions').glob('*.json'))
    save_json(args.out/'protocol.json',{
        'stage':'development only; same120 examined families/360 images',
        'hypothesis':'head location disambiguates body-only fits without claiming face direction',
        'weights':WEIGHTS,'confidence_min':NOSE_SCORE_MIN,'tolerance':HEAD_TOLERANCE,'penalty_cap':PENALTY_CAP,
        'formula':'old keepfirst blend75 score + weight*min(1,max(0,head_distance-.25)); score<.7 disables hint',
        'fixed':'same observed Human-Art XY,body mask,normalization,all1981 gallery poses,previous fitted cameras,first117 result; no new camera fitting/queryZ',
        'selection':'strong Top5>=baseline+3; all3 Top1/Top5 nondecreasing; maximize strongTop5,totalTop5; tie smaller weight',
        'ranking_inputs':'predicted XY and nose score, gallery geometry and previous unlabelled costs only; no GT/group/source camera',
        'failure':'all3 existing extraction failures retained in strong denominator',
        'limits':'Head is an approximation, not precise nose or face direction; first result cannot improve; all queries already examined',
        'frozen_sources':{p:sha256(ROOT/p) for p in CODE},'input_hashes':{str(p):sha256(p) for p in paths}})
    poses=load_library(args.library_root);points=normalized_joints(np.stack([p.joints for p in poses]))
    originals={r['image']:r for r in read(args.blend_out/'rankings.json')}
    manifest={r['image']:r for r in read(args.source/'manifest.json')};rows=[]
    for old in read(args.refinement_out/'rankings.json'):
        name=old['image'];path=args.source/'predictions'/(Path(name).stem+'.json')
        assert sha256(path)==old['prediction_sha256'] and sha256(args.source/'images'/name)==manifest[name]['sha256']
        pred=read(path);query=valid_query(pred);row={'image':name,'prediction_sha256':sha256(path)}
        if query is None:
            assert old.get('error');row['error']=old['error'];rows.append(row);continue
        xy,mask=query;row['shared_xy_sha256']=hashlib.sha256(xy.tobytes()).hexdigest();row['shared_mask']=mask.tolist()
        assert row['shared_xy_sha256']==old['shared_xy_sha256'] and row['shared_mask']==old['shared_mask']
        path=args.refinement_out/'costs'/(Path(name).stem+'.npz');assert sha256(path)==old['costs_sha256']
        with np.load(path,allow_pickle=False) as data:base,fit,params=[data[k] for k in ('baseline','refined','parameters')]
        score=blend_cost(base,fit,.75);bo=order(base,score)
        row[BASE]=[poses[i].pose_id for i in bo[:5]];assert row[BASE]==originals[name][BASE]
        nose=float(pred['scores'][0]);error,penalty=head_hint(points,params,normalize_batch(xy),nose)
        row['nose_score']=nose;row['head_hint_enabled']=bool(np.isfinite(nose) and nose>=NOSE_SCORE_MIN)
        for arm,weight in WEIGHTS.items():
            indices=order(base,score+weight*penalty);row[arm]=[poses[i].pose_id for i in indices[:5]]
            if not row['head_hint_enabled']:assert row[arm]==row[BASE]
            assert row[arm][0]==row[BASE][0]
        dest=args.out/'costs'/(Path(name).stem+'.npz');np.savez_compressed(dest,head_error=error,head_penalty=penalty)
        row['costs_sha256']=sha256(dest);row['original_costs_sha256']=sha256(path);rows.append(row)
    assert len(rows)==360;verify(args)
    save_json(args.out/'rankings.json',rows);save_json(args.out/'ranked.json',{'sha256':sha256(args.out/'rankings.json'),'answers_parsed':False})


def evaluate(args):
    verify(args);assert not (args.out/'results.json').exists()
    assert sha256(args.out/'rankings.json')==read(args.out/'ranked.json')['sha256']
    rows=read(args.out/'rankings.json');answers={r['image']:r for r in read(args.source/'answers.json')}
    assert len(rows)==len(answers)==360 and {r['image'] for r in rows}==set(answers)
    counts={a:{g:{'top1':0,'top5':0} for g in GROUPS} for a in (BASE,*WEIGHTS)}
    applied={g:0 for g in GROUPS};failed={g:0 for g in GROUPS}
    changes={a:{g:{'gains':0,'losses':0} for g in GROUPS} for a in WEIGHTS}
    for row in rows:
        a=answers[row['image']];g=a['group'];target=a['pose_id']
        applied[g]+=row.get('head_hint_enabled',False);failed[g]+=bool(row.get('error'))
        for arm in counts:
            ids=row.get(arm,[]);counts[arm][g]['top1']+=bool(ids and ids[0]==target);counts[arm][g]['top5']+=target in ids
            if arm!=BASE:
                before=target in row.get(BASE,[]);after=target in ids
                changes[arm][g]['gains']+=after and not before;changes[arm][g]['losses']+=before and not after
    eligible=select_policy(counts)
    result={'stage':'development only','counts':counts,'applied':applied,'failures':failed,'top5_changes':changes,
            'eligible':eligible,'selected':eligible[0] if eligible else None}
    save_json(args.out/'results.json',result);print(result,flush=True)


def parse():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('rank','evaluate'))
    for name in ('source','refinement-out','blend-out','out','library-root','model'):p.add_argument('--'+name,type=Path,required=True)
    return p.parse_args()


if __name__=='__main__':
    args=parse();args.out.parent.mkdir(parents=True,exist_ok=True)
    with (args.out.parent/'.head-hint-dev.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);globals()[args.stage](args)
