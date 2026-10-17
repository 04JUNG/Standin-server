"""Development-only fixed score regularization grid after camera-fit diagnosis."""
import argparse
import fcntl
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiment_camera_refinement import verify as verify_refinement,GROUPS
from scripts.experiment_near_camera import read
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import save_json,sha256
from standin_eval.dual_label_search import merged_order

ALPHAS=(.25,.5,.75)
POLICIES={('keepfirst_' if keep else '')+f'blend{int(a*100)}':(a,keep)
          for a in ALPHAS for keep in (False,True)}
BASELINES=('pooled_117','keepfirst_195')


def blend_cost(base,fitted,alpha):
    base=np.asarray(base,dtype=float);fitted=np.asarray(fitted,dtype=float)
    if (base.ndim!=1 or base.shape!=fitted.shape or not 0<=alpha<=1
        or not np.isfinite(base).all() or not np.isfinite(fitted).all()
        or np.any(base<0) or np.any(fitted<0)):
        raise ValueError('finite nonnegative same-length costs, alpha in[0,1] required')
    return (1-alpha)*base+alpha*fitted


def select_policy(counts):
    eligible=[]
    for arm in POLICIES:
        if (counts[arm]['strong_perspective']['top5']>=counts['keepfirst_195']['strong_perspective']['top5']+6
            and all(counts[arm][g]['top1']>=counts['pooled_117'][g]['top1'] for g in GROUPS)
            and all(counts[arm][g]['top5']>=max(counts[b][g]['top5'] for b in BASELINES) for g in GROUPS[:1]+GROUPS[2:])):
            eligible.append(arm)
    eligible.sort(key=lambda a:(counts[a]['strong_perspective']['top5'],sum(counts[a][g]['top5'] for g in GROUPS),
        sum(counts[a][g]['top1'] for g in GROUPS),-POLICIES[a][0],POLICIES[a][1]),reverse=True)
    return eligible


def verify(args):
    verify_refinement(SimpleNamespace(out=args.refinement_out,source=args.source,library_root=args.library_root,model=args.model))
    policy=read(args.out/'protocol.json')
    assert sha256(Path(__file__))==policy['script_sha256']
    for path,digest in policy['input_hashes'].items():
        assert sha256(Path(path))==digest


def rank(args):
    source=SimpleNamespace(out=args.refinement_out,source=args.source,library_root=args.library_root,model=args.model)
    verify_refinement(source)
    assert read(args.refinement_out/'audit.json')['ranking_sha256']==sha256(args.refinement_out/'rankings.json')
    args.out.mkdir(parents=True,exist_ok=False)
    paths=[args.refinement_out/f for f in ('protocol.json','rankings.json','results.json','audit.json')]
    paths+=[args.source/'answers.json']
    save_json(args.out/'protocol.json',{
        'stage':'development only, same examined120 families/360 images, after unregularized fit failed selection',
        'hypothesis':'unrestricted camera fit favors wrong flexible candidates; retain fixed117 geometric evidence',
        'formula':'score=(1-alpha)*pooled117_cost+alpha*refined_cost, same torso-distance units',
        'policies':POLICIES,'no_routing':'single global alpha and optional pooled117 first result; no group labels/GT/camera routing',
        'selection':'strong Top5 >=keepfirst195+6; all3 Top1>=pooled117; moderate/control Top5>=both baselines; maximize strongTop5,totalTop5,totalTop1,then smaller alpha,then keepfirst',
        'fixed':'existing XY/mask,1981 gallery and camera-refinement results; no additional fitting or inference',
        'input_hashes':{str(p):sha256(p) for p in paths},'script_sha256':sha256(Path(__file__))})
    poses=load_library(args.library_root)
    rows=[]
    for source_row in read(args.refinement_out/'rankings.json'):
        row={k:source_row[k] for k in ('image','prediction_sha256')}
        if source_row.get('error'):
            row['error']=source_row['error'];rows.append(row);continue
        path=args.refinement_out/'costs'/(Path(row['image']).stem+'.npz')
        assert sha256(path)==source_row['costs_sha256']
        row['costs_sha256']=source_row['costs_sha256']
        row['shared_xy_sha256']=source_row['shared_xy_sha256']
        row['shared_mask']=source_row['shared_mask']
        with np.load(path,allow_pickle=False) as data:
            base,fitted=data['baseline'],data['refined']
        for arm,(alpha,keep) in POLICIES.items():
            score=blend_cost(base,fitted,alpha)
            # preserve_first uses baseline argmin, pooled costs must remain score.
            order=merged_order(score,score)
            if keep:
                first=int(np.argmin(base))
                order=np.concatenate(([first],order[order!=first]))
            row[arm]=[poses[int(i)].pose_id for i in order[:5]]
        for arm in BASELINES:
            row[arm]=source_row[arm]
        rows.append(row)
    verify(args)
    save_json(args.out/'rankings.json',rows)
    save_json(args.out/'ranked.json',{'sha256':sha256(args.out/'rankings.json'),'answers_parsed':False})


def evaluate(args):
    verify(args)
    assert not (args.out/'results.json').exists()
    assert sha256(args.out/'rankings.json')==read(args.out/'ranked.json')['sha256']
    rows=read(args.out/'rankings.json')
    answers={r['image']:r for r in read(args.source/'answers.json')}
    assert len(rows)==len(answers)==len({r['image'] for r in rows})==360
    assert {r['image'] for r in rows}==set(answers)
    counts={a:{g:{'top1':0,'top5':0} for g in GROUPS} for a in (*BASELINES,*POLICIES)}
    for row in rows:
        target=answers[row['image']]
        for arm in counts:
            ids=row.get(arm,[])
            counts[arm][target['group']]['top1']+=bool(ids and ids[0]==target['pose_id'])
            counts[arm][target['group']]['top5']+=target['pose_id'] in ids
    eligible=select_policy(counts)
    result={'stage':'development only, six fixed alternatives','counts':counts,'eligible':eligible,'selected':eligible[0] if eligible else None}
    save_json(args.out/'results.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('rank','evaluate'))
    for name in ('source','refinement-out','out','library-root','model'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.out.parent.mkdir(parents=True,exist_ok=True)
    with (args.out.parent/'.refinement-blend-dev.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);globals()[args.stage](args)
