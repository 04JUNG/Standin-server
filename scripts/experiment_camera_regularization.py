"""Precommitted development test of weak camera-distance regularization."""
import argparse
import fcntl
import hashlib
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiment_camera_refinement import verify as verify_source, GROUPS
from scripts.experiment_refinement_blend import blend_cost
from scripts.experiment_near_camera import read, near_cameras
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import save_json,sha256
from scripts.confirm_camera_fit import geometry_hash
from scripts.diagnose_dual_camera_coverage import valid_query
from standin_eval.perspective_search import camera_grid,build_camera_index,normalize_batch
from standin_eval.camera_refinement import normalized_joints
from standin_eval.camera_regularization import regularized_costs

WEIGHTS={'reg005':.005,'reg020':.02,'reg050':.05}
BASE='keepfirst_blend75'
CODE=('scripts/experiment_camera_regularization.py','standin_eval/camera_regularization.py',
      'standin_eval/camera_refinement.py','standin_eval/perspective_search.py',
      'scripts/experiment_refinement_blend.py','standin_eval/dual_label_search.py')


def order(base,cost):
    first=int(np.argmin(base))
    ordered=np.argsort(blend_cost(base,cost,.75),kind='stable')
    return np.concatenate(([first],ordered[ordered!=first]))


def select_policy(counts,energies):
    eligible=[arm for arm in WEIGHTS if
        all(counts[arm][g][m]>=counts[BASE][g][m] for g in GROUPS for m in ('top1','top5'))
        and energies[arm] <= .9*energies[BASE]]
    return sorted(eligible,key=lambda a:(-counts[a]['strong_perspective']['top5'],
        -sum(counts[a][g]['top5'] for g in GROUPS),energies[a],WEIGHTS[a]))


def verify(args):
    verify_source(SimpleNamespace(out=args.refinement_out,source=args.source,
                                   library_root=args.library_root,model=args.model))
    policy=read(args.out/'protocol.json')
    assert {p:sha256(ROOT/p) for p in CODE}==policy['frozen_sources']
    assert all(sha256(Path(p))==h for p,h in policy['input_hashes'].items())


def rank(args):
    verify_source(SimpleNamespace(out=args.refinement_out,source=args.source,
                                   library_root=args.library_root,model=args.model))
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'costs').mkdir()
    paths=[args.source/'answers.json',args.source/'manifest.json',args.blend_out/'rankings.json',
           args.blend_out/'audit.json',args.refinement_out/'rankings.json',args.refinement_out/'audit.json']
    paths+=sorted((args.source/'predictions').glob('*.json'))
    save_json(args.out/'protocol.json',{
        'stage':'development only; same previously examined120 families/360 images; no new inference',
        'hypothesis':'weak distance prior reduces near-camera overfitting without losing Top5 retrieval',
        'weights':WEIGHTS,'formula':'fitted = min_cameras(XY_body_cost + weight*inverse_distance^2); score=.25*base117+.75*fitted',
        'fixed':'Human-Art XY/mask,dual labels,all1981 poses,3 regularized coarse seeds per label,4 scales and2 sweeps,bounds unchanged; first117 pose preserved',
        'selection':'all3 groups Top1 and Top5 >= keepfirst_blend75; sum inverse_distance^2 at ranks2..5 <=90% baseline; then maximize strongTop5,totalTop5,minimize energy,weight',
        'energy_warning':'camera proximity proxy, NOT human quality or anatomy; excludes unchanged first result',
        'ranking_inputs':'image-derived XY and gallery only; no group,answerID,GT camera or XY; answers hashed only',
        'failure':'all extraction failures retained in denominator',
        'frozen_sources':{p:sha256(ROOT/p) for p in CODE},
        'input_hashes':{str(p):sha256(p) for p in paths}})
    poses=load_library(args.library_root)
    assert geometry_hash(poses)==read(args.source/'protocol.json')['hashes']['library_geometry']
    cams=camera_grid()['yaw_pitch_perspective_117']+near_cameras()
    index=build_camera_index(poses,cams)
    points=normalized_joints(np.stack([p.joints for p in poses]))
    originals={r['image']:r for r in read(args.blend_out/'rankings.json')}
    manifest={r['image']:r for r in read(args.source/'manifest.json')}
    rows=[]
    for source in read(args.refinement_out/'rankings.json'):
        name=source['image'];path=args.source/'predictions'/(Path(name).stem+'.json')
        assert sha256(path)==source['prediction_sha256']
        assert sha256(args.source/'images'/name)==manifest[name]['sha256']
        query=valid_query(read(path));row={'image':name,'prediction_sha256':sha256(path)}
        if query is None:
            assert source.get('error');row['error']=source['error'];rows.append(row);continue
        xy,mask=query;q=normalize_batch(xy)
        row['shared_xy_sha256']=hashlib.sha256(xy.tobytes()).hexdigest()
        row['shared_mask']=mask.tolist()
        assert row['shared_xy_sha256']==source['shared_xy_sha256'] and row['shared_mask']==source['shared_mask']
        oldpath=args.refinement_out/'costs'/(Path(name).stem+'.npz')
        assert sha256(oldpath)==source['costs_sha256']
        with np.load(oldpath,allow_pickle=False) as data:
            base,fitted,params=[data[k] for k in ('baseline','refined','parameters')]
        bo=order(base,fitted)[:5]
        row[BASE]=[poses[i].pose_id for i in bo]
        assert row[BASE]==originals[name][BASE]
        row['inverse_distance_squared']={BASE:float(np.sum(params[bo[1:],2].astype(float)**2))}
        output={'baseline':base};start=time.perf_counter()
        for arm,weight in WEIGHTS.items():
            cost,rp=regularized_costs(points,index,cams,q,mask,weight)
            selected=order(base,cost)[:5]
            assert selected[0]==bo[0]
            row[arm]=[poses[i].pose_id for i in selected]
            row['inverse_distance_squared'][arm]=float(np.sum(rp[selected[1:],2].astype(float)**2))
            output[arm+'_cost']=cost;output[arm+'_parameters']=rp
        row['fit_seconds_all_three']=time.perf_counter()-start
        dest=args.out/'costs'/(Path(name).stem+'.npz');np.savez_compressed(dest,**output)
        row['costs_sha256']=sha256(dest);rows.append(row)
        if len(rows)%30==0:print('regularized',len(rows),'/360',flush=True)
    assert len(rows)==360
    verify(args);save_json(args.out/'rankings.json',rows)
    save_json(args.out/'ranked.json',{'sha256':sha256(args.out/'rankings.json'),'answers_parsed':False})


def evaluate(args):
    verify(args);assert not (args.out/'results.json').exists()
    assert sha256(args.out/'rankings.json')==read(args.out/'ranked.json')['sha256']
    rows=read(args.out/'rankings.json');answers={a['image']:a for a in read(args.source/'answers.json')}
    assert len(rows)==len(answers)==360 and {r['image'] for r in rows}==set(answers)
    arms=(BASE,*WEIGHTS);counts={a:{g:{'top1':0,'top5':0} for g in GROUPS} for a in arms}
    energy={a:0. for a in arms};failures={g:0 for g in GROUPS}
    for row in rows:
        answer=answers[row['image']];g=answer['group'];target=answer['pose_id']
        failures[g]+=bool(row.get('error'))
        for arm in arms:
            ids=row.get(arm,[])
            counts[arm][g]['top1']+=bool(ids and ids[0]==target)
            counts[arm][g]['top5']+=target in ids
            energy[arm]+=row.get('inverse_distance_squared',{}).get(arm,0)
    eligible=select_policy(counts,energy)
    result={'stage':'development only','counts':counts,'failures':failures,
            'sum_inverse_distance_squared_ranks2to5':energy,
            'energy_ratio_to_baseline':{a:energy[a]/energy[BASE] for a in arms},
            'eligible':eligible,'selected':eligible[0] if eligible else None}
    save_json(args.out/'results.json',result);print(result,flush=True)


def parse():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('rank','evaluate'))
    for name in ('source','refinement-out','blend-out','out','library-root','model'):
        p.add_argument('--'+name,type=Path,required=True)
    return p.parse_args()


if __name__=='__main__':
    args=parse();args.out.parent.mkdir(parents=True,exist_ok=True)
    with (args.out.parent/'.camera-regularization-dev.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);globals()[args.stage](args)
