"""Factorial fixed-camera study of model confidence and rotation consistency."""
import argparse
import fcntl
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiment_confidence_cost import verify as verify_confidence,read,BASE,GROUPS
from scripts.diagnose_rotation_consistency import verify as verify_rotation
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import save_json,sha256
from scripts.experiment_refinement_blend import blend_cost
from scripts.experiment_head_hint import order
from scripts.confirm_camera_fit import geometry_hash
from scripts.diagnose_dual_camera_coverage import valid_query
from standin_eval.perspective_search import normalize_batch
from standin_eval.camera_refinement import normalized_joints,project_parameters
from standin_eval.rotation_consistency import agreement
from standin_eval.confidence_cost import joint_weights
from standin_eval.consistency_cost import consistency_weights,fixed_weighted_cost

CONFIDENCE='confidence100';CONSISTENCY='consistency_half';COMBINED='confidence_consistency'
ARMS=(BASE,CONFIDENCE,CONSISTENCY,COMBINED)
CODE=('scripts/experiment_consistency_cost.py','standin_eval/consistency_cost.py',
      'standin_eval/rotation_consistency.py','standin_eval/confidence_cost.py',
      'standin_eval/camera_refinement.py','standin_eval/dual_label_search.py',
      'scripts/experiment_refinement_blend.py','scripts/experiment_head_hint.py')


def verify_inputs(args):
    verify_confidence(SimpleNamespace(source=args.source,refinement_out=args.refinement_out,
        blend_out=args.blend_out,out=args.confidence_out,library_root=args.library_root,model=args.model))
    verify_rotation(SimpleNamespace(out=args.rotation_out))


def verify(args):
    verify_inputs(args);p=read(args.out/'protocol.json')
    assert all(sha256(ROOT/f)==h for f,h in p['frozen_sources'].items())
    assert all(sha256(Path(f))==h for f,h in p['input_hashes'].items())


def select_policy(counts):
    choices=[a for a in ARMS[2:] if counts[a]['strong_perspective']['top5']>=counts[BASE]['strong_perspective']['top5']+3
             and counts[a]['strong_perspective']['top5']>counts[CONFIDENCE]['strong_perspective']['top5']
             and all(counts[a][g][k]>=max(counts[b][g][k] for b in ARMS[:2]) for g in GROUPS for k in ('top1','top5'))]
    return sorted(choices,key=lambda a:(-counts[a]['strong_perspective']['top5'],
        -sum(counts[a][g]['top5'] for g in GROUPS),a!=CONSISTENCY))


def rank(args):
    verify_inputs(args)
    assert read(args.confidence_out/'audit.json')['ranking_sha256']==sha256(args.confidence_out/'rankings.json')
    assert read(args.rotation_out/'audit.json')['results_sha256']==sha256(args.rotation_out/'results.json')
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'costs').mkdir()
    paths=[args.confidence_out/n for n in ('protocol.json','rankings.json','audit.json')]
    paths+=[args.rotation_out/n for n in ('protocol.json','inferred.json','audit.json')]
    paths+=[args.source/'answers.json',args.source/'manifest.json',args.refinement_out/'rankings.json']
    save_json(args.out/'protocol.json',{
        'stage':'development only on same examined120 families/360images',
        'arms':ARMS,'hypothesis':'observation-only rotation disagreement supplies useful information beyond model score',
        'consistency_weight':'half if disagreement>=.25 observed torso and support>=2; otherwise1 including unknown support; no joint removal',
        'confidence_weight':'same prior .25+.75*clip((score-.3)/.4,0,1)',
        'combined':'product of confidence and consistency weights; valid joint weights>=.125',
        'cost':'weighted mean at same stored camera and original label; final .25*pooled117+.75*weightedcost; first117 preserved',
        'fixed':'same original HumanArt XY/mask/normalization/gallery1981/cameras/labels; no consensus XY, refit, queryZ or head cue',
        'selection':'strongTop5>=original+3 AND >confidence100; all3 Top1/Top5>=bothcomparators; maximize strong,total; tie consistency-only',
        'ranking_inputs':'recompute agreement from four image-only predictions, never read diagnostic GT error records; no answers/group/sourcecamera',
        'failure':'retain original3 strong failures; no replacement',
        'no_signal':'if no valid joint is downweighted, preserve original fitted costs/ranks EXACTLY; combined preserves prior confidence costs/ranks',
        'limits':'post-diagnostic development; no weight sweep or calibration claim; fixed camera, correlated model views',
        'frozen_sources':{p:sha256(ROOT/p) for p in CODE},'input_hashes':{str(p):sha256(p) for p in paths}})
    poses=load_library(args.library_root)
    assert geometry_hash(poses)==read(args.source/'protocol.json')['hashes']['library_geometry']
    points=normalized_joints(np.stack([p.joints for p in poses]))
    previous={r['image']:r for r in read(args.refinement_out/'rankings.json')}
    manifest={r['image']:r for r in read(args.source/'manifest.json')}
    inferred={r['image']:r for r in read(args.rotation_out/'inferred.json')['records']};rows=[]
    for old in read(args.confidence_out/'rankings.json'):
        name=old['image'];predpath=args.source/'predictions'/(Path(name).stem+'.json')
        assert sha256(predpath)==old['prediction_sha256'] and sha256(args.source/'images'/name)==manifest[name]['sha256']
        pred=read(predpath);query=valid_query(pred);row={k:old[k] for k in ('image','prediction_sha256')}
        if query is None:
            assert old.get('error');row['error']=old['error'];rows.append(row);continue
        xy,mask=query;q=normalize_batch(xy)
        row['shared_xy_sha256']=hashlib.sha256(xy.tobytes()).hexdigest();row['shared_mask']=mask.tolist()
        assert row['shared_xy_sha256']==old['shared_xy_sha256'] and row['shared_mask']==old['shared_mask']
        path=args.refinement_out/'costs'/(Path(name).stem+'.npz');assert sha256(path)==previous[name]['costs_sha256']
        with np.load(path,allow_pickle=False) as d:base,fit,params=[d[k] for k in ('baseline','refined','parameters')]
        path=args.confidence_out/'costs'/(Path(name).stem+'.npz');assert sha256(path)==old['costs_sha256']
        with np.load(path,allow_pickle=False) as d:conf=d['weighted']
        path=args.rotation_out/'hypotheses'/(Path(name).stem+'.json');assert sha256(path)==inferred[name]['sha256']
        h=read(path);signal=agreement(pred,h['hypotheses']);assert signal==h['agreement']
        weights=consistency_weights(signal);cw=joint_weights(pred['scores'])
        active=bool(np.any(weights[5:][mask[5:]]<1));row['signal_active']=active
        row['downweighted_joints']=[j for j in range(5,17) if mask[j] and weights[j]<1]
        features,valid=project_parameters(points,params);assert valid.all()
        reconstructed,_=fixed_weighted_cost(features,q,mask,cw,fit);assert np.array_equal(reconstructed,conf)
        if active:
            consistency,labels=fixed_weighted_cost(features,q,mask,weights,fit)
            combined,labels2=fixed_weighted_cost(features,q,mask,weights*cw,fit);assert np.array_equal(labels,labels2)
        else:consistency,combined=fit.copy(),conf.copy()
        costs={BASE:fit,CONFIDENCE:conf,CONSISTENCY:consistency,COMBINED:combined}
        for arm,cost in costs.items():
            row[arm]=[poses[i].pose_id for i in order(base,blend_cost(base,cost,.75))[:5]]
            assert row[arm][0]==row[BASE][0]
        assert row[BASE]==old[BASE] and row[CONFIDENCE]==old[CONFIDENCE]
        if not active:assert row[CONSISTENCY]==row[BASE] and row[COMBINED]==row[CONFIDENCE]
        path=args.out/'costs'/(Path(name).stem+'.npz')
        np.savez_compressed(path,consistency=consistency,combined=combined,weights=weights)
        row['costs_sha256']=sha256(path);rows.append(row)
    assert len(rows)==360;verify(args)
    save_json(args.out/'rankings.json',rows);save_json(args.out/'ranked.json',{'sha256':sha256(args.out/'rankings.json'),'answers_parsed':False})


def evaluate(args):
    verify(args);assert not (args.out/'results.json').exists()
    assert sha256(args.out/'rankings.json')==read(args.out/'ranked.json')['sha256']
    rows=read(args.out/'rankings.json');answers={r['image']:r for r in read(args.source/'answers.json')}
    assert len(rows)==len(answers)==360 and set(answers)=={r['image'] for r in rows}
    counts={a:{g:{'top1':0,'top5':0} for g in GROUPS} for a in ARMS}
    changes={a:{b:{g:{'gains':0,'losses':0} for g in GROUPS} for b in ARMS[:2]} for a in ARMS[2:]}
    failed={g:0 for g in GROUPS};active={g:0 for g in GROUPS}
    for row in rows:
        t=answers[row['image']];g,pid=t['group'],t['pose_id'];failed[g]+=bool(row.get('error'));active[g]+=row.get('signal_active',False)
        for arm in ARMS:
            ids=row.get(arm,[]);counts[arm][g]['top1']+=bool(ids and ids[0]==pid);counts[arm][g]['top5']+=pid in ids
        for arm in ARMS[2:]:
            for b in ARMS[:2]:
                before,after=pid in row.get(b,[]),pid in row.get(arm,[])
                changes[arm][b][g]['gains']+=after and not before;changes[arm][b][g]['losses']+=before and not after
    eligible=select_policy(counts)
    result={'stage':'development only','counts':counts,'changes':changes,'failures':failed,'active':active,
            'eligible':eligible,'selected':eligible[0] if eligible else None}
    save_json(args.out/'results.json',result);print(result,flush=True)


def parse():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('rank','evaluate'))
    for name in ('source','refinement-out','blend-out','confidence-out','rotation-out','out','library-root','model'):
        p.add_argument('--'+name,type=Path,required=True)
    return p.parse_args()


if __name__=='__main__':
    args=parse();args.out.parent.mkdir(parents=True,exist_ok=True)
    with (args.out.parent/'.consistency-cost-dev.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);globals()[args.stage](args)
