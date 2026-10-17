"""Fresh-family fixed-size confirmation of the selected distance prior."""
import argparse
import fcntl
import hashlib
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.confirm_perspective_isolation import select,strong_camera,projection_strength,PRIOR
from scripts.experiment_near_camera import read,near_cameras
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import render_pose,save_json,sha256
from scripts.experiment_perspective_images import infer as image_infer
from scripts.confirm_camera_fit import geometry_hash,paired_comparison
from scripts.diagnose_dual_camera_coverage import valid_query
from scripts.experiment_camera_regularization import order
from standin_eval.camera_refinement import normalized_joints,refine_costs
from standin_eval.camera_regularization import regularized_costs
from standin_eval.perspective_search import Camera,camera_grid,build_camera_index,score_camera_index,normalize_batch
from standin_eval.dual_label_search import alternate_labels

SEED=20261103
GROUPS=('perspective','strong_perspective','orthographic_control')
BASE='keepfirst_blend75';SELECTED='reg050';ARMS=(BASE,SELECTED)
DEV=ROOT/'out/camera-regularization-dev-20261013'
CODE=('scripts/confirm_camera_regularization.py','scripts/audit_regularization_confirmation.py',
      'scripts/experiment_camera_regularization.py','standin_eval/camera_regularization.py',
      'standin_eval/camera_refinement.py','scripts/experiment_refinement_blend.py',
      'scripts/confirm_perspective_isolation.py','scripts/experiment_near_camera.py',
      'scripts/experiment_z_axis_search.py','scripts/experiment_z_rendered.py',
      'scripts/experiment_perspective_images.py','scripts/confirm_camera_fit.py',
      'scripts/diagnose_dual_camera_coverage.py','standin_eval/perspective_search.py',
      'standin_eval/dual_label_search.py','standin_eval/z_diagnostics.py','src/bvh.py')


def sources():return {p:sha256(ROOT/p) for p in CODE}


def verify(args):
    p=read(args.out/'protocol.json')
    assert sources()==p['frozen_sources']
    assert sha256(args.model)==p['hashes']['model']
    assert sha256(args.library_root/'poses.db')==p['hashes']['library_db']
    assert all(sha256(Path(path))==digest for path,digest in p['input_hashes'].items())
    return p


def prepared(args):
    verify(args);snap=read(args.out/'prepared.json')
    assert snap['strong_ratio_ge_1_5']>=100
    assert sha256(args.out/'manifest.json')==snap['manifest_sha256']
    assert sha256(args.out/'answers.json')==snap['answers_sha256']


def prepare(args):
    assert read(DEV/'results.json')['selected']==SELECTED
    audit=read(DEV/'audit.json')
    assert audit['ranking_sha256']==sha256(DEV/'rankings.json')
    assert audit['counts_and_selection_independently_recomputed']
    # Check the development selection really uses the same frozen code.
    dp=read(DEV/'protocol.json')
    assert all(sha256(ROOT/p)==h for p,h in dp['frozen_sources'].items())
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'images').mkdir();(args.out/'costs').mkdir()
    poses=load_library(args.library_root)
    prior=[ROOT/'out'/p/'answers.json' for p in (*PRIOR,'perspective-isolation-confirm-20261013',
            'near-camera-confirm-20261013','refinement-blend-confirm-20261013')]
    excluded={r['family_id'] for p in prior for r in read(p)};assert len(excluded)==760
    chosen=select(poses,excluded,seed=SEED,count=120)
    inputs=prior+[DEV/'protocol.json',DEV/'results.json',DEV/'audit.json']
    save_json(args.out/'protocol.json',{
        'stage':'fresh120 family confirmation, fixed sample size; no optional stopping',
        'seed':SEED,'selected':SELECTED,'weight':.05,'alpha':.75,'arms':ARMS,'groups':GROUPS,
        'primary':'sum inverse_distance_squared for ranks2..5 across three groups, paired family bootstrap ratio',
        'success_gate':'energy ratio<=.90 and95CI upper<1; all3 Top5 counts>=baseline and paired95CI lower>=-.05; all3 Top1 identical',
        'statistics':'20000 family resamples,seed20261103 for energy; group Top5 paired statistics seed20261019; descriptive accuracy noninferiority is not zero-harm proof',
        'formula':'fit=min(XY_body_cost+.05*inverse_distance^2); score=.25*pooled117+.75*fit; stable ties; keep first117',
        'fixed':'Human-Art XY/mask/normalization/dual labels fixed; three regularized195 camera starts; original4 scales,2 sweeps,pitch+-60,invdistance0..1.25; all1981 poses; no queryZ orGT',
        'sampling':'exclude760 used families, uniform120 families then uniform member; no replacement of failures',
        'cameras':'same moderate offgrid, strong targetdepthratio2, old4 ortho controls',
        'limits':['synthetic mannequin contours,not real roughs','family not sequence/semantic holdout',
                  'proximity energy is not anatomy or human quality','closed-set source pose remains in gallery',
                  'Top1 fixed by construction; energy excludes unchanged first result'],
        'excluded_families':sorted(excluded),'test_families':[p.family_id for p in chosen],
        'test_pose_ids':[p.pose_id for p in chosen],'prior_answers':[str(p) for p in prior],
        'input_hashes':{str(p):sha256(p) for p in inputs},'frozen_sources':sources(),
        'hashes':{'model':sha256(args.model),'library_db':sha256(args.library_root/'poses.db'),'geometry':geometry_hash(poses)}})
    manifest=[];answers=[]
    for i,pose in enumerate(chosen):
        yaw,pitch=(15,75,135,225)[i%4],(-25,20)[(i//4)%2]
        cams=(Camera(yaw,pitch,(4.5,6.5)[(i//8)%2]),strong_camera(pose.joints,yaw,pitch),Camera((0,45,90,180)[i%4]))
        for group,cam in zip(GROUPS,cams):
            name=f'{group}_{i:03d}.png'
            im,xy=render_pose(pose.joints,cam.yaw,style='rough',seed=SEED+i,pitch=cam.pitch,distance=cam.distance)
            im.save(args.out/'images'/name);manifest.append({'image':name,'sha256':sha256(args.out/'images'/name)})
            answers.append({'image':name,'group':group,'pose_id':pose.pose_id,'family_id':pose.family_id,
                            'camera':asdict(cam),'body_depth_ratio':projection_strength(pose.joints,cam),'ground_truth_xy':xy.tolist()})
        if (i+1)%30==0:print('rendered',i+1,'/120',flush=True)
    save_json(args.out/'manifest.json',manifest);save_json(args.out/'answers.json',answers)
    strong=sum(r['group']==GROUPS[1] and r['body_depth_ratio']>=1.5 for r in answers)
    save_json(args.out/'prepared.json',{'manifest_sha256':sha256(args.out/'manifest.json'),
        'answers_sha256':sha256(args.out/'answers.json'),'strong_ratio_ge_1_5':strong})
    assert strong>=100


def infer(args):prepared(args);image_infer(args)


def rank(args):
    prepared(args);assert not (args.out/'rankings.json').exists()
    poses=load_library(args.library_root);assert geometry_hash(poses)==verify(args)['hashes']['geometry']
    cams=camera_grid()['yaw_pitch_perspective_117']+near_cameras()
    index=build_camera_index(poses,cams);bi=(index[0][:117],index[1][:117])
    points=normalized_joints(np.stack([p.joints for p in poses]));rows=[]
    for item in read(args.out/'manifest.json'):
        name=item['image'];assert sha256(args.out/'images'/name)==item['sha256']
        path=args.out/'predictions'/(Path(name).stem+'.json');pred=read(path)
        row={'image':name,'prediction_sha256':sha256(path)};query=valid_query(pred)
        if query is None:
            row['error']=pred.get('error','insufficient Human-Art body joints/anchors');rows.append(row);continue
        xy,mask=query;q=normalize_batch(xy);alt,am=alternate_labels(q,mask)
        row['shared_xy_sha256']=hashlib.sha256(xy.tobytes()).hexdigest();row['shared_mask']=mask.tolist()
        base=np.minimum(score_camera_index(bi,q,mask),score_camera_index(bi,alt,am))
        fit,fp=refine_costs(points,index,cams,q,mask)
        reg,rp=regularized_costs(points,index,cams,q,mask,.05)
        row['inverse_distance_squared']={}
        for arm,cost,params in ((BASE,fit,fp),(SELECTED,reg,rp)):
            selected=order(base,cost)[:5];row[arm]=[poses[i].pose_id for i in selected]
            row['inverse_distance_squared'][arm]=float(np.sum(params[selected[1:],2].astype(float)**2))
        assert row[BASE][0]==row[SELECTED][0]
        dest=args.out/'costs'/(Path(name).stem+'.npz')
        np.savez_compressed(dest,baseline=base,refined=fit,parameters=fp,regularized=reg,regularized_parameters=rp)
        row['costs_sha256']=sha256(dest);rows.append(row)
        if len(rows)%30==0:print('ranked',len(rows),'/360',flush=True)
    assert len(rows)==360;verify(args)
    save_json(args.out/'rankings.json',rows);save_json(args.out/'ranked.json',{'sha256':sha256(args.out/'rankings.json'),'answers_parsed':False})


def energy_stats(baseline,selected):
    b=np.asarray(baseline,float);s=np.asarray(selected,float)
    if b.ndim!=1 or b.shape!=s.shape or not np.isfinite(b).all() or not np.isfinite(s).all() or np.any(b<0) or np.any(s<0) or b.sum()<=0:
        raise ValueError('paired nonnegative finite family energies required')
    rng=np.random.default_rng(SEED);indices=rng.integers(0,len(b),size=(20000,len(b)))
    den=b[indices].sum(axis=1);num=s[indices].sum(axis=1)
    if np.any(den<=0):raise ValueError('bootstrap denominator is zero')
    return {'baseline_sum':float(b.sum()),'selected_sum':float(s.sum()),'ratio':float(s.sum()/b.sum()),
            'family_bootstrap_95ci':np.quantile(num/den,[.025,.975]).tolist(),'families':len(b)}


def success_gate(groups,energy):
    return bool(energy['ratio']<=.9 and energy['family_bootstrap_95ci'][1]<1 and
        all(groups[g]['counts'][SELECTED]['top5']>=groups[g]['counts'][BASE]['top5'] and
            groups[g]['selected_vs_baseline']['paired_bootstrap_95ci'][0]>=-.05 and
            groups[g]['counts'][SELECTED]['top1']==groups[g]['counts'][BASE]['top1'] for g in GROUPS))


def evaluate(args):
    prepared(args);assert not (args.out/'results.json').exists()
    assert sha256(args.out/'rankings.json')==read(args.out/'ranked.json')['sha256']
    rows=read(args.out/'rankings.json');answers={r['image']:r for r in read(args.out/'answers.json')}
    assert len(rows)==len(answers)==len({r['image'] for r in rows})==360
    assert set(answers)=={r['image'] for r in rows}
    groups={};family={f:{a:0. for a in ARMS} for f in verify(args)['test_families']}
    for g in GROUPS:
        records=[]
        for row in rows:
            target=answers[row['image']]
            if target['group']!=g:continue
            rec={k:target[k] for k in ('image','pose_id','family_id','body_depth_ratio')}
            if row.get('error'):rec['error']=row['error']
            for arm in ARMS:
                ids=row.get(arm,[]);rec[arm+'_rank']=ids.index(target['pose_id'])+1 if target['pose_id'] in ids else None
                family[target['family_id']][arm]+=row.get('inverse_distance_squared',{}).get(arm,0.)
            records.append(rec)
        assert len(records)==len({r['family_id'] for r in records})==120
        counts={a:{f'top{k}':sum(r[a+'_rank'] is not None and r[a+'_rank']<=k for r in records) for k in (1,5)} for a in ARMS}
        groups[g]={'n':120,'extraction_failures':sum('error' in r for r in records),'counts':counts,
                   'selected_vs_baseline':paired_comparison(records,SELECTED,BASE),'records':records}
        print(g,{k:v for k,v in groups[g].items() if k!='records'},flush=True)
    energy=energy_stats([v[BASE] for v in family.values()],[v[SELECTED] for v in family.values()])
    result={'groups':groups,'energy':energy,'gate_passed':success_gate(groups,energy)}
    save_json(args.out/'results.json',result);print('energy',energy,'gate',result['gate_passed'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','infer','rank','evaluate'))
    for name in ('out','library-root','model'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.out.parent.mkdir(parents=True,exist_ok=True)
    with (args.out.parent/'.regularization-confirm.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);globals()[args.stage](args)
