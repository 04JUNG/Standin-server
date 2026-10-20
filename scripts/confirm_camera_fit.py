"""One precommitted fresh 120-family confirmation of dev-selected XY fitting.

Separate preparation, image-only inference, answer-free ranking and evaluation.
"""
import argparse,json,sys,hashlib,time,math
from pathlib import Path
from dataclasses import asdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import render_pose,save_json,sha256
from scripts.experiment_perspective_images import infer
from standin_eval.perspective_search import Camera,camera_grid,build_camera_index,score_camera_index,normalize_batch
from standin_eval.camera_fit import score_fit_camera_index


def geometry_hash(poses):
    digest=hashlib.sha256()
    for p in poses:
        digest.update(p.pose_id.encode());digest.update(np.asarray(p.joints,dtype=np.float32).tobytes())
    return digest.hexdigest()


def frozen_sources():
    return {p:sha256(ROOT/p) for p in ('scripts/confirm_camera_fit.py','standin_eval/camera_fit.py',
        'standin_eval/perspective_search.py','standin_eval/z_diagnostics.py','scripts/experiment_z_rendered.py',
        'scripts/experiment_perspective_images.py')}


def prepare(a):
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'images').mkdir()
    selection=ROOT/'out/camera-fit-dev-20261012/results.json'
    selected=json.loads(selection.read_text())['selection']['selected']
    assert selected in ('fit_117','weighted_fit_117')
    poses=load_library(a.library_root)
    paths=[ROOT/'out'/x/'answers.json' for x in ('z-rough-20261012/experiment','z-diagnosis-20261012-heldout',
        'perspective-images-20261012','perspective-replication-20261012','camera-fit-dev-20261012')]
    excluded={r['family_id'] for p in paths for r in json.loads(p.read_text())}
    families={}
    for pose in poses:
        if pose.family_id not in excluded:families.setdefault(pose.family_id,pose)
    pool=list(families.values());rng=np.random.default_rng(20261018)
    chosen=[pool[int(i)] for i in rng.choice(len(pool),120,replace=False)]
    policy={'stage':'fresh fixed-size confirmation; one batch; no optional stopping or per-image tuning',
        'samples_families':120,'seed':20261018,'gallery_poses':len(poses),'selected':selected,
        'arms':['anchor_4','anchor_117',selected],
        'primary':'selected fit versus anchor_117, isolates fitting from camera coverage',
        'secondary':'selected fit and anchor_117 versus anchor_4, full local method comparison',
        'groups':['perspective','orthographic_control'],'group_dependence':'two renders per family, never pool as 240 independent poses',
        'camera':'same off-grid query recipe and four-view control as previous tests; 117-view fixed gallery',
        'input':'one Human-Art M inference per image shared exactly by all arms; RGB crop, 4 rotations, body median score',
        'validity':'all arms: >=6 body joints with scores>=0.3 and all shoulder/hip anchors; failures retained',
        'metric':'positive-scale/translation fit using same query joints; no reflection, rotation, correspondence swap or target information',
        'success_gate':'perspective fit vs anchor_117 Top5 delta>=0.05, paired bootstrap 95% lower>0 and exact two-sided McNemar p<0.05; control Top5 and Top1 counts not lower than anchor_117',
        'statistics':'20000 paired bootstrap resamples by family within group, seed20261019; single primary contrast; secondary contrasts descriptive',
        'limitations':['synthetic contour renders, not human roughs','closed-set original-ID retrieval',
                       'family metadata exclusion, not guaranteed motion-sequence or semantic independence','not deployed pipeline'],
        'excluded_families':sorted(excluded),'test_families':[p.family_id for p in chosen],
        'test_pose_ids':[p.pose_id for p in chosen],
        'hashes':{'model':sha256(a.model),'library_db':sha256(a.library_root/'poses.db'),
                  'library_geometry':geometry_hash(poses),'selection':sha256(selection)},
        'frozen_sources':frozen_sources(),'prior_answers':{str(p.relative_to(ROOT)):sha256(p) for p in paths}}
    save_json(a.out/'protocol.json',policy)
    manifest=[];answers=[]
    for i,pose in enumerate(chosen):
        cam=Camera((15,75,135,225)[i%4],(-25,20)[(i//4)%2],(4.5,6.5)[(i//8)%2])
        for group,c in [('perspective',cam),('orthographic_control',Camera((0,45,90,180)[i%4]))]:
            name=f'{group}_{i:03d}.png'
            image,gt=render_pose(pose.joints,c.yaw,style='rough',seed=20261018+i,pitch=c.pitch,distance=c.distance)
            image.save(a.out/'images'/name)
            manifest.append({'image':name,'sha256':sha256(a.out/'images'/name)})
            answers.append({'image':name,'group':group,'pose_id':pose.pose_id,'family_id':pose.family_id,
                'camera':asdict(c),'ground_truth_xy':gt.tolist()})
        if (i+1)%10==0:print('render',i+1,'/120',flush=True)
    save_json(a.out/'manifest.json',manifest);save_json(a.out/'answers.json',answers)


def rank(a):
    policy=json.loads((a.out/'protocol.json').read_text())
    assert frozen_sources()==policy['frozen_sources']
    poses=load_library(a.library_root);assert geometry_hash(poses)==policy['hashes']['library_geometry']
    indices={4:build_camera_index(poses,camera_grid()['baseline_4']),
             117:build_camera_index(poses,camera_grid()['yaw_pitch_perspective_117'])}
    predictions=a.out/'predictions';rows=[];timing={arm:[] for arm in policy['arms']}
    for item in json.loads((a.out/'manifest.json').read_text()):
        image=item['image'];path=predictions/(Path(image).stem+'.json')
        p=json.loads(path.read_text());row={'image':image,'prediction_sha256':sha256(path)}
        if p.get('error'):row['error']=p['error'];rows.append(row);continue
        xy=np.asarray(p['keypoints2d'],dtype=np.float32);scores=np.asarray(p['scores'],dtype=np.float32);mask=scores>=.3
        row['shared_xy_sha256']=hashlib.sha256(xy.tobytes()).hexdigest();row['shared_mask']=mask.tolist()
        if not np.isfinite(xy).all() or mask[5:].sum()<6 or not mask[[5,6,11,12]].all():
            row['error']='insufficient Human-Art body joints/anchors';rows.append(row);continue
        for arm in policy['arms']:
            start=time.perf_counter()
            if arm.startswith('anchor_'):cost=score_camera_index(indices[int(arm.split('_')[-1])],normalize_batch(xy),mask)
            else:cost=score_fit_camera_index(indices[117],xy,mask,scores if arm=='weighted_fit_117' else None)
            order=np.argsort(cost,kind='stable');timing[arm].append(time.perf_counter()-start)
            row[arm]=[poses[int(i)].pose_id for i in order if np.isfinite(cost[int(i)])]
        rows.append(row)
        if len(rows)%20==0:print('rank',len(rows),'/240',flush=True)
    save_json(a.out/'rankings.json',rows);save_json(a.out/'timing.json',timing)


def paired_comparison(records,arm,base,k=5):
    hit=lambda r,a:r.get(a+'_rank') is not None and r[a+'_rank']<=k
    diff=np.array([int(hit(r,arm))-int(hit(r,base)) for r in records])
    rng=np.random.default_rng(20261019)
    interval=np.quantile(diff[rng.integers(0,len(diff),size=(20000,len(diff)))].mean(axis=1),[.025,.975]).tolist()
    gains=int((diff==1).sum());losses=int((diff==-1).sum());n=gains+losses
    exact=min(1.,2*sum(math.comb(n,j) for j in range(min(gains,losses)+1))/2**n) if n else 1.
    return {'gains':gains,'losses':losses,'delta':float(diff.mean()),'paired_bootstrap_95ci':interval,'exact_mcnemar_p':exact}


def evaluate(a):
    policy=json.loads((a.out/'protocol.json').read_text());arms=policy['arms'];selected=policy['selected']
    answers={r['image']:r for r in json.loads((a.out/'answers.json').read_text())}
    rows=json.loads((a.out/'rankings.json').read_text());result={}
    assert len(rows)==len(answers)==240
    for group in policy['groups']:
        records=[]
        for row in rows:
            answer=answers[row['image']]
            if answer['group']!=group:continue
            r={k:answer[k] for k in ('image','pose_id','family_id')}
            if row.get('error'):r['error']=row['error']
            for arm in arms:
                candidates=row.get(arm,[]);r[arm+'_rank']=candidates.index(answer['pose_id'])+1 if answer['pose_id'] in candidates else None
                r[arm+'_top5']=candidates[:5]
            records.append(r)
        assert len(records)==120
        counts={arm:{f'top{k}':sum(r[arm+'_rank'] is not None and r[arm+'_rank']<=k for r in records) for k in (1,5)} for arm in arms}
        result[group]={'n':len(records),'extraction_failures':sum('error' in r for r in records),'counts':counts,
            'fit_vs_camera':paired_comparison(records,selected,'anchor_117'),
            'fit_vs_baseline':paired_comparison(records,selected,'anchor_4'),
            'camera_vs_baseline':paired_comparison(records,'anchor_117','anchor_4'),'records':records}
        print(group,{k:v for k,v in result[group].items() if k!='records'},flush=True)
    contrast=result['perspective']['fit_vs_camera'];control=result['orthographic_control']['counts']
    result['gate_passed']=bool(contrast['delta']>=.05 and contrast['paired_bootstrap_95ci'][0]>0 and contrast['exact_mcnemar_p']<.05
        and all(control[selected][k]>=control['anchor_117'][k] for k in ('top1','top5')))
    result['selected']=selected;save_json(a.out/'results.json',result);print('gate',result['gate_passed'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','infer','rank','evaluate'))
    p.add_argument('--library-root',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();globals()[a.stage](a)
