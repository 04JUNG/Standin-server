"""Frozen, fresh-family confirmation of the dev-selected two-label policy."""
import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import save_json, sha256, render_pose
from scripts.experiment_perspective_images import infer
from scripts.confirm_camera_fit import geometry_hash, paired_comparison
from standin_eval.perspective_search import Camera, camera_grid, build_camera_index, score_camera_index, normalize_batch
from standin_eval.dual_label_search import alternate_labels, merged_order


def frozen_sources():
    return {p: sha256(ROOT/p) for p in (
        'scripts/confirm_dual_labels.py', 'standin_eval/dual_label_search.py',
        'standin_eval/perspective_search.py', 'standin_eval/z_diagnostics.py',
        'scripts/experiment_z_rendered.py', 'scripts/experiment_perspective_images.py',
        'scripts/experiment_z_axis_search.py', 'src/bvh.py')}


def prepare(a):
    selection = ROOT/'out/dual-label-dev-20261012/results.json'
    selected = json.loads(selection.read_text())['selected']
    assert selected in ('pooled_117', 'keepfirst_117')
    poses = load_library(a.library_root)
    prior = [ROOT/'out'/d/'answers.json' for d in (
        'z-rough-20261012/experiment', 'z-diagnosis-20261012-heldout',
        'perspective-images-20261012', 'perspective-replication-20261012',
        'camera-fit-dev-20261012', 'camera-fit-confirm-20261012')]
    excluded = {r['family_id'] for p in prior for r in json.loads(p.read_text())}
    pool = {}
    for pose in poses:
        if pose.family_id not in excluded: pool.setdefault(pose.family_id, []).append(pose)
    candidates = list(pool.values())
    seed = 20261023
    rng = np.random.default_rng(seed)
    selected_families = [candidates[int(i)] for i in rng.choice(len(candidates), 120, replace=False)]
    chosen = [members[int(rng.integers(len(members)))] for members in selected_families]
    a.out.mkdir(parents=True, exist_ok=False); (a.out/'images').mkdir()
    policy = {'stage': 'fresh 120-family fixed-size confirmation, one batch, no optional stopping',
        'seed': seed, 'samples_families':120, 'gallery_poses':len(poses), 'selected':selected,
        'arms':['anchor_4','anchor_117',selected], 'groups':['perspective','orthographic_control'],
        'primary':'selected versus anchor117; isolates two-label interpretation, not Z estimation',
        'secondary':'selected and anchor117 versus original four cameras, descriptive',
        'input':'identical Human-Art XY and mask; alternate hypothesis permutes label correspondence and its mask, never changes observed positions',
        'camera':'unchanged four-view baseline and 117-view grid; off-grid perspective queries',
        'validity':'score>=.3, >=6 body joints plus all shoulder/hip anchors; failures count as misses',
        'success_gate':'primary perspective Top5 gain>=.05, paired bootstrap95 lower>0, exact two-sided McNemar p<.05; neither group Top1 nor control Top5 lower than anchor117',
        'statistics':'20000 paired resamples by family within group, seed20261019; one primary contrast; groups not pooled as independent images',
        'scoring':'exact original pose ID; no mirror credit, no source identity or camera in ranking',
        'member_sampling':'after selecting 120 unseen families, choose a member uniformly within each using the same seeded RNG; do not always pick lexicographic non-mirror representative',
        'preflight_revision':'initial preparation used first family member; abandoned before any inference or ranking after detecting 1/220 mirror IDs in development; this revision fixed before new inference',
        'limits':['procedural mannequin contours, not human roughs','family holdout, not guaranteed motion-sequence/semantic independence','closed-set gallery, no deployment'],
        'excluded_families':sorted(excluded), 'test_families':[p.family_id for p in chosen],
        'test_pose_ids':[p.pose_id for p in chosen],
        'hashes':{'model':sha256(a.model),'selection':sha256(selection),
                  'library_db':sha256(a.library_root/'poses.db'),'library_geometry':geometry_hash(poses)},
        'frozen_sources':frozen_sources(),'prior_answers':{str(p.relative_to(ROOT)):sha256(p) for p in prior}}
    save_json(a.out/'protocol.json',policy)
    print('excluded',len(excluded),'selected fresh families',len(chosen),flush=True)
    manifest=[];answers=[]
    for i,pose in enumerate(chosen):
        perspective=Camera((15,75,135,225)[i%4],(-25,20)[(i//4)%2],(4.5,6.5)[(i//8)%2])
        for group,c in [('perspective',perspective),('orthographic_control',Camera((0,45,90,180)[i%4]))]:
            name=f'{group}_{i:03d}.png'
            image,gt=render_pose(pose.joints,c.yaw,style='rough',seed=seed+i,pitch=c.pitch,distance=c.distance)
            image.save(a.out/'images'/name)
            manifest.append({'image':name,'sha256':sha256(a.out/'images'/name)})
            answers.append({'image':name,'group':group,'pose_id':pose.pose_id,'family_id':pose.family_id,
                            'camera':asdict(c),'ground_truth_xy':gt.tolist()})
        if (i+1)%20==0:print('render',i+1,'/120',flush=True)
    save_json(a.out/'manifest.json',manifest);save_json(a.out/'answers.json',answers)


def rank(a):
    policy=json.loads((a.out/'protocol.json').read_text())
    assert frozen_sources()==policy['frozen_sources']
    poses=load_library(a.library_root)
    assert geometry_hash(poses)==policy['hashes']['library_geometry']
    selected=policy['selected']
    indices={n:build_camera_index(poses,camera_grid()['baseline_4' if n==4 else 'yaw_pitch_perspective_117']) for n in (4,117)}
    rows=[];timings={arm:[] for arm in policy['arms']}
    for item in json.loads((a.out/'manifest.json').read_text()):
        name=item['image'];path=a.out/'predictions'/(Path(name).stem+'.json')
        p=json.loads(path.read_text());row={'image':name,'prediction_sha256':sha256(path)}
        if p.get('error'):row['error']=p['error'];rows.append(row);continue
        xy=np.asarray(p['keypoints2d'],dtype=np.float32);mask=np.asarray(p['scores'],dtype=np.float32)>=.3
        row['shared_xy_sha256']=hashlib.sha256(xy.tobytes()).hexdigest();row['shared_mask']=mask.tolist()
        if not np.isfinite(xy).all() or mask[5:].sum()<6 or not mask[[5,6,11,12]].all():
            row['error']='insufficient Human-Art body joints/anchors';rows.append(row);continue
        q=normalize_batch(xy)
        for n in (4,117):
            start=time.perf_counter()
            direct=score_camera_index(indices[n],q,mask)
            order=np.argsort(direct,kind='stable')
            row[f'anchor_{n}']=[poses[int(i)].pose_id for i in order[:5]]
            timings[f'anchor_{n}'].append(time.perf_counter()-start)
        start=time.perf_counter()
        alt,alt_mask=alternate_labels(q,mask)
        other=score_camera_index(indices[117],alt,alt_mask)
        order=merged_order(direct,other,preserve_first=selected.startswith('keepfirst'))
        row[selected]=[poses[int(i)].pose_id for i in order[:5]]
        timings[selected].append(timings['anchor_117'][-1]+time.perf_counter()-start)
        rows.append(row)
        if len(rows)%40==0:print('rank',len(rows),'/240',flush=True)
    save_json(a.out/'rankings.json',rows);save_json(a.out/'timing.json',timings)


def evaluate(a):
    policy=json.loads((a.out/'protocol.json').read_text())
    assert frozen_sources()==policy['frozen_sources']
    answers={r['image']:r for r in json.loads((a.out/'answers.json').read_text())}
    rows=json.loads((a.out/'rankings.json').read_text())
    assert len(rows)==len(answers)==240
    selected=policy['selected'];arms=policy['arms'];result={}
    for group in policy['groups']:
        records=[]
        for row in rows:
            answer=answers[row['image']]
            if answer['group']!=group:continue
            r={k:answer[k] for k in ('image','pose_id','family_id')}
            if row.get('error'):r['error']=row['error']
            for arm in arms:
                candidates=row.get(arm,[])
                r[arm+'_rank']=candidates.index(answer['pose_id'])+1 if answer['pose_id'] in candidates else None
                r[arm+'_top5']=candidates
            records.append(r)
        assert len(records)==120
        counts={arm:{f'top{k}':sum(r[arm+'_rank'] is not None and r[arm+'_rank']<=k for r in records) for k in (1,5)} for arm in arms}
        result[group]={'n':120,'extraction_failures':sum('error' in r for r in records),'counts':counts,
            'dual_vs_camera':paired_comparison(records,selected,'anchor_117'),
            'dual_vs_baseline':paired_comparison(records,selected,'anchor_4'),
            'records':records}
        print(group,{k:v for k,v in result[group].items() if k!='records'},flush=True)
    contrast=result['perspective']['dual_vs_camera']
    control=result['orthographic_control']['counts'];pers=result['perspective']['counts']
    result['gate_passed']=bool(contrast['delta']>=.05 and contrast['paired_bootstrap_95ci'][0]>0 and contrast['exact_mcnemar_p']<.05
        and all(control[selected][k]>=control['anchor_117'][k] for k in ('top1','top5'))
        and pers[selected]['top1']>=pers['anchor_117']['top1'])
    result['selected']=selected
    save_json(a.out/'results.json',result);print('gate',result['gate_passed'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','infer','rank','evaluate'))
    p.add_argument('--library-root',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();globals()[a.stage](a)
