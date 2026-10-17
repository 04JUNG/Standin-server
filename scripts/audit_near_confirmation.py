"""Independent full-index check of near-camera confirmation identities and ranks."""
import argparse
import hashlib
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.confirm_near_camera import read,verify,prepared,GROUPS,ARMS
from scripts.confirm_perspective_isolation import select
from scripts.experiment_near_camera import near_cameras
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import save_json,sha256
from scripts.confirm_camera_fit import geometry_hash
from standin_eval.perspective_search import Camera,camera_points,project,normalize_batch,camera_grid,build_camera_index,score_camera_index


def audit(args):
    prepared(args)
    policy=verify(args)
    assert sha256(args.out/'rankings.json')==read(args.out/'ranked.json')['ranking_sha256']
    poses=load_library(args.library_root)
    assert geometry_hash(poses)==policy['hashes']['library_geometry']
    excluded={r['family_id'] for path in policy['prior_answers'] for r in read(ROOT/path)}
    assert len(excluded)==520 and excluded==set(policy['excluded_families'])
    chosen=select(poses,excluded,seed=policy['seed'],count=120)
    assert [p.pose_id for p in chosen]==policy['test_pose_ids']
    assert [p.family_id for p in chosen]==policy['test_families']
    assert not {p.family_id for p in chosen}&excluded
    manifest=read(args.out/'manifest.json')
    answers={r['image']:r for r in read(args.out/'answers.json')}
    rows={r['image']:r for r in read(args.out/'rankings.json')}
    expected_names={f'{g}_{i:03d}.png' for g in GROUPS for i in range(120)}
    assert len(manifest)==len(answers)==len(rows)==360
    assert set(rows)==set(answers)=={r['image'] for r in manifest}==expected_names
    full=build_camera_index(poses,camera_grid()['yaw_pitch_perspective_117']+near_cameras())
    base=(full[0][:117],full[1][:117])
    permutation=np.array([0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15])
    checked=failures=0
    ratios={g:[] for g in GROUPS}
    for item in manifest:
        name=item['image'];target=answers[name];row=rows[name]
        i=int(Path(name).stem.rsplit('_',1)[1]);pose=chosen[i]
        assert target['pose_id']==pose.pose_id and target['family_id']==pose.family_id
        assert name==f"{target['group']}_{i:03d}.png"
        camera=Camera(**target['camera'])
        yaw,pitch=(15,75,135,225)[i%4],(-25,20)[(i//4)%2]
        if target['group']=='orthographic_control':
            assert camera==Camera((0,45,90,180)[i%4])
        elif target['group']=='perspective':
            assert camera==Camera(yaw,pitch,(4.5,6.5)[(i//8)%2])
        else:
            p=camera_points(pose.joints,Camera(yaw,pitch))
            d=max(float(p[5:,2].max()-2*p[5:,2].min()),.65-float(p[:,2].min()))
            assert camera.yaw==yaw and camera.pitch==pitch and abs(camera.distance-d)<1e-6
        xy,valid=project(pose.joints,camera)
        assert valid and np.allclose(normalize_batch(xy),normalize_batch(target['ground_truth_xy']),atol=1e-5)
        ratio=1.
        if camera.distance is not None:
            depth=camera_points(pose.joints,camera)[5:,2]+camera.distance
            ratio=float(depth.max()/depth.min())
        assert abs(ratio-target['body_depth_ratio'])<1e-6
        ratios[target['group']].append(ratio)
        assert sha256(args.out/'images'/name)==item['sha256']
        path=args.out/'predictions'/(Path(name).stem+'.json')
        assert sha256(path)==row['prediction_sha256']
        pred=read(path)
        if pred.get('error'):
            assert row.get('error') and all(a not in row for a in ARMS)
            failures+=1
            continue
        xy=np.asarray(pred['keypoints2d'],dtype=np.float32)
        mask=np.asarray(pred['scores'],dtype=np.float32)>=.3
        if not np.isfinite(xy).all() or mask[5:].sum()<6 or not mask[[5,6,11,12]].all():
            assert row.get('error') and all(a not in row for a in ARMS)
            failures+=1
            continue
        assert hashlib.sha256(xy.tobytes()).hexdigest()==row['shared_xy_sha256']
        assert mask.tolist()==row['shared_mask']
        q=normalize_batch(xy)
        costs=[np.minimum(score_camera_index(index,q,mask),score_camera_index(index,q[permutation],mask[permutation])) for index in (base,full)]
        orders=[[poses[int(j)].pose_id for j in np.argsort(c,kind='stable') if np.isfinite(c[int(j)])] for c in costs]
        assert row['pooled_117']==orders[0][:5]
        first=orders[0][0]
        assert row['keepfirst_195']==([first]+[pid for pid in orders[1] if pid!=first])[:5]
        checked+=1
        if checked%60==0:
            print('audited',checked,flush=True)
    strong=sum(v>=1.5 for v in ratios['strong_perspective'])
    assert strong>=100
    result={'frozen_sources_and_geometry_verified':True,'family_selection_replayed':True,
            'heldout_families':120,'excluded_families':520,'family_overlap':0,
            'mirror_source_count':sum(p.pose_id.endswith('_mirror') for p in chosen),
            'images_and_projected_answers_verified':360,'prediction_hashes_checked':360,
            'both_rankings_recomputed':checked,'keepfirst_top1_preserved':checked,
            'failures_retained':failures,'strong_ratio_ge_1_5':strong,
            'depth_ratio_min_median_max':{g:np.quantile(v,[0,.5,1]).tolist() for g,v in ratios.items()},
            'audit_script_sha256':sha256(Path(__file__)),'ranking_sha256':sha256(args.out/'rankings.json')}
    save_json(args.out/'audit.json',result)
    print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--library-root',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    audit(parser.parse_args())
