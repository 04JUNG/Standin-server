"""Recompute ranking and verify heldout/input contracts, without changing them."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.confirm_dual_labels import frozen_sources
from scripts.confirm_camera_fit import geometry_hash
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import save_json,sha256
from standin_eval.z_axis_search import build_search_index,distances
from standin_eval.perspective_search import camera_grid,build_camera_index,score_camera_index,normalize_batch


def run(a):
    policy=json.loads((a.out/'protocol.json').read_text())
    assert frozen_sources()==policy['frozen_sources']
    assert sha256(ROOT/'out/dual-label-dev-20261012/results.json')==policy['hashes']['selection']
    poses=load_library(a.library_root)
    assert geometry_hash(poses)==policy['hashes']['library_geometry']
    assert sha256(a.library_root/'poses.db')==policy['hashes']['library_db']
    for path,h in policy['prior_answers'].items():assert sha256(ROOT/path)==h
    answers=json.loads((a.out/'answers.json').read_text());manifest=json.loads((a.out/'manifest.json').read_text())
    rankings={r['image']:r for r in json.loads((a.out/'rankings.json').read_text())}
    families={r['family_id'] for r in answers}
    assert len(families)==120 and families==set(policy['test_families'])
    assert not families & set(policy['excluded_families'])
    assert len(answers)==len(manifest)==len(rankings)==240
    assert {r['image'] for r in answers}=={r['image'] for r in manifest}==set(rankings)
    # Replay random family-member sampling, rather than merely trust the protocol.
    groups={}
    for pose in poses:
        if pose.family_id not in set(policy['excluded_families']):groups.setdefault(pose.family_id,[]).append(pose)
    pool=list(groups.values());rng=np.random.default_rng(policy['seed'])
    picked=[pool[int(i)] for i in rng.choice(len(pool),120,replace=False)]
    chosen=[members[int(rng.integers(len(members)))] for members in picked]
    assert [p.pose_id for p in chosen]==policy['test_pose_ids']
    assert {r['pose_id'] for r in answers}==set(policy['test_pose_ids'])
    old=build_search_index(poses)
    dense=build_camera_index(poses,camera_grid()['yaw_pitch_perspective_117'])
    perm=np.array([0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15])
    selected=policy['selected'];checked=0;failed=0
    for item in manifest:
        name=item['image'];r=rankings[name]
        assert sha256(a.out/'images'/name)==item['sha256']
        pp=a.out/'predictions'/(Path(name).stem+'.json');pred=json.loads(pp.read_text())
        assert sha256(pp)==r['prediction_sha256']
        if pred.get('error'):
            assert r.get('error');assert all(arm not in r for arm in policy['arms'])
            failed+=1;continue
        xy=np.asarray(pred['keypoints2d'],dtype=np.float32);mask=np.asarray(pred['scores'],dtype=np.float32)>=.3
        assert hashlib.sha256(xy.tobytes()).hexdigest()==r['shared_xy_sha256']
        assert mask.tolist()==r['shared_mask']
        if r.get('error'):
            assert not np.isfinite(xy).all() or mask[5:].sum()<6 or not mask[[5,6,11,12]].all()
            assert all(arm not in r for arm in policy['arms']);failed+=1;continue
        q=normalize_batch(xy)
        values=distances(old,q,np.zeros(17),valid_mask=mask,z_weight=0)
        seen=set();expected=[]
        for i in np.argsort(values,kind='stable'):
            pid=old.pose_ids[int(i)]
            if pid not in seen:seen.add(pid);expected.append(pid)
            if len(expected)==5:break
        assert r['anchor_4']==expected
        d=score_camera_index(dense,q,mask)
        alt=score_camera_index(dense,q[perm],mask[perm])
        assert r['anchor_117']==[poses[int(i)].pose_id for i in np.argsort(d,kind='stable')[:5]]
        order=np.argsort(np.minimum(d,alt),kind='stable').tolist()
        if selected.startswith('keepfirst'):
            first=int(np.argmin(d));order=[first]+[i for i in order if i!=first]
        assert r[selected]==[poses[i].pose_id for i in order[:5]]
        assert len(set(r[selected]))==5
        checked+=1
        if checked%40==0:print('audit',checked,flush=True)
    audit={'frozen_sources_verified':True,'gallery_geometry_and_db_verified':True,
        'source_selection_hash_verified':True,'images_verified':len(manifest),
        'prediction_hashes_and_masks_verified':len(rankings),'heldout_families':120,
        'overlap_with_prior_families':0,'random_family_member_selection_replayed':True,
        'mirror_source_families':sum(p.pose_id.endswith('_mirror') for p in chosen),
        'original_4view_baseline_matches':checked,'dense_and_dual_top5_recomputed':checked,
        'failures_retained':failed,'ranking_sha256':sha256(a.out/'rankings.json')}
    save_json(a.out/'audit.json',audit);print(json.dumps(audit,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--library-root',type=Path,required=True);run(p.parse_args())
