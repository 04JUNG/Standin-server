"""Audit source separation, frozen hashes, renderer and baseline equivalence."""
import io
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import render_pose,sha256,save_json
from standin_eval.perspective_search import build_camera_index,camera_grid,score_camera_index
from standin_eval.z_axis_search import build_search_index,distances


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--library-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    poses=load_library(a.library_root);by_id={p.pose_id:p for p in poses}
    old=build_search_index(poses);new=build_camera_index(poses,camera_grid()['baseline_4'])
    oldset=json.loads((ROOT/'out/humanart-plus-z-20261012/results.json').read_text())['synthetic']['records']
    checked=0;max_error=0.
    for r in oldset:
        if r.get('error'):continue
        xy=np.asarray(r['shared_xy'],dtype=np.float32);mask=np.asarray(r['shared_mask'],bool)
        values=distances(old,xy,np.zeros(17),valid_mask=mask,z_weight=0)
        old_min={}
        for pid,d in zip(old.pose_ids,values):old_min[pid]=min(old_min.get(pid,np.inf),float(d))
        expected=np.array([old_min[p.pose_id] for p in poses])
        actual=score_camera_index(new,xy,mask)
        max_error=max(max_error,float(np.max(np.abs(actual-expected))))
        assert np.allclose(actual,expected,atol=2e-5)
        target=next(i for i,p in enumerate(poses) if p.pose_id==r['target_pose_id'])
        rank=int(np.flatnonzero(np.argsort(actual,kind='stable')==target)[0])+1
        assert rank==r['xy_rank'],(r['image'],rank,r['xy_rank'])
        checked+=1
    previous=json.loads((ROOT/'out/z-diagnosis-20261012-heldout/answers.json').read_text())
    for i,r in enumerate(previous[:4]):
        image,gt=render_pose(by_id[r['pose_id']].joints,r['yaw'],style='rough',seed=20261013+i)
        blob=io.BytesIO();image.save(blob,format='PNG')
        assert hashlib.sha256(blob.getvalue()).hexdigest()==r['sha256']
        assert np.array_equal(gt,np.array(r['gt_projected_xy']))
    policy=json.loads((a.out/'protocol.json').read_text())
    for key,path in [('script',ROOT/'scripts/experiment_perspective_images.py'),
                     ('renderer',ROOT/'scripts/experiment_z_rendered.py'),
                     ('camera_module',ROOT/'standin_eval/perspective_search.py')]:
        assert sha256(path)==policy['hashes'][key]
    manifest=json.loads((a.out/'manifest.json').read_text())
    for item in manifest:assert sha256(a.out/'images'/item['image'])==item['sha256']
    answers=json.loads((a.out/'answers.json').read_text())
    development=json.loads((ROOT/'out/perspective-coordinates-20261012/protocol.json').read_text())
    families={r['family_id'] for r in answers}
    assert len(families)==40
    assert not families&set(development['excluded_families']+development['dev_families'])
    save_json(a.out/'audit.json',{'baseline_rank_matches':checked,'baseline_max_distance_error':max_error,
        'old_renderer_png_byte_matches':4,'new_input_hashes_verified':len(manifest),
        'frozen_script_renderer_camera_hashes_verified':True,'query_family_overlap_with_development':0,
        'new_families':len(families),'limitations':'family metadata is not sequence-level or semantic-duplicate separation'})
    print('PASS baseline ranks',checked,'renderer PNGs',4,'new inputs',len(manifest),'max distance error',max_error)
