"""Recompute real-rough rankings and verify every saved comparison asset."""
import argparse
import hashlib
import sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.compare_refinement_real_roughs import read,sha256,save_json,CACHE,CONFIRM,ARMS
from scripts.experiment_near_camera import near_cameras
from scripts.experiment_z_axis_search import load_library
from scripts.confirm_camera_fit import geometry_hash
from standin_eval.perspective_search import camera_grid,build_camera_index,score_camera_index
from standin_eval.camera_refinement import normalized_joints,refine_costs
from standin_eval.dual_label_search import alternate_labels


def run(args):
    policy=read(args.out/'protocol.json')
    assert sha256(ROOT/'scripts/compare_refinement_real_roughs.py')==policy['script_sha256']
    for p,digest in policy['input_hashes'].items():assert sha256(Path(p))==digest
    for p,digest in policy['confirmed_sources'].items():assert sha256(ROOT/p)==digest
    poses=load_library(args.library_root);assert geometry_hash(poses)==read(CONFIRM/'protocol.json')['hashes']['geometry']
    cameras=camera_grid()['yaw_pitch_perspective_117']+near_cameras()
    index=build_camera_index(poses,cameras);baseindex=(index[0][:117],index[1][:117])
    index4=build_camera_index(poses,camera_grid()['baseline_4'])
    points=normalized_joints(np.stack([p.joints for p in poses]))
    original={r['image']:r for r in read(CACHE)['real']['records']};rows=read(args.out/'rankings.json')
    assert len(rows)==len(original)==len({r['image'] for r in rows})==10
    valid=failures=0;assets=read(args.out/'asset-manifest.json')
    for row in rows:
        item=original[row['image']]
        assert sha256(Path(row['path']))==row['sha256']==item['sha256']
        cached_xy=np.asarray(item['shared_xy'],dtype=np.float32)
        assert hashlib.sha256(cached_xy.tobytes()).hexdigest()==item['shared_xy_sha256']
        assert sha256(args.out/row['sheet'])==row['sheet_sha256']
        with Image.open(args.out/row['sheet']) as im:
            assert im.size==(1500,1120);im.verify()
        if item.get('error'):
            assert row.get('error') and all(a not in row for a in ARMS);failures+=1;continue
        q=np.asarray(item['shared_xy'],dtype=np.float32);mask=np.asarray(item['shared_mask'],bool)
        assert hashlib.sha256(q.tobytes()).hexdigest()==row['shared_xy_sha256']
        assert row['shared_mask']==mask.tolist()
        alt,am=alternate_labels(q,mask)
        base=np.minimum(score_camera_index(baseindex,q,mask),score_camera_index(baseindex,alt,am))
        fit,params=refine_costs(points,index,cameras,q,mask)
        first=int(np.argmin(base));score=.25*base.astype(float)+.75*fit.astype(float)
        order=np.argsort(score,kind='stable').tolist();order=[first]+[j for j in order if j!=first]
        originalcost=score_camera_index(index4,q,mask)
        orders={'anchor_4':np.argsort(originalcost,kind='stable').tolist(),'pooled_117':np.argsort(base,kind='stable').tolist(),ARMS[2]:order}
        with np.load(args.out/'costs'/(Path(row['image']).stem+'.npz'),allow_pickle=False) as saved:
            assert np.array_equal(fit,saved['refined']) and np.array_equal(base,saved['baseline'])
            assert np.array_equal(params,saved['parameters'])
        for arm,indices in orders.items():
            assert [c['pose_id'] for c in row[arm]]==[poses[j].pose_id for j in indices[:5]]
            for c in row[arm]:
                assert poses[c['pose_index']].pose_id==c['pose_id']
                for key in ('preview','canonical_preview'):
                    path=args.out/c[key];assert sha256(path)==assets[path.name]['sha256']
        assert row[ARMS[2]][0]['camera']==row['pooled_117'][0]['camera']
        valid+=1
    for name,asset in assets.items():
        path=args.out/'assets'/name;assert sha256(path)==asset['sha256']
        with Image.open(path) as im:assert im.size==(240,240);im.verify()
    summary=read(args.out/'summary.json')
    assert valid==8 and failures==2 and summary['accuracy'] is None and summary['human_ratings_received']==0
    result={'input_images_verified':10,'normalized_XY_hashes_verified':10,'ranking_and_camera_fits_recomputed':valid,
            'failures_retained':failures,'sheets_checked':10,'preview_assets_checked':len(assets),
            'selected_first_pose_and_view_preserved':valid,'no_accuracy_claim':True,
            'audit_script_sha256':sha256(Path(__file__)),'ranking_sha256':sha256(args.out/'rankings.json')}
    save_json(args.out/'audit.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--library-root',type=Path,required=True)
    run(p.parse_args())
