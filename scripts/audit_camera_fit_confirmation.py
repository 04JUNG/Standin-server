"""Verify fixed inputs and baseline equivalence without modifying predictions."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.confirm_camera_fit import frozen_sources,geometry_hash
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import save_json,sha256
from standin_eval.z_axis_search import build_search_index,distances
from standin_eval.perspective_search import normalize_batch


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--library-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    protocol=json.loads((a.out/'protocol.json').read_text());assert protocol['frozen_sources']==frozen_sources()
    poses=load_library(a.library_root);assert geometry_hash(poses)==protocol['hashes']['library_geometry']
    for path,digest in protocol['prior_answers'].items():assert sha256(ROOT/path)==digest
    answers=json.loads((a.out/'answers.json').read_text());manifest=json.loads((a.out/'manifest.json').read_text())
    ranked={r['image']:r for r in json.loads((a.out/'rankings.json').read_text())}
    families={r['family_id'] for r in answers};assert len(families)==120
    assert families==set(protocol['test_families']) and not families&set(protocol['excluded_families'])
    assert len(answers)==len(manifest)==len(ranked)==240
    assert {r['image'] for r in answers}=={r['image'] for r in manifest}==set(ranked)
    old=build_search_index(poses);id_set={p.pose_id for p in poses}
    baseline_checks=0;failures=0
    for item in manifest:
        image=item['image'];assert sha256(a.out/'images'/image)==item['sha256']
        pred_path=a.out/'predictions'/(Path(image).stem+'.json');r=ranked[image]
        assert sha256(pred_path)==r['prediction_sha256'];pred=json.loads(pred_path.read_text())
        if pred.get('error'):
            assert r.get('error');failures+=1;continue
        xy=np.asarray(pred['keypoints2d'],dtype=np.float32);mask=np.asarray(pred['scores'],dtype=np.float32)>=.3
        assert hashlib.sha256(xy.tobytes()).hexdigest()==r['shared_xy_sha256']
        assert mask.tolist()==r['shared_mask']
        if r.get('error'):
            assert not np.isfinite(xy).all() or mask[5:].sum()<6 or not mask[[5,6,11,12]].all()
            assert all(arm not in r for arm in protocol['arms']);failures+=1;continue
        for arm in protocol['arms']:assert len(r[arm])==len(set(r[arm]))==len(poses) and set(r[arm])==id_set
        values=distances(old,normalize_batch(xy),np.zeros(17),valid_mask=mask,z_weight=0)
        seen=set();expected=[]
        for i in np.argsort(values,kind='stable'):
            pid=old.pose_ids[int(i)]
            if pid not in seen:seen.add(pid);expected.append(pid)
            if len(expected)==5:break
        assert r['anchor_4'][:5]==expected,(image,r['anchor_4'][:5],expected)
        baseline_checks+=1
    audit={'frozen_sources_verified':True,'library_geometry_verified':True,'input_images_verified':len(manifest),
        'prediction_files_and_shared_xy_verified':len(ranked),'heldout_families':len(families),
        'overlap_with_prior_families':0,'old_baseline_top5_matches':baseline_checks,'failures_retained':failures}
    save_json(a.out/'audit.json',audit);print(audit)
