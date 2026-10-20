"""Read-only rank diagnosis on reviewed roughs: does one cue sacrifice others?"""
import argparse
import fcntl
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_z_rendered import save_json,sha256
from src.features import normalize_skeleton
from standin_eval.perspective_search import Camera,project,normalize_batch
from standin_eval.body_part_evidence import body_part_errors


def read(p):return json.loads(p.read_text())


def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    files=[a.gallery,a.original/'rankings.json',a.before/'rankings.json',a.after/'rankings.json']
    files+=sorted((a.original/'predictions').glob('*.json'))+sorted((a.original/'costs').glob('*.npz'))
    hashes={str(p):sha256(p) for p in files}
    save_json(a.out/'protocol.json',{'purpose':'post-review diagnostic only; not new ranking or accuracy evaluation',
        'fixed':'historical1981 gallery, original Human-Art XY/mask; compare stored previous joint-shape Top1 to current manually constrained Top1',
        'regions':'mean L2 over observed normalized XY; one whole-body direct/swap correspondence, then same mapping for all regions',
        'query178':'excluded from paired XY comparison because after image crop changed inference',
        'features_vs_whole_pose':'report body,arms,legs,torso costs together; do not tune thresholds or choose a policy from these9 observations',
        'limits':['model-relative mismatch, not human relevance or source-ID correctness','manual constraints add information','camera-relative torso tilt is not anatomical bend'],
        'files':hashes,'code':{n:sha256(ROOT/n) for n in ('scripts/diagnose_manual_feature_tradeoffs.py','standin_eval/body_part_evidence.py')}})
    with np.load(a.gallery,allow_pickle=False) as d:ids=d['pose_ids'].tolist();joints=d['joints']
    original={r['id']:r for r in read(a.original/'rankings.json')};before={r['id']:r for r in read(a.before/'rankings.json')};rows=[]
    for after in read(a.after/'rankings.json'):
        name=after['id'];old=original[name];row={'id':name}
        if old.get('error'):row['excluded']='original extraction failure; new crop XY is a separate input arm';rows.append(row);continue
        pred=read(a.original/'predictions'/(name+'.json'));xy=np.asarray(pred['keypoints2d'],np.float32);confidence=np.asarray(pred['scores'],np.float32);mask=confidence>=.3
        q=normalize_skeleton(xy,confidence,valid_mask=mask).reshape(17,2)
        assert hashlib.sha256(q.tobytes()).hexdigest()==old['shared_xy_sha256']
        with np.load(a.original/'costs'/(name+'.npz'),allow_pickle=False) as d:params=d['parameters'].copy()
        oldid=before[name]['top5']['joint_shape'][0];j=ids.index(oldid);yaw,pitch,inverse=map(float,params[j]);oldcam=Camera(yaw,pitch,None if inverse==0 else 1/inverse)
        new=after['candidates'][0];k=ids.index(new['pose_id']);newcam=Camera(**new['camera'])
        views={}
        for method,index,cam in (('before',j,oldcam),('after',k,newcam)):
            projected,valid=project(joints[index],cam);assert valid
            views[method]=body_part_errors(q,mask,normalize_batch(projected))
        row.update(before_pose_id=oldid,after_pose_id=new['pose_id'],parts=views,
                   changes={part:views['after'][part]-views['before'][part] for part in ('body','arms','legs','torso')})
        rows.append(row)
    assert len(rows)==10
    result={'records':rows,'accuracy':None,'ranking_changed_by_this_diagnostic':False,
            'paired_queries':sum('parts' in r for r in rows),'unchanged_input_mask':True,
            'region_worsened_counts':{part:sum(r.get('changes',{}).get(part,0)>1e-5 for r in rows) for part in ('body','arms','legs','torso')},
            'source_hashes_unchanged':all(sha256(Path(p))==h for p,h in hashes.items())}
    assert result['source_hashes_unchanged'];save_json(a.out/'results.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('gallery','original','before','after','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    # Same per-study locks prevent accidental overlap with the preceding runs.
    with (ROOT/'out/.perspective-rough-review.lock').open('a+') as rough_lock, (ROOT/'out/.joint-shape.lock').open('a+') as shape_lock, (ROOT/'out/.first-rank-confirm.lock').open('a+') as fresh_lock:
        for lock in (rough_lock,shape_lock,fresh_lock):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(a)
