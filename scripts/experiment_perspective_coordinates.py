"""Coordinate-only camera coverage diagnostic, NOT image/rough accuracy."""
import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import save_json,sha256
from standin_eval.perspective_search import Camera,project,normalize_batch,camera_grid,build_camera_index,score_camera_index


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--library-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--exclude-answers',nargs='+',type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    poses=load_library(a.library_root)
    excluded={r['family_id'] for path in a.exclude_answers for r in json.loads(path.read_text())}
    families={}
    for pose in poses:
        if pose.family_id not in excluded: families.setdefault(pose.family_id,pose)
    pool=list(families.values()); rng=np.random.default_rng(20261014)
    chosen=[pool[int(i)] for i in rng.choice(len(pool),60,replace=False)]
    dev=chosen[:20]; reserved=chosen[20:]
    grids=camera_grid()
    save_json(a.out/'protocol.json',{'stage':'coordinate-only development diagnostic; no Human-Art inference',
        'seed':20261014,'gallery_poses':len(poses),'dev_families':[p.family_id for p in dev],
        'reserved_future_image_test_families':[p.family_id for p in reserved],
        'reserved_future_image_test_pose_ids':[p.pose_id for p in reserved],
        'excluded_families':sorted(excluded),'camera_grids':{k:[asdict(c) for c in v] for k,v in grids.items()},
        'query_yaw':[15,75,135,225],'query_pitch':[-25,20],'query_distance':[4.5,6.5],
        'query_camera_not_equal_to_any_gallery_camera':True,'noise_xy_torso':[0,.08],
        'warning':'Queries projected from GT XYZ. This tests gallery coverage only, NOT real image performance.',
        'script_sha256':sha256(Path(__file__))})
    queries=[]
    for i,pose in enumerate(dev):
        cam=Camera((15,75,135,225)[i%4],(-25,20)[(i//4)%2],(4.5,6.5)[(i//8)%2])
        xy,valid=project(pose.joints,cam)
        if not valid: raise ValueError('invalid predetermined camera; do not silently resample')
        norm=normalize_batch(xy)
        noise=rng.normal(0,.08,size=norm.shape).astype(np.float32)
        queries.append({'pose_id':pose.pose_id,'camera':asdict(cam),'clean':norm,'noisy':normalize_batch(norm+noise)})
    save_json(a.out/'queries.json',[{**q,'clean':q['clean'].tolist(),'noisy':q['noisy'].tolist()} for q in queries])
    results={}
    for name,cameras in grids.items():
        ix=build_camera_index(poses,cameras); rows=[]
        for q in queries:
            row={'pose_id':q['pose_id']}
            for mode in ('clean','noisy'):
                order=np.argsort(score_camera_index(ix,q[mode]),kind='stable')
                row[mode]=next(i+1 for i,pid in enumerate(order) if poses[int(pid)].pose_id==q['pose_id'])
            rows.append(row)
        summary={mode:{'top1':sum(r[mode]==1 for r in rows)/len(rows),'top5':sum(r[mode]<=5 for r in rows)/len(rows)} for mode in ('clean','noisy')}
        results[name]={'summary':summary,'records':rows}
        print(name,summary,flush=True)
    save_json(a.out/'results.json',results)


if __name__=='__main__': main()
