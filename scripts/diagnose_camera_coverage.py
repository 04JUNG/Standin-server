"""Post-hoc yaw/pitch/perspective ablation; does not change primary test results."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import save_json
from standin_eval.perspective_search import camera_grid,build_camera_index,score_camera_index,normalize_batch

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--library-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    poses=load_library(a.library_root);rows={}
    for arm in ('yaw_13','yaw_pitch_39'):
        index=build_camera_index(poses,camera_grid()[arm])
        for item in json.loads((a.out/'manifest.json').read_text()):
            name=item['image'];pred=json.loads((a.out/'predictions'/(Path(name).stem+'.json')).read_text())
            row=rows.setdefault(name,{'image':name})
            if pred.get('error'):row['error']=pred['error'];continue
            xy=np.asarray(pred['keypoints2d'],dtype=np.float32);mask=np.asarray(pred['scores'],dtype=np.float32)>=.3
            if not np.isfinite(xy).all() or mask[5:].sum()<6 or not mask[[5,6,11,12]].all():
                row['error']='insufficient Human-Art body joints/anchors';continue
            ranking=np.argsort(score_camera_index(index,normalize_batch(xy),mask),kind='stable')
            row[arm]=[poses[int(i)].pose_id for i in ranking]
        print('rank',arm,flush=True)
    save_json(a.out/'coverage-diagnostic-rankings.json',list(rows.values()))
    # Ground-truth access starts here, after all rankings have been saved.
    answers=json.loads((a.out/'answers.json').read_text());result={}
    main=json.loads((a.out/'results.json').read_text())
    for group in ('perspective','orthographic_control'):
        records=[]
        for answer in answers:
            if answer['group']!=group:continue
            row=rows[answer['image']];r={'image':answer['image'],'pose_id':answer['pose_id']}
            for arm in ('yaw_13','yaw_pitch_39'):
                candidates=row.get(arm,[]);r[arm+'_rank']=candidates.index(answer['pose_id'])+1 if answer['pose_id'] in candidates else None
            records.append(r)
        counts={arm:{f'top{k}':sum(r[arm+'_rank'] is not None and r[arm+'_rank']<=k for r in records) for k in (1,5)} for arm in ('yaw_13','yaw_pitch_39')}
        counts['anchor_4']=main[group]['counts']['anchor_4'];counts['anchor_117']=main[group]['counts']['anchor_117']
        result[group]={'n':len(records),'counts':counts,'records':records};print(group,counts,flush=True)
    result['warning']='Post-hoc descriptive ablation, not a new holdout or a replacement primary test. No parameter was changed.'
    save_json(a.out/'coverage-diagnostics.json',result)
