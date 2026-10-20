"""Post-test oracle diagnostics. Never used to choose test ranking parameters."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import save_json
from standin_eval.perspective_search import camera_grid,build_camera_index,score_camera_index,normalize_batch

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--library-root',type=Path,required=True);a=p.parse_args()
    answers=json.loads((a.out/'answers.json').read_text())
    poses=load_library(a.library_root);positions={p.pose_id:i for i,p in enumerate(poses)}
    indices={arm:build_camera_index(poses,cameras) for arm,cameras in camera_grid().items()}
    records=[]
    for answer in answers:
        pred=json.loads((a.out/'predictions'/(Path(answer['image']).stem+'.json')).read_text())
        gt=normalize_batch(np.array(answer['ground_truth_xy']))
        rec={k:answer[k] for k in ('image','group','pose_id')}
        if not pred.get('error'):
            xy=normalize_batch(np.array(pred['keypoints2d'])); mask=np.array(pred['scores'])>=.3
            rec['mean_xy_error_torso']=float(np.linalg.norm(xy[5:]-gt[5:],axis=-1).mean())
            rec['turn']=pred['turn']
        for arm,index in indices.items():
            ranking=np.argsort(score_camera_index(index,gt),kind='stable')
            rec[arm+'_oracle_rank']=int(np.flatnonzero(ranking==positions[answer['pose_id']])[0])+1
        records.append(rec)
    summaries={}
    for group in ('perspective','orthographic_control'):
        rows=[r for r in records if r['group']==group]
        summary={'n':len(rows),'oracle_counts':{arm:{f'top{k}':sum(r[arm+'_oracle_rank']<=k for r in rows) for k in (1,5)} for arm in indices},
                 'median_xy_error_torso':float(np.median([r['mean_xy_error_torso'] for r in rows if 'mean_xy_error_torso' in r]))}
        summaries[group]=summary;print(group,summary,flush=True)
    save_json(a.out/'oracle-diagnostics.json',{'warning':'Uses exact source XY after evaluation. NOT image accuracy, NOT a deployed correction.',
        'summary':summaries,'records':records})
