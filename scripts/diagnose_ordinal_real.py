"""Real rough coverage/candidate-change diagnostics; no original pose labels."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import save_json,sha256
from standin_eval.perspective_search import camera_grid,camera_points,build_camera_index
from standin_eval.ordinal_depth import query_relations,ordinal_penalty,rerank

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--library-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    cache=ROOT/'out/humanart-plus-z-20261012/results.json';items=json.loads(cache.read_text())['real']['records']
    save_json(a.out/'protocol.json',{'stage':'real rough coverage and candidate-change diagnostic, not accuracy',
        'weight':.1,'top_n':50,'agreement_limit':.5,'margin':.1,'views':[4,117],
        'selection':'no ordinal method passed synthetic development; these are unchanged exploratory settings, not adopted improvements',
        'xy':'same cached Human-Art XY, mask and failures','z':'cached corrected RTMW depth',
        'cache_sha256':sha256(cache),'script_sha256':sha256(Path(__file__))})
    poses=load_library(a.library_root);joints=np.stack([p.joints for p in poses]);rows={}
    for n in (4,117):
        cameras=camera_grid()['baseline_4' if n==4 else 'yaw_pitch_perspective_117']
        features,valid=build_camera_index(poses,cameras);zs=np.stack([camera_points(joints,c)[...,2] for c in cameras])
        for item in items:
            row=rows.setdefault(item['image'],{'image':item['image'],'input_path':item['path'],'accuracy':None})
            if item.get('error'):row['error']=item['error'];continue
            xy=np.array(item['shared_xy'],dtype=np.float32);mask=np.array(item['shared_mask'],bool)
            body=np.arange(5,17)[mask[5:]]
            cost=np.linalg.norm(features[:,:,body]-xy[body],axis=-1).mean(axis=-1);cost=np.where(valid,cost,np.inf)
            view=cost.argmin(axis=0);baseline=cost[view,np.arange(len(poses))]
            order=np.argsort(baseline,kind='stable');row[f'xy_{n}']=[poses[int(i)].pose_id for i in order[:5]]
            for gate,limit in [('ungated',None),('aligned',.5)]:
                signs,use=query_relations(item['normalized_z'],mask,item['human']['keypoints2d'],item['rtmw']['keypoints2d'],limit)
                row[gate+'_relations']={'signs':signs.tolist(),'used':use.tolist(),'count':int(use.sum())}
                penalty=ordinal_penalty(zs[view,np.arange(len(poses))],signs,use)
                candidates=rerank(baseline,penalty,.1,50)
                row[f'ordinal_{n}_{gate}']=[poses[int(i)].pose_id for i in candidates[:5]]
    summary={'n':len(rows),'failures':sum('error' in r for r in rows.values()),'accuracy':None,
        'reason':'No verified source-pose IDs; candidate changes are not improvements.'}
    success=[r for r in rows.values() if 'error' not in r]
    for gate in ('ungated','aligned'):
        summary[gate]={'queries_with_relations':sum(r[gate+'_relations']['count']>0 for r in success),
                       'used_pairs':sum(r[gate+'_relations']['count'] for r in success)}
        for n in (4,117):
            summary[gate][f'views_{n}']={'top1_changed':sum(r[f'xy_{n}'][0]!=r[f'ordinal_{n}_{gate}'][0] for r in success),
                'top5_set_changed':sum(set(r[f'xy_{n}'])!=set(r[f'ordinal_{n}_{gate}']) for r in success)}
    save_json(a.out/'results.json',{'summary':summary,'records':list(rows.values())});print(summary)
