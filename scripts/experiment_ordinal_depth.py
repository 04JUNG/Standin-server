"""Reused-cache DEVELOPMENT ONLY: ordinal Z versus frozen Human-Art XY."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import save_json,sha256
from standin_eval.perspective_search import Camera,camera_grid,camera_points,build_camera_index
from standin_eval.ordinal_depth import PAIRS,query_relations,ordinal_penalty,rerank


def rank(a):
    a.out.mkdir(parents=True,exist_ok=False)
    cache=ROOT/'out/humanart-plus-z-20261012/results.json'
    records=json.loads(cache.read_text())['synthetic']['records']
    # Do not select a variant based on answers or inspect new holdout data.
    save_json(a.out/'protocol.json',{'stage':'development only; reuse 40 already-examined synthetic queries',
        'hypothesis':'relative depth order may survive metric-depth error',
        'pairs':PAIRS.tolist(),'margin_normalized_torso':.1,'query_torso_prior_m':.5,
        'camera_grids':[4,117],'rerank_top_n':50,'weights':[.1,.25],
        'gates':{'ungated':None,'aligned':.5},
        'gate_definition':'both pair endpoints must agree in Human-Art and RTMW XY within 0.5 Human-Art torso lengths; not calibrated depth confidence',
        'query':'identical cached Human-Art XY and mask, corrected RTMW normalized depth',
        'view_selection':'minimum XY-only distance per pose; depth does not choose a different view',
        'unknown':'query depth difference below margin ignored; gallery unknown relation costs 0.5',
        'selection_rule':'117-view variant must gain >=2 Top5 and not lose Top1 over 117-view XY, and eligible depth-order agreement>0.6; then fresh verification required',
        'hashes':{'cache':sha256(cache),'script':sha256(Path(__file__)),
                  'ordinal':sha256(ROOT/'standin_eval/ordinal_depth.py'),'library_db':sha256(a.library_root/'poses.db')}})
    poses=load_library(a.library_root);joints=np.stack([p.joints for p in poses]);rows={}
    for size in (4,117):
        cameras=camera_grid()['baseline_4' if size==4 else 'yaw_pitch_perspective_117']
        features,valid=build_camera_index(poses,cameras)
        zs=np.stack([camera_points(joints,c)[...,2] for c in cameras])
        for cached in records:
            name=cached['image'];row=rows.setdefault(name,{'image':name,'relations':{}})
            if cached.get('error'):row['error']=cached['error'];continue
            xy=np.asarray(cached['shared_xy'],dtype=np.float32);mask=np.asarray(cached['shared_mask'],bool)
            body=np.arange(5,17)[mask[5:]]
            by_view=np.linalg.norm(features[:,:,body]-xy[body],axis=-1).mean(axis=-1)
            by_view=np.where(valid,by_view,np.inf);best_view=np.argmin(by_view,axis=0)
            costs=by_view[best_view,np.arange(len(poses))];best_z=zs[best_view,np.arange(len(poses))]
            row[f'xy_{size}']=[poses[int(i)].pose_id for i in np.argsort(costs,kind='stable')]
            for gate,limit in [('ungated',None),('aligned',.5)]:
                signs,use=query_relations(cached['normalized_z'],mask,cached['human']['keypoints2d'],
                    cached['rtmw']['keypoints2d'],limit)
                row['relations'][gate]={'signs':signs.tolist(),'used':use.tolist(),'count':int(use.sum())}
                penalty=ordinal_penalty(best_z,signs,use)
                for w in (.1,.25):
                    arm=f'ordinal_{size}_{gate}_{w}'
                    row[arm]=[poses[int(i)].pose_id for i in rerank(costs,penalty,w,50)]
        print('rank',size,flush=True)
    save_json(a.out/'rankings.json',list(rows.values()))


def evaluate(a):
    ranked=json.loads((a.out/'rankings.json').read_text())
    answers={r['image']:r for r in json.loads((ROOT/'out/z-diagnosis-20261012-heldout/answers.json').read_text())}
    poses={p.pose_id:p for p in load_library(a.library_root)}
    old={r['image']:r for r in json.loads((ROOT/'out/humanart-plus-z-20261012/results.json').read_text())['synthetic']['records']}
    arms=[k for k in ranked[0] if k.startswith(('xy_','ordinal_'))]
    records=[];diagnostics={g:[] for g in ('ungated','aligned')}
    for row in ranked:
        answer=answers[row['image']];r={'image':row['image'],'pose_id':answer['pose_id']}
        if row.get('error'):r['error']=row['error']
        for arm in arms:
            candidates=row.get(arm,[]);r[arm+'_rank']=candidates.index(answer['pose_id'])+1 if answer['pose_id'] in candidates else None
        assert r['xy_4_rank']==old[row['image']].get('xy_rank')
        records.append(r)
        truth=camera_points(poses[answer['pose_id']].joints,Camera(answer['yaw']))[:,2]
        delta=truth[PAIRS[:,0]]-truth[PAIRS[:,1]]
        for gate,rel in row.get('relations',{}).items():
            used=np.array(rel['used'],bool);eligible=used&(np.abs(delta)>=.1)
            signs=np.array(rel['signs']);n=int(eligible.sum())
            diagnostics[gate].append({'image':row['image'],'used':int(used.sum()),'eligible':n,
                'correct':int((signs[eligible]==np.sign(delta[eligible])).sum()),
                'always_front_correct':int((delta[eligible]<0).sum()),'always_back_correct':int((delta[eligible]>0).sum())})
    counts={arm:{f'top{k}':sum(r[arm+'_rank'] is not None and r[arm+'_rank']<=k for r in records) for k in (1,5)} for arm in arms}
    order_summary={}
    for gate,rows in diagnostics.items():
        n=sum(r['eligible'] for r in rows);correct=sum(r['correct'] for r in rows)
        order_summary[gate]={'queries_with_relations':sum(r['used']>0 for r in rows),'used_pairs':sum(r['used'] for r in rows),
            'eligible_pairs_with_gt_order':n,'correct_pairs':correct,'agreement':correct/n if n else None,
            'always_front_agreement':sum(r['always_front_correct'] for r in rows)/n if n else None,
            'always_back_agreement':sum(r['always_back_correct'] for r in rows)/n if n else None}
    selection=[]
    for arm in arms:
        if not arm.startswith('ordinal_117_'):continue
        gate=arm.split('_')[2];acc=order_summary[gate]['agreement']
        if counts[arm]['top5']>=counts['xy_117']['top5']+2 and counts[arm]['top1']>=counts['xy_117']['top1'] and acc is not None and acc>.6:
            selection.append(arm)
    results={'n':len(records),'counts':counts,'order_diagnostics':order_summary,'eligible_variants':selection,
        'development_only':True,'records':records,'pair_diagnostics':diagnostics,
        'warning':'GT used only after saved rankings for scoring/diagnostics; conditional pair agreement not image retrieval accuracy or independent-pair statistics'}
    save_json(a.out/'results.json',results)
    print('counts',counts,flush=True);print('order',order_summary,flush=True);print('eligible',selection,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('rank','evaluate'))
    p.add_argument('--library-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();globals()[a.stage](a)
