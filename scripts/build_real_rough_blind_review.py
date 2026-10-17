"""Apply frozen retrieval to existing roughs; no fabricated relevance labels."""
import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import render_pose,save_json,sha256
from scripts.confirm_camera_fit import geometry_hash
from standin_eval.perspective_search import camera_grid,build_camera_index
from standin_eval.dual_label_search import alternate_labels,merged_order

ARMS=('anchor_4','anchor_117','pooled_117')


def run(a):
    cache=ROOT/'out/humanart-plus-z-20261012/results.json'
    confirmed=json.loads((ROOT/'out/dual-label-confirm-balanced-20261012/protocol.json').read_text())
    for module in ('standin_eval/dual_label_search.py','standin_eval/perspective_search.py'):
        assert sha256(ROOT/module)==confirmed['frozen_sources'][module]
    items=json.loads(cache.read_text())['real']['records'];assert len(items)==10
    poses=load_library(a.library_root);assert geometry_hash(poses)==confirmed['hashes']['library_geometry']
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'assets').mkdir()
    study_id='real-rough-dual-20261012-v1'
    policy={'study_id':study_id,'stage':'human relevance review pending; no known source3D GT',
        'arms':ARMS,'queries':10,'seed':20261025,'cache_sha256':sha256(cache),
        'gallery_geometry_sha256':geometry_hash(poses),
        'rules':'same cached Human-Art XY and failures; three frozen methods, top5 exact IDs; randomized method positions per query; blinded names and costs',
        'ratings':'yes/no/uncertain with no default; all 120 candidates must be reviewed before primary aggregation',
        'metric':'human-rated useful Top1/Top5; unknowns retained as uncertain, not silently counted as no; extraction failures count as no',
        'limits':['already explored convenience sample, not independent real-world test',
                  'approximate BVH body previews; face/hands excluded',
                  'human self-reported ratings, not objective original-ID accuracy',
                  'rough_309 has rough-related gallery metadata; not an unseen-source success'],
        'frozen_sources':{p:sha256(ROOT/p) for p in ('scripts/build_real_rough_blind_review.py','scripts/templates/rough_blind_review.html','standin_eval/dual_label_search.py')}}
    save_json(a.out/'protocol.json',policy)
    grids={4:camera_grid()['baseline_4'],117:camera_grid()['yaw_pitch_perspective_117']}
    indices={n:build_camera_index(poses,c) for n,c in grids.items()}
    records=[]
    for item in items:
        source=Path(item['path']);assert sha256(source)==item['sha256']
        row={'image':item['image'],'input_path':str(source),'input_sha256':item['sha256']}
        if item.get('error'):row['error']=item['error'];records.append(row);continue
        xy=np.asarray(item['shared_xy'],dtype=np.float32);mask=np.asarray(item['shared_mask'],bool)
        assert hashlib.sha256(xy.tobytes()).hexdigest()==item['shared_xy_sha256']
        row['shared_xy_sha256']=item['shared_xy_sha256'];row['shared_mask']=mask.tolist()
        alt,am=alternate_labels(xy,mask)
        cache_cost={}
        for n in (4,117):
            features,valid=indices[n];sel=np.arange(5,17)[mask[5:]]
            costs=np.linalg.norm(features[:,:,sel]-xy[sel],axis=-1).mean(axis=-1)
            costs=np.where(valid,costs,np.inf);best=costs.argmin(axis=0)
            direct=costs[best,np.arange(len(poses))];cache_cost[n]=(direct,best)
            order=np.argsort(direct,kind='stable')
            row[f'anchor_{n}']=[{'pose_id':poses[int(i)].pose_id,'pose_index':int(i),'camera':asdict(grids[n][int(best[i])]),'distance':float(direct[i]),'hypothesis':'direct'} for i in order[:5]]
        assert [c['pose_id'] for c in row['anchor_4']]==[c['pose_id'] for c in item['xy'][:5]]
        features,valid=indices[117];sel=np.arange(5,17)[am[5:]]
        costs=np.linalg.norm(features[:,:,sel]-alt[sel],axis=-1).mean(axis=-1)
        costs=np.where(valid,costs,np.inf);abest=costs.argmin(axis=0)
        other=costs[abest,np.arange(len(poses))];direct,best=cache_cost[117]
        order=merged_order(direct,other);out=[]
        for i in order[:5]:
            alternate=bool(other[i]<direct[i]);vi=int(abest[i] if alternate else best[i])
            out.append({'pose_id':poses[int(i)].pose_id,'pose_index':int(i),'camera':asdict(grids[117][vi]),
                        'distance':float(min(direct[i],other[i])),'hypothesis':'alternate' if alternate else 'direct'})
        row['pooled_117']=out;records.append(row)
    save_json(a.out/'rankings-private.json',records)
    rng=np.random.default_rng(policy['seed']);queries=[];key={};all_ids=[];render_cache={}
    for number,index in enumerate(rng.permutation(len(records)),1):
        row=records[int(index)];qid=f'Q{number:02d}'
        input_name=f'assets/{qid}-input.png';shutil.copyfile(row['input_path'],a.out/input_name)
        q={'id':qid,'input':input_name,'failed':bool(row.get('error')),'groups':[]}
        key[qid]={'image':row['image'],'failed':q['failed'],'slots':{}}
        methods=[ARMS[int(i)] for i in rng.permutation(3)]
        for slot,arm in zip('ABC',methods):
            key[qid]['slots'][slot]=arm
            if q['failed']:continue
            candidates=[]
            for rank,c in enumerate(row[arm],1):
                cid=f'{qid}-{slot}-{rank}';all_ids.append(cid)
                filename='assets/'+hashlib.sha256(cid.encode()).hexdigest()[:16]+'.png'
                cachekey=(c['pose_id'],json.dumps(c['camera'],sort_keys=True))
                if cachekey not in render_cache:
                    camera=c['camera'];im,_=render_pose(poses[c['pose_index']].joints,camera['yaw'],size=448,
                        pitch=camera['pitch'],distance=camera['distance'])
                    render_cache[cachekey]=im
                render_cache[cachekey].save(a.out/filename)
                candidates.append({'id':cid,'image':filename})
            q['groups'].append({'slot':slot,'candidates':candidates})
        queries.append(q);print('review',qid,'failed' if q['failed'] else '15 candidates',flush=True)
    public={'study_id':study_id,'queries':queries,'candidate_ids':all_ids,'failed_queries':sum(q['failed'] for q in queries)}
    public['public_data_sha256']=hashlib.sha256(json.dumps(public,sort_keys=True).encode()).hexdigest()
    save_json(a.out/'review-data.json',public)
    save_json(a.out/'blinding-key-private.json',{'study_id':study_id,'public_data_sha256':public['public_data_sha256'],'queries':key})
    template=(ROOT/'scripts/templates/rough_blind_review.html').read_text()
    html=template.replace('__STUDY_DATA__',json.dumps(public,ensure_ascii=False).replace('<','\\u003c'))
    (a.out/'review.html').write_text(html)
    valid=[r for r in records if 'error' not in r]
    summary={'queries':10,'searchable':len(valid),'failed':10-len(valid),'human_ratings_received':0,'accuracy':None,
        'pairwise_changes':{f'{arm}_vs_{base}':{
            'top1_changed':sum(r[arm][0]['pose_id']!=r[base][0]['pose_id'] for r in valid),
            'top5_set_changed':sum({c['pose_id'] for c in r[arm]}!={c['pose_id'] for c in r[base]} for r in valid)}
            for arm,base in [('anchor_117','anchor_4'),('pooled_117','anchor_117'),('pooled_117','anchor_4')]},
        'warning':'Candidate changes are not improvement; no human ratings have been filled in.'}
    save_json(a.out/'summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--library-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);run(p.parse_args())
