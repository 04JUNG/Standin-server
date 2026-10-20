"""Separate prepare / image-only inference / blinded ranking / final scoring.

Closed-set held-out query families, not unseen-gallery or human-drawn accuracy.
Camera policy was selected on a disjoint 20-family coordinate development set.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import render_pose,save_json,sha256
from scripts.experiment_humanart_z import read_rgb
from standin_eval.perspective_search import Camera,camera_grid,build_camera_index,score_camera_index,normalize_batch
from standin_eval.z_diagnostics import infer_pose


def prepare(a):
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'images').mkdir()
    source=json.loads((a.development/'protocol.json').read_text())
    poses=load_library(a.library_root)
    by_id={p.pose_id:p for p in poses}
    ids=source['reserved_future_image_test_pose_ids']
    assert len(ids)==len(set(ids))==40
    assert not ({by_id[i].family_id for i in ids}&set(source['dev_families']+source['excluded_families']))
    cameras=[Camera((15,75,135,225)[i%4],(-25,20)[(i//4)%2],(4.5,6.5)[(i//8)%2]) for i in range(40)]
    policy={'stage':'image-only Human-Art XY camera-search evaluation',
        'primary_comparison':['baseline_4','yaw_pitch_perspective_117'],
        'secondary_ablation':['yaw_13','yaw_pitch_39'],
        'selection':'117-view policy selected on separate coordinate development set; no image results inspected',
        'groups':{'perspective':'40 reserved families, off-grid yaw/pitch/distance',
                  'orthographic_control':'same 40 families, old four yaw views, zero pitch, no perspective'},
        'query_cameras':[asdict(c) for c in cameras],
        'gallery_poses':len(poses),'camera_grids':{k:[asdict(c) for c in v] for k,v in camera_grid().items()},
        'human_xy':'RGB Human-Art M 192x256, pixel crop and four rotation hypotheses; median body score selects',
        'mask':'scores>=0.3; >=6 body joints and all four shoulder/hip anchors required',
        'failure':'keep failed images in denominator, no replacement',
        'rank_metric':'mean body joint XY distance after same hip/torso normalization; min across views per pose ID',
        'target':'exact pose ID, no mirror/family credit',
        'uncertainty':'paired bootstrap by family, 10000 resamples, seed20261015; separate groups, do not pool n=80',
        'success_gate':'perspective Top5 improves with paired 95% CI lower bound >0; orthographic Top5 does not decrease',
        'limitations':['procedural ellipsoid contours, not human roughs','head/body frame approximate',
                       'library contains target poses; only query families held out from development',
                       'fixed gallery camera orientation/discretization','not production full pipeline'],
        'hashes':{'development_protocol':sha256(a.development/'protocol.json'),
                  'library_db':sha256(a.library_root/'poses.db'),'model':sha256(a.model),
                  'script':sha256(Path(__file__)),'renderer':sha256(ROOT/'scripts/experiment_z_rendered.py'),
                  'camera_module':sha256(ROOT/'standin_eval/perspective_search.py')}}
    save_json(a.out/'protocol.json',policy)
    answers=[];manifest=[]; previews=[]
    for i,pid in enumerate(ids):
        pose=by_id[pid]
        for group,cam in [('perspective',cameras[i]),('orthographic_control',Camera((0,45,90,180)[i%4]))]:
            name=f'{group}_{i:03d}.png'
            image,gt=render_pose(pose.joints,cam.yaw,size=512,style='rough',seed=20261015+i,
                                 pitch=cam.pitch,distance=cam.distance)
            image.save(a.out/'images'/name)
            manifest.append({'image':name,'sha256':sha256(a.out/'images'/name)})
            answers.append({'image':name,'group':group,'family_id':pose.family_id,'pose_id':pid,
                            'camera':asdict(cam),'ground_truth_xy':gt.tolist()})
            if i<8: previews.append((name,image))
        print('render',i+1,flush=True)
    save_json(a.out/'answers.json',answers)
    save_json(a.out/'manifest.json',manifest)
    sheet=Image.new('RGB',(1024,8*280),'white'); draw=ImageDraw.Draw(sheet)
    for i,(name,img) in enumerate(previews):
        x=(i%4)*256; y=(i//4)*560
        sheet.paste(img.resize((256,256)),(x,y+24));draw.text((x+4,y+4),name,fill='black')
    # Four rows of alternating perspective/control, not an accuracy visualization.
    sheet.crop((0,0,1024,4*560)).save(a.out/'render-check.png')


def infer(a):
    from rtmlib import RTMPose
    manifest=json.loads((a.out/'manifest.json').read_text())
    assert sha256(a.model)==json.loads((a.out/'protocol.json').read_text())['hashes']['model']
    model=RTMPose(str(a.model),model_input_size=(192,256),backend='onnxruntime',device='cpu')
    dest=a.out/'predictions';dest.mkdir(exist_ok=True)
    for item in manifest:
        path=a.out/'images'/item['image']; target=dest/(path.stem+'.json')
        assert sha256(path)==item['sha256']
        if target.exists(): continue
        start=time.perf_counter()
        try:
            rec={'image':path.name,**infer_pose(read_rgb(path),model,crop=True,turns=(0,1,2,3),is_3d=False)}
        except Exception as e:
            rec={'image':path.name,'error':type(e).__name__+': '+str(e)}
        rec['inference_seconds']=time.perf_counter()-start
        save_json(target,rec);print('infer',path.name,flush=True)


def rank(a):
    # Deliberately never opens answers.json: ranking accepts image-derived XY only.
    manifest=json.loads((a.out/'manifest.json').read_text())
    policy=json.loads((a.out/'protocol.json').read_text())
    assert sha256(a.library_root/'poses.db')==policy['hashes']['library_db']
    poses=load_library(a.library_root)
    rows={}; timing={}
    for arm,cameras in camera_grid().items():
        start=time.perf_counter(); index=build_camera_index(poses,cameras)
        timing[arm]={'build_seconds':time.perf_counter()-start,'query_seconds':[]}
        for item in manifest:
            pred=json.loads((a.out/'predictions'/(Path(item['image']).stem+'.json')).read_text())
            rec=rows.setdefault(item['image'],{'image':item['image']})
            if pred.get('error'): rec['error']=pred['error'];continue
            xy=np.asarray(pred['keypoints2d'],dtype=np.float32)
            scores=np.asarray(pred['scores']);mask=scores>=.3
            if not np.isfinite(xy).all() or mask[5:].sum()<6 or not mask[[5,6,11,12]].all():
                rec['error']='insufficient Human-Art body joints/anchors';continue
            shared=normalize_batch(xy)
            start=time.perf_counter()
            distances=score_camera_index(index,shared,mask)
            order=np.argsort(distances,kind='stable')
            timing[arm]['query_seconds'].append(time.perf_counter()-start)
            rec[arm]=[poses[int(i)].pose_id for i in order]
        print('rank',arm,flush=True)
    save_json(a.out/'rankings.json',list(rows.values())); save_json(a.out/'timing.json',timing)


def evaluate(a):
    answers={r['image']:r for r in json.loads((a.out/'answers.json').read_text())}
    ranks=json.loads((a.out/'rankings.json').read_text()); results={}
    for group in ('perspective','orthographic_control'):
        records=[]
        for row in ranks:
            target=answers[row['image']]
            if target['group']!=group: continue
            rec={'image':row['image'],'pose_id':target['pose_id'],'family_id':target['family_id']}
            if row.get('error'): rec['error']=row['error']
            for arm in camera_grid():
                rec[arm+'_rank']=row[arm].index(target['pose_id'])+1 if arm in row else None
                rec[arm+'_top5']=row.get(arm,[])[:5]
            records.append(rec)
        summary={arm:{f'top{k}':sum(r[arm+'_rank'] is not None and r[arm+'_rank']<=k for r in records) for k in (1,5)} for arm in camera_grid()}
        hit=lambda r,arm: r[arm+'_rank'] is not None and r[arm+'_rank']<=5
        pairs=np.array([int(hit(r,'yaw_pitch_perspective_117'))-int(hit(r,'baseline_4')) for r in records])
        rng=np.random.default_rng(20261015)
        boot=pairs[rng.integers(0,len(pairs),size=(10000,len(pairs)))].mean(axis=1)
        interval=np.quantile(boot,[.025,.975]).tolist()
        results[group]={'n':len(records),'extraction_failures':sum('error' in r for r in records),
                        'counts':summary,'top5_gains':int((pairs==1).sum()),'top5_losses':int((pairs==-1).sum()),
                        'top5_delta':float(pairs.mean()),'paired_bootstrap_95ci':interval,'records':records}
        print(group,{k:v for k,v in results[group].items() if k!='records'},flush=True)
    results['gate_passed']=results['perspective']['paired_bootstrap_95ci'][0]>0 and results['orthographic_control']['top5_delta']>=0
    save_json(a.out/'results.json',results)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=('prepare','infer','rank','evaluate'))
    p.add_argument('--library-root',type=Path,required=True)
    p.add_argument('--development',type=Path,default=ROOT/'out/perspective-coordinates-20261012')
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();globals()[args.stage](args)
