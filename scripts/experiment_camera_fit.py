"""Development-only distance selection; inference and ranking do not read answers."""
import argparse,json,sys,time,hashlib
from pathlib import Path
from dataclasses import asdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import render_pose,save_json,sha256
from scripts.experiment_perspective_images import infer
from standin_eval.perspective_search import Camera,camera_grid,build_camera_index,score_camera_index,normalize_batch
from standin_eval.camera_fit import score_fit_camera_index


ARMS={f'{metric}_{views}':{'metric':metric,'views':views}
      for metric in ('anchor','fit','weighted_fit') for views in (4,117)}


def prepare(a):
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'images').mkdir()
    source=ROOT/'out/perspective-coordinates-20261012'
    queries=json.loads((source/'queries.json').read_text())
    poses=load_library(a.library_root);by_id={p.pose_id:p for p in poses}
    assert len(queries)==20
    save_json(a.out/'protocol.json',{'stage':'development ONLY, coordinate-development families reused as new images',
        'selection_gate':'fit_117 or weighted_fit_117 must strictly improve perspective Top5 and not reduce orthographic Top5 versus anchor_117; best total Top5 then total Top1; uniform wins ties',
        'arms':ARMS,'gallery_size':len(poses),'input':'identical frozen Human-Art raw XY in all arms',
        'mask':'scores>=0.3, six body joints and all shoulder/hip anchors required in all arms',
        'fit':'positive scale and translation only, no in-plane rotation, mirrors, label swaps or query GT',
        'weights':'uniform or clipped Human-Art scores in [0,1], NOT probabilities',
        'hashes':{'model':sha256(a.model),'script':sha256(Path(__file__)),
                  'camera_fit':sha256(ROOT/'standin_eval/camera_fit.py'),
                  'source_queries':sha256(source/'queries.json'),
                  'library_db':sha256(a.library_root/'poses.db')}})
    manifest=[];answers=[]
    for i,q in enumerate(queries):
        pose=by_id[q['pose_id']]
        for group,cam in [('perspective',Camera(**q['camera'])),('orthographic_control',Camera((0,45,90,180)[i%4]))]:
            name=f'{group}_{i:03d}.png'
            image,gt=render_pose(pose.joints,cam.yaw,style='rough',seed=20261017+i,pitch=cam.pitch,distance=cam.distance)
            image.save(a.out/'images'/name)
            manifest.append({'image':name,'sha256':sha256(a.out/'images'/name)})
            answers.append({'image':name,'group':group,'pose_id':pose.pose_id,'family_id':pose.family_id,
                            'camera':asdict(cam),'ground_truth_xy':gt.tolist()})
        print('render dev',i+1,flush=True)
    save_json(a.out/'manifest.json',manifest);save_json(a.out/'answers.json',answers)


def rank(a):
    policy=json.loads((a.out/'protocol.json').read_text())
    assert sha256(a.library_root/'poses.db')==policy['hashes']['library_db']
    poses=load_library(a.library_root);rows={};timing={}
    for views in (4,117):
        key='baseline_4' if views==4 else 'yaw_pitch_perspective_117'
        index=build_camera_index(poses,camera_grid()[key])
        for metric in ('anchor','fit','weighted_fit'):
            arm=f'{metric}_{views}';timing[arm]=[]
            for item in json.loads((a.out/'manifest.json').read_text()):
                image=item['image'];pred=json.loads((a.out/'predictions'/(Path(image).stem+'.json')).read_text())
                row=rows.setdefault(image,{'image':image})
                if pred.get('error'):row['error']=pred['error'];continue
                xy=np.asarray(pred['keypoints2d'],dtype=np.float32);scores=np.asarray(pred['scores'],dtype=np.float32)
                mask=scores>=.3
                row['shared_xy_sha256']=hashlib.sha256(xy.tobytes()).hexdigest()
                row['shared_mask']=mask.tolist()
                if not np.isfinite(xy).all() or mask[5:].sum()<6 or not mask[[5,6,11,12]].all():
                    row['error']='insufficient Human-Art body joints/anchors';continue
                start=time.perf_counter()
                if metric=='anchor':cost=score_camera_index(index,normalize_batch(xy),mask)
                else:cost=score_fit_camera_index(index,xy,mask,scores if metric=='weighted_fit' else None)
                order=np.argsort(cost,kind='stable')
                timing[arm].append(time.perf_counter()-start)
                row[arm]=[poses[int(i)].pose_id for i in order if np.isfinite(cost[int(i)])]
            print('rank',arm,flush=True)
    save_json(a.out/'rankings.json',list(rows.values()));save_json(a.out/'timing.json',timing)


def evaluate(a):
    answers={r['image']:r for r in json.loads((a.out/'answers.json').read_text())}
    rows=json.loads((a.out/'rankings.json').read_text());result={}
    for group in ('perspective','orthographic_control'):
        records=[]
        for row in rows:
            target=answers[row['image']]
            if target['group']!=group:continue
            rec={k:target[k] for k in ('image','pose_id','family_id')}
            if 'error' in row:rec['error']=row['error']
            for arm in ARMS:
                candidates=row.get(arm,[])
                rec[arm+'_rank']=candidates.index(target['pose_id'])+1 if target['pose_id'] in candidates else None
            records.append(rec)
        summary={arm:{f'top{k}':sum(r[arm+'_rank'] is not None and r[arm+'_rank']<=k for r in records) for k in (1,5)} for arm in ARMS}
        result[group]={'n':len(records),'extraction_failures':sum('error' in r for r in records),'counts':summary,'records':records}
        print(group,summary,flush=True)
    perspective=result['perspective']['counts'];control=result['orthographic_control']['counts']
    eligible=[arm for arm in ('fit_117','weighted_fit_117') if perspective[arm]['top5']>perspective['anchor_117']['top5'] and control[arm]['top5']>=control['anchor_117']['top5']]
    eligible.sort(key=lambda arm:(perspective[arm]['top5']+control[arm]['top5'],perspective[arm]['top1']+control[arm]['top1']),reverse=True)
    result['selection']={'selected':eligible[0] if eligible else None,'development_only':True,
        'reason':'development gate passed' if eligible else 'no fit variant improves perspective Top5 without reducing control Top5',
        'eligible':eligible}
    save_json(a.out/'results.json',result);print('selection',result['selection'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','infer','rank','evaluate'))
    p.add_argument('--library-root',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();globals()[a.stage](a)
