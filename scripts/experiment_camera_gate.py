"""Known-data development of conditional camera expansion, not new accuracy."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import save_json,sha256
from standin_eval.perspective_search import camera_grid,build_camera_index,score_camera_index,normalize_batch
from standin_eval.camera_gate import use_expanded

DATASETS=['camera-fit-dev-20261012','perspective-images-20261012','perspective-replication-20261012','camera-fit-confirm-20261012']
POLICIES={f'cost{c}_gain{g}':{'min_coarse_cost':c,'min_relative_gain':g}
          for c in (0.,.15,.3,.5) for g in (0.,.15,.3)}


def features(a):
    a.out.mkdir(parents=True,exist_ok=False)
    save_json(a.out/'protocol.json',{'stage':'development ONLY, all previous image results reused and explicitly retired as holdouts',
        'datasets':DATASETS,'policies':POLICIES,
        'rule':'expand only if baseline best cost exceeds threshold AND relative improvement meets threshold',
        'selection':'perspective Top5>=always-expanded; control Top5>=coarse baseline; control Top1>=always-expanded; overall Top5>=always-expanded; must preserve coarse for some and expand for some; maximize total Top5, then total Top1, then minimize expansion',
        'metric':'unchanged Human-Art XY and valid mask; no fitted scale, depth, mirror or view label oracle',
        'cost_warning':'both indices are evaluated to decide; no speedup claim',
        'script_sha256':sha256(Path(__file__)),'gate_sha256':sha256(ROOT/'standin_eval/camera_gate.py'),
        'source_hashes':{d:{f:sha256(ROOT/'out'/d/f) for f in ('protocol.json','answers.json','rankings.json')} for d in DATASETS}})
    poses=load_library(a.library_root);indices={n:build_camera_index(poses,camera_grid()['baseline_4' if n==4 else 'yaw_pitch_perspective_117']) for n in (4,117)}
    records=[]
    for d in DATASETS:
        root=ROOT/'out'/d
        for item in json.loads((root/'manifest.json').read_text()):
            name=item['image'];path=root/'predictions'/(Path(name).stem+'.json');p=json.loads(path.read_text())
            r={'dataset':d,'image':name,'prediction_sha256':sha256(path)}
            if p.get('error'):r['error']=p['error'];records.append(r);continue
            xy=np.array(p['keypoints2d'],dtype=np.float32);mask=np.array(p['scores'],dtype=np.float32)>=.3
            r['shared_xy_sha256']=hashlib.sha256(xy.tobytes()).hexdigest()
            if not np.isfinite(xy).all() or mask[5:].sum()<6 or not mask[[5,6,11,12]].all():
                r['error']='insufficient Human-Art body joints/anchors';records.append(r);continue
            query=normalize_batch(xy)
            for n in (4,117):
                cost=score_camera_index(indices[n],query,mask);order=np.argsort(cost,kind='stable')
                r[f'cost_{n}']=float(cost[order[0]]);r[f'top5_{n}']=[poses[int(i)].pose_id for i in order[:5]]
            records.append(r)
        print('features',d,flush=True)
    save_json(a.out/'features.json',records)


def evaluate(a):
    records=json.loads((a.out/'features.json').read_text())
    truth={(d,r['image']):r for d in DATASETS for r in json.loads((ROOT/'out'/d/'answers.json').read_text())}
    families=[r['family_id'] for r in truth.values()];assert len(set(families))==len(families)//2==220
    scored=[]
    for row in records:
        target=truth[(row['dataset'],row['image'])]
        r={'dataset':row['dataset'],'image':row['image'],'group':target['group'],'family_id':target['family_id'],'arms':{},'expanded':{}}
        if row.get('error'):r['error']=row['error']
        for arm in ['coarse','expanded']+list(POLICIES):
            expand=arm=='expanded' or (arm in POLICIES and not row.get('error') and use_expanded(row['cost_4'],row['cost_117'],**POLICIES[arm]))
            candidates=row.get('top5_117' if expand else 'top5_4',[])
            r['arms'][arm]={'top1':bool(candidates and candidates[0]==target['pose_id']),'top5':target['pose_id'] in candidates}
            r['expanded'][arm]=bool(expand and not row.get('error'))
        scored.append(r)
    summary={}
    for arm in ['coarse','expanded']+list(POLICIES):
        summary[arm]={}
        for group in ('perspective','orthographic_control'):
            rows=[r for r in scored if r['group']==group]
            summary[arm][group]={k:sum(r['arms'][arm][k] for r in rows) for k in ('top1','top5')}
            summary[arm][group]['expanded']=sum(r['expanded'][arm] for r in rows)
    eligible=[]
    for arm in POLICIES:
        s=summary[arm];p=s['perspective'];c=s['orthographic_control']
        n=p['expanded']+c['expanded'];maxn=sum('error' not in r for r in scored)
        if p['top5']>=summary['expanded']['perspective']['top5'] and c['top5']>=summary['coarse']['orthographic_control']['top5'] and c['top1']>=summary['expanded']['orthographic_control']['top1'] and p['top5']+c['top5']>=sum(v['top5'] for v in summary['expanded'].values()) and 0<n<maxn:
            eligible.append(arm)
    eligible.sort(key=lambda arm:(sum(v['top5'] for v in summary[arm].values()),sum(v['top1'] for v in summary[arm].values()),-sum(v['expanded'] for v in summary[arm].values())),reverse=True)
    result={'stage':'development only','families':220,'images':len(scored),'failures':sum('error' in r for r in scored),
        'counts':summary,'selected':eligible[0] if eligible else None,'eligible':eligible,'records':scored}
    save_json(a.out/'results.json',result)
    for arm,s in summary.items():print(arm,s,flush=True)
    print('selected',result['selected'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('features','evaluate'))
    p.add_argument('--library-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();globals()[a.stage](a)
