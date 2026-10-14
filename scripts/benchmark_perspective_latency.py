"""Local cached-XY latency only; no inference, accuracy evaluation or deployment."""
import argparse
import fcntl
import json
import platform
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_z_rendered import save_json,sha256
from scripts.review_perspective_roughs import best_views,rgb
from standin_eval.perspective_search import Camera,camera_grid,build_camera_index
from standin_eval.z_axis_search import Pose3D
from standin_eval.dual_label_search import alternate_labels
from standin_eval.camera_refinement import normalized_joints,refine_costs
from standin_eval.contour_letterbox import prepare_letterboxed_query
from standin_eval.contour_rerank import candidate_evidence
from src.features import normalize_skeleton


def timed(fn):
    start=time.perf_counter();result=fn();return result,time.perf_counter()-start


def stats(values):
    return {'n':len(values),'median_ms':float(np.median(values)*1000),'p95_ms':float(np.percentile(values,95)*1000),'mean_ms':float(np.mean(values)*1000)}


def run(a):
    rows=json.loads((a.source/'rankings.json').read_text());rows=[r for r in rows if not r.get('error')]
    input_files=[a.gallery,a.source/'rankings.json']+[a.source/'predictions'/(r['id']+'.json') for r in rows]+[Path(r['path']) for r in rows]
    hashes={str(p):sha256(p) for p in input_files}
    a.out.mkdir(parents=True,exist_ok=False)
    code=('scripts/benchmark_perspective_latency.py','scripts/review_perspective_roughs.py','standin_eval/camera_refinement.py','standin_eval/contour_rerank.py','standin_eval/contour_letterbox.py')
    policy={'purpose':'read-only local performance diagnostic, no accuracy or new holdout',
            'queries':[r['id'] for r in rows],'selection':'all searchable9 of original10;178 original failure remains excluded from algorithm timing, not replaced',
            'repetitions':a.repetitions,'warmup':'one complete run for first searchable cached query, excluded',
            'arms':['4 single-label cached XY scan','117 single-label cached XY scan','117 dual-label cached XY scan','117 dual+195-camera refinement+.25/.75 blend+Top20 contour with first free'],
            'order':'reverse arm order every other query/repetition to limit fixed-order bias; serial local execution',
            'timed_scope':'algorithm operations only; input image/model/prediction loading and index construction excluded; separately report cached index construction',
            'excluded':['Human-Art inference','VLM','network','API queue','DB read','manual-cue gallery predicates','basic-mannequin display/JPEG delivery'],
            'limits':['small repeated local workload, not production p95 or load test','latest unconstrained research stack, not every manual-feature branch','contour errors trigger full contour fallback and are counted'],
            'machine':{'platform':platform.platform(),'python':platform.python_version()},
            'inputs':hashes,'code':{n:sha256(ROOT/n) for n in code}}
    save_json(a.out/'protocol.json',policy)
    with np.load(a.gallery,allow_pickle=False) as d:poses=[Pose3D(str(pid),str(fam),p,s) for pid,fam,p,s in zip(d['pose_ids'],d['family_ids'],d['joints'],d['scores'])]
    assert len(poses)==1981
    cameras117=camera_grid()['yaw_pitch_perspective_117'];cameras195=cameras117+[Camera(c.yaw,c.pitch,d) for c in camera_grid()['yaw_pitch_39'] for d in (2.,3.)]
    index4,t4=timed(lambda:build_camera_index(poses,camera_grid()['baseline_4']))
    index195,t195=timed(lambda:build_camera_index(poses,cameras195));index117=(index195[0][:117],index195[1][:117])
    points=normalized_joints(np.stack([p.joints for p in poses]));queries=[]
    for row in rows:
        pred=json.loads((a.source/'predictions'/(row['id']+'.json')).read_text());xy=np.asarray(pred['keypoints2d'],np.float32);scores=np.asarray(pred['scores'],np.float32);mask=scores>=.3
        q=normalize_skeleton(xy,scores,valid_mask=mask).reshape(17,2);alt,am=alternate_labels(q,mask)
        assert mask[5:].sum()>=6 and mask[[5,6,11,12]].all()
        queries.append((row['id'],q,mask,alt,am,rgb(Path(row['path'])),xy))
    def stack(query):
        name,q,mask,alt,am,image,raw_xy=query;start=time.perf_counter()
        direct=best_views(index117,q,mask)[0];other=best_views(index117,alt,am)[0];coarse_end=time.perf_counter()
        fit,params=refine_costs(points,index195,cameras195,q,mask);fit_end=time.perf_counter()
        score=.25*np.minimum(direct,other).astype(float)+.75*fit.astype(float);head=np.argsort(score,kind='stable')[:20];errors=[];penalties=np.zeros(20)
        try:
            evidence,_=prepare_letterboxed_query(image,raw_xy)
            for k,j in enumerate(head):
                try:penalties[k]=candidate_evidence(evidence,poses[j].joints,params[j])[0]
                except ValueError as error:errors.append(str(error))
        except ValueError as error:errors.append(str(error))
        final=np.argsort(score[head]+(0 if errors else penalties),kind='stable');assert len(final)==20
        end=time.perf_counter()
        return {'total_seconds':end-start,'dual117_seconds':coarse_end-start,'refine_seconds':fit_end-coarse_end,
                'contour_and_sort_seconds':end-fit_end,'contour_fallback':bool(errors),'contour_error_count':len(errors)}
    samples=[]
    stack(queries[0]);best_views(index4,queries[0][1],queries[0][2]);best_views(index117,queries[0][1],queries[0][2])
    arms=('four','single117','dual117','full_stack')
    for repetition in range(a.repetitions):
        for i,query in enumerate(queries):
            name,q,mask,alt,am,_,_=query;sample={'query':name,'repeat':repetition}
            for arm in (arms if (i+repetition)%2==0 else arms[::-1]):
                if arm=='full_stack':sample[arm]=stack(query)
                else:
                    fn=(lambda:best_views(index4,q,mask)) if arm=='four' else (lambda:best_views(index117,q,mask)) if arm=='single117' else (lambda:(best_views(index117,q,mask),best_views(index117,alt,am)))
                    _,seconds=timed(fn);sample[arm]=seconds
            samples.append(sample)
        print('completed timing repetition',repetition+1,flush=True)
    summaries={arm:stats([s[arm] for s in samples]) for arm in ('four','single117','dual117')}
    for phase in ('total_seconds','dual117_seconds','refine_seconds','contour_and_sort_seconds'):
        summaries[phase]=stats([s['full_stack'][phase] for s in samples])
    delta=stats([s['full_stack']['total_seconds']-s['four'] for s in samples])
    result={'gallery_count':len(poses),'query_count':len(queries),'samples':len(samples),'summaries':summaries,
            'paired_full_stack_minus_four':delta,'index_build_seconds':{'four':t4,'all195':t195},
            'contour_fallback_count':sum(s['full_stack']['contour_fallback'] for s in samples),
            'production_latency':None,'rankings_modified':False}
    save_json(a.out/'samples.json',samples);save_json(a.out/'results.json',result)
    for p,h in hashes.items():assert sha256(Path(p))==h
    for n,h in policy['code'].items():assert sha256(ROOT/n)==h
    print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for n in ('source','gallery','out'):parser.add_argument('--'+n,type=Path,required=True)
    parser.add_argument('--repetitions',type=int,default=3);a=parser.parse_args();assert a.repetitions>0
    with (ROOT/'out/.perspective-rough-review.lock').open('a+') as rough,(ROOT/'out/.joint-shape.lock').open('a+') as shape,(ROOT/'out/.first-rank-confirm.lock').open('a+') as fresh:
        for lock in (rough,shape,fresh):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(a)
