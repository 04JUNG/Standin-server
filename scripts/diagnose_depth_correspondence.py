"""Old cached Z experiment: XY-only model correspondence, no retrieval changes."""
import argparse
import fcntl
import sys
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiment_z_rendered import save_json,sha256
from scripts.experiment_z_axis_search import load_library
from standin_eval.depth_correspondence import map_depth
from standin_eval.ordinal_depth import PAIRS
from standin_eval.dual_label_search import LEFT_RIGHT
from standin_eval.perspective_search import camera_points,Camera

CACHE=ROOT/'out/humanart-plus-z-20261012/results.json'
ANSWERS=ROOT/'out/z-diagnosis-20261012-heldout/answers.json'
OLD=ROOT/'out/ordinal-depth-dev-20261012/results.json'
ARMS=('direct','xy_aligned');GATES=('all','xy_gate')


def read(p):return json.loads(p.read_text())


def verify(args):
    p=read(args.out/'protocol.json')
    assert all(sha256(Path(n))==h for n,h in p['hashes'].items())


def predict(args):
    args.out.mkdir(parents=True,exist_ok=False)
    paths=[CACHE,ANSWERS,OLD,args.library_root/'poses.db',Path(__file__),ROOT/'standin_eval/depth_correspondence.py',
           ROOT/'standin_eval/ordinal_depth.py',ROOT/'standin_eval/dual_label_search.py',ROOT/'standin_eval/perspective_search.py']
    save_json(args.out/'protocol.json',{
        'stage':'posthoc known-development diagnostic; cached synthetic40 and real10, no inference/search changes',
        'mapping':'direct vs whole-body LR swap of RTMW output minimizing mean pixel XY distance over fixed Human-Art valid body slots; divide by Human-Art projected torso; tie direct; no fit/translation/GT',
        'depth':'raw corrected RTMW z_meters permuted BEFORE Human-Art masking, centered on mapped hips then /0.5m same prior',
        'eligibility':'Human-Art original mask, abs(querydepthdifference)>=.1; optional both endpoint XY errors<=.5 HumanArt torso; no RTMWscore gate',
        'truth':'eight same-side limb-anchor pairs, true difference>=.1 normalized3Dtorso; original source yaw, no pitch/perspective in this40',
        'truth_frames':'semanticCOCO replicates old diagnostic; observedHumanFrame uses GT ONLY to choose global HumanArt-to-projectedGT mapping for diagnostic scoring, never mapping RTMW outputs',
        'paired':'report common eligible pairs within each frame/gate and each arm own denominator; constant front/back controls on exact same subsets',
        'failure':'retain all50 original records; real2 nonsearchable remain failures with no correspondence output',
        'limits':'same old40 not fresh holdout, model pixel agreement not depth accuracy, pair observations correlated, real10 no3Dtruth; score threshold unchanged',
        'hashes':{str(p):sha256(p) for p in paths}})
    cache=read(CACHE);rows=[]
    assert sha256(CACHE)==read(ROOT/'out/ordinal-depth-dev-20261012/protocol.json')['hashes']['cache']
    for dataset in ('synthetic','real'):
        records=cache[dataset]['records'];assert len(records)==(40 if dataset=='synthetic' else 10)
        for old in records:
            assert sha256(Path(old['path']))==old['sha256']
            row={'dataset':dataset,'image':old['image'],'searchable':not bool(old.get('error'))}
            if old.get('error'):
                row['error']=old['error'];rows.append(row);continue
            assert old['human']['turn']==old['rtmw']['turn'] and not old['z_fallback']
            h,r=old['human'],old['rtmw'];mask=np.asarray(old['shared_mask'],bool)
            result=map_depth(h['keypoints2d'],mask,r['keypoints2d'],r['z_meters'],r['scores'])
            assert np.allclose(result['hypotheses']['direct']['normalized_z'],old['normalized_z'],atol=1e-7,rtol=0)
            row.update(result);row['rtmw_body_score_ge03']=int((np.asarray(r['scores'])[5:]>=.3).sum())
            for arm,values in row['hypotheses'].items():
                z=np.asarray(values['normalized_z'],np.float32);delta=z[PAIRS[:,0]]-z[PAIRS[:,1]]
                use=mask[PAIRS].all(axis=1)&(abs(delta)>=.1)
                gate=(np.asarray(values['xy_error_torso'])[PAIRS]<=.5).all(axis=1)
                values.update(signs=np.sign(delta).astype(int).tolist(),all=use.tolist(),xy_gate=(use&gate).tolist())
            rows.append(row)
    verify(args);save_json(args.out/'predictions.json',rows)
    save_json(args.out/'predicted.json',{'sha256':sha256(args.out/'predictions.json'),'answers_parsed':False})


def counts(correct,sign,eligible):
    return {'n':int(eligible.sum()),'correct':int((correct&eligible).sum()),
            'always_front':int(((sign<0)&eligible).sum()),'always_back':int(((sign>0)&eligible).sum())}


def evaluate(args):
    verify(args);assert not (args.out/'results.json').exists()
    assert sha256(args.out/'predictions.json')==read(args.out/'predicted.json')['sha256']
    rows=read(args.out/'predictions.json');answers={r['image']:r for r in read(ANSWERS)}
    originals={r['image']:r for r in read(CACHE)['synthetic']['records']}
    poses={p.pose_id:p for p in load_library(args.library_root)};scored=[]
    for row in rows:
        if row['dataset']!='synthetic':continue
        a=answers[row['image']];h=originals[row['image']]['human'];mask=np.asarray(originals[row['image']]['shared_mask'],bool)
        xy=np.asarray(h['keypoints2d']);gt=np.asarray(a['gt_projected_xy']);body=np.flatnonzero(mask[5:])+5
        swap=bool(np.linalg.norm(xy[body]-gt[LEFT_RIGHT][body],axis=1).mean()<np.linalg.norm(xy[body]-gt[body],axis=1).mean())
        truth=camera_points(poses[a['pose_id']].joints,Camera(a['yaw']))[:,2]
        record={'image':row['image'],'oracle_human_gt_swap':swap,'frames':{}}
        for frame,z in (('semantic',truth),('observed_human',truth[LEFT_RIGHT] if swap else truth)):
            delta=z[PAIRS[:,0]]-z[PAIRS[:,1]];valid=abs(delta)>=.1;sign=np.sign(delta)
            output={}
            for gate in GATES:
                eligible={arm:valid&np.asarray(row['hypotheses'][arm][gate],bool) for arm in ARMS}
                correct={arm:np.asarray(row['hypotheses'][arm]['signs'])==sign for arm in ARMS}
                common=eligible['direct']&eligible['xy_aligned']
                output[gate]={'own':{arm:counts(correct[arm],sign,eligible[arm]) for arm in ARMS},
                              'common':{arm:counts(correct[arm],sign,common) for arm in ARMS},
                              'gains':int((~correct['direct']&correct['xy_aligned']&common).sum()),
                              'losses':int((correct['direct']&~correct['xy_aligned']&common).sum())}
            record['frames'][frame]=output
        scored.append(record)
    aggregates={}
    for frame in ('semantic','observed_human'):
        aggregates[frame]={}
        for gate in GATES:
            cells=[r['frames'][frame][gate] for r in scored]
            aggregates[frame][gate]={scope:{arm:{key:sum(c[scope][arm][key] for c in cells)
                for key in ('n','correct','always_front','always_back')} for arm in ARMS} for scope in ('own','common')}
            aggregates[frame][gate].update(gains=sum(c['gains'] for c in cells),losses=sum(c['losses'] for c in cells))
    old=read(OLD)['order_diagnostics']['ungated'];raw=aggregates['semantic']['all']['own']['direct']
    assert raw['n']==old['eligible_pairs_with_gt_order'] and raw['correct']==old['correct_pairs']
    descriptive={}
    for dataset in ('synthetic','real'):
        rr=[r for r in rows if r['dataset']==dataset];valid=[r for r in rr if r['searchable']]
        descriptive[dataset]={'images':len(rr),'searchable':len(valid),'mapped_swaps':sum(r['swapped'] for r in valid),
            'direct_xy_error_median':float(np.median([r['mean_direct'] for r in valid])),
            'aligned_xy_error_median':float(np.median([min(r['mean_direct'],r['mean_swapped']) for r in valid])),
            'used_pairs':{arm:{g:sum(sum(r['hypotheses'][arm][g]) for r in valid) for g in GATES} for arm in ARMS}}
    result={'stage':'diagnostic only','descriptive':descriptive,'depth_order':aggregates,'records':scored,
            'oracle_human_gt_swaps':sum(r['oracle_human_gt_swap'] for r in scored),'retrieval_accuracy':None}
    save_json(args.out/'results.json',result);print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


def parse():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('predict','evaluate'))
    for name in ('out','library-root'):p.add_argument('--'+name,type=Path,required=True)
    return p.parse_args()


if __name__=='__main__':
    args=parse();args.out.parent.mkdir(parents=True,exist_ok=True)
    with (args.out.parent/'.depth-correspondence.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);globals()[args.stage](args)
