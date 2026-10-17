"""Fixed joint-feature and first-rank release ablations; cached observed XY only."""
import argparse
import fcntl
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import sha256,save_json
from scripts.review_perspective_roughs import rgb,keep_first_order
from standin_eval.camera_refinement import normalized_joints,project_parameters
from standin_eval.perspective_search import normalize_batch
from standin_eval.contour_rerank import candidate_evidence
from standin_eval.contour_letterbox import prepare_letterboxed_query
from standin_eval.joint_shape import shape_penalties,reorder_shortlist

ARMS=('prior','unlock','elbow','leg_cross','joint_shape')
GROUPS=('perspective','strong_perspective','orthographic_control')
WEIGHT=.15


def read(path):return json.loads(path.read_text())


def verify(a):
    p=read(a.out/'protocol.json')
    for n,h in p['inputs'].items():assert sha256(Path(n))==h,n
    for n,h in p['code'].items():assert sha256(ROOT/n)==h,n
    return p


def rank(a):
    original=read(a.contour/'rankings.json')
    audit=read(a.contour/'audit.json');assert audit['ranking_sha256']==sha256(a.contour/'rankings.json')
    oldpolicy=read(a.contour/'protocol.json')
    for n in ('standin_eval/contour_rerank.py','standin_eval/perspective_search.py','scripts/experiment_z_rendered.py'):
        assert sha256(ROOT/n)==oldpolicy['code'][n]
    a.out.mkdir(parents=True,exist_ok=False)
    for name in ('records','masks'): (a.out/name).mkdir()
    inputs=[a.contour/n for n in ('protocol.json','rankings.json','audit.json','results.json')]
    inputs+=[a.source/n for n in ('manifest.json','answers.json')]+[a.library_root/'poses.db']
    inputs+=sorted((a.source/'predictions').glob('*.json'))+sorted((a.source/'images').glob('*.png'))+sorted((a.fits/'costs').glob('*.npz'))
    code=['scripts/experiment_joint_shape.py','scripts/review_perspective_roughs.py',
          'standin_eval/joint_shape.py','standin_eval/contour_letterbox.py','standin_eval/contour_rerank.py',
          'standin_eval/camera_refinement.py','standin_eval/perspective_search.py','standin_eval/dual_label_search.py',
          'scripts/experiment_z_axis_search.py','scripts/experiment_z_rendered.py']
    save_json(a.out/'protocol.json',{
        'stage':'known120-family development,360 paired views; no new holdout or inference',
        'arms':ARMS,'angle_weight':WEIGHT,'cross_weight':WEIGHT,'shortlist':20,
        'features':'mean observed elbow angle error/180; proper 2D leg segment crossing mismatch; all six leg joints required; normalized bone>=.05; contact/endpoint margin.025 torso; candidate ambiguous=.5',
        'query':'same cached Human-Art XY and>=.3 mask; no semantic IDs/GT/user preferred rank in scorer',
        'correspondence':'choose one global direct/swap per candidate from fitted-camera XY error, then evaluate both features',
        'first_rank':'unlock all20 including previous first; compute actual contour cost for former first too; original tail unchanged',
        'fallback':'any20 renderer error or query error removes contour for whole query; joint evidence still used if query joint features valid',
        'gate':'joint_shape strongTop1>=prior+3 and>=unlock+2; all groups Top1/Top5>=both prior and unlock; otherwise no confirmation claim',
        'fixed':'same cameras/poses/fit scores, .25*base+.75*fit; contour weight1 cap.5; no feature-weight tuning after evaluation',
        'limits':['2D crossing not3D front/back proof','observed joint errors may mislead angles','first release may regress','real user reviewed10 are development feedback, no source3D GT'],
        'inputs':{str(p):sha256(p) for p in inputs},'code':{n:sha256(ROOT/n) for n in code}})
    poses=load_library(a.library_root); points=normalized_joints(np.stack([p.joints for p in poses]));rows=[]
    for old in original:
        name=old['image'];path=a.source/'predictions'/(Path(name).stem+'.json');pred=read(path)
        assert sha256(path)==old['prediction_sha256']
        row={'image':name,'prediction_sha256':sha256(path)}
        if old.get('error'):
            row['error']=old['error'];rows.append(row);continue
        xy=np.asarray(pred['keypoints2d'],np.float32);mask=np.asarray(pred['scores'],np.float32)>=.3
        assert mask[5:].sum()>=6 and mask[[5,6,11,12]].all()
        row['shared_xy_sha256']=hashlib.sha256(xy.tobytes()).hexdigest();row['shared_mask']=mask.tolist()
        assert row['shared_xy_sha256']==old['shared_xy_sha256'] and row['shared_mask']==old['shared_mask']
        costpath=a.fits/'costs'/(Path(name).stem+'.npz');assert sha256(costpath)==old['original_costs_sha256']
        row['costs_sha256']=sha256(costpath)
        with np.load(costpath,allow_pickle=False) as d:base,fit,params=[d[k] for k in ('baseline','refined','parameters')]
        score=.25*base.astype(float)+.75*fit.astype(float);order=keep_first_order(base,score);short=order[:20]
        assert short.tolist()==old['shortlist_indices']
        row['shortlist_indices']=short.tolist();row['shortlist']=old['shortlist'];row['prior']=old['contour100']
        projected,valid=project_parameters(points[short],params[short]);assert valid.all()
        angle,cross,meta=shape_penalties(normalize_batch(xy),mask,projected)
        row['angle_costs']=angle.tolist();row['cross_costs']=cross.tolist();row['feature_metadata']=meta
        contour=np.zeros(20);errors={};masks={}
        try:
            query,_=prepare_letterboxed_query(rgb(a.source/'images'/name),xy)
            row['query_torso']=query['torso']
            if len(rows)<3:masks['query']=query['mask']
            for k,j in enumerate(short):
                try:
                    value,affine,pixels=candidate_evidence(query,poses[j].joints,params[j]);contour[k]=value
                    if k>=1:assert value==old['contour_costs'][k]
                    if len(rows)<3:masks['candidate_'+str(k)]=pixels
                except ValueError as exc:errors[str(k)]=str(exc)
        except ValueError as exc:errors['query']=str(exc)
        row['contour_costs']=contour.tolist();row['contour_errors']=errors
        used=np.zeros(20) if errors else contour
        for arm,add in (('unlock',used),('elbow',used+WEIGHT*angle),('leg_cross',used+WEIGHT*cross),('joint_shape',used+WEIGHT*(angle+cross))):
            ranked=reorder_shortlist(order,score,add,1.)
            row[arm]=[poses[j].pose_id for j in ranked[:5]];row[arm+'_order20']=ranked[:20].tolist()
            assert np.array_equal(ranked[20:],order[20:])
        if masks:
            path=a.out/'masks'/(Path(name).stem+'.npz');np.savez_compressed(path,**masks);row['masks_sha256']=sha256(path)
        rows.append(row);save_json(a.out/'records'/(Path(name).stem+'.json'),row)
        if len(rows)%30==0:print('joint-shape',len(rows),'/360',flush=True)
    assert len(rows)==360;verify(a);save_json(a.out/'rankings.json',rows)
    save_json(a.out/'ranked.json',{'sha256':sha256(a.out/'rankings.json'),'answers_parsed':False})


def evaluate(a):
    verify(a);assert not (a.out/'results.json').exists()
    assert sha256(a.out/'rankings.json')==read(a.out/'ranked.json')['sha256']
    rows=read(a.out/'rankings.json');answers={r['image']:r for r in read(a.source/'answers.json')}
    counts={arm:{g:dict(top1=0,top5=0) for g in GROUPS} for arm in ARMS}
    for row in rows:
        truth=answers[row['image']];g,target=truth['group'],truth['pose_id']
        for arm in ARMS:
            ids=row.get(arm,[]);counts[arm][g]['top1']+=bool(ids and ids[0]==target);counts[arm][g]['top5']+=target in ids
    gate=(counts['joint_shape']['strong_perspective']['top1']>=counts['prior']['strong_perspective']['top1']+3
          and counts['joint_shape']['strong_perspective']['top1']>=counts['unlock']['strong_perspective']['top1']+2
          and all(counts['joint_shape'][g][k]>=counts[b][g][k] for b in ('prior','unlock') for g in GROUPS for k in ('top1','top5')))
    result={'counts':counts,'eligible_for_confirmation':gate,'stage':'development only',
            'extract_failures':sum(bool(r.get('error')) for r in rows),'contour_fallbacks':sum(bool(r.get('contour_errors')) for r in rows),
            'cross_signal_images':sum(r.get('feature_metadata',{}).get('used_crossing',False) for r in rows),
            'arm_signal_images':sum(r.get('feature_metadata',{}).get('used_arms',0)>0 for r in rows)}
    save_json(a.out/'results.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('rank','evaluate'))
    for n in ('source','contour','fits','library-root','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
    with (a.out.parent/'.joint-shape.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);globals()[a.stage](a)
