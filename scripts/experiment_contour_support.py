"""One preregistered preprocessing ablation on already-examined development data."""
import argparse
import fcntl
import hashlib
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiment_near_camera import read
from scripts.experiment_z_rendered import sha256,save_json
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_humanart_z import read_rgb
from scripts.experiment_refinement_blend import blend_cost
from scripts.experiment_head_hint import order
from scripts.diagnose_dual_camera_coverage import valid_query
from scripts.confirm_camera_fit import geometry_hash
from standin_eval.contour_rerank import prepare_query,candidate_evidence,contour_cost,rerank
from standin_eval.contour_support import prepare_support_query

SOURCE=ROOT/'out/near-camera-confirm-20261013'
OLD=ROOT/'out/contour-rerank-dev-20261013'
FIT=ROOT/'out/camera-refinement-dev-20261013'
BASE='contour100'
NEW='native_support100'
GROUPS=('perspective','strong_perspective','orthographic_control')
CODE=('scripts/experiment_contour_support.py','standin_eval/contour_support.py',
      'standin_eval/contour_letterbox.py','standin_eval/contour_rerank.py',
      'scripts/experiment_z_rendered.py','scripts/experiment_head_hint.py',
      'scripts/experiment_refinement_blend.py','scripts/experiment_humanart_z.py',
      'scripts/diagnose_dual_camera_coverage.py')


def verify(out):
    p=read(out/'protocol.json')
    for name,digest in p['inputs'].items(): assert sha256(Path(name))==digest,name
    for name,digest in p['code'].items(): assert sha256(ROOT/name)==digest,name


def eligible(counts):
    return (counts[NEW]['strong_perspective']['top5']>=counts[BASE]['strong_perspective']['top5']+3
            and all(counts[NEW][g][k]>=counts[BASE][g][k] for g in GROUPS for k in ('top1','top5')))


def rank(args):
    assert read(OLD/'audit.json')['ranking_sha256']==sha256(OLD/'rankings.json')
    oldpolicy=read(OLD/'protocol.json')
    for n,h in oldpolicy['code'].items(): assert sha256(ROOT/n)==h
    for n,h in oldpolicy['inputs'].items(): assert sha256(Path(n))==h
    args.out.mkdir(parents=True,exist_ok=False)
    for name in ('records','masks'): (args.out/name).mkdir()
    inputs=[OLD/'protocol.json',OLD/'rankings.json',OLD/'audit.json',OLD/'results.json',
            SOURCE/'manifest.json',SOURCE/'answers.json',args.library_root/'poses.db']
    inputs+=sorted((SOURCE/'predictions').glob('*.json'))+sorted((FIT/'costs').glob('*.npz'))
    inputs+=sorted((SOURCE/'images').glob('*.png'))
    save_json(args.out/'protocol.json',{
        'stage':'known-development120 families,360 paired images; not fresh validation',
        'single_hypothesis':'native dark-pixel support before area resize preserves useful thin query lines',
        'baseline':BASE,'selected_candidate':NEW,'weight':1.,'limit':20,'size':256,
        'query_change':'minRGB<200 at native resolution; float32 mask INTER_AREA resize; output>0; white square padding if needed',
        'fixed':'same XY/mask, camera costs/parameters, gallery, candidate renderer/raster, costcap.5, weights, first and tail',
        'fallback':'ANY candidate or query error keeps full XY camera baseline, no sample replacement',
        'gate':'strong Top5>=old contour100+3; all groups Top1/Top5 nondecreasing; otherwise no fresh confirmation',
        'failure':'all original3 extraction failures retained; no GT loaded until rankings frozen',
        'risks':['thicker lines and background noise retained','same-renderer bias','no actual rough accuracy claim'],
        'inputs':{str(p):sha256(p) for p in inputs},'code':{p:sha256(ROOT/p) for p in CODE}})
    poses=load_library(args.library_root)
    expected=read(ROOT/'out/refinement-blend-confirm-20261013/protocol.json')['hashes']
    assert geometry_hash(poses)==expected['geometry'] and sha256(args.library_root/'poses.db')==expected['library_db']
    rows=[]
    for old in read(OLD/'rankings.json'):
        name=old['image']; predpath=SOURCE/'predictions'/(Path(name).stem+'.json')
        assert sha256(predpath)==old['prediction_sha256']
        row={'image':name,'prediction_sha256':old['prediction_sha256']}; valid=valid_query(read(predpath))
        if valid is None:
            assert old.get('error'); row['error']=old['error']; rows.append(row); continue
        xy,mask=valid
        row['shared_xy_sha256']=hashlib.sha256(xy.tobytes()).hexdigest(); row['shared_mask']=mask.tolist()
        assert row['shared_xy_sha256']==old['shared_xy_sha256'] and row['shared_mask']==old['shared_mask']
        costpath=FIT/'costs'/(Path(name).stem+'.npz'); assert sha256(costpath)==old['original_costs_sha256']
        row['costs_sha256']=sha256(costpath)
        with np.load(costpath,allow_pickle=False) as d: base,fit,params=[d[k] for k in ('baseline','refined','parameters')]
        score=blend_cost(base,fit,.75); original=order(base,score); short=original[:20]
        assert short.tolist()==old['shortlist_indices']
        row[BASE]=old[BASE]; row['shortlist_indices']=short.tolist(); row['shortlist']=old['shortlist']
        row['contour_costs']=[0.]*20; row['candidate_errors']={}; row['affines']={}
        rgb=read_rgb(SOURCE/'images'/name); oldquery=prepare_query(rgb,xy); masks={}
        try:
            query,pad=prepare_support_query(rgb,xy)
            assert np.array_equal(query['xy'],oldquery['xy']) and query['torso']==oldquery['torso']
            row['query_torso']=query['torso']; row['old_line_pixels']=int(oldquery['mask'].sum()); row['new_line_pixels']=int(query['mask'].sum())
            assert np.all(query['mask'][oldquery['mask']])
            if len(rows)<3: masks.update(query=query['mask'],old_query=oldquery['mask'])
            for k,j in enumerate(short[1:],1):
                try:
                    value,affine,pixels=candidate_evidence(query,poses[j].joints,params[j])
                    assert str(k) not in old['candidate_errors']
                    assert contour_cost(oldquery,pixels)==old['contour_costs'][k]
                    assert affine.tolist()==old['affines'][str(k)]
                    row['contour_costs'][k]=value; row['affines'][str(k)]=affine.tolist()
                    if len(rows)<3: masks[f'candidate_{k}']=pixels
                except ValueError as exc: row['candidate_errors'][str(k)]='ValueError: '+str(exc)
        except ValueError as exc: row['query_error']=str(exc)
        assert row['candidate_errors']==old['candidate_errors']
        row['fallback']=bool(row['candidate_errors'] or row.get('query_error'))
        ranked=original if row['fallback'] else rerank(original,score,row['contour_costs'],1.)
        assert ranked[0]==original[0] and np.array_equal(ranked[20:],original[20:])
        row[NEW]=[poses[i].pose_id for i in ranked[:5]]
        if masks:
            path=args.out/'masks'/(Path(name).stem+'.npz'); np.savez_compressed(path,**masks); row['masks_sha256']=sha256(path)
        rows.append(row); save_json(args.out/'records'/(Path(name).stem+'.json'),row)
        if len(rows)%30==0: print('ranked',len(rows),'/360',flush=True)
    assert len(rows)==360; verify(args.out); save_json(args.out/'rankings.json',rows)
    save_json(args.out/'ranked.json',{'sha256':sha256(args.out/'rankings.json'),'answers_parsed':False})


def evaluate(args):
    verify(args.out); assert not (args.out/'results.json').exists()
    assert sha256(args.out/'rankings.json')==read(args.out/'ranked.json')['sha256']
    rows=read(args.out/'rankings.json'); answers={r['image']:r for r in read(SOURCE/'answers.json')}
    assert len(rows)==len(answers)==360
    counts={arm:{g:dict(top1=0,top5=0) for g in GROUPS} for arm in (BASE,NEW)}
    changes={g:dict(gains=0,losses=0) for g in GROUPS}; failures={g:0 for g in GROUPS}; fallback={g:0 for g in GROUPS}
    for r in rows:
        a=answers[r['image']]; g=a['group']; target=a['pose_id']
        failures[g]+=bool(r.get('error')); fallback[g]+=bool(r.get('fallback'))
        for arm in counts:
            ids=r.get(arm,[]); counts[arm][g]['top1']+=bool(ids and ids[0]==target); counts[arm][g]['top5']+=target in ids
        before=target in r.get(BASE,[]); after=target in r.get(NEW,[])
        changes[g]['gains']+=after and not before; changes[g]['losses']+=before and not after
    assert counts[BASE]==read(OLD/'results.json')['counts'][BASE]
    result={'counts':counts,'changes':changes,'failures':failures,'fallback_images':fallback,
            'eligible_for_fresh_confirmation':eligible(counts),'stage':'development only; same renderer'}
    save_json(args.out/'results.json',result); print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=('rank','evaluate'))
    p.add_argument('--out',type=Path,required=True); p.add_argument('--library-root',type=Path,required=True)
    args=p.parse_args()
    with (ROOT/'out/.contour-support.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); globals()[args.stage](args)
