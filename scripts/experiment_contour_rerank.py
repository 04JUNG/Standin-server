"""Known-development full-boundary reranking; not isolated occlusion evidence."""
import argparse
import fcntl
import hashlib
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiment_camera_refinement import verify as verify_source,GROUPS
from scripts.experiment_near_camera import read
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import save_json,sha256
from scripts.experiment_humanart_z import read_rgb
from scripts.experiment_refinement_blend import blend_cost
from scripts.experiment_head_hint import order
from scripts.diagnose_dual_camera_coverage import valid_query
from standin_eval.contour_rerank import prepare_query,candidate_evidence,rerank

BASE='keepfirst_blend75'
WEIGHTS={'contour025':.25,'contour050':.5,'contour100':1.}
CODE=('scripts/experiment_contour_rerank.py','standin_eval/contour_rerank.py',
      'scripts/experiment_z_rendered.py','scripts/experiment_head_hint.py',
      'scripts/experiment_refinement_blend.py','scripts/experiment_humanart_z.py',
      'scripts/diagnose_dual_camera_coverage.py','standin_eval/perspective_search.py')


def select_policy(counts):
    choices=[a for a in WEIGHTS if counts[a]['strong_perspective']['top5']>=counts[BASE]['strong_perspective']['top5']+3
             and all(counts[a][g][k]>=counts[BASE][g][k] for g in GROUPS for k in ('top1','top5'))]
    return sorted(choices,key=lambda a:(-counts[a]['strong_perspective']['top5'],
                                       -sum(counts[a][g]['top5'] for g in GROUPS),WEIGHTS[a]))


def verify(args):
    p=read(args.out/'protocol.json')
    assert all(sha256(ROOT/n)==h for n,h in p['code'].items())
    assert all(sha256(Path(n))==h for n,h in p['inputs'].items())


def rank(args):
    verify_source(SimpleNamespace(source=args.source,out=args.refinement_out,library_root=args.library_root,model=args.model))
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'records').mkdir();(args.out/'masks').mkdir()
    for folder in (args.refinement_out,args.blend_out):
        assert read(folder/'audit.json')['ranking_sha256']==sha256(folder/'rankings.json')
    inputs=[args.source/n for n in ('manifest.json','answers.json')]
    inputs += [folder/n for folder in (args.refinement_out,args.blend_out) for n in ('rankings.json','audit.json')]
    inputs += sorted((args.source/'predictions').glob('*.json'))
    inputs += [args.library_root/'poses.db']
    save_json(args.out/'protocol.json',{
        'stage':'development on examined120 families/360 images; no new holdout or inference',
        'hypothesis':'candidate image boundaries add evidence beyond fixed XY; boundaries include silhouette, proportions and occlusion; NOT an occlusion-only ablation',
        'baseline':BASE,'weights':WEIGHTS,'shortlist':20,
        'query':'RGB resize256 area, pixel-center XY resize, dark minRGB<200; >=10line pixels; square images only',
        'candidate':'unchanged ellipsoid renderer256 rough seed0 at stored fitted camera; hip translation and positive torso scale to unchanged HumanXY, no new optimization, nearest raster warp',
        'cost':'symmetric mean nearest-boundary Euclidean distance /HumanTorso; each direction pixel distances clipped at.5; score=originalblend75+weight*contour_cost',
        'fixed':'original HumanArt XY/mask, full1981gallery, stored camera, pooled117 first result; rerank ranks2..20 only, tail unchanged; no queryZ, target ID, GT camera/joints used',
        'fallback':'if query raster invalid or ANY shortlist rank2..20 render/scoring fails, entire image retains baseline for allweights; record errors, no pose/sample replacement',
        'failure':'original3 extraction failures preserved in denominator',
        'selection':'strongTop5>=baseline+3; all3 Top1/Top5 nondecreasing; maximize strongTop5,totalTop5 then smaller weight',
        'limits':['same procedural renderer for query/candidate is favorable bias; not actual rough accuracy',
                  'head/hand/foot/body geometry approximate; whole boundary effect not solely depth',
                  'source camera fitted to noisy joints, no guarantee of pixel alignment',
                  'target outside original20 cannot be recovered; square white-background pilot'],
        'code':{n:sha256(ROOT/n) for n in CODE},'inputs':{str(p):sha256(p) for p in inputs}})
    poses=load_library(args.library_root);rows=[]
    originals={r['image']:r for r in read(args.blend_out/'rankings.json')}
    manifest={r['image']:r for r in read(args.source/'manifest.json')}
    for old in read(args.refinement_out/'rankings.json'):
        name=old['image'];path=args.source/'predictions'/(Path(name).stem+'.json')
        assert sha256(path)==old['prediction_sha256']
        assert sha256(args.source/'images'/name)==manifest[name]['sha256']
        row={'image':name,'prediction_sha256':sha256(path)};query=valid_query(read(path))
        if query is None:
            assert old.get('error');row['error']=old['error'];rows.append(row);continue
        xy,mask=query;row['shared_xy_sha256']=hashlib.sha256(xy.tobytes()).hexdigest();row['shared_mask']=mask.tolist()
        assert row['shared_xy_sha256']==old['shared_xy_sha256'] and row['shared_mask']==old['shared_mask']
        costpath=args.refinement_out/'costs'/(Path(name).stem+'.npz')
        assert sha256(costpath)==old['costs_sha256'];row['original_costs_sha256']=old['costs_sha256']
        with np.load(costpath,allow_pickle=False) as d:base,fit,params=[d[k] for k in ('baseline','refined','parameters')]
        scores=blend_cost(base,fit,.75);original=order(base,scores);short=original[:20]
        row[BASE]=[poses[i].pose_id for i in original[:5]];assert row[BASE]==originals[name][BASE]
        row['shortlist']=[poses[i].pose_id for i in short];row['shortlist_indices']=short.tolist()
        row['contour_costs']=[0.]*20;row['candidate_errors']={};row['affines']={}
        start=time.perf_counter();masks=[]
        try:
            evidence=prepare_query(read_rgb(args.source/'images'/name),xy)
            row['query_torso']=evidence['torso'];row['query_line_pixels']=int(evidence['mask'].sum())
            for rank_index,index in enumerate(short[1:],1):
                try:
                    cost,affine,pixels=candidate_evidence(evidence,poses[index].joints,params[index])
                    row['contour_costs'][rank_index]=cost;row['affines'][str(rank_index)]=affine.tolist()
                    if len(rows)<3:masks.append((rank_index,pixels))
                except Exception as exc:row['candidate_errors'][str(rank_index)]=type(exc).__name__+': '+str(exc)
            if len(rows)<3:
                # Scientific evidence arrays, not edited user artwork.
                np.savez_compressed(args.out/'masks'/(Path(name).stem+'.npz'),query=evidence['mask'],
                                    **{f'candidate_{j}':pixels for j,pixels in masks})
        except Exception as exc:row['query_error']=type(exc).__name__+': '+str(exc)
        row['contour_seconds']=time.perf_counter()-start
        row['fallback']=bool(row['candidate_errors'] or row.get('query_error'))
        for arm,weight in WEIGHTS.items():
            ranked=original if row['fallback'] else rerank(original,scores,row['contour_costs'],weight)
            row[arm]=[poses[i].pose_id for i in ranked[:5]]
            assert row[arm][0]==row[BASE][0] and np.array_equal(ranked[20:],original[20:])
        rows.append(row);save_json(args.out/'records'/(Path(name).stem+'.json'),row)
        if len(rows)%15==0:print('contours',len(rows),'/360',flush=True)
    assert len(rows)==360;verify(args)
    save_json(args.out/'rankings.json',rows)
    save_json(args.out/'ranked.json',{'sha256':sha256(args.out/'rankings.json'),'answers_parsed':False})


def evaluate(args):
    verify(args);assert not (args.out/'results.json').exists()
    assert sha256(args.out/'rankings.json')==read(args.out/'ranked.json')['sha256']
    rows=read(args.out/'rankings.json');answers={r['image']:r for r in read(args.source/'answers.json')}
    assert len(rows)==len(answers)==360 and {r['image'] for r in rows}==set(answers)
    counts={a:{g:dict(top1=0,top5=0) for g in GROUPS} for a in (BASE,*WEIGHTS)}
    changes={a:{g:dict(gains=0,losses=0) for g in GROUPS} for a in WEIGHTS}
    failures={g:0 for g in GROUPS};fallback={g:0 for g in GROUPS};ceiling={g:0 for g in GROUPS}
    for row in rows:
        a=answers[row['image']];g,target=a['group'],a['pose_id']
        failures[g]+=bool(row.get('error'));fallback[g]+=bool(row.get('fallback'));ceiling[g]+=target in row.get('shortlist',[])
        for arm in counts:
            ids=row.get(arm,[]);counts[arm][g]['top1']+=bool(ids and ids[0]==target);counts[arm][g]['top5']+=target in ids
            if arm!=BASE:
                before=target in row.get(BASE,[]);after=target in ids
                changes[arm][g]['gains']+=after and not before;changes[arm][g]['losses']+=before and not after
    selected=select_policy(counts)
    result={'stage':'development only; same-renderer bias','counts':counts,'changes':changes,
            'failures':failures,'fallback_images':fallback,'target_in_original20':ceiling,
            'eligible':selected,'selected':selected[0] if selected else None,
            'seconds_median':float(np.median([r['contour_seconds'] for r in rows if not r.get('error')]))}
    save_json(args.out/'results.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('rank','evaluate'))
    for name in ('source','refinement-out','blend-out','out','library-root','model'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.out.parent.mkdir(parents=True,exist_ok=True)
    with (args.out.parent/'.contour-rerank.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);globals()[args.stage](args)
