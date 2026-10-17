"""Audit native line-support masks, fixed rankings and development rejection."""
import argparse
import hashlib
import sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiment_contour_support import verify,SOURCE,OLD,FIT,BASE,NEW,GROUPS
from scripts.experiment_near_camera import read
from scripts.experiment_z_rendered import sha256,save_json
from scripts.experiment_humanart_z import read_rgb
from scripts.experiment_z_axis_search import load_library
from scripts.diagnose_dual_camera_coverage import valid_query
from scripts.audit_contour_rerank import brute_cost
from standin_eval.contour_support import prepare_support_query
from standin_eval.contour_rerank import candidate_evidence


def native_support_independent(rgb,size=256):
    """Exact nonzero area support via integer-overlap bounds and summed areas."""
    h,w=rgb.shape[:2]; side=max(h,w); left,top=(side-w)//2,(side-h)//2
    dark=np.zeros((side,side),np.int64)
    dark[top:top+h,left:left+w]=(np.min(rgb,axis=2)<200)
    integral=np.pad(dark.cumsum(0).cumsum(1),((1,0),(1,0)))
    start=(np.arange(size)*side)//size
    end=((np.arange(1,size+1)*side)+size-1)//size
    sums=(integral[end[:,None],end[None,:]]-integral[start[:,None],end[None,:]]
          -integral[end[:,None],start[None,:]]+integral[start[:,None],start[None,:]])
    return sums>0


def main(args):
    verify(args.out); rows=read(args.out/'rankings.json'); result=read(args.out/'results.json')
    assert read(args.out/'ranked.json')['sha256']==sha256(args.out/'rankings.json')
    originals={r['image']:r for r in read(OLD/'rankings.json')}
    answers={r['image']:r for r in read(SOURCE/'answers.json')}
    assert len(rows)==len({r['image'] for r in rows})==len(answers)==360
    poses=load_library(args.library_root); checks=0; replays=0; maximum=0.; maskchecks=0
    for row in rows:
        name=row['image']; old=originals[name]; path=SOURCE/'predictions'/(Path(name).stem+'.json')
        assert sha256(path)==row['prediction_sha256']==old['prediction_sha256']
        valid=valid_query(read(path))
        if row.get('error'): assert valid is None and old.get('error'); continue
        xy,mask=valid
        assert hashlib.sha256(xy.tobytes()).hexdigest()==row['shared_xy_sha256']==old['shared_xy_sha256']
        assert row['shared_mask']==mask.tolist()==old['shared_mask']
        rgb=read_rgb(SOURCE/'images'/name); query,_=prepare_support_query(rgb,xy)
        assert np.array_equal(query['mask'],native_support_independent(rgb)); maskchecks+=1
        assert query['torso']==row['query_torso'] and query['mask'].sum()==row['new_line_pixels']
        assert row['old_line_pixels']==old['query_line_pixels'] and row[BASE]==old[BASE]
        costpath=FIT/'costs'/(Path(name).stem+'.npz')
        assert sha256(costpath)==row['costs_sha256']==old['original_costs_sha256']
        with np.load(costpath,allow_pickle=False) as d: base,fit,params=[d[k] for k in ('baseline','refined','parameters')]
        scores=.25*base.astype(float)+.75*fit.astype(float); first=int(np.argmin(base))
        order=sorted(range(len(poses)),key=lambda i:(scores[i],i)); order=[first]+[i for i in order if i!=first]
        assert row['shortlist_indices']==order[:20] and row['shortlist']==[poses[i].pose_id for i in order[:20]]
        assert row['fallback']==bool(row['candidate_errors'] or row.get('query_error'))
        assert row['candidate_errors']==old['candidate_errors']
        ranked=order[:]
        if not row['fallback']:
            middle=sorted(range(1,20),key=lambda k:(scores[order[k]]+row['contour_costs'][k],k))
            ranked=[first]+[order[k] for k in middle]+order[20:]
        assert row[NEW]==[poses[i].pose_id for i in ranked[:5]]
        assert ranked[0]==first and ranked[20:]==order[20:]
        if row.get('masks_sha256'):
            maskpath=args.out/'masks'/(Path(name).stem+'.npz'); assert sha256(maskpath)==row['masks_sha256']
            with np.load(maskpath,allow_pickle=False) as masks:
                assert np.array_equal(masks['query'],query['mask'])
                for key in masks.files:
                    if not key.startswith('candidate_'): continue
                    k=int(key.split('_')[1]); delta=abs(brute_cost(masks['query'],masks[key],query['torso'])-row['contour_costs'][k])
                    maximum=max(maximum,delta); assert delta<2e-7; checks+=1
                    if k in (1,19):
                        value,affine,pixels=candidate_evidence(query,poses[order[k]].joints,params[order[k]])
                        assert value==row['contour_costs'][k] and np.array_equal(pixels,masks[key])
                        assert affine.tolist()==row['affines'][str(k)]; replays+=1
    counts={a:{g:dict(top1=0,top5=0) for g in GROUPS} for a in (BASE,NEW)}
    changes={g:dict(gains=0,losses=0) for g in GROUPS}; failures={g:0 for g in GROUPS}; fallback={g:0 for g in GROUPS}
    regressions=[]
    for r in rows:
        truth=answers[r['image']]; g=truth['group']; target=truth['pose_id']
        failures[g]+=bool(r.get('error')); fallback[g]+=bool(r.get('fallback'))
        for arm in counts:
            ids=r.get(arm,[]); counts[arm][g]['top1']+=bool(ids and ids[0]==target); counts[arm][g]['top5']+=target in ids
        before=target in r.get(BASE,[]); after=target in r.get(NEW,[])
        changes[g]['gains']+=after and not before; changes[g]['losses']+=before and not after
        if before and not after: regressions.append({'image':r['image'],'pose_id':target,'group':g})
    gate=(counts[NEW]['strong_perspective']['top5']>=counts[BASE]['strong_perspective']['top5']+3
          and all(counts[NEW][g][k]>=counts[BASE][g][k] for g in GROUPS for k in ('top1','top5')))
    assert counts==result['counts'] and changes==result['changes'] and failures==result['failures'] and fallback==result['fallback_images']
    assert gate==result['eligible_for_fresh_confirmation']
    # Existing real roughs: preprocessing diagnostic only, no rankings or ratings.
    real=read(ROOT/'out/humanart-plus-z-20261012/results.json')['real']['records']
    oldreal={r['image']:r for r in read(ROOT/'out/contour-real-roughs-20261013/rankings.json')}
    realrows=[]; sheet=Image.new('RGB',(1040,8*290),'#f3f5f7'); draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('/System/Library/Fonts/AppleSDGothicNeo.ttc',18)
    for r in real:
        if r.get('error'): continue
        imagepath=Path(r['path']); assert sha256(imagepath)==r['sha256']
        rgb=read_rgb(imagepath); q,_=prepare_support_query(rgb,np.asarray(r['human']['keypoints2d'],np.float32))
        assert np.array_equal(q['mask'],native_support_independent(rgb))
        with np.load(ROOT/'out/contour-real-roughs-20261013/masks'/(Path(r['image']).stem+'.npz'),allow_pickle=False) as d: oldmask=d['query']
        assert oldmask.sum()==oldreal[r['image']]['query_line_pixels']
        y=len(realrows)*290; draw.text((12,y+10),r['image']+' · 채점 아님',font=font,fill='#152c40')
        for col,(label,pixels) in enumerate((('기존 축소 후 판정',oldmask),('원본 선 먼저 보존',q['mask']))):
            x=330+col*330; draw.text((x,y+10),label,font=font,fill='#152c40')
            sheet.paste(Image.fromarray(np.where(pixels,0,255).astype(np.uint8)).convert('RGB'),(x,y+34))
        im=Image.fromarray(rgb); im.thumbnail((280,240)); sheet.paste(im,(12,y+40))
        realrows.append({'image':r['image'],'input_sha256':r['sha256'],'old_pixels':int(oldmask.sum()),'new_pixels':int(q['mask'].sum()),'accuracy':None})
    sheet.save(args.out/'real-line-support.png')
    output={'inputs':360,'native_support_masks_independently_checked':maskchecks,
            'bruteforce_costs':checks,'max_cost_difference':maximum,'render_replays':replays,
            'regressions':regressions,'real_preprocessing_only':realrows,
            'ranking_sha256':sha256(args.out/'rankings.json'),'results_sha256':sha256(args.out/'results.json'),
            'real_diagnostic_sha256':sha256(args.out/'real-line-support.png'),'auditor_sha256':sha256(Path(__file__))}
    save_json(args.out/'audit.json',output); print(output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); p.add_argument('--library-root',type=Path,required=True)
    main(p.parse_args())
