"""Independent ranking/padding checks and all-pixel boundary-cost audit."""
import argparse
import hashlib
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.compare_contour_real_roughs import verify,SOURCE,CACHE,BASE,NEW
from scripts.experiment_near_camera import read
from scripts.experiment_z_rendered import save_json,sha256
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_humanart_z import read_rgb
from scripts.audit_contour_rerank import brute_cost
from standin_eval.contour_rerank import candidate_evidence,distance_field


def run(args):
    verify(args.out); rows=read(args.out/'rankings.json'); summary=read(args.out/'summary.json')
    cached={r['image']:r for r in read(CACHE)['real']['records']}
    originals={r['image']:r for r in read(SOURCE/'rankings.json')}
    assert len(rows)==len(cached)==len({r['image'] for r in rows})==10
    poses=load_library(args.library_root); checks=0; replays=0; maximum=0.; max_affine=0.; line_diagnostics=[]
    for row in rows:
        item=cached[row['image']]; old=originals[row['image']]
        assert sha256(Path(row['path']))==row['sha256']==item['sha256']
        assert sha256(args.out/row['sheet'])==row['sheet_sha256']
        with Image.open(args.out/row['sheet']) as im: assert im.size==(1540,720); im.verify()
        if row.get('error'):
            assert old.get('error') and item.get('error') and BASE not in row and NEW not in row
            continue
        xy=np.asarray(item['human']['keypoints2d'],float); rgb=read_rgb(Path(row['path']))
        h,w=rgb.shape[:2]; side=max(h,w); left,top=(side-w)//2,(side-h)//2
        padded=np.pad(rgb,((top,side-h-top),(left,side-w-left),(0,0)),constant_values=255)
        pixels=cv2.resize(padded,(256,256),interpolation=cv2.INTER_AREA)
        observed=np.min(pixels,axis=2)<200
        # Match input float32 precision used in the frozen Human-Art cache.
        target=(xy.astype(np.float32).astype(float)+[left,top]+.5)*256/side-.5
        torso=float(np.linalg.norm(target[[5,6]].mean(0)-target[[11,12]].mean(0)))
        assert row['padding']=={'left':left,'top':top,'side':side}
        assert abs(torso-row['query_torso'])<1e-10 and observed.sum()==row['query_line_pixels']
        q=np.asarray(item['shared_xy'],np.float32)
        assert hashlib.sha256(q.tobytes()).hexdigest()==row['shared_xy_sha256']==old['shared_xy_sha256']
        assert row['shared_mask']==item['shared_mask']==old['shared_mask']
        normalized=target-target[[11,12]].mean(0); normalized/=torso
        normalized[~np.asarray(row['shared_mask'],bool)]=0
        assert np.allclose(normalized,q,atol=2e-6)
        costpath=SOURCE/'costs'/(Path(row['image']).stem+'.npz')
        assert sha256(costpath)==row['costs_sha256']
        with np.load(costpath,allow_pickle=False) as d: base,fit,params=[d[k] for k in ('baseline','refined','parameters')]
        scores=.25*base.astype(float)+.75*fit.astype(float); first=int(np.argmin(base))
        ordered=sorted(range(len(poses)),key=lambda i:(scores[i],i)); ordered=[first]+[i for i in ordered if i!=first]
        assert row['shortlist_indices']==ordered[:20]==row['order20_before']
        assert row['shortlist']==[poses[i].pose_id for i in ordered[:20]]
        assert row[BASE]==[c['pose_id'] for c in old[BASE]]==[poses[i].pose_id for i in ordered[:5]]
        maskpath=args.out/'masks'/(Path(row['image']).stem+'.npz'); assert sha256(maskpath)==row['masks_sha256']
        evidence={'mask':observed,'xy':target,'torso':torso,'size':256,'distance':distance_field(observed)}
        with np.load(maskpath,allow_pickle=False) as masks:
            assert np.array_equal(masks['query'],observed)
            for k in range(1,20):
                key=f'candidate_{k}'
                if str(k) in row['candidate_errors']: assert key not in masks; continue
                candidate=masks[key]
                error=abs(brute_cost(observed,candidate,torso)-row['contour_costs'][k]); maximum=max(maximum,error)
                assert error<2e-7; checks+=1
                if k in (1,19):
                    value,affine,replay=candidate_evidence(evidence,poses[ordered[k]].joints,params[ordered[k]])
                    assert abs(value-row['contour_costs'][k])<1e-10 and np.array_equal(replay,candidate)
                    delta=float(np.max(np.abs(affine-np.asarray(row['affines'][str(k)]))))
                    max_affine=max(max_affine,delta)
                    assert delta<1e-10; replays+=1
        assert row['fallback']==bool(row['candidate_errors'] or row.get('query_error'))
        updated=ordered[:]
        if not row['fallback']:
            middle=sorted(range(1,20),key=lambda k:(scores[ordered[k]]+row['contour_costs'][k],k))
            updated=[first]+[ordered[k] for k in middle]+ordered[20:]
        assert row['order20_after']==updated[:20]
        assert row[NEW]==[poses[i].pose_id for i in updated[:5]] and updated[20:]==ordered[20:]
        # Diagnostic only, NOT an alternative retrieval arm or accuracy proxy.
        dark=np.min(padded,axis=2)<200
        support=cv2.resize(dark.astype(np.float32),(256,256),interpolation=cv2.INTER_AREA)>0
        line_diagnostics.append({'image':row['image'],'original_hw':[h,w],
            'native_dark_pixels':int(dark.sum()),'downsampled_dark_pixels':int(observed.sum()),
            'native_dark_support_cells':int(support.sum()),
            'support_cells_no_longer_dark':int((support&~observed).sum()),
            'support_loss_fraction':float((support&~observed).sum()/max(support.sum(),1)),
            'note':'loss of native <200 line support after area resize; not ground-truth line recall'})
        print('audited',row['image'],flush=True)
    valid=[r for r in rows if not r.get('error')]
    assert summary['queries']==10 and summary['searchable']==len(valid)==8 and summary['extraction_failed']==2
    assert summary['render_or_query_fallback']==sum(r['fallback'] for r in valid)
    assert summary['top1_changed']==sum(r[BASE][0]!=r[NEW][0] for r in valid)==0
    assert summary['top5_set_changed']==sum(set(r[BASE])!=set(r[NEW]) for r in valid)
    assert summary['top5_order_changed']==sum(r[BASE]!=r[NEW] for r in valid)
    assert summary['new_candidates_per_image']=={r['image']:len(set(r[NEW])-set(r[BASE])) for r in valid}
    assert summary['accuracy'] is None and summary['human_ratings_received']==0
    result={'inputs':10,'valid':8,'failures_retained':2,'sheets_verified':10,
            'padding_xy_masks_and_ranks_verified':8,'bruteforce_costs':checks,'max_cost_error':maximum,
            'render_replays':replays,'max_independent_affine_error':max_affine,'line_support_diagnostics':line_diagnostics,
            'rankings_sha256':sha256(args.out/'rankings.json'),'summary_sha256':sha256(args.out/'summary.json'),
            'auditor_sha256':sha256(Path(__file__))}
    save_json(args.out/'audit.json',result); print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); p.add_argument('--library-root',type=Path,required=True)
    run(p.parse_args())
