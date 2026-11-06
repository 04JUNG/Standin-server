"""Label-free P4/P5 experiment runner; GT evaluation lives in another process."""
import argparse,json,hashlib
from pathlib import Path
import cv2
import numpy as np
from experiments.refine_v5.learning_data import BONES,COCO,axes,frame,geometry,image_features,normalized_points,visibility,digest,source_camera,point_visibility
from experiments.refine_v5.learned import load_model
from experiments.refine_v5.iterative import solve,hierarchy_digest
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--previous',required=True);p.add_argument('--repo',required=True);p.add_argument('--body',required=True);p.add_argument('--fusion',choices=['observation','learned_balanced'],default='observation');p.add_argument('--without-learning',action='store_true');a=p.parse_args();run=Path(a.run);previous=Path(a.previous);repo=Path(a.repo)
    out=run/('iterative-geometric' if a.without_learning else ('iterative' if a.fusion=='observation' else 'iterative-balanced'));out.mkdir(exist_ok=False)
    rigs=json.loads((run/'native-data/records.json').read_text());seeds={r['case_id']:r for r in json.loads((previous/'all-seeds/seeds.json').read_text())};reviews={r['case_id']:r for r in json.loads((repo/'evaluation/refine_v5/visual_review_v2.json').read_text())['records']}
    chosen=json.loads((run/'models/selection.json').read_text())['selected'];checkpoint=run/'models'/(chosen+'.pt');model,ck=(None,{}) if a.without_learning else load_model(checkpoint);torch.set_num_threads(2)
    results=[];inputs=[]
    for rig in rigs:
        if rig['status']!='ready':continue
        case=rig['case_id'];seed=seeds[case];review=reviews[case];mask=visibility(review)
        q,pmask=normalized_points(seed['keypoints'],seed['scores']);pmask=pmask.astype(bool)
        # Visibility is a conservative draft; low detector scores do not establish ownership.
        pmask &= point_visibility(mask)
        by,center,scale,body=frame(rig['bones']);camera=source_camera(rig['bones'],seed['view'])
        source_sha=digest(rig['base_fbx']);rig={**rig,'source_sha256':source_sha};im=cv2.imread(seed['image'],cv2.IMREAD_GRAYSCALE)
        frozen=image_features(im,np.array(seed['keypoints'])[COCO]);base_axes=axes(rig['bones'])
        def propose(world=None):
            if world is None:b=base_axes
            else:
                names=[b['name'] for b in rig['bones']];ix={n:i for i,n in enumerate(names)}
                b=np.array([world[ix[n],:3,:2].T@body for n in BONES]);b/=np.linalg.norm(b,axis=-1,keepdims=True)
            if a.without_learning:return dict(axes=b.tolist(),uncertainty_rad=[float(np.pi/2)]*13)
            g=geometry(b,q,normalized_points(seed['keypoints'],seed['scores'])[1],review['representation'],camera)
            with torch.no_grad():pred,lv=model(torch.tensor(g[None],dtype=torch.float32),torch.tensor(frozen[None],dtype=torch.float32),torch.tensor(b[None],dtype=torch.float32))
            return dict(axes=pred[0].numpy().tolist(),uncertainty_rad=np.maximum(np.exp(lv[0].numpy()/2),ck['validation_error_floor_rad']).tolist())
        prediction=propose();evidence=dict(points=q.tolist(),point_mask=pmask.tolist(),part_mask=mask.tolist(),camera=camera.tolist(),ownership_verified=False,offline_ownership_assumption=review['target_identity']!='ambiguous',support_verified=False,orientation_observed=[False]*5)
        # Model priors never become observed palm/sole/head direction annotations.
        for prof in ('conservative12','experimental30'):
            profile=dict(name=prof,body_asset_sha256=digest(a.body),hierarchy_sha256=hierarchy_digest(rig['bones']),offline_only=True,approved=False,fusion=a.fusion)
            selection=dict(kind='offline_fixture_selection',source_sha256=source_sha,body_asset_sha256=profile['body_asset_sha256'])
            for stage,context in [('source',None),('selected_target',selection)]:
                for count in (1,3):
                    # Equal maximum nfev budget: 1x60 vs 3x20, including rejected rounds.
                    result=solve(rig,evidence,prediction,profile=profile,selection=context,rounds=count,max_nfev=60//count,seconds=30,redetect=propose if count>1 else None)
                    results.append(dict(case_id=case,split=rig['split'],stage=stage,rounds=count,**result))
        inputs.append(dict(case_id=case,seed_original_ownership_verified=seed['ownership_verified'],experimental_ownership_assumption=evidence['offline_ownership_assumption'],source_sha256=source_sha,image_sha256=seed['image_sha256'],profile=profile))
        (out/'results.json').write_text(json.dumps(results));print('V5_SOLVED',case,flush=True)
    (out/'manifest.json').write_text(json.dumps(dict(schema='v5-p4-p5-run-v1',model=None if a.without_learning else chosen,checkpoint_sha256=None if a.without_learning else digest(checkpoint),gt_opened_by_runner=False,test_used=False,orientation_observed=False,profiles_certified=False,inference_ownership='offline_assumption_only',max_nfev_total=60,records=inputs),indent=2))
if __name__=='__main__':main()
