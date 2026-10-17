"""Prespecified local crop hypotheses; confidence selection, no GT coordinates."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_z_rendered import save_json,sha256
from scripts.review_perspective_roughs import rgb
from standin_eval.z_diagnostics import undo_quarter_turn


def run(a):
    from rtmlib import RTMPose
    a.out.mkdir(parents=True,exist_ok=False)
    settings=[(x,y) for x in (0.,.1,.2) for y in (0.,.2,.35)]
    save_json(a.out/'protocol.json',{'purpose':'rough178 development crop diagnostic only, not pipeline improvement evidence',
        'settings':settings,'turns':[0,1,2,3],'selection':'maximum median body confidence only',
        'policy':'same score.3 mask; missing joints never invented; full search gate unchanged',
        'image_sha256':sha256(a.image),'model_sha256':sha256(a.model),'code_sha256':sha256(Path(__file__))})
    image=rgb(a.image);h,w=image.shape[:2]
    model=RTMPose(str(a.model),model_input_size=(192,256),backend='onnxruntime',device='cpu')
    rows=[]
    for x,y in settings:
        left,top=int(w*x),int(h*y);crop=image[top:,left:w-left]
        for turn in (0,1,2,3):
            rotated=np.ascontiguousarray(np.rot90(crop,turn));rh,rw=rotated.shape[:2]
            points,scores=model(rotated,bboxes=[[0,0,rw,rh]])
            points=undo_quarter_turn(points[0,:17],turn,crop.shape[1],crop.shape[0])+[left,top]
            confidence=np.asarray(scores[0,:17]);mask=confidence>=.3
            rows.append({'crop':[left,top,w-left,h],'turn':turn,'keypoints2d':points.tolist(),'scores':confidence.tolist(),
                'median_body_confidence':float(np.median(confidence[5:])),
                'valid_body':int(mask[5:].sum()),'valid_upper_body':int(mask[5:11].sum()),
                'full_searchable':bool(mask[5:].sum()>=6 and mask[[5,6,11,12]].all()),
                'upper_searchable':bool(mask[5:11].all())})
    selected=max(rows,key=lambda r:r['median_body_confidence'])
    save_json(a.out/'results.json',{'records':rows,'confidence_selected':selected,
        'any_full_searchable':any(r['full_searchable'] for r in rows),
        'any_upper_searchable':any(r['upper_searchable'] for r in rows),
        'limits':'a high-confidence skeleton still requires visual verification; no result-quality-based crop selection'})
    print({k:v for k,v in selected.items() if k not in ('keypoints2d','scores')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('image','model','out'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
