"""Image-only input corrections and translation/scale-fit retrieval experiments.

No helper here accepts answer IDs or reference query joints. Gallery XYZ is
permitted, as in any retrieval system; source query XYZ is never a model input.
"""
from __future__ import annotations

import cv2
import numpy as np


def foreground_bbox(image):
    """White/neutral-background pilot crop, inferred from pixels, not GT.

    This is deliberately not a general comic panel person detector.
    """
    border = np.concatenate((image[0], image[-1], image[:,0], image[:,-1]))
    background = np.median(border,axis=0)
    foreground = np.max(np.abs(image.astype(float)-background),axis=2)>25
    yy,xx=np.where(foreground)
    if len(xx)<20:
        return [0,0,image.shape[1],image.shape[0]]
    return [int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]


def undo_quarter_turn(points, turns, width, height):
    """Undo np.rot90 pixel coordinates; anatomical labels are unchanged."""
    out=np.asarray(points,dtype=np.float32).copy()
    x,y=out[:,0].copy(),out[:,1].copy()
    if turns==1:
        out[:,0],out[:,1]=width-1-y,x
    elif turns==2:
        out[:,0],out[:,1]=width-1-x,height-1-y
    elif turns==3:
        out[:,0],out[:,1]=y,height-1-x
    return out


def infer_pose(image, model, *, crop=False, turns=(0,), is_3d=True):
    """Choose upright hypothesis using model confidence only, never retrieval/GT."""
    hypotheses=[]
    for turn in turns:
        rotated=np.ascontiguousarray(np.rot90(image,turn))
        kwargs={"bboxes":[foreground_bbox(rotated)]} if crop else {}
        result=model(rotated,**kwargs)
        if is_3d:
            k3,scores,_,k2=result
            z=np.asarray(k3[0,:17,2],dtype=float)
        else:
            k2,scores=result
            z=np.zeros(17)
        points=undo_quarter_turn(k2[0,:17],turn,image.shape[1],image.shape[0])
        confidence=np.asarray(scores[0,:17],dtype=float)
        hypotheses.append({"keypoints2d":points.tolist(),"z_meters":z.tolist(),
                           "scores":confidence.tolist(),"turn":int(turn),
                           "selection_score":float(np.median(confidence[5:]))})
    return max(hypotheses,key=lambda h:h["selection_score"])


def fit_distances(gallery_xy, query_xy, valid_mask):
    """Fit each candidate by translation and positive scale, not rotation.

    Fixes the sensitivity to one erroneous hip/shoulder anchor. Both operands
    are centered over exactly the same visible body joints, then scale is
    fitted by least squares. RMS error is normalized by query body RMS.
    """
    selected=np.arange(5,17)[np.asarray(valid_mask,dtype=bool)[5:]]
    if len(selected)<6:
        return np.full(len(gallery_xy),np.inf,dtype=np.float32)
    query=np.asarray(query_xy,dtype=np.float32)[selected]
    library=np.asarray(gallery_xy,dtype=np.float32)[:,selected]
    query=query-query.mean(axis=0)
    library=library-library.mean(axis=1,keepdims=True)
    scale=np.einsum('nji,ji->n',library,query)/np.maximum(
        np.square(library).sum(axis=(1,2)),1e-8)
    scale=np.maximum(scale,0)
    residual=library*scale[:,None,None]-query
    norm=max(float(np.square(query).sum()/len(selected)),1e-8)
    return np.sqrt(np.square(residual).sum(axis=(1,2))/len(selected)/norm)
