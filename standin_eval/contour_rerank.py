"""Offline image-boundary evidence at frozen candidate cameras; no query GT."""
import cv2
import numpy as np


def line_mask(rgb):
    image=np.asarray(rgb)
    if image.ndim!=3 or image.shape[2]!=3:
        raise ValueError('RGB image required')
    return np.min(image,axis=2)<200


def distance_field(mask):
    mask=np.asarray(mask,bool)
    if mask.ndim!=2 or mask.sum()<10:raise ValueError('insufficient line pixels')
    return cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)


def prepare_query(rgb,xy,size=256):
    rgb=np.asarray(rgb);xy=np.asarray(xy,float)
    if xy.shape!=(17,2) or not np.isfinite(xy).all():raise ValueError('finite COCO17 XY required')
    h,w=rgb.shape[:2]
    if h!=w:raise ValueError('pilot expects square images; do not distort aspect ratio')
    target=(xy+.5)*(size/w)-.5
    torso=float(np.linalg.norm(target[[5,6]].mean(0)-target[[11,12]].mean(0)))
    if torso<1:raise ValueError('degenerate torso')
    small=cv2.resize(rgb,(size,size),interpolation=cv2.INTER_AREA)
    mask=line_mask(small)
    return {'mask':mask,'distance':distance_field(mask),'xy':target,'torso':torso,'size':size}


def anchor_affine(candidate_xy,query_xy):
    """Same hip/torso alignment as XY search; no extra camera or shape fit."""
    c,q=np.asarray(candidate_xy,float),np.asarray(query_xy,float)
    if c.shape!=(17,2) or q.shape!=(17,2) or not np.isfinite(c).all() or not np.isfinite(q).all():
        raise ValueError('finite matching COCO17 XY required')
    ch,qh=c[[11,12]].mean(0),q[[11,12]].mean(0)
    ct,qt=np.linalg.norm(c[[5,6]].mean(0)-ch),np.linalg.norm(q[[5,6]].mean(0)-qh)
    if ct<1e-6 or qt<1:raise ValueError('degenerate torso')
    scale=qt/ct;translation=qh-scale*ch
    return np.array([[scale,0,translation[0]],[0,scale,translation[1]]],dtype=float)


def contour_cost(query,candidate_mask):
    candidate=np.asarray(candidate_mask,bool)
    if candidate.shape!=query['mask'].shape:raise ValueError('matching raster shapes required')
    distance=distance_field(candidate)
    forward=np.minimum(query['distance'][candidate]/query['torso'],.5)
    backward=np.minimum(distance[query['mask']]/query['torso'],.5)
    return float((forward.mean()+backward.mean())/2)


def candidate_evidence(query,joints,parameters):
    # Import lazily to keep numerical scoring independent of rendering.
    from scripts.experiment_z_rendered import render_pose
    yaw,pitch,inverse=map(float,parameters)
    if inverse<0:raise ValueError('negative inverse distance')
    image,xy=render_pose(joints,yaw,size=query['size'],style='rough',seed=0,
                         pitch=pitch,distance=None if inverse==0 else 1/inverse)
    affine=anchor_affine(xy,query['xy'])
    mask=cv2.warpAffine(line_mask(np.asarray(image)).astype(np.uint8),affine,
                        (query['size'],query['size']),flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    return contour_cost(query,mask),affine,mask


def rerank(original_order,scores,penalties,weight,limit=20):
    order=np.asarray(original_order,dtype=int);scores=np.asarray(scores,float)
    penalty=np.asarray(penalties,float)
    if (len(order)<limit or penalty.shape!=(limit,) or weight<0 or not np.isfinite(penalty).all()
        or np.any(penalty<0) or np.any(penalty>.5)):
        raise ValueError('finite shortlist penalties required')
    if len(set(order.tolist()))!=len(order) or len(scores)!=len(order):raise ValueError('unique full order required')
    if weight==0:return order.copy()
    # First result and original tail are immutable; ties keep original shortlist order.
    head=order[1:limit]
    perm=np.argsort(scores[head]+weight*penalty[1:],kind='stable')
    return np.concatenate((order[:1],head[perm],order[limit:]))
