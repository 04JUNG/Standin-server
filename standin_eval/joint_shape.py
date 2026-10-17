"""Observed 2D elbow bend and leg crossing at each candidate's fitted camera.

No source IDs, human preferred ranks, or true query depth are inputs.
"""
import numpy as np
from standin_eval.dual_label_search import alternate_labels,LEFT_RIGHT

ARM_TRIPLES=((5,7,9),(6,8,10))
LEG_JOINTS=(11,12,13,14,15,16)
SEGMENTS=(((11,13),(12,14)),((11,13),(14,16)),((13,15),(12,14)),((13,15),(14,16)))
MIN_BONE=.05
MARGIN=.025


def elbow_angles(xy,mask):
    """Degrees; None for unobserved or very short projected bones."""
    xy=np.asarray(xy,float); mask=np.asarray(mask,bool); result=[]
    for shoulder,elbow,wrist in ARM_TRIPLES:
        u,v=xy[shoulder]-xy[elbow],xy[wrist]-xy[elbow]
        norms=np.linalg.norm(u),np.linalg.norm(v)
        if not mask[[shoulder,elbow,wrist]].all() or min(norms)<MIN_BONE:
            result.append(None)
        else:result.append(float(np.degrees(np.arccos(np.clip(np.dot(u,v)/np.prod(norms),-1,1)))))
    return result


def cross2(a,b):
    return float(a[0]*b[1]-a[1]*b[0])


def point_segment_distance(p,a,b):
    v=b-a; t=np.clip(np.dot(p-a,v)/max(np.dot(v,v),1e-12),0,1)
    return float(np.linalg.norm(p-(a+t*v)))


def segment_relation(a,b,c,d):
    """True proper interior crossing; None near contact; False separated."""
    u,v=b-a,d-c; determinant=cross2(u,v)
    if abs(determinant)>1e-10:
        t=cross2(c-a,v)/determinant; s=cross2(c-a,u)/determinant
        if 0<t<1 and 0<s<1:
            clearance=min(t*np.linalg.norm(u),(1-t)*np.linalg.norm(u),s*np.linalg.norm(v),(1-s)*np.linalg.norm(v))
            return True if clearance>=MARGIN else None
    distance=min(point_segment_distance(a,c,d),point_segment_distance(b,c,d),
                 point_segment_distance(c,a,b),point_segment_distance(d,a,b))
    return None if distance<MARGIN else False


def leg_crossing(xy,mask):
    xy=np.asarray(xy,float); mask=np.asarray(mask,bool)
    if not mask[list(LEG_JOINTS)].all():return None
    if any(np.linalg.norm(xy[b]-xy[a])<MIN_BONE for a,b in ((11,13),(13,15),(12,14),(14,16))):return None
    values=[segment_relation(xy[a],xy[b],xy[c],xy[d]) for (a,b),(c,d) in SEGMENTS]
    if True in values:return True
    return None if None in values else False


def shape_penalties(query_xy,query_mask,candidate_xy):
    """Use one global label correspondence selected from XY, then both features."""
    q=np.asarray(query_xy,float); mask=np.asarray(query_mask,bool); candidates=np.asarray(candidate_xy,float)
    if q.shape!=(17,2) or mask.shape!=(17,) or candidates.ndim!=3 or candidates.shape[1:]!=(17,2):
        raise ValueError('COCO17 query and candidate batch required')
    if not np.isfinite(q).all() or not np.isfinite(candidates).all():raise ValueError('finite joints required')
    selected=np.flatnonzero(mask[5:])+5
    if len(selected)<6:raise ValueError('at least6 observed body joints required')
    alt,am=alternate_labels(q,mask); other_selected=np.flatnonzero(am[5:])+5
    direct=np.linalg.norm(candidates[:,selected]-q[selected],axis=-1).mean(1)
    alternate=np.linalg.norm(candidates[:,other_selected]-alt[other_selected],axis=-1).mean(1)
    swap=alternate<direct
    aligned=candidates.copy(); aligned[swap]=aligned[swap][:,LEFT_RIGHT]
    angles=elbow_angles(q,mask); crossing=leg_crossing(q,mask)
    angle_cost=np.zeros(len(aligned)); cross_cost=np.zeros(len(aligned)); all_valid=np.ones(17,bool)
    candidate_angles=[]; candidate_cross=[]
    for i,c in enumerate(aligned):
        a=elbow_angles(c,all_valid); cr=leg_crossing(c,all_valid)
        errors=[1. if a[j] is None else abs(a[j]-value)/180 for j,value in enumerate(angles) if value is not None]
        if errors:angle_cost[i]=float(np.mean(errors))
        if crossing is not None:cross_cost[i]=.5 if cr is None else float(cr!=crossing)
        candidate_angles.append(a);candidate_cross.append(cr)
    return angle_cost,cross_cost,{'query_angles_degrees':angles,'query_crossing':crossing,
        'used_arms':sum(a is not None for a in angles),'used_crossing':crossing is not None,
        'swapped_candidates':swap.tolist(),'candidate_angles_degrees':candidate_angles,'candidate_crossing':candidate_cross}


def reorder_shortlist(original_order,scores,penalties,weight,limit=20):
    """All shortlist ranks including1 can change; tail stays immutable."""
    order=np.asarray(original_order,int); values=np.asarray(scores,float); p=np.asarray(penalties,float)
    if len(order)<limit or p.shape!=(limit,) or len(values)!=len(order) or weight<0 or not np.isfinite(p).all():
        raise ValueError('finite shortlist evidence and nonnegative weight required')
    head=order[:limit]; permutation=np.argsort(values[head]+weight*p,kind='stable')
    return np.concatenate((head[permutation],order[limit:]))
