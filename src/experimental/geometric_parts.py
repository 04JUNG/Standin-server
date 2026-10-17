"""Full-first, observed-part geometry references. Offline/opt-in experiment only.

Does not change full candidates, VLM routing, feature schema, confidence or refine.
Upper/lower references compare only observed joints in a symmetric anchor-width
frame. A part match is NOT a corrected full-body pose or proof of extraction truth.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from time import perf_counter

import numpy as np

from src.features import _BONES
from src.search import pose_family_id

PARTS = {
    'upper': {'joints':tuple(range(5,11)), 'anchors':(5,6), 'bones':tuple(range(4)), 'unit':'shoulder_width'},
    'lower': {'joints':tuple(range(11,17)), 'anchors':(11,12), 'bones':tuple(range(4,8)), 'unit':'hip_width'},
}


@dataclass(frozen=True)
class PartPolicy:
    min_bones: int = 3
    min_box_px: float = 64.
    min_box_height_ratio: float = .05
    min_anchor_box_ratio: float = .05
    owner_padding: float = .15
    min_segment_widths: float = .05
    upper_max_segment_widths: float = 3.
    lower_max_segment_widths: float = 5.
    min_relative_gain: float = .25
    min_absolute_gain: float = .1

    def __post_init__(self):
        values=asdict(self)
        if any(not np.isfinite(value) or value<0 for value in values.values()):
            raise ValueError('part thresholds must be finite and nonnegative')
        if isinstance(self.min_bones,bool) or not isinstance(self.min_bones,int) or not 1<=self.min_bones<=4:
            raise ValueError('min_bones must be an integer in [1,4]')
        if any(values[name]>1 for name in ('min_box_height_ratio','min_anchor_box_ratio','owner_padding','min_relative_gain')):
            raise ValueError('part ratios must be in [0,1]')
        if self.min_box_px<=0 or self.min_anchor_box_ratio<=0 or self.min_segment_widths<=0:
            raise ValueError('size and degeneration guards must be positive')
        if min(self.upper_max_segment_widths,self.lower_max_segment_widths)<=self.min_segment_widths:
            raise ValueError('max segment lengths must exceed minimum')


def part_frame(points, part, mask=None):
    """One or N COCO17 projections -> same region normalization on both sides.

    Re-centering/rescaling the existing hip/torso library features cancels their
    original affine normalization. Missing points stay masked, never reconstructed.
    """
    spec=PARTS[part];p=np.asarray(points,dtype=np.float32)
    if p.shape[-2:]!=(17,2):raise ValueError('expected (...,17,2)')
    m=np.ones(p.shape[:-1],bool) if mask is None else np.broadcast_to(np.asarray(mask,bool),p.shape[:-1])
    a,b=spec['anchors'];center=(p[...,a,:]+p[...,b,:])*.5
    width=np.linalg.norm(p[...,a,:]-p[...,b,:],axis=-1)
    valid=m[...,a]&m[...,b]&np.isfinite(center).all(axis=-1)&np.isfinite(width)&(width>1e-6)
    out=np.zeros_like(p)
    for j in spec['joints']:
        observed=m[...,j]&np.isfinite(p[...,j,:]).all(axis=-1)
        valid &= (~m[...,j])|observed
        safe=np.where(observed[...,None],p[...,j,:],center)
        out[...,j,:]=(safe-center)/np.where(width>1e-6,width,1.)[...,None]
    out=np.where(valid[...,None,None],out,0.)
    return out,valid


def _inside(point,box,padding=0.):
    x1,y1,x2,y2=box;dx=(x2-x1)*padding;dy=(y2-y1)*padding
    return x1-dx<=point[0]<=x2+dx and y1-dy<=point[1]<=y2+dy


def assess_part(points,joint_mask,bone_mask,part,*,owner_box,peer_boxes=(),image_size,policy=PartPolicy(),suspect_limbs=()):
    spec=PARTS[part];p=np.asarray(points,float);m=np.asarray(joint_mask,bool);bm=np.asarray(bone_mask,bool)
    if p.shape!=(17,2) or m.shape!=(17,) or bm.shape!=(12,):raise ValueError('invalid part evidence shape')
    reasons=[];a,b=spec['anchors'];indices=[j for j in spec['joints'] if m[j]]
    image_shape=np.asarray(image_size,float)
    valid_image=image_shape.shape==(2,) and np.isfinite(image_shape).all() and (image_shape>0).all()
    if not valid_image:reasons.append('invalid_image_size')
    valid_peers=[]
    for peer in peer_boxes:
        if peer is None:continue
        box=np.asarray(peer,float)
        if box.shape!=(4,) or not np.isfinite(box).all() or box[2]<=box[0] or box[3]<=box[1]:
            reasons.append('invalid_peer_box')
        else:valid_peers.append(box)
    bones=[i for i in spec['bones'] if bm[i] and m[_BONES[i][0]] and m[_BONES[i][1]]]
    if not m[[a,b]].all():reasons.append('missing_region_anchors')
    if len(bones)<policy.min_bones:reasons.append('insufficient_region_bones')
    if not np.isfinite(p[indices]).all():reasons.append('nonfinite_observed_joints')
    region_limbs=('left_arm','right_arm') if part=='upper' else ('left_leg','right_leg')
    if set(region_limbs)&set(suspect_limbs):reasons.append('suspect_region_ownership_or_structure')
    if owner_box is None:reasons.append('owner_box_unknown')
    else:
        box=np.asarray(owner_box,float)
        if box.shape!=(4,) or not np.isfinite(box).all() or box[2]<=box[0] or box[3]<=box[1]:reasons.append('invalid_owner_box')
        else:
            h=box[3]-box[1]
            if h<policy.min_box_px or (valid_image and h/image_shape[1]<policy.min_box_height_ratio):reasons.append('owner_box_too_small')
            if any(not _inside(p[j],box,policy.owner_padding) for j in indices):reasons.append('region_outside_owner')
            if m[[a,b]].all():
                center=(p[a]+p[b])*.5;width=float(np.linalg.norm(p[a]-p[b]))
                if not np.isfinite(width) or width<max(1e-6,h*policy.min_anchor_box_ratio):reasons.append('region_anchor_projection_degenerate')
                if any(_inside(center,peer) for peer in valid_peers):reasons.append('region_center_in_peer_box')
                max_length=policy.upper_max_segment_widths if part=='upper' else policy.lower_max_segment_widths
                lengths=[float(np.linalg.norm(p[_BONES[i][0]]-p[_BONES[i][1]])) for i in bones]
                if lengths and (min(lengths)<policy.min_segment_widths*width or max(lengths)>max_length*width):reasons.append('region_segment_length_outlier')
    return {'sufficient':not reasons,'reasons':reasons,'part':part,'observed_joints':indices,
            'observed_bones':bones,'unit':spec['unit'],'thresholds':asdict(policy)}


class PartGeometryIndex:
    """Immutable projections; no tag/view priors or VLM meaning in ranking."""
    def __init__(self,entries):
        self.entries=tuple(entries)
        self.features=np.stack([e.feature.reshape(17,2) for e in self.entries]) if self.entries else np.empty((0,17,2),np.float32)
        if not np.isfinite(self.features).all():raise ValueError('nonfinite library feature')
        self.frames={part:part_frame(self.features,part) for part in PARTS}
        self.lookup={(e.pose_id,e.view.value):i for i,e in enumerate(self.entries)}
        self.features.setflags(write=False)
        for points,valid in self.frames.values():points.setflags(write=False);valid.setflags(write=False)

    def distances(self,points,mask,part):
        q,valid=part_frame(points,part,mask)
        selected=[i for i in PARTS[part]['joints'] if mask[i]]
        if not valid or len(selected)<4:return np.full(len(self.entries),np.inf)
        library,usable=self.frames[part]
        ds=np.linalg.norm(library[:,selected]-q[selected],axis=2).mean(axis=1)
        return np.where(usable,ds,np.inf)

    def search(self,points,mask,part,*,top_k=5,quarantined=()):
        if top_k<=0:return []
        distances=self.distances(points,mask,part);seen=set();hits=[]
        for index in np.argsort(distances,kind='stable'):
            entry=self.entries[index];family=pose_family_id(entry.pose_id,entry.meta)
            if not np.isfinite(distances[index]) or entry.pose_id in quarantined or family in seen:continue
            seen.add(family);hits.append({'pose_id':entry.pose_id,'view':entry.view.value,'part':part,
                'part_distance':float(distances[index]),'distance_unit':PARTS[part]['unit'],
                'bvh_path':entry.bvh_path,'kind':'observed_part_geometry_reference',
                'confidence':'low','refine_allowed':False})
            if len(hits)>=top_k:break
        return hits


def full_first_supplement(index,full_candidates,points,joint_mask,bone_mask,*,full_trusted,
        owner_box,peer_boxes=(),image_size,policy_block=False,suspect_limbs=(),policy=PartPolicy(),quarantined=()):
    """Keep full results verbatim. Add references ONLY where the full result lacks.

    Trusted full path performs zero part searches. Unobserved/ambiguous regions
    never trigger geometry search. A weak full candidate's already-matching region
    is not supplemented without both absolute and relative improvement.
    """
    started=perf_counter()
    output={'policy':'full-first-parts-v1-offline','full_candidates':full_candidates,
        'parts':{},'part_search_calls':0,'refine_allowed':False,'synthetic_pose_created':False,
        'release_allowed':False}
    if policy_block:output['status']='blocked_existing_policy';return output
    if full_candidates and full_trusted:
        output.update(status='full_preserved',elapsed_ms=(perf_counter()-started)*1000);return output
    if points is None:output['status']='no_observed_skeleton';return output
    mask=np.asarray(joint_mask,bool)
    for part in PARTS:
        evidence=assess_part(points,mask,bone_mask,part,owner_box=owner_box,peer_boxes=peer_boxes,
            image_size=image_size,policy=policy,suspect_limbs=suspect_limbs)
        record={'evidence':evidence,'search_ran':False,'hits':[],'decision':'not_observable'}
        if evidence['sufficient']:
            output['part_search_calls']+=1;record['search_ran']=True
            hits=index.search(points,mask,part,quarantined=quarantined)
            base_distance=None
            if full_candidates:
                top=full_candidates[0];row=index.lookup.get((top['pose_id'],top['view']))
                if row is not None:base_distance=float(index.distances(points,mask,part)[row])
            record['full_candidate_part_distance']=base_distance
            gain=(base_distance-hits[0]['part_distance']) if hits and base_distance is not None else None
            if not hits:record['decision']='no_part_candidate'
            elif base_distance is not None and (gain<policy.min_absolute_gain or gain/max(base_distance,1e-8)<policy.min_relative_gain):
                record['decision']='existing_part_preserved_no_material_gain'
            else:record.update(hits=hits,decision='supplement_reference_only')
        output['parts'][part]=record
    output['status']='part_references_added' if any(r['hits'] for r in output['parts'].values()) else 'no_safe_part_supplement'
    output['elapsed_ms']=(perf_counter()-started)*1000
    return output
