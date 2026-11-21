"""Geometric shortlist scoring and bone evidence."""

import numpy as np
from .evaluate import rank
from .evaluate import validation
from src.camera_search import frame

LIMITS = (80, 120, 200)


BONES = ((5,6),(11,12),(5,11),(6,12),(5,7),(7,9),(6,8),(8,10),
         (11,13),(13,15),(12,14),(14,16))


def normalized(vectors):
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(lengths, 1e-6), lengths[..., 0] > 1e-6


class BoneEvidence:
    def __init__(self, camera):
        self.camera = camera
        start, stop = np.array(BONES).T
        projected = camera.projections
        self.vectors, self.valid = normalized(projected[:,:,stop] - projected[:,:,start])

    def position_and_direction(self, evidence, part):
        camera = self.camera
        mask = np.asarray(evidence['joint_mask'], bool)
        selected = np.asarray(range(5,17) if part=='full' else range(5,11))
        selected = selected[mask[selected]]
        start, stop = np.asarray(BONES).T
        used = mask[start] & mask[stop]
        if part=='upper':
            used &= (start<11) & (stop<11)
        points = np.asarray(evidence['keypoints'], dtype=np.float32)
        vectors, observed = normalized(points[stop] - points[start])
        used &= observed
        query, ok = frame(points, part, mask)
        if used.sum()<3 or not ok:
            return {}
        positions, valid = camera.frames[part]
        xy = np.linalg.norm(positions[:,:,selected]-query[selected],axis=-1).mean(axis=-1)
        dot = np.einsum('cpbd,bd->cpb',self.vectors[:,:,used],vectors[used])
        cost = np.where(self.valid[:,:,used],(1.-np.clip(dot,-1.,1.))*.5,1.).mean(axis=-1)
        present = np.all(camera.scores[:,selected]>=.3,axis=-1)
        mixed = np.where(camera.valid & valid & present[None,:],xy + .35*cost,np.inf)
        return dict(zip(camera.pose_ids,mixed.min(axis=0)))


def geometry(engine, evidence, required, bone):
    part = 'upper' if evidence['coverage_class']=='upper_only' else 'full'
    original = engine._clean(engine.camera.search(evidence['keypoints'],evidence['joint_mask'],part),'geometry')
    mixed = bone.position_and_direction(evidence,part) if bone else {}
    accepted=[]
    for hit in original:
        if not 0<=hit['distance']<=engine.cutoffs[part]:
            continue
        cost=0.
        if required and required['active']:
            checked,cost=validation(engine.validator.check(required,hit['pose_id']),'unknown_soft')
            if checked['status'] not in ('passed','near_match'):
                continue
            hit=dict(hit,whole_pose_validation=checked)
        accepted.append((hit['distance']+cost,mixed.get(hit['pose_id'],hit['distance'])+cost,hit))
    position=[h for _,_,h in sorted(accepted,key=lambda row:row[0])]
    direction=[h for _,_,h in sorted(accepted,key=lambda row:row[1])]
    # Preserve the strongest position matches while allocating the rest to
    # candidates with strong joint-direction evidence.
    reserved=position[:40]
    ids={h['pose_id'] for h in reserved}
    diverse=(reserved+[h for h in direction if h['pose_id'] not in ids])[:80]
    options={f'position_{k}':position[:k] for k in LIMITS}
    options.update(direction_80=direction[:80],position_direction_80=diverse)
    return options,position,direction,original
