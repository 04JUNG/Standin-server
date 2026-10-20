"""Image-derived heading hypotheses, with explicit evidence and alternatives."""
import math
import numpy as np

YAW={'front':0.,'front_left':-45.,'front_right':45.,'left':-90.,'right':90.,'back_left':-135.,'back_right':135.,'back':180.}


def owns_observations(box,points,image_size,padding=.15):
    box=np.asarray(box,float);size=np.asarray(image_size,float);extent=box[2:]-box[:2]
    valid=[]
    for name,value in (points or {}).items():
        p=np.asarray(value,float)
        if p.shape!=(2,) or not np.isfinite(p).all():continue
        p=p*size/1000;inside=bool(np.all(p>=box[:2]-extent*padding) and np.all(p<=box[2:]+extent*padding))
        if name=='head_center':return inside
        valid.append(inside)
    return bool(valid and np.mean(valid)>=.5)


def infer_heading(person,observation,view_scores,peers,image_size):
    owner=owns_observations(person['bbox'],observation.get('visible_points'),image_size)
    label=observation.get('body_heading_best_guess','unknown')
    if owner and label in YAW:
        basis=observation.get('heading_basis','insufficient')
        return dict(yaw_degrees=YAW[label],label=label,status='image_heading_hypothesis',confidence=observation.get('heading_confidence','low'),basis=basis,ownership_valid=True,source='fresh_full_scene_and_target',alternative=observation.get('heading_alternative','unknown'),direct_torso_observation=basis=='directly_visible_torso')
    scores={k:float(view_scores.get(k,-1)) for k in ('front','back','left','right','down')}
    center=(person['bbox'][0]+person['bbox'][2])/2
    peers=[p for p in peers if p['person_index']!=person['person_index']]
    if not peers:return dict(yaw_degrees=None,status='insufficient_heading_evidence',ownership_valid=owner)
    nearest=min(peers,key=lambda p:abs((p.get('head_center') or [(p['bbox'][0]+p['bbox'][2])/2])[0]-center))
    peer_x=(nearest.get('head_center') or [(nearest['bbox'][0]+nearest['bbox'][2])/2])[0]
    side=max(scores['left'],scores['right']);front_back=max(scores['front'],scores['back'])
    if side>=front_back-.01 and side>.2:
        towards='right' if peer_x>center else 'left'
        prefix='back_' if scores['back']>scores['front'] else 'front_'
        label=prefix+towards
        return dict(yaw_degrees=YAW[label],label=label,status='image_heading_hypothesis',confidence='low',basis='image_side_back_evidence_and_peer_facing_prior',ownership_valid=owner,source='fresh_contrastive_image_scores+relative_peer_position',alternative=towards,direct_torso_observation=False,peer_person_index=nearest['person_index'],view_scores=scores,assumption='partial actor faces nearest interlocutor; may be wrong')
    if front_back>.2 and abs(scores['front']-scores['back'])>.015:
        label='front' if scores['front']>scores['back'] else 'back'
        return dict(yaw_degrees=YAW[label],label=label,status='image_heading_hypothesis',confidence='low',basis='image_front_back_comparison',ownership_valid=owner,source='fresh_contrastive_image_scores',alternative='unknown',direct_torso_observation=False)
    return dict(yaw_degrees=None,status='insufficient_heading_evidence',ownership_valid=owner)
