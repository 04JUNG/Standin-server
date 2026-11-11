"""Map COCO17 to the same native joints without consulting manual targets."""
import numpy as np

MAPPING = {'RightArm': [6], 'RightForeArm': [8], 'RightHand': [10],
           'LeftArm': [5], 'LeftForeArm': [7], 'LeftHand': [9],
           'Neck': [5, 6], 'LeftLeg': [13]}


def map_targets(record, image_hash, image_size, min_score=0.3):
    if not 0 <= min_score <= 1:
        raise ValueError('min_score must be between 0 and 1 for this experiment')
    develop = record.get('producer') == 'develop HumanArt M multiview consensus'
    if (record.get('producer') != 'rtmlib.Body' and not develop) or record.get('keypoint_format') != 'COCO17':
        raise ValueError('Expected Body or develop HumanArt COCO17 observations')
    if record.get('image_sha256') != image_hash or record.get('image_size') != image_size:
        raise ValueError('Observation image does not match fixture')
    kp = np.asarray(record['keypoints'], dtype=float)
    scores = np.asarray(record['scores'], dtype=float)
    if kp.shape != (17, 2) or scores.shape != (17,):
        raise ValueError('Invalid COCO17 shapes')
    mask = np.asarray(record['joint_mask'], dtype=bool) if develop else scores >= min_score
    if mask.shape != (17,):
        raise ValueError('Invalid joint mask')
    valid = np.isfinite(kp).all(axis=1) & np.isfinite(scores) & mask
    valid &= (kp >= 0).all(axis=1) & (kp < np.asarray(image_size)).all(axis=1)
    targets, diagnostics = {}, {}
    for name, indices in MAPPING.items():
        accepted = bool(valid[indices].all())
        diagnostics[name] = dict(coco_indices=indices, score=float(np.min(scores[indices])),
                                 accepted=accepted, derived=name == 'Neck')
        if accepted:
            targets[name] = kp[indices].mean(axis=0).tolist()
    # A hip midpoint is required: never quietly retain the hand-authored hip anchor.
    if not valid[[11, 12]].all() or not valid[[5, 6]].all():
        raise ValueError('Valid shoulders and hips required for automatic camera/torso anchors')
    if len(targets) < 5:
        raise ValueError('Insufficient valid automatic targets')
    return targets, kp[[11, 12]].mean(axis=0), diagnostics
