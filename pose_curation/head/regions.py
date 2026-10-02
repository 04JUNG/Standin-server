"""Image regions proposed by existing observed body landmarks, not facial fits."""
import numpy as np


def body_head_regions(people, size):
    """Crop proposals only: a body head/face point never becomes a face landmark."""
    width, height = size
    regions = []
    for person in people:
        p, s = np.asarray(person['keypoints'], float)[:5], np.asarray(person['scores'], float)[:5]
        valid = (s >= .55) & np.isfinite(p).all(1) & (p >= 0).all(1) & (p < [width,height]).all(1)
        if valid.sum() < 3 or not valid[0]:
            continue
        selected = p[valid]
        extent = np.ptp(selected,axis=0)
        if max(extent) < 12:
            continue
        center = (selected.min(0) + selected.max(0)) / 2
        side = max(80., float(max(extent)*2.8))
        # Eye/nose/ear points sit above the face center; include chin and forehead.
        center[1] += .08 * side
        x0, y0 = np.maximum(np.floor(center-side/2),0).astype(int)
        x1, y1 = np.minimum(np.ceil(center+side/2),[width,height]).astype(int)
        if min(x1-x0,y1-y0) < 48:
            continue
        regions.append({'bbox':[int(x0),int(y0),int(x1),int(y1)], 'proposal_person':person['person']})
    return regions
