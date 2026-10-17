"""Do not satisfy a manual cue by importing a distant whole-pose candidate."""
import numpy as np


def guarded_shortlist(scores,eligible,limit=20):
    s=np.asarray(scores,float);e=np.asarray(eligible,bool)
    if s.ndim!=1 or e.shape!=s.shape or limit<1 or np.isnan(s).any():
        raise ValueError('matching scores and eligibility with positive limit required')
    finite=np.flatnonzero(np.isfinite(s));order=finite[np.argsort(s[finite],kind='stable')]
    short=order[:limit];matches=short[e[short]]
    first=int(matches[0]) if len(matches) else None
    return {'whole_pose_shortlist':short.tolist(),'cue_matches':matches.tolist(),
            'cue_first':first,'cue_abstention':first is None,
            'unconstrained_first':int(order[0]) if len(order) else None,
            'scope':'cached XY blend Top20 only; no semantic success guarantee'}
