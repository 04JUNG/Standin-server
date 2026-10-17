"""Positive, well-observed pose cues; absence of a crossing is not evidence."""
import numpy as np
from standin_eval.joint_shape import ARM_TRIPLES,segment_relation,SEGMENTS


def focused_costs(metadata,xy,scores):
    xy=np.asarray(xy,float); scores=np.asarray(scores,float)
    n=len(metadata['candidate_crossing']);angle=np.zeros(n);cross=np.zeros(n);used=[]
    for side,triple in enumerate(ARM_TRIPLES):
        target=metadata['query_angles_degrees'][side]
        a,b,c=triple
        if (target is not None and 70<=target<=110 and scores[list(triple)].min()>=.7
            and min(np.linalg.norm(xy[a]-xy[b]),np.linalg.norm(xy[c]-xy[b]))>=.15):used.append(side)
    if used:
        for i,values in enumerate(metadata['candidate_angles_degrees']):
            angle[i]=np.mean([1. if values[j] is None else abs(values[j]-metadata['query_angles_degrees'][j])/180 for j in used])
    # A clearly observed crossing is a positive cue. An uncrossed projection
    # does not establish that legs are uncrossed in3D.
    crossing=metadata['query_crossing'] is True
    if crossing:
        cross=np.array([.5 if value is None else float(value is False) for value in metadata['candidate_crossing']])
    return angle,cross,{'used_right_angle_sides':used,'used_positive_crossing':crossing}
