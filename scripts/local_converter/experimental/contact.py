"""Conservative swept bounds for non-neighbor triangle pairs of one mesh."""
import numpy as np
from scipy.spatial import cKDTree


def contact_guard(start, end, faces, pair_budget=250000):
    """Reject when separation throughout a straight displacement is unproven.

    A fixed separating axis at both endpoints certifies separation throughout
    linear motion. Uncertain pairs (including pre-existing contact) fail closed.
    Topological neighbors are excluded, so this is a partial intersection test.
    """
    moving = np.any(start != end, axis=1)[faces].any(axis=1)
    old, new = start[faces], end[faces]
    lower = np.minimum(old.min(1), new.min(1))
    upper = np.maximum(old.max(1), new.max(1))
    centers = (lower+upper)/2
    radius = ((upper-lower)/2).max(axis=1)
    tree = cKDTree(centers)
    extent = np.ptp(start,axis=0).max()
    max_radius = radius.max()
    checked = 0
    for i in np.flatnonzero(moving):
        near = np.asarray(tree.query_ball_point(centers[i], radius[i]+max_radius,
                                                p=np.inf), dtype=int)
        near = near[(near != i) & (~moving[near] | (near > i))]
        near = near[((upper[near] >= lower[i]) & (lower[near] <= upper[i])).all(1)]
        for j in near:
            if np.isin(faces[i], faces[j]).any():
                continue
            checked += 1
            if checked > pair_budget:
                return dict(passCheck=False, reason="contact_budget", pairs=checked)
            separated = False
            for first, second in ((old[i],old[j]),(new[i],new[j])):
                a = np.roll(first,-1,axis=0)-first
                b = np.roll(second,-1,axis=0)-second
                na, nb = np.cross(a[0],a[1]), np.cross(b[0],b[1])
                axes = np.vstack([na, nb, np.cross(a[:,None],b).reshape(-1,3),
                                  np.cross(na,a), np.cross(nb,b)])
                left = np.vstack([old[i],new[i]]) @ axes.T
                right = np.vstack([old[j],new[j]]) @ axes.T
                # Insist on a positive separation with a roundoff margin.
                tol = 1e-12*np.linalg.norm(axes,axis=1)*extent
                if ((left.max(0)+tol < right.min(0)) |
                    (right.max(0)+tol < left.min(0))).any():
                    separated = True
                    break
            if not separated:
                return dict(passCheck=False, reason="contact_unresolved", pairs=checked,
                            faces=[int(i),int(j)])
    return dict(passCheck=True, pairs=checked)
