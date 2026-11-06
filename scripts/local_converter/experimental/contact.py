"""Conservative swept bounds for non-neighbor triangle pairs of one mesh."""
import numpy as np
from scipy.spatial import cKDTree


def _axes(first, second):
    a = np.roll(first,-1,axis=0)-first
    b = np.roll(second,-1,axis=0)-second
    na, nb = np.cross(a[0],a[1]), np.cross(b[0],b[1])
    return np.vstack([na, nb, np.cross(a[:,None],b).reshape(-1,3),
                      np.cross(na,a), np.cross(nb,b)])


def static_relation(first, second, extent):
    axes = _axes(first, second)
    if np.any(np.linalg.norm(axes[:2], axis=1) < extent**2*1e-14):
        return "degenerate"
    left, right = first @ axes.T, second @ axes.T
    margin = 1e-12*np.linalg.norm(axes,axis=1)*extent
    if ((left.max(0)+margin < right.min(0)) |
        (right.max(0)+margin < left.min(0))).any():
        return "separated"
    return "overlap_or_touch_within_tolerance"


def interval_separated(a0, a1, b0, b1, extent, depth=0):
    left_vertices = np.vstack([a0,a1])
    right_vertices = np.vstack([b0,b1])
    for first, second in ((a0,b0),(a1,b1)):
        axes = _axes(first, second)
        left, right = left_vertices @ axes.T, right_vertices @ axes.T
        margin = 1e-12*np.linalg.norm(axes,axis=1)*extent
        if ((left.max(0)+margin < right.min(0)) |
            (right.max(0)+margin < left.min(0))).any():
            return True
    if depth >= 5:
        return False
    am, bm = (a0+a1)/2, (b0+b1)/2
    if static_relation(am,bm,extent) != "separated":
        return False
    return (interval_separated(a0,am,b0,bm,extent,depth+1) and
            interval_separated(am,a1,bm,b1,extent,depth+1))


def potential_pairs(faces, moving, lower, upper):
    centers = (lower+upper)/2
    radius = ((upper-lower)/2).max(axis=1)
    tree = cKDTree(centers)
    max_radius = radius.max()
    for i in np.flatnonzero(moving):
        near = np.asarray(tree.query_ball_point(centers[i], radius[i]+max_radius,
                                                p=np.inf), dtype=int)
        near = near[(near != i) & (~moving[near] | (near > i))]
        near = near[((upper[near] >= lower[i]) & (lower[near] <= upper[i])).all(1)]
        for j in near:
            if np.isin(faces[i], faces[j]).any():
                continue
            yield i, j


def baseline_contact_pins(points, faces, free, pair_budget=250000):
    """Keep the six input coordinates of each possible existing contact fixed."""
    triangles = points[faces]
    extent = np.ptp(points,axis=0).max()
    pinned = np.zeros(len(points),bool)
    count = 0
    for checked,(i,j) in enumerate(potential_pairs(
            faces,free[faces].any(1),triangles.min(1),triangles.max(1)), start=1):
        if checked > pair_budget:
            return free.copy(), dict(complete=False, checked=checked)
        if static_relation(triangles[i],triangles[j],extent) != "separated":
            pinned[faces[[i,j]].ravel()] = True
            count += 1
    return pinned, dict(complete=True, pairs=count, freePinned=int((free&pinned).sum()))


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
    extent = np.ptp(start,axis=0).max()
    checked = 0
    for i,j in potential_pairs(faces,moving,lower,upper):
        checked += 1
        if checked > pair_budget:
            return dict(passCheck=False, reason="contact_budget", pairs=checked)
        if not interval_separated(old[i],new[i],old[j],new[j],extent):
            return dict(passCheck=False, reason="contact_unresolved", pairs=checked,
                        faces=[int(i),int(j)],
                        baseline=static_relation(old[i],old[j],extent),
                        proposed=static_relation(new[i],new[j],extent))
    return dict(passCheck=True, pairs=checked)
