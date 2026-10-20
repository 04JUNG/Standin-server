"""Two anatomical-label hypotheses from the same measured XY, no source GT."""
import numpy as np

LEFT_RIGHT = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])


def alternate_labels(xy, mask):
    """Permute observations AND their validity together; never mirror pixels."""
    xy = np.asarray(xy)
    mask = np.asarray(mask, dtype=bool)
    if xy.shape != (17, 2) or mask.shape != (17,):
        raise ValueError('expected (17,2) XY and (17,) mask')
    return xy[LEFT_RIGHT].copy(), mask[LEFT_RIGHT].copy()


def merged_order(direct, alternate, *, preserve_first=False):
    """Unique pose-index order by min cost; optionally keep original top result.

    No pose-family merging or mirror GT credit. Exact ties use gallery order.
    """
    direct = np.asarray(direct, dtype=float)
    alternate = np.asarray(alternate, dtype=float)
    if direct.ndim != 1 or direct.shape != alternate.shape:
        raise ValueError('expected same-length cost vectors')
    direct = np.where(np.isfinite(direct) & (direct >= 0), direct, np.inf)
    alternate = np.where(np.isfinite(alternate) & (alternate >= 0), alternate, np.inf)
    pooled = np.minimum(direct, alternate)
    order = np.argsort(pooled, kind='stable')
    order = order[np.isfinite(pooled[order])]
    if preserve_first and np.isfinite(direct).any():
        first = int(np.argmin(direct))
        order = np.concatenate(([first], order[order != first]))
    return order
