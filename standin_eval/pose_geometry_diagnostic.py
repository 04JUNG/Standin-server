"""Post-hoc body geometry measurement, never a query/search feature or new GT."""
import numpy as np

BODY_LEFT_RIGHT = np.array([6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])


def body_label_errors(query, target, mask):
    """Compare normalized XY with original versus globally swapped GT labels.

    Oracle diagnostic only. Query coordinates and confidence mask never change.
    """
    q, t = np.asarray(query), np.asarray(target)
    m = np.asarray(mask, dtype=bool)[5:17]
    if q.shape != (17, 2) or t.shape != (17, 2) or not m.any():
        raise ValueError('expected XY bodies and nonempty body mask')
    if not np.isfinite(q).all() or not np.isfinite(t).all():
        raise ValueError('nonfinite XY')
    direct = np.linalg.norm(q[5:17][m] - t[5:17][m], axis=-1).mean()
    swapped = np.linalg.norm(q[5:17][m] - t[BODY_LEFT_RIGHT][m], axis=-1).mean()
    return float(direct), float(swapped)


def normalize_body(joints):
    p = np.asarray(joints, dtype=np.float64)
    if p.shape[-2:] != (17, 3) or not np.isfinite(p).all():
        raise ValueError('expected finite (...,17,3) joints')
    hip = (p[..., 11, :] + p[..., 12, :]) / 2
    shoulder = (p[..., 5, :] + p[..., 6, :]) / 2
    torso = np.linalg.norm(shoulder - hip, axis=-1)
    if np.any(torso < 1e-6):
        raise ValueError('degenerate torso')
    return ((p - hip[..., None, :]) / torso[..., None, None])[..., 5:17, :]


def yaw_aligned_rms(reference, candidates):
    """RMS over 12 labeled body joints after optimal yaw only.

    Inputs are hip-centered, torso-normalized bodies. Does not reflect,
    swap left/right, fit pitch/roll, or independently scale limbs.
    The analytic angle minimizes squared 3D error, not screen-space error.
    """
    q = np.asarray(reference, dtype=np.float64)
    c = np.asarray(candidates, dtype=np.float64)
    if q.shape != (12, 3) or c.shape[-2:] != (12, 3):
        raise ValueError('expected one (12,3) reference and (...,12,3) candidates')
    if not np.isfinite(q).all() or not np.isfinite(c).all():
        raise ValueError('nonfinite body')
    a = (c[..., 0] * q[:, 0] + c[..., 2] * q[:, 2]).sum(axis=-1)
    b = (c[..., 2] * q[:, 0] - c[..., 0] * q[:, 2]).sum(axis=-1)
    angle = np.arctan2(b, a)
    co, si = np.cos(angle)[..., None], np.sin(angle)[..., None]
    aligned = c.copy()
    aligned[..., 0] = co * c[..., 0] + si * c[..., 2]
    aligned[..., 2] = -si * c[..., 0] + co * c[..., 2]
    return np.sqrt(np.mean(np.sum((aligned - q) ** 2, axis=-1), axis=-1))
