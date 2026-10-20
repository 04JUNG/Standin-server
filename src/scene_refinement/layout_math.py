"""Image-space anchor fitting and silhouette separation. No case-specific rules."""

import numpy as np, cv2
from scipy import ndimage, signal, optimize


def project(points, translation, camera):
    p = np.asarray(points) + np.asarray(translation)
    depth = p[:, 1] - camera["position_blender"][1]
    f = camera["focal_pixels"]
    cx, cy = camera["principal_pixels"]
    if depth.min() <= 0.05:
        raise ValueError("body behind camera")
    return np.column_stack(
        (
            cx + f * p[:, 0] / depth,
            cy - f * (p[:, 2] - camera["position_blender"][2]) / depth,
        )
    )


def box(points):
    return np.r_[np.min(points, axis=0), np.max(points, axis=0)]


def fit_head(
    points,
    target,
    camera,
    depth=None,
    initial=None,
    scale_points=None,
    scale_target=None,
):
    target = np.asarray(target, float)
    center = (target[:2] + target[2:]) / 2
    size = target[2:] - target[:2]
    local = np.mean(points, axis=0)
    f = camera["focal_pixels"]
    cx, cy = camera["principal_pixels"]
    camy = camera["position_blender"][1]
    camz = camera["position_blender"][2]
    if depth is None:
        geom = box(points[:, [0, 2]])
        dim = geom[2:] - geom[:2]
        d = f * np.sqrt(np.prod(dim) / np.prod(size))
        if scale_points is not None:
            d = f * np.ptp(scale_points[:, 2]) / (scale_target[3] - scale_target[1])
        d = np.clip(d, 0.65, 80)
    else:
        d = depth
    initial = (
        np.array(
            [
                (center[0] - cx) * d / f - local[0],
                camy + d - local[1],
                camz + (cy - center[1]) * d / f - local[2],
            ]
        )
        if initial is None
        else np.asarray(initial, float)
    )

    def residual(t):
        try:
            b = box(project(points, t, camera))
        except ValueError:
            return (
                np.array([100.0, 100.0, 100.0])
                if depth is None
                else np.array([100.0, 100.0])
            )
        out = list(
            ((b[:2] + b[2:]) / 2 - center) / np.sqrt(camera["width"] * camera["height"])
        )
        if depth is None:
            if scale_points is None:
                out.append(
                    0.2 * np.log(np.sqrt(np.prod(b[2:] - b[:2]) / np.prod(size)))
                )
            else:
                body = box(project(scale_points, t, camera))
                out.append(
                    0.2
                    * np.log((body[3] - body[1]) / (scale_target[3] - scale_target[1]))
                )
        return np.array(out)

    if depth is None:
        lower = np.array([-200, camy + 0.4 - points[:, 1].min(), -200])
        upper = np.array([200, camy + 100, 200])
        initial = np.maximum(lower + 0.001, np.minimum(upper - 0.001, initial))
        res = optimize.least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
            max_nfev=100,
        )
        t = res.x
    else:
        fixed_y = camy + depth - local[1]

        def r2(xz):
            return residual([xz[0], fixed_y, xz[1]])

        res = optimize.least_squares(
            r2, initial[[0, 2]], xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=50
        )
        t = np.array([res.x[0], fixed_y, res.x[1]])
    return t, box(project(points, t, camera))


def silhouette(points, triangles, translation, camera, scale):
    xy = project(points, translation, camera) * scale
    w, h = round(camera["width"] * scale), round(camera["height"] * scale)
    mask = np.zeros((h, w), np.uint8)
    tri = xy[triangles]
    valid = (
        (tri.max(1)[:, 0] >= 0)
        & (tri.min(1)[:, 0] < w)
        & (tri.max(1)[:, 1] >= 0)
        & (tri.min(1)[:, 1] < h)
    )
    for pts in tri[valid]:
        cv2.fillConvexPoly(
            mask, np.rint(np.clip(pts, -32760, 32760)).astype(np.int32), 1
        )
    return mask


def overlaps(masks):
    return [
        dict(a=i, b=j, pixels=int(np.logical_and(a, b).sum()))
        for i, a in enumerate(masks)
        for j, b in enumerate(masks)
        if j > i
    ]


class PairField:
    def __init__(self, a, b, i, j, clearance):
        self.i = i
        self.j = j

        def crop(m):
            ys, xs = np.where(m)
            assert len(xs)
            return m[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1], np.array(
                [xs.min(), ys.min()]
            )

        kernel = np.ones((2 * clearance + 1, 2 * clearance + 1), np.uint8)
        a = cv2.dilate(a, kernel)
        b = cv2.dilate(b, kernel)
        aa, ao = crop(a)
        bb, bo = crop(b)
        blocked = (
            signal.fftconvolve(
                aa.astype(float), bb[::-1, ::-1].astype(float), mode="full"
            )
            > 0.5
        )
        pad = 3
        blocked = np.pad(blocked, pad)
        self.field = ndimage.distance_transform_edt(
            ~blocked
        ) - ndimage.distance_transform_edt(blocked)
        self.near = ndimage.distance_transform_edt(
            blocked, return_distances=False, return_indices=True
        )
        self.origin = bo - ao + np.array([bb.shape[1] - 1, bb.shape[0] - 1]) + pad
        self.gy, self.gx = np.gradient(self.field)

    def coord(self, d):
        return self.origin + d[self.j] - d[self.i]

    def value(self, d):
        xy = self.coord(d)
        h, w = self.field.shape
        x, y = xy
        if x < 0 or y < 0 or x > w - 1 or y > h - 1:
            return float(max(-x, -y, x - w + 1, y - h + 1) + 1.0)
        return float(
            ndimage.map_coordinates(self.field, [[y], [x]], order=1, mode="nearest")[0]
        )

    def jac(self, d, n):
        x, y = self.coord(d)
        h, w = self.field.shape
        out = np.zeros((n, 2))
        if not (0 <= x <= w - 1 and 0 <= y <= h - 1):
            return out.ravel()
        x0 = min(int(x), w - 2)
        y0 = min(int(y), h - 2)
        tx = x - x0
        ty = y - y0
        f = self.field
        dx = (1 - ty) * (f[y0, x0 + 1] - f[y0, x0]) + ty * (
            f[y0 + 1, x0 + 1] - f[y0 + 1, x0]
        )
        dy = (1 - tx) * (f[y0 + 1, x0] - f[y0, x0]) + tx * (
            f[y0 + 1, x0 + 1] - f[y0, x0 + 1]
        )
        gradient = np.array([dx, dy])
        out[self.i] = -gradient
        out[self.j] = gradient
        return out.ravel()

    def escape(self, d):
        xy = np.rint(self.coord(d)).astype(int)
        h, w = self.field.shape
        x, y = xy
        if not (0 <= x < w and 0 <= y < h) or self.field[y, x] > 0:
            return np.zeros(2)
        ny, nx = self.near[:, y, x]
        v = np.array([nx - x, ny - y], float)
        norm = np.linalg.norm(v)
        return v * (1 + 2 / max(norm, 1))


def separate_masks(masks, anchors, accumulated=None, clearance=1):
    n = len(masks)
    h, w = masks[0].shape
    anchors = np.asarray(anchors)
    base = np.zeros((n, 2)) if accumulated is None else np.asarray(accumulated)
    pairs = [
        PairField(masks[i], masks[j], i, j, clearance)
        for i in range(n)
        for j in range(i + 1, n)
    ]
    if all(p.value(np.zeros((n, 2))) >= 0.2 for p in pairs):
        return np.zeros((n, 2)), dict(success=True, objective=0.0, starts=0)
    limit = 0.4 * min(w, h)
    bounds = []
    for i, (x, y) in enumerate(anchors):
        bounds += [
            (max(-limit, -x), min(limit, w - x)),
            (max(-limit, -y), min(limit, h - y)),
        ]

    def fun(x):
        total = x.reshape(n, 2) + base
        return float(np.sum(total * total))

    def jac(x):
        return 2 * (x + base.ravel())

    cons = [
        dict(
            type="ineq",
            fun=lambda x, p=p: p.value(x.reshape(n, 2)) - 0.15,
            jac=lambda x, p=p: p.jac(x.reshape(n, 2), n),
        )
        for p in pairs
    ]
    # Keep the original horizontal actor order; depth and physical scale are fixed.
    order = np.argsort(anchors[:, 0])
    for ia, ib in zip(order, order[1:]):

        def fc(x, a=ia, b=ib):
            return (
                anchors[b, 0]
                + x.reshape(n, 2)[b, 0]
                - anchors[a, 0]
                - x.reshape(n, 2)[a, 0]
            )

        def jc(x, a=ia, b=ib):
            v = np.zeros((n, 2))
            v[a, 0] = -1
            v[b, 0] = 1
            return v.ravel()

        cons.append(dict(type="ineq", fun=fc, jac=jc))
    seeds = [np.zeros((n, 2))]
    for direction in (1.0, -1.0):
        d = np.zeros((n, 2))
        for _ in range(8):
            for p in pairs:
                v = p.escape(d)
                if np.linalg.norm(v):
                    d[p.i] -= v * (0.5 if direction == 1 else 0.3)
                    d[p.j] += v * (0.5 if direction == 1 else 0.7)
        seeds.append(d)
    # A small deterministic seed lets SLSQP leave a symmetric collision plateau.
    rng = np.random.default_rng(289)
    seeds.extend([rng.normal(0, 3, (n, 2)) for _ in range(2)])
    candidates = []
    for seed in seeds:
        x0 = np.array([np.clip(v, lo, hi) for v, (lo, hi) in zip(seed.ravel(), bounds)])
        res = optimize.minimize(
            fun,
            x0,
            jac=jac,
            bounds=bounds,
            constraints=cons,
            method="SLSQP",
            options=dict(maxiter=180, ftol=1e-6),
        )
        feasible = min(c["fun"](res.x) for c in cons) >= -0.2
        if feasible:
            candidates.append(res)
    if not candidates:
        return np.zeros((n, 2)), dict(
            success=False, reason="no_feasible_small_translation", starts=len(seeds)
        )
    best = min(candidates, key=lambda r: r.fun)
    return best.x.reshape(n, 2), dict(
        success=True,
        objective=float(best.fun),
        starts=len(seeds),
        iterations=int(best.nit),
    )
