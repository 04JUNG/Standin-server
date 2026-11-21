"""Baseline-relative safeguards for experimental positional corrections.

No connectivity, bone or weight edits. An accepted step certifies only the
listed endpoint constraints and non-collapse along its linear trajectory.
It does not certify absence of self intersections.
"""
import numpy as np
from converter.local import surface


class DeformationGuard:
    def __init__(self, rest, posed, faces, fixed, seam_pairs=None):
        self.rest = np.asarray(rest, dtype=float).copy()
        self.baseline = np.asarray(posed, dtype=float).copy()
        self.faces = np.asarray(faces, dtype=int).copy()
        self.fixed = np.asarray(fixed, dtype=bool).copy()
        self.seams = np.asarray(seam_pairs if seam_pairs is not None else [],
                                dtype=int).reshape(-1, 2)
        if self.rest.shape != self.baseline.shape or not np.isfinite(self.baseline).all():
            raise ValueError("Invalid position arrays")
        self.scale = float(np.ptp(self.rest, axis=0).max())
        if self.scale <= 0:
            raise ValueError("Zero mesh extent")
        self.graph = surface.Neighborhood(self.faces, self.rest)
        self.lengths = surface.lengths(self.rest[self.graph.a]-self.rest[self.graph.b])
        self.area, _ = surface.triangle_geometry(self.rest, self.faces)
        if np.any(self.lengths <= self.scale*1e-12) or np.any(self.area <= self.scale**2*1e-14):
            raise ValueError("Degenerate source geometry requires separate inspection")
        # A rest tangent basis measures strain independently of triangle orientation.
        triangle = self.rest[self.faces]
        first = triangle[:, 1]-triangle[:, 0]
        second = triangle[:, 2]-triangle[:, 0]
        axis = first / surface.lengths(first)[:, None]
        along = (axis*second).sum(1)
        across = surface.lengths(second-axis*along[:, None])
        self.inverse = np.zeros((len(self.faces), 2, 2))
        self.inverse[:, 0, 0] = 1/surface.lengths(first)
        self.inverse[:, 1, 1] = 1/across
        self.inverse[:, 0, 1] = -along*self.inverse[:, 0, 0]/across
        self.reference = self.measure(self.baseline)
        self.rest_bend = self.measure(self.rest)["bend"]
        self.upper = np.maximum(2., self.reference["stretch"])
        self.lower = np.minimum(.2, self.reference["compression"])
        self.edge_upper = np.maximum(2., self.reference["edge"])
        self.edge_lower = np.minimum(.2, self.reference["edge"])
        self.bend_limit = np.maximum.reduce([
            np.full_like(self.rest_bend, np.pi/3),
            np.abs(self.rest_bend), np.abs(self.reference["bend"])])
        # Keep healthy hinges below the reporting threshold despite solver tolerance.
        healthy = (np.abs(self.rest_bend) < np.pi/6) & (
            np.abs(self.reference["bend"]) < np.pi/3)
        self.bend_limit[healthy] = np.maximum(
            np.abs(self.reference["bend"][healthy]), np.pi/3-1e-5)
        # Authored creases preserve their signed angle relative to rest.
        self.crease = np.abs(self.rest_bend) > np.pi/6
        self.crease_limit = np.maximum(np.pi/18, np.abs(
            self.angle_difference(self.reference["bend"], self.rest_bend)))
        self.area_floor = np.minimum(.01, self.reference["area"])

    @staticmethod
    def angle_difference(a, b):
        return np.arctan2(np.sin(a-b), np.cos(a-b))

    def measure(self, points):
        triangle = points[self.faces]
        edges = np.stack((triangle[:, 1]-triangle[:, 0],
                          triangle[:, 2]-triangle[:, 0]), axis=2)
        singular = np.linalg.svd(edges @ self.inverse, compute_uv=False)
        area, normals = surface.triangle_geometry(points, self.faces)
        left, right = self.graph.face_pairs.T
        # A stable shared-edge orientation defines a signed hinge angle.
        lf, rf = self.faces[left], self.faces[right]
        shared = (lf[:, :, None] == rf[:, None, :]).any(2)
        endpoints = lf[shared].reshape(-1, 2)
        direction = points[endpoints[:, 1]]-points[endpoints[:, 0]]
        direction /= np.maximum(surface.lengths(direction)[:, None], 1e-30)
        bend = np.arctan2((np.cross(normals[left], normals[right])*direction).sum(1),
                          (normals[left]*normals[right]).sum(1))
        return dict(stretch=singular[:, 0], compression=singular[:, 1],
                    area=area/self.area, bend=bend,
                    edge=surface.lengths(points[self.graph.a]-points[self.graph.b])/self.lengths,
                    seam=surface.lengths(points[self.seams[:, 0]]-points[self.seams[:, 1]]))

    def violations(self, points):
        if not np.isfinite(points).all():
            return {"nonfinite": 1}
        m = self.measure(points)
        tolerance = 1e-7
        checks = dict(
            stretch=m["stretch"] > self.upper+tolerance,
            compression=m["compression"] < self.lower-tolerance,
            edge_stretch=m["edge"] > self.edge_upper+tolerance,
            edge_compression=m["edge"] < self.edge_lower-tolerance,
            collapse=m["area"] < self.area_floor-tolerance,
            bend=np.abs(m["bend"]) > self.bend_limit+tolerance,
            crease=self.crease & (np.abs(self.angle_difference(m["bend"], self.rest_bend))
                                   > self.crease_limit+tolerance),
            seam=m["seam"] > self.reference["seam"]+self.scale*1e-8,
            fixed=np.any(points[self.fixed] != self.baseline[self.fixed], axis=1))
        return {k: int(v.sum()) for k, v in checks.items() if v.any()}

    def trajectory_safe(self, start, end):
        # Cross product along a linear vertex trajectory is quadratic.
        # Its projection onto the starting face normal must stay positive.
        a = start[self.faces]
        delta = end[self.faces]-a
        u, v = a[:, 1]-a[:, 0], a[:, 2]-a[:, 0]
        du, dv = delta[:, 1]-delta[:, 0], delta[:, 2]-delta[:, 0]
        normal = np.cross(u, v)
        size = surface.lengths(normal)
        if np.any(size <= self.scale**2*1e-14):
            return False
        normal /= size[:, None]
        linear = (normal*(np.cross(du, v)+np.cross(u, dv))).sum(1)
        quadratic = (normal*np.cross(du, dv)).sum(1)
        time = np.clip(-linear/np.where(abs(quadratic)>1e-30, 2*quadratic, 1.), 0, 1)
        minimum = np.minimum(size, size+linear+quadratic)
        convex = quadratic > 0
        minimum[convex] = np.minimum(minimum[convex],
            (size+linear*time+quadratic*time*time)[convex])
        return bool(np.all(minimum > size*1e-6))

    def accept(self, current, proposal, attempts=18):
        rejected = {}
        for iteration in range(attempts):
            amount = 2.**-iteration
            candidate = current+(proposal-current)*amount
            candidate[self.fixed] = self.baseline[self.fixed]
            rejected = self.violations(candidate)
            if not rejected and self.trajectory_safe(current, candidate):
                return candidate, dict(accepted=True, fraction=amount)
        return current.copy(), dict(accepted=False, reason="constraints_blocked",
                                    violations=rejected)
