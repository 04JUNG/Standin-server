"""One search's immutable rest geometry; all posed safety checks still run.

A context belongs to one mesh/rig/rest-position array and is never shared across
jobs, imports, shape changes, or final baking. Arithmetic and traversal order
match the frozen ankle_swing._surface_metrics implementation.
"""
import math


class RestSurface:
    def __init__(self, v, mesh, arm, rt, table, side, rest, policy, weights):
        self.v, self.rest = v, rest
        self.failed = None
        measure = policy['measurement']
        inv = arm.matrix_world.inverted()
        knee = inv @ rt._rest_world(arm, table[f'leg.{side}']).translation
        ankle = inv @ rt._rest_world(arm, table[f'foot.{side}']).translation
        toe = inv @ rt._rest_world(arm, table[f'toe.{side}']).translation
        self.leg_length = (ankle - knee).length
        foot_length = (toe - ankle).length
        if self.leg_length <= v._EPS or foot_length <= v._EPS:
            self.failed = v.SurfaceMetrics(False, 'target_rest_edges_degenerate')
            return
        shin = (knee - ankle).normalized()
        forward = (toe - ankle).normalized()
        radius = max(measure['roi_radius_leg_ratio'] * self.leg_length,
                     measure['roi_radius_foot_ratio'] * foot_length)
        roi, foot, calf = set(), [], []
        for index, (point, row) in enumerate(zip(rest, weights)):
            wl, wf, wt = (row.get(table[f'{name}.{side}'], 0.0) for name in ('leg', 'foot', 'toe'))
            if wl + wf + wt >= measure['minimum_limb_weight'] and (point - ankle).length <= radius:
                roi.add(index)
            fp = (point - ankle).dot(forward) / foot_length
            if wf + wt >= measure['foot_weight_min'] and fp >= measure['foot_distal_min_ratio']:
                foot.append(index)
            cp = (point - ankle).dot(shin) / self.leg_length
            if wl >= measure['calf_weight_min'] and measure['calf_proximal_min_ratio'] <= cp <= measure['calf_proximal_max_ratio']:
                calf.append(index)
        mesh.data.calc_loop_triangles()
        triangles = [tuple(tri.vertices) for tri in mesh.data.loop_triangles]
        self.triangles = [tri for tri in triangles if sum(i in roi for i in tri) >= 2]
        self.edges = [(e.vertices[0], e.vertices[1]) for e in mesh.data.edges
                      if e.vertices[0] in roi and e.vertices[1] in roi]
        self.counts = (len(roi), len(self.triangles), len(self.edges), len(foot), len(calf))
        self.foot, self.calf = foot, calf
        minimum = int(measure['minimum_clearance_vertices'])
        if (len(roi) < int(measure['minimum_roi_vertices'])
                or len(self.triangles) < int(measure['minimum_roi_triangles'])
                or len(self.edges) < int(measure['minimum_roi_triangles'])
                or len(foot) < minimum or len(calf) < minimum):
            self.failed = v.SurfaceMetrics(False, 'ankle_mesh_roi_unmeasurable', *self.counts)
            return
        self.areas = [v._triangle_area(rest, tri) for tri in self.triangles]
        self.lengths = [(rest[a] - rest[b]).length for a, b in self.edges]
        edge_to_triangles = {}
        for i, tri in enumerate(self.triangles):
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                edge_to_triangles.setdefault(tuple(sorted((a, b))), []).append(i)
        normals = [v._triangle_normal(rest, tri) for tri in self.triangles]
        self.adjacency = []
        for adjacent in edge_to_triangles.values():
            if len(adjacent) != 2:
                continue
            a, b = adjacent
            if None in (normals[a], normals[b]):
                continue
            angle = math.degrees(math.acos(v._clamp(normals[a].dot(normals[b]), -1.0, 1.0)))
            self.adjacency.append((a, b, angle))

    def measure(self, positions):
        v = self.v
        if self.failed is not None:
            return v.SurfaceMetrics(**vars(self.failed))
        areas, edges, strain = [], [], []
        for tri, rest_area in zip(self.triangles, self.areas):
            posed_area = v._triangle_area(positions, tri)
            if rest_area > v._EPS:
                areas.append(posed_area / rest_area)
        for (a, b), rest_length in zip(self.edges, self.lengths):
            length = (positions[a] - positions[b]).length
            if rest_length > v._EPS and length > v._EPS:
                ratio = length / rest_length
                edges.append(ratio)
                strain.append(abs(math.log(ratio)))
        if not areas or not edges:
            return v.SurfaceMetrics(False, 'ankle_surface_distributions_empty')
        normals = [v._triangle_normal(positions, tri) for tri in self.triangles]
        changes, sharp = [], 0
        for a, b, rest_angle in self.adjacency:
            if None in (normals[a], normals[b]):
                continue
            angle = math.degrees(math.acos(v._clamp(normals[a].dot(normals[b]), -1.0, 1.0)))
            changes.append(abs(angle - rest_angle))
            if rest_angle < 100.0 and angle > 130.0:
                sharp += 1
        clearance_min, clearance_p01 = v._clearance(positions, self.foot, self.calf, self.leg_length)
        return v.SurfaceMetrics(True, 'ok', *self.counts,
            min(areas), v._quantile(areas, 0.01), min(edges), max(edges), v._quantile(strain, 0.99),
            clearance_min, clearance_p01, v._quantile(changes, 0.99) if changes else 0.0, sharp)
