"""Bounded prop placement against the evaluated body; never a review approval."""

from mathutils import Vector
from mathutils.bvhtree import BVHTree
from converter.hand_contact import triangle_crossing


class BodyClearance:
    """Cache body triangles once while trying small prop translations."""

    def __init__(self):
        import bpy

        graph = bpy.context.evaluated_depsgraph_get()
        self.vertices, self.triangles = [], []
        for obj in bpy.context.scene.objects:
            if (
                obj.type != "MESH"
                or not obj.vertex_groups
                or obj.name.startswith("StandinScene__")
            ):
                continue
            evaluated = obj.evaluated_get(graph)
            mesh = evaluated.to_mesh()
            try:
                mesh.calc_loop_triangles()
                offset = len(self.vertices)
                self.vertices.extend(
                    evaluated.matrix_world @ v.co for v in mesh.vertices
                )
                self.triangles.extend(
                    tuple(offset + i for i in t.vertices) for t in mesh.loop_triangles
                )
            finally:
                evaluated.to_mesh_clear()
        self.tree = BVHTree.FromPolygons(
            self.vertices, self.triangles, all_triangles=True
        )

    def outside(self, point):
        near, normal, _, gap = self.tree.find_nearest(point)
        if gap <= 1e-4 or (point - near).dot(normal) >= -1e-4:
            return True
        # All three parities must agree on outside; an open seam is not clearance.
        for raw in ((1, 0.317, 0.173), (-0.271, 1, 0.419), (0.311, -0.221, 1)):
            direction = Vector(raw).normalized()
            start, hits = point.copy(), 0
            for _ in range(100):
                hit = self.tree.ray_cast(start, direction, 100)[0]
                if hit is None:
                    break
                hits += 1
                start = hit + direction * 1e-5
            else:
                return False
            if hits % 2:
                return False
        return True

    def clear_mesh(self, obj, shift):
        obj.data.calc_loop_triangles()
        vertices = [obj.matrix_world @ v.co + shift for v in obj.data.vertices]
        faces = [tuple(t.vertices) for t in obj.data.loop_triangles]
        other = BVHTree.FromPolygons(vertices, faces, all_triangles=True)
        for a, b in self.tree.overlap(other):
            if triangle_crossing(
                [self.vertices[i] for i in self.triangles[a]],
                [vertices[i] for i in faces[b]],
            ):
                return False
        return all(self.outside(v) for v in vertices)

    def translate_clear(self, obj, shifts):
        import bpy

        bpy.context.view_layer.update()
        for shift in sorted(shifts, key=lambda v: v.length):
            if self.clear_mesh(obj, shift):
                obj.location += shift
                bpy.context.view_layer.update()
                return list(shift)
        raise ValueError("no bounded clear placement for " + obj.name)


def clear_connector(name, start, end, radius, tree, axes, scale):
    """Route a short handle bracket around skin, with at most one bend."""
    from converter.scene_primitives import rod

    def clear(a, b):
        delta = b - a
        if delta.length < 1e-7:
            return False
        if tree.ray_cast(a, delta.normalized(), delta.length)[0] is not None:
            return False
        return all(
            tree.find_nearest(a + delta * (i / 40))[3] >= radius + 0.0007 * scale
            for i in range(41)
        )

    if clear(start, end):
        return [rod(name, start, end, radius)]
    candidates = []
    for extent in (1, 2, 3):
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(-2, 3):
                    corner = (start + end) / 2 + (
                        axes[0] * i + axes[1] * j + axes[2] * k
                    ) * 0.018 * scale * extent
                    if clear(start, corner) and clear(corner, end):
                        candidates.append(
                            ((corner - start).length + (end - corner).length, corner)
                        )
        if candidates:
            break
    if not candidates:
        raise ValueError("no clear connector path: " + name)
    corner = min(candidates, key=lambda row: row[0])[1]
    return [rod(name, start, corner, radius), rod(name, corner, end, radius)]
