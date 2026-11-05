"""Screen hand skin against non-adjacent body regions on an exported FBX.

The wrist's own forearm is adjacent skin and is excluded. Hand-to-hand and
same-hand checks are separate. A clear screen is not a full anatomy certificate.
"""

from collections import Counter

VERSION = "hand-nonadjacent-body-skin-v1"


def screen(meshes):
    import bpy
    from mathutils.bvhtree import BVHTree
    from converter.hand_contact import triangle_crossing

    points, faces, roles = [], [], []
    for obj in meshes:
        if obj.type != "MESH" or obj.name.startswith("StandinScene__"):
            continue
        names = {g.index: g.name for g in obj.vertex_groups}
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            offset = len(points)
            points.extend(evaluated.matrix_world @ v.co for v in mesh.vertices)
            for tri in mesh.loop_triangles:
                weights = Counter()
                for index in tri.vertices:
                    for group in mesh.vertices[index].groups:
                        weights[names.get(group.group, "unbound")] += group.weight
                faces.append(tuple(offset + i for i in tri.vertices))
                roles.append(weights.most_common(1)[0][0] if weights else "unbound")
        finally:
            evaluated.to_mesh_clear()
    counts = Counter()
    for side in ("Left", "Right"):
        hand = [f for f, r in zip(faces, roles) if side + "Hand" in r]
        body = [
            (f, r)
            for f, r in zip(faces, roles)
            if "Hand" not in r and side + "ForeArm" not in r
        ]
        if not hand or not body:
            return dict(status="unavailable", reason="missing_skin_regions")
        a = BVHTree.FromPolygons(points, hand, all_triangles=True)
        b = BVHTree.FromPolygons(points, [f for f, _ in body], all_triangles=True)
        for i, j in a.overlap(b):
            if set(hand[i]) & set(body[j][0]):
                continue
            if triangle_crossing(
                [points[k] for k in hand[i]], [points[k] for k in body[j][0]]
            ):
                counts[side + "Hand / " + body[j][1]] += 1
    return dict(
        method=VERSION,
        status="intersection" if counts else "clear_screen",
        intersection_pairs=sum(counts.values()),
        regions=dict(counts),
        complete_collision_certificate=False,
    )
