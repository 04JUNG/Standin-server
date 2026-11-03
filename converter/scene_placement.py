"""Fit primitive scene guides to the evaluated hand and pelvis surfaces.

No skeleton edits. Distance samples help place guides; final mesh crossings and
visual review remain separate requirements. A fit is not a contact certificate."""

import numpy as np
from mathutils import Vector


def body_surface(arm, side=None):
    import bpy
    from mathutils.bvhtree import BVHTree

    graph = bpy.context.evaluated_depsgraph_get()
    vertices, faces, selected = [], [], []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.name.startswith(
            ("StandinScene__", "Guide", "Blade", "Guard")
        ):
            continue
        if not any(m.type == "ARMATURE" and m.object == arm for m in obj.modifiers):
            continue
        names = {g.index: g.name for g in obj.vertex_groups}
        evaluated = obj.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            offset = len(vertices)
            points = [evaluated.matrix_world @ v.co for v in mesh.vertices]
            vertices.extend(points)
            faces.extend(
                tuple(offset + i for i in t.vertices) for t in mesh.loop_triangles
            )
            if side:
                selected.extend(
                    p
                    for v, p in zip(mesh.vertices, points)
                    if sum(
                        g.weight
                        for g in v.groups
                        if side + "Hand" in names.get(g.group, "")
                    )
                    > 0.5
                )
        finally:
            evaluated.to_mesh_clear()
    return BVHTree.FromPolygons(vertices, faces, all_triangles=True), selected


def fit_clearance(center, axis, along, normal, width, radius, tree, vertices):
    lo, hi = (
        min((v - center).dot(axis) for v in vertices),
        max((v - center).dot(axis) for v in vertices),
    )
    offsets = [lo + (hi - lo) * i / 32 for i in range(33)]

    def clearance(candidate):
        values = []
        for offset in offsets:
            p = candidate + axis * offset
            nearest, outward, _, distance = tree.find_nearest(p)
            values.append(distance if (p - nearest).dot(outward) >= 0 else -distance)
        return min(values)

    initial = clearance(center)
    best = center.copy()
    best_distance = initial
    # Stay in the same curled grip cavity; do not solve by floating away.
    desired = min(radius, width * 0.16)
    best_score = min(initial, desired)
    for a in range(-4, 5):
        for b in range(-4, 5):
            offset = (along * a + normal * b) * width * 0.065
            candidate = center + offset
            distance = clearance(candidate)
            score = min(distance, desired) - offset.length * 0.2
            if score > best_score:
                best, best_distance, best_score = candidate, distance, score
    fitted_radius = max(width * 0.045, min(desired, best_distance - width * 0.015))
    return best, fitted_radius, initial, best_distance


def fit_two_hand_line(fits, preferred, arm):
    """Search a nearby straight grip that clears both existing hand meshes."""
    center = (fits["L"]["center"] + fits["R"]["center"]) / 2
    axis = (fits["L"]["center"] - fits["R"]["center"]).normalized()
    if axis.dot(preferred) < 0:
        axis = -axis
    lateral = fits["R"]["along"]
    lateral = (lateral - axis * lateral.dot(axis)).normalized()
    normal = axis.cross(lateral).normalized()
    tree, _ = body_surface(arm)
    points = fits["L"]["surface"] + fits["R"]["surface"]
    width = min(fits[s]["width"] for s in ("L", "R"))
    desired = min(fits[s]["radius"] for s in ("L", "R"))
    best = None
    for i in range(-3, 4):
        for j in range(-3, 4):
            direction = (axis + lateral * i * 0.12 + normal * j * 0.12).normalized()
            transverse = (lateral - direction * lateral.dot(direction)).normalized()
            candidate, radius, _, distance = fit_clearance(
                center,
                direction,
                transverse,
                direction.cross(transverse).normalized(),
                width,
                desired,
                tree,
                points,
            )
            score = (
                min(distance, desired)
                - (candidate - center).length * 0.10
                - (i * i + j * j) ** 0.5 * width * 0.005
            )
            if best is None or score > best[0]:
                best = (score, candidate, direction, radius)
    return best[1:]


def hand_fit(arm, side):
    def point(suffix):
        return arm.matrix_world @ arm.pose.bones["mixamorig:" + side + suffix].head

    wrist = point("Hand")
    knuckles = [
        point("Hand" + finger + "1") for finger in ("Index", "Middle", "Ring", "Pinky")
    ]
    across = (knuckles[0] - knuckles[-1]).normalized()
    along = (knuckles[1] - wrist).normalized()
    along = (along - across * along.dot(across)).normalized()
    normal = across.cross(along).normalized()
    width = (knuckles[0] - knuckles[-1]).length
    palm_length = (knuckles[1] - wrist).length
    curves = []
    points = []
    for finger in ("Index", "Middle", "Ring", "Pinky"):
        chain = [point("Hand" + finger + str(i)) for i in (1, 2, 3)]
        chain.append(
            arm.matrix_world
            @ arm.pose.bones["mixamorig:" + side + "Hand" + finger + "3"].tail
        )
        points.extend(chain)
        first, last = (
            (chain[1] - chain[0]).normalized(),
            (chain[3] - chain[2]).normalized(),
        )
        if first.dot(last) > 0.7:
            continue
        coords = np.array(
            [((p - wrist).dot(along), (p - wrist).dot(normal)) for p in chain]
        )
        # Fit a circle to the bent finger's centerline in its flexion plane.
        matrix = np.column_stack((2 * coords[:, 0], 2 * coords[:, 1], np.ones(4)))
        cx, cy, constant = np.linalg.lstsq(
            matrix, (coords * coords).sum(axis=1), rcond=None
        )[0]
        radius = max(0, constant + cx * cx + cy * cy) ** 0.5
        if width * 0.08 < radius < palm_length * 0.7 and abs(cy) < palm_length:
            curves.append((cx, cy, radius))
    if curves:
        cx, cy, radius = np.median(curves, axis=0)
        center = (
            wrist
            + along * float(cx)
            + normal * float(cy)
            + across * np.mean([(k - wrist).dot(across) for k in knuckles])
        )
        radius = max(width * 0.055, min(float(radius) - width * 0.11, width * 0.22))
        method = "curled-finger-circle"
    else:
        bend = point("HandMiddle2") - point("HandMiddle1")
        if bend.dot(normal) < 0:
            normal = -normal
        center = (
            sum(knuckles, Vector()) / 4
            - along * palm_length * 0.12
            + normal * palm_length * 0.22
        )
        radius = width * 0.13
        method = "open-palm-fallback"
    tree, surface = body_surface(arm, side)
    initial = final = None
    if curves and surface:
        center, radius, initial, final = fit_clearance(
            center, across, along, normal, width, radius, tree, surface
        )
    return dict(
        center=center,
        axis=across,
        normal=normal,
        along=along,
        radius=radius,
        width=width,
        points=points,
        surface=surface,
        method=method,
        initial_axis_clearance=initial,
        fitted_axis_clearance=final,
    )


def seat_surface(arm, height):
    """Find actual pelvis/thigh underside, excluding dangling lower legs."""
    import bpy

    hips = arm.matrix_world @ arm.pose.bones["mixamorig:Hips"].head
    graph = bpy.context.evaluated_depsgraph_get()
    candidates = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.name.startswith("StandinScene__"):
            continue
        names = {group.index: group.name for group in obj.vertex_groups}
        evaluated = obj.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        try:
            for vertex in mesh.vertices:
                relevant = sum(
                    g.weight
                    for g in vertex.groups
                    if names.get(g.group, "").endswith(
                        ("Hips", "LeftUpLeg", "RightUpLeg")
                    )
                )
                p = evaluated.matrix_world @ vertex.co
                if (
                    relevant > 0.5
                    and (Vector((p.x - hips.x, p.y - hips.y, 0))).length < height * 0.14
                ):
                    candidates.append(p.z)
        finally:
            evaluated.to_mesh_clear()
    if not candidates:
        raise ValueError("no pelvis surface available for support placement")
    return min(candidates) - height * 0.002
