"""Shield and bow guides fitted to evaluated hand geometry."""

from mathutils import Vector, Matrix
from converter.scene_primitives import box as _box, rod as _rod
from converter.scene_placement import hand_fit, body_surface


def add_surface(arm, kind, grip, scale, *, bow_state="drawn"):
    def p(name):
        return arm.matrix_world @ arm.pose.bones["mixamorig:" + name].head

    side = "Left" if grip == "L" else "Right"
    fit = hand_fit(arm, side)
    center = fit["center"]
    axis = fit["axis"]
    if kind == "shield":
        normal = (p(side + "Hand") - p(side + "ForeArm")).normalized()
        vertical = Vector((0, 0, 1))
        vertical = vertical - normal * vertical.dot(normal)
        if vertical.length < 0.05:
            vertical = fit["along"] - normal * fit["along"].dot(normal)
        vertical.normalize()
        lateral = normal.cross(vertical).normalized()
        thickness = 0.035 * scale
        offset = (
            max((v - center).dot(normal) for v in fit["surface"])
            + thickness / 2
            + 0.008 * scale
        )
        panel = _box(
            "Shield panel",
            center + normal * offset,
            (0.42 * scale, thickness, 0.54 * scale),
        )
        panel.rotation_euler = (
            Matrix((lateral, normal, vertical)).transposed().to_euler()
        )
        low = min((v - center).dot(axis) for v in fit["surface"]) - 0.008 * scale
        high = max((v - center).dot(axis) for v in fit["surface"]) + 0.008 * scale
        _rod("Shield grip", center + axis * low, center + axis * high, fit["radius"])
        for v in (low, high):
            a = center + axis * v
            b = a + normal * (offset - thickness / 2 - (a - center).dot(normal))
            _rod("Shield bracket", a, b, 0.005 * scale)
        return
    aim = (center - p("Head")).normalized()
    if axis.z < 0:
        axis = -axis
    arc = [
        center + axis * (t * 0.46 * scale) - aim * (t * t * 0.13 * scale)
        for t in [i / 6 for i in range(-6, 7)]
    ]
    for a, b in zip(arc, arc[1:]):
        _rod("Bow limb", a, b, min(0.008 * scale, fit["radius"]))
    if bow_state == "undrawn":
        _rod("Bowstring", arc[0], arc[-1], 0.002 * scale)
        return
    other = hand_fit(arm, "Right" if grip == "L" else "Left")
    draw = other["center"]
    forward = (center - draw).normalized()
    tree, _ = body_surface(arm)

    def clearance(a, b):
        result = 100
        for i in range(101):
            point = a + (b - a) * (i / 100)
            nearest, normal, _, dist = tree.find_nearest(point)
            if (point - nearest).dot(normal) < 0:
                dist = -dist
            result = min(result, dist)
        return result

    best = None
    for step in range(41):
        candidate = draw + forward * (step * 0.002 * scale)
        gap = min(clearance(arc[0], candidate), clearance(candidate, arc[-1]))
        if best is None or gap > best[0]:
            best = (gap, candidate)
        if gap > 0.003 * scale:
            break
    draw = best[1]
    _rod("Bowstring", arc[0], draw, 0.002 * scale)
    _rod("Bowstring", draw, arc[-1], 0.002 * scale)
    return
