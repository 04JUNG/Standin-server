"""Containers, reading surfaces and hand-held household guides.

Geometry follows the actual hand surfaces. Body/finger corrections belong to
the explicit scene policy stage; this module never changes bones.
"""

from mathutils import Vector, Matrix
from converter.scene_primitives import box as _box, rod as _rod
from converter.scene_clearance import clear_connector, BodyClearance

KINDS = frozenset(
    "crate chest basket bag pouch lantern dryer book letter photo scroll".split()
)


def add_container(arm, kind, grip, fits, scale):
    def p(name):
        return arm.matrix_world @ arm.pose.bones["mixamorig:" + name].head

    selected = [fits[s] for s in ("L", "R") if grip == "B" or s == grip]
    from converter.scene_placement import body_surface

    tree, _ = body_surface(arm)
    up = Vector((0, 0, 1))
    center = sum((f["center"] for f in selected), Vector()) / len(selected)
    if kind in {"crate", "chest", "basket"} and grip == "B":
        lateral = (fits["R"]["center"] - fits["L"]["center"]).normalized()
        vertical = (up - lateral * up.dot(lateral)).normalized()
        forward = vertical.cross(lateral).normalized()
        if forward.dot(center - p("Spine")) < 0:
            forward = -forward
            vertical = -vertical
        # Side faces stay between the two complete hand surfaces.
        lo = (
            max((v - center).dot(lateral) for v in fits["L"]["surface"]) + 0.008 * scale
        )
        hi = (
            min((v - center).dot(lateral) for v in fits["R"]["surface"]) - 0.008 * scale
        )
        if hi - lo < 0.06 * scale:
            raise ValueError("no clear width between hands")
        width = hi - lo
        depth = 0.26 * scale
        height = 0.26 * scale
        rear = (
            max((v - center).dot(forward) for f in selected for v in f["surface"])
            + 0.003 * scale
        )
        box_center = (
            center
            + lateral * ((lo + hi) / 2)
            + forward * (rear + depth / 2)
            + up * (height * 0.4)
        )
        obj = _box("Container", box_center, (width, depth, height))
        obj.rotation_euler = (
            Matrix((lateral, forward, lateral.cross(forward))).transposed().to_euler()
        )
        for f in selected:
            axis = f["axis"]
            c = f["center"]
            low = min((v - c).dot(axis) for v in f["surface"]) - 0.008 * scale
            high = max((v - c).dot(axis) for v in f["surface"]) + 0.008 * scale
            _rod("Side handle", c + axis * low, c + axis * high, f["radius"])
            for edge in (low, high):
                a = c + axis * edge
                b = a + forward * (rear + 0.01 * scale - (a - center).dot(forward))
                target = (
                    box_center
                    + lateral
                    * max(-width / 2, min(width / 2, (b - box_center).dot(lateral)))
                    - forward * depth / 2
                )
                target.z = b.z
                clear_connector(
                    "Handle bracket",
                    a,
                    b,
                    0.004 * scale,
                    tree,
                    (lateral, forward, up),
                    scale,
                )
                clear_connector(
                    "Handle return",
                    b,
                    target,
                    0.004 * scale,
                    tree,
                    (lateral, forward, up),
                    scale,
                )
    elif kind in {"bag", "pouch", "lantern"}:
        f = selected[0]
        axis = f["axis"]
        c = f["center"]
        low = min((v - c).dot(axis) for v in f["surface"]) - 0.008 * scale
        high = max((v - c).dot(axis) for v in f["surface"]) + 0.008 * scale
        _rod(
            "Carry handle",
            c + axis * low,
            c + axis * high,
            min(f["radius"], 0.006 * scale),
        )
        height = (
            0.18 if kind == "lantern" else 0.13 if kind == "pouch" else 0.28
        ) * scale
        width = (
            0.13 if kind == "lantern" else 0.10 if kind == "pouch" else 0.26
        ) * scale
        top = min(v.z for v in f["surface"]) - 0.025 * scale
        body_center = Vector((c.x, c.y, top - height / 2))
        obj = _box("Hanging body", body_center, (width, width * 0.60, height))
        clearance = BodyClearance()
        shifts = [
            Vector((i * 0.003 * scale, j * 0.003 * scale, 0))
            for i in range(-6, 7)
            for j in range(-6, 7)
        ]
        shift = Vector(clearance.translate_clear(obj, shifts))
        for edge in (low, high):
            a = c + axis * edge
            b = Vector((a.x, a.y, top)) + shift
            clear_connector(
                "Handle strap",
                a,
                b,
                0.004 * scale,
                tree,
                (f["axis"], f["along"], f["normal"]),
                scale,
            )
    elif kind == "dryer":
        f = selected[0]
        axis = f["axis"]
        c = f["center"]
        if axis.dot(up) < 0:
            axis = -axis
        low = min((v - c).dot(axis) for v in f["surface"]) - 0.008 * scale
        high = max((v - c).dot(axis) for v in f["surface"]) + 0.04 * scale
        _rod("Dryer handle", c + axis * low, c + axis * high, f["radius"])
        top = c + axis * high
        direction = (p("Head") - top).normalized()
        direction = (direction - axis * direction.dot(axis)).normalized()
        _rod(
            "Dryer nozzle",
            top - direction * 0.035 * scale,
            top + direction * 0.12 * scale,
            0.027 * scale,
        )
    elif kind in {"book", "letter", "photo", "scroll"}:
        normal = (p("Head") - center).normalized()
        lateral = (
            (fits["R"]["center"] - fits["L"]["center"]).normalized()
            if grip == "B"
            else selected[0]["axis"]
        )
        lateral = (lateral - normal * lateral.dot(normal)).normalized()
        vertical = lateral.cross(normal).normalized()
        normal = vertical.cross(lateral).normalized()
        width = (
            (fits["R"]["center"] - fits["L"]["center"]).length + 0.07 * scale
            if grip == "B"
            else 0.22 * scale
        )
        thickness = (0.015 if kind == "book" else 0.003) * scale
        offset = (
            min((v - center).dot(normal) for f in selected for v in f["surface"])
            - thickness / 2
            - 0.002 * scale
        )
        obj = _box(
            "Reading surface",
            center + normal * offset,
            (width, thickness, 0.26 * scale),
        )
        obj.rotation_euler = Matrix((lateral, normal, vertical)).transposed().to_euler()
    else:
        raise ValueError("unsupported container grip: " + kind + ":" + grip)
