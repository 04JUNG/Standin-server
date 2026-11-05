"""Surface-fitted small tools and bows with a straight, nonpenetrating grip."""

from mathutils import Vector
from converter.scene_placement import body_surface
from converter.scene_primitives import rod, box, sphere
from converter.scene_clearance import BodyClearance, clear_connector
from converter.scene_contact_frames import span


def thin_tool(arm, kind, grip, fits, scale):
    f = fits[grip]
    axis = f["axis"]
    axis = axis if axis.z >= 0 else -axis
    c = f["center"]
    low, high = span(f, axis, scale)
    radius = min(f["radius"], 0.003 * scale)
    end = c + axis * (high + 0.07 * scale)
    obj = rod("Tool grip", c + axis * low, end, radius)
    BodyClearance().translate_clear(
        obj, [Vector(), f["normal"] * 0.002 * scale, -f["normal"] * 0.002 * scale]
    )
    if kind == "spoon":
        sphere("Spoon bowl", end, 0.016 * scale, (1, 0.35, 1.4))
    elif kind in {"knife", "comb"}:
        box(
            "Tool head",
            end + axis * 0.04 * scale,
            (0.03 * scale, 0.004 * scale, 0.08 * scale),
            axis,
        )
    elif kind == "key":
        box("Key bit", end, (0.02 * scale, 0.006 * scale, 0.012 * scale), axis)


def bow(arm, grip, fits, scale, undrawn=False):
    f = fits[grip]
    c = f["center"]
    axis = f["axis"]
    axis = axis if axis.z >= 0 else -axis

    def p(n):
        return arm.matrix_world @ arm.pose.bones["mixamorig:" + n].head

    aim = (c - p("Head")).normalized()
    lo, hi = span(f, axis, scale, 0.015)
    a, b = c + axis * lo, c + axis * hi
    rod("Bow grip", a, b, min(f["radius"], 0.005 * scale))
    ends = []
    for sign, start in ((-1, a), (1, b)):
        curve = [
            start + axis * sign * (t * 0.40 * scale) - aim * t * t * 0.10 * scale
            for t in [i / 8 for i in range(9)]
        ]
        for v, w in zip(curve, curve[1:]):
            rod("Bow limb", v, w, 0.006 * scale)
        ends.append(curve[-1])
    tree, _ = body_surface(arm)
    if undrawn:
        clear_connector(
            "Bowstring", *ends, 0.001 * scale, tree, (axis, aim, axis.cross(aim)), scale
        )
    else:
        other = fits["R" if grip == "L" else "L"]
        draw = other["center"]
        for endpoint in ends:
            clear_connector(
                "Bowstring",
                endpoint,
                draw,
                0.001 * scale,
                tree,
                (other["axis"], other["along"], other["normal"]),
                scale,
            )
