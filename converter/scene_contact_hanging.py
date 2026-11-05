"""Hanging guides whose straps terminate inside the actual translated container."""

from mathutils import Vector
from converter.scene_primitives import box, rod
from converter.scene_clearance import BodyClearance, clear_connector
from converter.scene_placement import body_surface
from converter.scene_contact_frames import span


def lantern(arm, grip, fits, scale):
    f = fits[grip]
    center, axis = f["center"], f["axis"]
    lo, hi = span(f, axis, scale, 0.010)
    ends = [center + axis * edge for edge in (lo, hi)]
    rod("Lantern handle", *ends, min(f["radius"], 0.006 * scale))
    height, width, depth = 0.18 * scale, 0.13 * scale, 0.078 * scale
    top = min(v.z for v in f["surface"]) - 0.025 * scale
    obj = box(
        "Lantern body",
        Vector((center.x, center.y, top - height / 2)),
        (width, depth, height),
    )
    BodyClearance().translate_clear(
        obj,
        [
            Vector((i * 0.003 * scale, j * 0.003 * scale, 0))
            for i in range(-6, 7)
            for j in range(-6, 7)
        ],
    )
    tree, _ = body_surface(arm)
    margin = 0.003 * scale
    for start in ends:
        # A vertical drop from a wide/tilted grip can miss the small lantern.
        # Clamp both horizontal coordinates to its final top face, with overlap.
        end = Vector(
            (
                max(
                    obj.location.x - width / 2 + margin,
                    min(obj.location.x + width / 2 - margin, start.x),
                ),
                max(
                    obj.location.y - depth / 2 + margin,
                    min(obj.location.y + depth / 2 - margin, start.y),
                ),
                top - margin,
            )
        )
        clear_connector(
            "Attached lantern strap",
            start,
            end,
            0.004 * scale,
            tree,
            (f["axis"], f["along"], f["normal"]),
            scale,
        )
    return obj
