"""Keep cloth grips attached while fitting a thin sagging ribbon between them."""

import math
import bpy
from mathutils import Vector
from converter.scene_primitives import rod
from converter.scene_clearance import BodyClearance
from converter.scene_contact_frames import span


def flexible(arm, kind, grip, fits, scale):
    ends = []
    for side in ("L", "R") if grip == "B" else (grip,):
        f = fits[side]
        lo, hi = span(f, f["axis"], scale, 0.012)
        a, b = (f["center"] + f["axis"] * v for v in (lo, hi))
        rod("Folded fabric grip", a, b, min(0.003 * scale, f["radius"]))
        ends.append((a, b))
    if len(ends) == 2:
        a, b = min(
            ((a, b) for a in ends[0] for b in ends[1]),
            key=lambda ab: (ab[0] - ab[1]).length,
        )
    else:
        a = min(ends[0], key=lambda p: p.z)
        b = a - Vector((0, 0, 0.16 * scale))
    body = BodyClearance()
    delta = b - a
    width_axis = Vector((0, 0, -1))
    width_axis = (
        width_axis - delta.normalized() * width_axis.dot(delta.normalized())
    ).normalized()
    if width_axis.length < 0.1:
        width_axis = Vector((1, 0, 0))
    for sag in (0.02, 0.04, 0.07):
        for width in (0.05, 0.035, 0.02) if kind != "rope" else (0.003,):
            vertices = []
            for i in range(17):
                t = i / 16
                c = a + delta * t - Vector((0, 0, math.sin(t * math.pi) * sag * scale))
                w = (0.002 + math.sin(math.pi * t) * width) * scale
                vertices.extend((c - width_axis * w / 2, c + width_axis * w / 2))
            mesh = bpy.data.meshes.new("Fabric drape")
            mesh.from_pydata(
                vertices,
                [],
                [(i * 2, i * 2 + 1, i * 2 + 3, i * 2 + 2) for i in range(16)],
            )
            mesh.update()
            obj = bpy.data.objects.new("Fabric drape", mesh)
            bpy.context.collection.objects.link(obj)
            bpy.context.view_layer.update()
            if body.clear_mesh(obj, Vector()):
                obj["guide_kind"] = kind
                return
            bpy.data.objects.remove(obj, do_unlink=True)
    raise ValueError("no clear fabric drape without detaching grips")
