"""Place flat, carried and handheld objects against evaluated palm surfaces."""

import math
from mathutils import Vector, Matrix, Quaternion
from converter.scene_primitives import box, sphere
from converter.scene_clearance import BodyClearance
from converter.scene_contact_frames import palm


def carry(arm, kind, grip, fits, scale):
    up = Vector((0, 0, 1))
    sides = ("L", "R") if grip == "B" else (grip,)
    selected = [fits[s] for s in sides]
    palms = [palm(arm, {"L": "Left", "R": "Right"}[s]) for s in sides]
    center = sum((p[0] for p in palms), Vector()) / len(palms)
    lateral = (
        (palms[-1][0] - palms[0][0]).normalized()
        if len(palms) == 2
        else selected[0]["axis"]
    )
    vertical = (up - lateral * up.dot(lateral)).normalized()
    forward = vertical.cross(lateral).normalized()
    rotation = Matrix((lateral, forward, vertical)).transposed()
    surface = [v for f in selected for v in f["surface"]]
    flat = kind in {"tray", "plate", "cake", "ringbox"}
    if len(palms) == 2 and not flat:
        lo = (
            max((v - center).dot(lateral) for v in selected[0]["surface"])
            + 0.001 * scale
        )
        hi = (
            min((v - center).dot(lateral) for v in selected[1]["surface"])
            - 0.001 * scale
        )
        width = hi - lo
        if width < 0.04 * scale and kind != "ringbox":
            raise ValueError("hands too close for carried object")
        center += lateral * ((lo + hi) / 2)
    else:
        width = (
            (palms[-1][0] - palms[0][0]).length + 0.07 * scale
            if len(palms) == 2
            else 0.18 * scale
        )
    if kind in {"tray", "plate", "cake", "ringbox"}:
        if kind == "ringbox":
            center = palms[-1][0].copy()
            surface = selected[-1]["surface"]
        thickness = {"tray": 0.012, "plate": 0.008, "cake": 0.08, "ringbox": 0.045}[
            kind
        ] * scale
        base = max((v - center).dot(vertical) for v in surface) + 0.0007 * scale
        center += vertical * (base + thickness / 2)
        if len(palms) == 1:
            center += selected[0]["along"] * 0.04 * scale
        if kind == "ringbox":
            width = 0.06 * scale
        obj = box(
            "Supported " + kind,
            center,
            (
                width + 0.025 * scale,
                (0.05 if kind == "ringbox" else 0.15) * scale,
                thickness,
            ),
        )
    elif kind in {"boulder", "orb", "gem", "crown"}:
        radius = width / 2
        obj = sphere("Supported " + kind, center, radius, (1, 0.8, 1))
    else:
        depth = 0.16 * scale
        height = 0.22 * scale
        obj = box("Carried " + kind, center, (width, depth, height))
    obj.rotation_euler = rotation.to_euler()
    BodyClearance().translate_clear(
        obj,
        [
            vertical * i * 0.002 * scale + forward * j * 0.003 * scale
            for i in range(0, 11)
            for j in range(-5, 6)
        ],
    )


def reading(arm, kind, grip, fits, scale):
    selected = [
        palm(arm, {"L": "Left", "R": "Right"}[s])
        for s in ("L", "R")
        if grip == "B" or s == grip
    ]
    point, normal, f = selected[0]
    if len(selected) == 2:
        point = (point + selected[1][0]) / 2
        normal = (normal + selected[1][1]).normalized()
    width = (0.20 if kind == "book" else 0.15) * scale
    height = (0.26 if kind == "book" else 0.12) * scale
    thick = (0.016 if kind == "book" else 0.0015) * scale
    lateral = f["axis"]
    lateral = (lateral - normal * lateral.dot(normal)).normalized()
    vertical = lateral.cross(normal).normalized()
    outward = max((v - point).dot(normal) for v in f["surface"])
    obj = box(
        "Palm supported " + kind,
        point
        + normal * (outward + thick / 2 + 0.0008 * scale)
        + f["along"] * 0.035 * scale,
        (width, thick, height),
    )
    obj.rotation_euler = Matrix((lateral, normal, vertical)).transposed().to_euler()
    axes = (normal, lateral, vertical)
    shifts = [
        axes[0] * i * 0.002 * scale
        + axes[1] * j * 0.004 * scale
        + axes[2] * k * 0.004 * scale
        for i in range(-2, 11)
        for j in range(-3, 4)
        for k in range(-3, 4)
    ]
    BodyClearance().translate_clear(obj, shifts)


def phone(arm, grip, fits, scale):
    side = "Left" if grip == "L" else "Right"
    point, normal, f = palm(arm, side)
    vertical = f["along"]
    lateral = normal.cross(vertical).normalized()
    # A phone rests against the fingers, lengthwise along the palm, not across it.
    base = Matrix((lateral, normal, vertical)).transposed()
    obj = box("Phone", point, (0.068 * scale, 0.006 * scale, 0.135 * scale))
    body = BodyClearance()
    for tilt in (0, 15, -15):
        q = Quaternion(lateral, math.radians(tilt))
        obj.rotation_euler = (q.to_matrix() @ base).to_euler()
        outward = max((v - point).dot(normal) for v in f["surface"])
        obj.location = (
            point + normal * (outward + 0.0035 * scale) + vertical * 0.035 * scale
        )
        shifts = [
            normal * i * 0.002 * scale
            + lateral * j * 0.004 * scale
            + vertical * k * 0.005 * scale
            for i in range(0, 8)
            for j in range(-3, 4)
            for k in range(-2, 3)
        ]
        try:
            body.translate_clear(obj, shifts)
            return
        except ValueError:
            pass
    raise ValueError("no phone contact fit")
