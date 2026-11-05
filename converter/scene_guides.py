"""Optional primitive prop guides for authored scenes, separate from the BVH.

Guides explain grip and scale. They are not production assets or collision
evidence. Body diagnostics run before these objects are added.
"""

import math
import bpy
from mathutils import Vector
from converter.scene_placement import (
    hand_fit,
    seat_surface,
    fit_clearance,
    body_surface,
    fit_two_hand_line,
)


from converter.scene_primitives import sphere as _sphere, box as _box, rod as _rod


def add_guides(arm, specification, support="", *, mirror=False, bow_state="drawn"):
    if not specification and not support:
        return {"count": 0, "kind": "none"}

    def point(name):
        return arm.matrix_world @ arm.pose.bones["mixamorig:" + name].head

    spine = [
        point(name) for name in ("Hips", "Spine", "Spine1", "Spine2", "Neck", "Head")
    ]
    height = sum((b - a).length for a, b in zip(spine, spine[1:])) / 0.65
    up = Vector((0, 0, 1))
    forward = Vector((0, -1, 0))
    fits = {s: hand_fit(arm, n) for s, n in (("L", "Left"), ("R", "Right"))}
    centers = {}
    for side in ("L", "R"):
        name = "Left" if side == "L" else "Right"
        centers[side] = fits[side]["center"]
    count = 0
    for item in specification.split(";") if specification else []:
        kind, grip, *orientation = item.split(":")
        center = (centers["L"] + centers["R"]) / 2 if grip == "B" else centers[grip]
        axis = (centers["L"] - centers["R"]).normalized() if grip == "B" else up
        if orientation:
            body_up = (point("Spine2") - point("Hips")).normalized()
            lateral = (point("LeftArm") - point("RightArm")).normalized()
            body_forward = lateral.cross(body_up).normalized()
            hand_side = "Left" if grip == "L" else "Right"
            aim = (
                point(hand_side + "Hand") - point(hand_side + "ForeArm")
            ).normalized()
            axis = {
                "up": up,
                "down": -up,
                "forward": forward,
                "body_up": body_up,
                "body_down": -body_up,
                "body_forward": body_forward,
                "aim": aim,
                "grip_line": (centers["L"] - centers["R"]).normalized(),
            }[orientation[0]]
        if not orientation and grip == "B" and axis.z < 0:
            axis = -axis
        scale = height
        if kind == "phone" and grip != "B":
            from converter.scene_phone import add_phone

            add_phone(fits[grip], scale)
            count += 1
            continue
        if kind in {"shield", "bow"} and grip != "B":
            from converter.scene_surfaces import add_surface

            add_surface(arm, kind, grip, scale, bow_state=bow_state)
            count += 1
            continue
        from converter.scene_containers import KINDS as CONTAINER_KINDS, add_container

        if kind in CONTAINER_KINDS:
            add_container(arm, kind, grip, fits, scale)
            count += 1
            continue
        if kind in {
            "sword",
            "greatsword",
            "dagger",
            "spear",
            "staff",
            "broom",
            "mop",
            "axe",
            "hammer",
            "pickaxe",
            "wand",
            "torch",
            "umbrella",
        }:
            lengths = {
                "dagger": 0.3,
                "wand": 0.3,
                "sword": 0.8,
                "greatsword": 1.2,
                "spear": 1.7,
                "staff": 1.6,
                "broom": 1.25,
                "mop": 1.25,
                "axe": 0.75,
                "hammer": 0.8,
                "pickaxe": 0.85,
                "torch": 0.5,
                "umbrella": 0.85,
            }
            length = lengths[kind] * scale
            fitted = (
                (centers["L"] - centers["R"]).normalized()
                if grip == "B"
                else fits[grip]["axis"]
            )
            axis = fitted if fitted.dot(axis) >= 0 else -fitted
            radius = min(
                fits[s]["radius"] for s in (("L", "R") if grip == "B" else (grip,))
            )
            if grip == "B":
                center, axis, radius = fit_two_hand_line(fits, axis, arm)
            lower = 0.72 if kind in {"broom", "mop"} else 0.18 if grip != "B" else 0.45
            start, end = (
                center - axis * length * lower,
                center + axis * length * (1 - lower),
            )
            grip_points = [
                p
                for side in (("L", "R") if grip == "B" else (grip,))
                for p in fits[side]["surface"]
            ]
            last_grip = min((p - center).dot(axis) for p in grip_points)
            if kind in {"sword", "greatsword", "dagger"}:
                start = center + axis * (last_grip - 0.025 * scale)
            elif kind not in {"broom", "mop"}:
                # A long rear shaft may hit the pelvis despite clear hand grips.
                # Shorten only the unused rear section; retain both grip spans.
                tree, _ = body_surface(arm)
                for i in range(1, 101):
                    distance = last_grip - (length * lower + last_grip) * i / 100
                    candidate = center + axis * distance
                    nearest, normal, _, gap = tree.find_nearest(candidate)
                    if (
                        gap < radius + 0.003 * scale
                        or (candidate - nearest).dot(normal) < 0
                    ):
                        start = center + axis * (distance + 0.02 * scale)
                        break
            _rod("Guide " + kind, start, end, radius)
            if kind in {"sword", "greatsword", "dagger"}:
                grip_points = [
                    p
                    for side in (("L", "R") if grip == "B" else (grip,))
                    for p in fits[side]["surface"]
                ]
                clearance = (
                    max((p - center).dot(axis) for p in grip_points) + 0.016 * scale
                )
                guard = center + axis * clearance
                blade_length = length * 0.65
                _box(
                    "Blade",
                    guard + axis * (0.013 * scale + blade_length / 2),
                    (0.055 * scale, 0.012 * scale, blade_length),
                    axis,
                )
                _box("Guard", guard, (0.16 * scale, 0.03 * scale, 0.025 * scale), axis)
            elif kind in {"axe", "hammer", "pickaxe", "broom", "mop"}:
                _box(
                    "Tool head",
                    start if kind in {"broom", "mop"} else end,
                    (0.26 * scale, 0.09 * scale, 0.15 * scale),
                )
            elif kind in {"spear", "torch"}:
                _sphere("Tip", end, 0.035 * scale, (1, 1, 2.3))
            elif kind == "umbrella":
                _sphere("Canopy guide", end, 0.40 * scale, (1, 1, 0.18))
        elif kind in {"cup", "goblet", "potion", "jar", "watering_can"}:
            radius = (0.038 if kind == "potion" else 0.044) * scale
            if grip != "B":
                fit = fits[grip]
                vessel_up = fit["axis"] if fit["axis"].dot(up) >= 0 else -fit["axis"]
                outward = fit["along"]
                if kind in {"potion", "jar"}:
                    _rod(
                        "Vessel guide",
                        center - vessel_up * 0.05 * scale,
                        center + vessel_up * 0.10 * scale,
                        fit["radius"],
                    )
                    count += 1
                    continue
                if kind == "goblet":
                    top = (
                        max((p - center).dot(vessel_up) for p in fit["surface"])
                        + 0.008 * scale
                    )
                    bottom = (
                        min((p - center).dot(vessel_up) for p in fit["surface"])
                        - 0.015 * scale
                    )
                    _rod(
                        "Stem",
                        center + vessel_up * bottom,
                        center + vessel_up * top,
                        min(fit["radius"], 0.006 * scale),
                    )
                    _rod(
                        "Goblet bowl",
                        center + vessel_up * top,
                        center + vessel_up * (top + 0.08 * scale),
                        radius,
                    )
                    count += 1
                    continue
                distance = (
                    max((p - center).dot(outward) for p in fit["surface"])
                    + radius
                    + 0.003 * scale
                )
                c = center + outward * distance
                half = 0.052 * scale
                _rod("Vessel guide", c - vessel_up * half, c + vessel_up * half, radius)
                lower = (
                    min((p - center).dot(vessel_up) for p in fit["surface"])
                    - 0.008 * scale
                )
                upper = (
                    max((p - center).dot(vessel_up) for p in fit["surface"])
                    + 0.008 * scale
                )
                _rod(
                    "Vessel handle",
                    center + vessel_up * lower,
                    center + vessel_up * upper,
                    min(fit["radius"], 0.006 * scale),
                )
                for edge in (lower, upper):
                    a = center + vessel_up * edge
                    corner = c + vessel_up * edge - outward * (radius * 0.85)
                    _rod("Handle joint", a, corner, 0.005 * scale)
                    _rod(
                        "Handle return",
                        corner,
                        c
                        + vessel_up * max(-half * 0.8, min(half * 0.8, edge))
                        - outward * (radius * 0.85),
                        0.005 * scale,
                    )
            else:
                c = center
                _rod(
                    "Vessel guide",
                    c - up * 0.045 * scale,
                    c + up * 0.07 * scale,
                    radius,
                )
            if kind == "goblet":
                _rod(
                    "Stem", c - up * 0.10 * scale, c - up * 0.045 * scale, 0.012 * scale
                )
        elif kind in {
            "phone",
            "book",
            "letter",
            "photo",
            "scroll",
            "shield",
            "tray",
            "plate",
            "lunchbox",
            "cake",
            "gift",
            "ringbox",
            "crate",
            "chest",
            "basket",
            "bag",
            "pouch",
            "pillow",
            "laundry",
            "teddy",
        }:
            sizes = {
                "phone": (0.07, 0.018, 0.14),
                "book": (0.26, 0.06, 0.30),
                "letter": (0.21, 0.008, 0.15),
                "photo": (0.16, 0.005, 0.11),
                "scroll": (0.35, 0.018, 0.23),
                "shield": (0.42, 0.05, 0.54),
                "tray": (0.52, 0.30, 0.025),
                "plate": (0.22, 0.22, 0.015),
                "lunchbox": (0.32, 0.20, 0.08),
                "cake": (0.28, 0.24, 0.10),
                "gift": (0.26, 0.20, 0.20),
                "ringbox": (0.06, 0.05, 0.05),
                "crate": (0.48, 0.32, 0.30),
                "chest": (0.52, 0.34, 0.30),
                "basket": (0.50, 0.32, 0.26),
                "bag": (0.28, 0.18, 0.30),
                "pouch": (0.10, 0.08, 0.13),
                "pillow": (0.42, 0.19, 0.28),
                "laundry": (0.40, 0.22, 0.24),
                "teddy": (0.38, 0.22, 0.42),
            }
            size = sizes[kind]
            c = center.copy()
            if kind in {
                "tray",
                "plate",
                "cake",
                "lunchbox",
                "crate",
                "chest",
                "basket",
            }:
                c.z += size[2] * scale * 0.42
            _box("Guide " + kind, c, tuple(v * scale for v in size))
        elif kind in {"orb", "gem", "boulder", "crown"}:
            radius = {"orb": 0.14, "gem": 0.035, "boulder": 0.24, "crown": 0.13}[
                kind
            ] * scale
            _sphere("Guide " + kind, center + up * radius * 0.5, radius)
        elif kind in {
            "pen",
            "spoon",
            "brush",
            "comb",
            "knife",
            "key",
            "flower",
            "bouquet",
        }:
            length = (0.38 if kind in {"flower", "bouquet"} else 0.18) * scale
            tool_axis = fits[grip]["axis"] if grip != "B" else axis
            if tool_axis.dot(up) < 0:
                tool_axis = -tool_axis
            _rod(
                "Guide " + kind,
                center - tool_axis * length * 0.2,
                center + tool_axis * length * 0.8,
                0.008 * scale,
            )
            if kind in {"flower", "bouquet"}:
                _sphere(
                    "Flowers",
                    center + tool_axis * length * 0.8,
                    (0.065 if kind == "flower" else 0.15) * scale,
                )
        elif kind in {"towel", "scarf", "handkerchief", "rope"}:
            if grip == "B":
                _rod(
                    "Cloth/rope guide",
                    centers["L"],
                    centers["R"],
                    (0.008 if kind == "rope" else 0.025) * scale,
                )
            else:
                _box(
                    "Cloth guide",
                    center - up * 0.055 * scale,
                    (0.12 * scale, 0.01 * scale, 0.12 * scale),
                )
        elif kind in {"pistol", "rifle", "crossbow"}:
            from converter.scene_firearms import add_firearm

            add_firearm(arm, kind, grip, fits, scale, mirror=mirror)
        elif kind == "bow":
            aim = (center - point("Head")).normalized()
            bow_up = fits[grip]["axis"] if grip != "B" else axis
            if bow_up.dot(up) < 0:
                bow_up = -bow_up
            arc = [
                center + bow_up * (t * 0.46 * scale) - aim * (t * t * 0.13 * scale)
                for t in [i / 6 for i in range(-6, 7)]
            ]
            for a, b in zip(arc, arc[1:]):
                _rod("Bow limb", a, b, 0.012 * scale)
            draw = centers["R"] if grip == "L" else centers["L"]
            _rod("Bowstring", arc[0], draw, 0.002 * scale)
            _rod("Bowstring", draw, arc[-1], 0.002 * scale)
        elif kind in {"barrier"}:
            continue
        elif kind == "dryer":
            _rod(
                "Dryer handle",
                center - up * 0.07 * scale,
                center + up * 0.04 * scale,
                0.025 * scale,
            )
            inward = (point("Head") - center).normalized()
            _rod(
                "Dryer nozzle",
                center + up * 0.08 * scale,
                center + up * 0.08 * scale + inward * 0.15 * scale,
                0.04 * scale,
            )
        elif kind == "lantern":
            _box(
                "Lantern",
                center - up * 0.12 * scale,
                (0.13 * scale, 0.13 * scale, 0.18 * scale),
            )
            _rod(
                "Lantern handle",
                center - up * 0.03 * scale,
                center + up * 0.04 * scale,
                0.009 * scale,
            )
        else:
            raise ValueError("unsupported prop guide: " + kind)
        count += 1
    if support == "seat":
        from converter.scene_support import add_seat

        add_seat(arm, height)
    return {
        "count": count,
        "specification": specification,
        "support": support,
        "kind": "primitive placement guides, separate from BVH; not contact/collision evidence",
        "placement_method": "evaluated-hand-and-seat-surface-v1",
        "hand_fits": {
            side: {
                key: fit[key]
                for key in (
                    "method",
                    "radius",
                    "width",
                    "initial_axis_clearance",
                    "fitted_axis_clearance",
                )
            }
            for side, fit in fits.items()
        },
    }
