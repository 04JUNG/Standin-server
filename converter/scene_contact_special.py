"""Explicit geometry for shared tools, offered blades and small contact objects."""

import bpy
from mathutils import Vector, Matrix
from converter.scene_placement import fit_two_hand_line, body_surface
from converter.scene_primitives import rod, box, sphere
from converter.scene_clearance import BodyClearance, clear_connector
from converter.scene_contact_frames import palm, span


def shared_tool(arm, kind, fits, scale):
    axis = (fits["L"]["center"] - fits["R"]["center"]).normalized()
    axis = axis if axis.z > 0 else -axis
    c, axis, radius = fit_two_hand_line(fits, axis, arm)
    surface = [v for f in fits.values() for v in f["surface"]]
    lo = min((v - c).dot(axis) for v in surface) - 0.02 * scale
    hi = (
        max((v - c).dot(axis) for v in surface)
        + (0.10 if kind == "pickaxe" else 0.04) * scale
    )
    rod("Shared tool handle", c + axis * lo, c + axis * hi, min(radius, 0.005 * scale))
    head = box(
        "Tool head",
        c + axis * hi,
        ((0.22 if kind == "pickaxe" else 0.035) * scale, 0.015 * scale, 0.025 * scale),
        axis,
    )
    BodyClearance().translate_clear(head, [Vector()])


def small_gem(arm, grip, fits, scale):
    side = "Left" if grip == "L" else "Right"
    f = fits[grip]
    a = arm.matrix_world @ arm.pose.bones["mixamorig:" + side + "HandThumb3"].tail
    b = arm.matrix_world @ arm.pose.bones["mixamorig:" + side + "HandIndex3"].tail
    obj = sphere("Small gem", (a + b) / 2, min(0.014 * scale, (a - b).length * 0.43))
    BodyClearance().translate_clear(
        obj,
        [
            f["along"] * i * 0.002 * scale + f["normal"] * j * 0.002 * scale
            for i in range(-5, 6)
            for j in range(-5, 6)
        ],
    )


def offered_blade(arm, kind, fits, scale):
    palms = [palm(arm, s)[0] for s in ("Left", "Right")]
    axis = (palms[1] - palms[0]).normalized()
    up = Vector((0, 0, 1))
    up = (up - axis * up.dot(axis)).normalized()
    width_axis = axis.cross(up).normalized()
    contacts = []
    for point, f in zip(palms, (fits["L"], fits["R"])):
        surface = [
            v for v in f["surface"] if abs((v - point).dot(width_axis)) < 0.024 * scale
        ]
        contacts.append(
            point + up * (max((v - point).dot(up) for v in surface) + 0.001 * scale)
        )
    axis = (contacts[1] - contacts[0]).normalized()
    center = (contacts[0] + contacts[1]) / 2
    length = (contacts[1] - contacts[0]).length + 0.22 * scale
    thickness = 0.007 * scale
    blade = box(
        "Offered blade",
        center + up * thickness / 2,
        (0.045 * scale, thickness, length),
        axis,
    )
    # Explicit right-handed blade frame keeps its thin side against the palms.
    width_axis = axis.cross(up).normalized()
    normal = axis.cross(width_axis).normalized()
    blade.rotation_euler = Matrix((width_axis, normal, axis)).transposed().to_euler()
    BodyClearance().translate_clear(blade, [up * i * 0.001 * scale for i in range(8)])
    end = blade.location - axis * length / 2
    rod("Offered grip", end - axis * 0.13 * scale, end, 0.009 * scale)
    box("Offered guard", end, (0.13 * scale, 0.012 * scale, 0.012 * scale), axis)


def hollow_crown(arm, fits, scale):
    import math
    from mathutils.kdtree import KDTree

    a, b = [palm(arm, s)[0] for s in ("Left", "Right")]
    center = (a + b) / 2
    lateral = (b - a).normalized()
    up = Vector((0, 0, 1))
    up = (up - lateral * up.dot(lateral)).normalized()
    depth = up.cross(lateral).normalized()
    hands = []
    for side in ("L", "R"):
        points = fits[side]["surface"]
        tree = KDTree(len(points))
        for i, p in enumerate(points):
            tree.insert(p, i)
        tree.balance()
        hands.append(tree)
    clear = BodyClearance()
    best = None
    # A hollow band must fit at the actual contact height, not inherit the
    # radius of an enclosing sphere constrained by unrelated finger extrema.
    for dz in (0, 0.015, -0.015, 0.03, -0.03, 0.045, -0.045):
        for ry in (0.09, 0.07, 0.11):
            for shrink in range(-20, 31):
                rx = (b - a).length / 2 - shrink * 0.002 * scale
                if rx < 0.035 * scale:
                    continue
                vertices = []
                for i in range(48):
                    angle = math.tau * i / 48
                    for inset, z in (
                        (0, -0.012),
                        (0, 0.012),
                        (0.004, 0.012),
                        (0.004, -0.012),
                    ):
                        vertices.append(
                            center
                            + up * (dz + z) * scale
                            + lateral * math.cos(angle) * (rx - inset * scale)
                            + depth * math.sin(angle) * (ry - inset) * scale
                        )
                gaps = [min(tree.find(v)[2] for v in vertices) for tree in hands]
                score = max(gaps)
                if score > 0.008 * scale or best and score >= best[0]:
                    continue
                faces = [
                    (
                        i * 4 + k,
                        ((i + 1) % 48) * 4 + k,
                        ((i + 1) % 48) * 4 + (k + 1) % 4,
                        i * 4 + (k + 1) % 4,
                    )
                    for i in range(48)
                    for k in range(4)
                ]
                mesh = bpy.data.meshes.new("Contact crown")
                mesh.from_pydata(vertices, [], faces)
                mesh.update()
                obj = bpy.data.objects.new("Contact crown band", mesh)
                bpy.context.collection.objects.link(obj)
                bpy.context.view_layer.update()
                if clear.clear_mesh(obj, Vector()):
                    if best:
                        bpy.data.objects.remove(best[1], do_unlink=True)
                    best = (score, obj, dict(gaps=gaps, rx=rx, ry=ry, dz=dz))
                    if score < 0.003 * scale:
                        break
                else:
                    bpy.data.objects.remove(obj, do_unlink=True)
            if best and best[0] < 0.003 * scale:
                break
        if best and best[0] < 0.003 * scale:
            break
    if best is None:
        raise ValueError("no attached clear crown band")
    best[1]["max_hand_gap_m"] = best[0]
    return best[1]


def bouquet(arm, fits, scale):
    # Wide wrapped bouquet: both palms hold its sides, rather than a floating stem.
    from converter.scene_contact_carry import carry

    before = set(bpy.data.objects)
    carry(arm, "bouquet", "B", fits, scale)
    wrapper = next(o for o in bpy.data.objects if o not in before and o.type == "MESH")
    center = wrapper.location.copy()
    top = center + Vector((0, 0, 0.14 * scale))
    sphere("Bouquet flowers", top, 0.10 * scale, (1, 0.7, 0.65))


def inspection_bow(arm, fits, scale):
    f = fits["L"]
    c = f["center"]
    axis = f["axis"]
    lo, hi = span(f, axis, scale, 0.025)
    rod("Bow grip", c + axis * lo, c + axis * hi, min(f["radius"], 0.004 * scale))
    forward = (
        (
            arm.matrix_world @ arm.pose.bones["mixamorig:LeftArm"].head
            - arm.matrix_world @ arm.pose.bones["mixamorig:RightArm"].head
        )
        .normalized()
        .cross(Vector((0, 0, 1)))
        .normalized()
    )
    ends = []
    tree, _ = body_surface(arm)
    axes = (f["normal"], f["along"], axis)
    for sign, start in ((-1, c + axis * lo), (1, c + axis * hi)):
        points = [
            start + axis * sign * t * 0.27 * scale + forward * t * t * 0.10 * scale
            for t in (i / 8 for i in range(9))
        ]
        for a, b in zip(points, points[1:]):
            clear_connector("Bow limb", a, b, 0.004 * scale, tree, axes, scale)
        ends.append(points[-1])
    clear_connector("Bowstring", *ends, 0.001 * scale, tree, axes, scale)


def goblet(arm, fits, scale):
    selected = [fits[s] for s in ("L", "R")]
    lateral = (selected[1]["center"] - selected[0]["center"]).normalized()
    c = (selected[1]["center"] + selected[0]["center"]) / 2
    lo = max((v - c).dot(lateral) for v in selected[0]["surface"]) + 0.001 * scale
    hi = min((v - c).dot(lateral) for v in selected[1]["surface"]) - 0.001 * scale
    radius = (hi - lo) / 2
    if radius < 0.012 * scale:
        raise ValueError("insufficient cup opening")
    c += lateral * (lo + hi) / 2
    axis = Vector((0, 0, 1))
    vessel = rod(
        "Goblet bowl", c - axis * 0.025 * scale, c + axis * 0.065 * scale, radius
    )
    BodyClearance().translate_clear(
        vessel, [axis * i * 0.002 * scale for i in range(8)]
    )
    c = vessel.location - axis * 0.02 * scale
    rod("Stem", c - axis * 0.10 * scale, c - axis * 0.045 * scale, 0.006 * scale)
    rod("Foot", c - axis * 0.104 * scale, c - axis * 0.10 * scale, radius * 0.8)
