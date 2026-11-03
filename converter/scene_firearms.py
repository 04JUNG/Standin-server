"""Place a grip and barrel against the evaluated hand surfaces."""

from converter.scene_placement import body_surface


def add_firearm(arm, kind, grip, fits, scale, *, mirror=False):
    from converter.scene_guides import _rod

    primary = "L" if grip == "L" or grip == "B" and mirror else "R"
    secondary = "R" if primary == "L" else "L"
    fit = fits[primary]
    center, up = fit["center"], fit["axis"]

    def point(name):
        return arm.matrix_world @ arm.pose.bones["mixamorig:" + name].head

    if up.dot(point("Spine2") - point("Hips")) < 0:
        up = -up
    lower = min((v - center).dot(up) for v in fit["surface"]) - 0.006 * scale
    upper = max((v - center).dot(up) for v in fit["surface"]) + 0.018 * scale
    tree, _ = body_surface(arm)
    start = center + up * upper
    side = "Left" if primary == "L" else "Right"
    aim = (point(side + "Hand") - point(side + "ForeArm")).normalized()
    aim = (aim - up * aim.dot(up)).normalized()
    length = (0.62 if kind == "rifle" else 0.45 if kind == "crossbow" else 0.27) * scale
    radius = 0.008 * scale

    def clearance(a, b):
        minimum = float("inf")
        for i in range(101):
            p = a + (b - a) * (i / 100)
            nearest, normal, _, distance = tree.find_nearest(p)
            minimum = min(
                minimum, distance if (p - nearest).dot(normal) >= 0 else -distance
            )
        return minimum

    best = None
    for rise in range(5):
        a = start + up * (rise * 0.008 * scale)
        goals = [a + aim * length]
        if kind in {"rifle", "crossbow"} and grip == "B":
            other = fits[secondary]
            goals = [
                other["center"]
                + other["normal"] * (i * 0.006 * scale)
                + other["along"] * (j * 0.006 * scale)
                for i in range(-5, 6)
                for j in range(-5, 6)
            ]
        for goal in goals:
            direction = (goal - a).normalized()
            b = a + direction * length
            distance = clearance(a - direction * 0.005 * scale, b)
            deviation = (
                (goal - fits[secondary]["center"]).length / scale if grip == "B" else 0
            )
            score = (
                (100 if distance >= radius + 0.001 * scale else distance / scale * 1000)
                - deviation
                - rise * 0.002
            )
            if best is None or score > best[0]:
                best = (score, a, b, distance)
    _, a, b, distance = best
    _rod("Grip", center + up * lower, a, fit["radius"])
    _rod("Aim guide", a - (b - a).normalized() * 0.005 * scale, b, radius)
    return {
        "barrel_axis_clearance": distance,
        "barrel_radius": radius,
        "primary_hand": primary,
        "collision_certified": False,
    }
