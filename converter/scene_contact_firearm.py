import bpy
from mathutils import Matrix
from converter.scene_placement import hand_fit
from converter.scene_contact_frames import palm as palm_frame
from converter.scene_contact_ik import move


def cradle(arm, meshes, *, mirror=False):
    from converter.hand_contact import screen_hands

    primary, secondary = ("Left", "Right") if mirror else ("Right", "Left")
    fits = {s: hand_fit(arm, s) for s in ("Left", "Right")}

    def p(n):
        return arm.matrix_world @ arm.pose.bones["mixamorig:" + n].head

    right = fits[primary]
    up = right["axis"]
    up = up if up.dot(p("Spine2") - p("Hips")) > 0 else -up
    aim = (p(primary + "Hand") - p(primary + "ForeArm")).normalized()
    aim = (aim - up * aim.dot(up)).normalized()
    bottom = right["center"] + up * (
        min((v - right["center"]).dot(up) for v in right["surface"]) - 0.007
    )
    palm, normal, left = palm_frame(arm, secondary)
    local = Matrix((left["along"], normal, left["along"].cross(normal))).transposed()
    target = Matrix((aim, up, aim.cross(up))).transposed()
    rotation = (target @ local.transposed()).to_quaternion()
    wrist = p(secondary + "Hand")
    local_surface = [rotation @ (v - palm) for v in left["surface"]]
    contact_patch = [v for v in local_surface if (v - up * v.dot(up)).length < 0.025]
    if not contact_patch:
        raise ValueError("missing support palm patch")
    projected = max(v.dot(up) for v in contact_patch)
    goal = bottom - up * (projected + 0.001) - rotation @ (palm - wrist)
    bases = {b.name: b.matrix_basis.copy() for b in arm.pose.bones}
    for down in (0, 0.004, 0.008, 0.012):
        for forward in (0, 0.01, -0.01, 0.02, -0.02):
            for b in arm.pose.bones:
                b.matrix_basis = bases[b.name]
            bpy.context.view_layer.update()
            try:
                result = move(
                    arm, secondary, goal - up * down + aim * forward, rotation
                )
            except ValueError:
                continue
            if screen_hands(meshes)["status"] == "clear_screen":
                result.update(clearance_down=down, clearance_forward=forward)
                return [result]
    raise ValueError("no bounded pistol cradle")
