"""Place a small seat under the evaluated pelvis, then check the full footprint."""

from mathutils import Vector
from converter.scene_primitives import box
from converter.scene_placement import seat_surface
from converter.scene_clearance import BodyClearance


def add_seat(arm, scale):
    hips = arm.matrix_world @ arm.pose.bones["mixamorig:Hips"].head
    top = seat_surface(arm, scale)
    seat = box(
        "Seat guide",
        Vector((hips.x, hips.y, top - 0.0225 * scale)),
        (0.32 * scale, 0.33 * scale, 0.045 * scale),
    )
    return BodyClearance().translate_clear(
        seat, [Vector((0, 0, -i * 0.002 * scale)) for i in range(41)]
    )
