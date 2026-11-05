"""Fit a thin phone to a reviewed phone grip, with a bounded skin clearance."""

from mathutils import Matrix, Vector
from converter.scene_primitives import box
from converter.scene_clearance import BodyClearance


def add_phone(fit, scale):
    width, thickness, height = 0.07 * scale, 0.008 * scale, 0.15 * scale
    center = fit["center"] + fit["along"] * (width / 2 - fit["radius"] * 0.5)
    obj = box("Phone", center, (width, thickness, height))
    obj.rotation_euler = (
        Matrix((fit["along"], fit["normal"], fit["axis"])).transposed().to_euler()
    )
    axes = (fit["normal"], fit["along"], fit["axis"])
    shifts = [
        sum((axes[n] * coordinate[n] * 0.003 * scale for n in range(3)), Vector())
        for coordinate in (
            (i, j, k)
            for i in range(-12, 13)
            for j in range(-8, 9)
            for k in range(-2, 3)
        )
    ]
    return BodyClearance().translate_clear(obj, shifts)
