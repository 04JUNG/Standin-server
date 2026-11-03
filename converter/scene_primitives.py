"""Small mesh constructors shared by scene placement families."""

import bpy


def sphere(name, center, radius, scale=(1, 1, 1)):
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=16, ring_count=8, radius=radius, location=center
    )
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    return obj


def box(name, center, size, direction=None):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = size
    if direction is not None:
        obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    return obj


def rod(name, start, end, radius):
    delta = end - start
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=16, radius=radius, depth=delta.length, location=(start + end) / 2
    )
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = delta.to_track_quat("Z", "Y").to_euler()
    return obj
