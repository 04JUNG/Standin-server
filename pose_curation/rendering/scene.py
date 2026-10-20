"""Blender-only camera and studio setup for service-style character thumbnails."""
from __future__ import annotations

import math
from pathlib import Path

import bpy
from mathutils import Vector
from converter.preview_style import configure_workbench

VIEWS = {"front": 0, "three_quarter": 45, "side": 90, "back": 180}


def _world_vertices() -> list[Vector]:
    points = []
    graph = bpy.context.evaluated_depsgraph_get()
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        evaluated = obj.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        try:
            points.extend(evaluated.matrix_world @ vertex.co for vertex in mesh.vertices)
        finally:
            evaluated.to_mesh_clear()
    if not points or not all(math.isfinite(v) for p in points for v in p):
        raise ValueError("character mesh is empty or non-finite")
    return points


def render_views(directory: Path, pose_id: str, *, resolution: int = 256) -> dict[str, Path]:
    scene = bpy.context.scene
    configure_workbench(scene)
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.quality = 78
    scene.display.render_aa = "32"

    points = _world_vertices()
    low = Vector(tuple(min(p[i] for p in points) for i in range(3)))
    high = Vector(tuple(max(p[i] for p in points) for i in range(3)))
    center = (low + high) / 2
    extent = max((high - low).length, 1)
    camera_data = bpy.data.cameras.new("Review Camera")
    camera = bpy.data.objects.new("Review Camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_data.type = "ORTHO"
    camera_data.clip_start = extent / 1000
    camera_data.clip_end = extent * 20
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for view, yaw in VIEWS.items():
        angle = math.radians(yaw)
        direction = Vector((math.sin(angle), -math.cos(angle), 0))
        camera.location = center + direction * extent * 3
        camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
        right = Vector((math.cos(angle), math.sin(angle), 0))
        horizontal = [p.dot(right) for p in points]
        camera_data.ortho_scale = max(high.z - low.z, max(horizontal) - min(horizontal)) * 1.16
        scene.render.filepath = str(directory / f"{pose_id}__{view}.jpg")
        bpy.ops.render.render(write_still=True)
        result[view] = Path(scene.render.filepath)
    return result
