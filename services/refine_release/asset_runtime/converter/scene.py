"""Compose final single-person rigged-rest FBXs, then verify an FBX round trip.

This module is outside the frozen retarget solver. Import bpy only in Blender.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from converter.scene_contract import SCENE_VERSION, validate_manifest


class SceneInputError(ValueError):
    """A supplied file is not a final single-person rigged-rest FBX."""


def _import(path):
    import bpy
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(path), use_anim=True, ignore_leaf_bones=False,
                             automatic_bone_orientation=False)
    return set(bpy.data.objects) - before


def _snapshot():
    import bpy
    graph = bpy.context.evaluated_depsgraph_get()
    result = {}
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            evaluated = obj.evaluated_get(graph)
            mesh = evaluated.to_mesh()
            try:
                result[obj.name] = {
                    "type": "MESH",
                    "points": [tuple(evaluated.matrix_world @ v.co) for v in mesh.vertices],
                    "rigs": sorted(m.object.name for m in obj.modifiers
                                   if m.type == "ARMATURE" and m.object),
                }
            finally:
                evaluated.to_mesh_clear()
        elif obj.type == "ARMATURE":
            result[obj.name] = {
                "type": "ARMATURE",
                "bones": {b.name: (b.parent.name if b.parent else None) for b in obj.data.bones},
                "points": [tuple(obj.matrix_world @ b.head_local) for b in obj.data.bones],
            }
    return result


def _verify(before, after, tolerance=0.0001):
    """World-space skin and rest-bone heads must survive within 0.1 mm."""
    if before.keys() != after.keys():
        raise ValueError("round trip changed mesh/armature membership")
    maximum = 0.0
    for name, expected in before.items():
        actual = after[name]
        if expected.keys() != actual.keys() or any(
            expected[key] != actual[key] for key in expected if key != "points"
        ):
            raise ValueError(f"round trip changed skeleton or skin binding: {name}")
        if len(expected["points"]) != len(actual["points"]):
            raise ValueError(f"round trip changed vertex/bone count: {name}")
        for a, b in zip(expected["points"], actual["points"]):
            error = math.dist(a, b)
            if not math.isfinite(error) or error > tolerance:
                raise ValueError(f"round trip geometry error: {name}: {error}")
            maximum = max(maximum, error)
    return maximum


def compose(manifest, fbx_paths, output_path):
    import bpy
    from mathutils import Matrix

    manifest = validate_manifest(manifest)
    if len(fbx_paths) != len(manifest["persons"]):
        raise SceneInputError("one FBX per person is required")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 1.0
    people = []
    rough = manifest.get("rough_layout")
    camera = None
    if rough:
        from converter.rough_layout import camera_for
        camera = camera_for(rough["width"], rough["height"])
    for row, path in zip(manifest["persons"], fbx_paths):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != row["fbx_sha256"]:
            raise SceneInputError("source FBX checksum mismatch")
        try:
            objects = _import(path)
        except Exception as exc:
            raise SceneInputError("source FBX could not be imported") from exc
        rigs = [o for o in objects if o.type == "ARMATURE"]
        meshes = [o for o in objects if o.type == "MESH"]
        if len(rigs) != 1 or not meshes or any(o.type not in {"MESH", "ARMATURE", "EMPTY"} for o in objects):
            raise SceneInputError("each source must contain one rig and its meshes")
        if any(o.animation_data and (o.animation_data.action or o.animation_data.nla_tracks)
               for o in objects):
            raise SceneInputError("only static rigged-rest sources are supported")
        if any(not any(m.type == "ARMATURE" and m.object == rigs[0] for m in o.modifiers)
               for o in meshes):
            raise SceneInputError("each mesh must be bound to its person's rig")
        prefix = f"person_{row['person_index']:04d}_"
        # Every object has a unique namespace; bone names stay compatible with its own skin.
        for obj in sorted(objects, key=lambda o: o.name):
            obj.name = prefix + obj.name
        group = bpy.data.objects.new(prefix + "placement", None)
        bpy.context.collection.objects.link(group)
        group["person_index"] = row["person_index"]
        group["cut_id"] = manifest["cut_id"]
        for obj in objects:
            if obj.parent not in objects:
                world = obj.matrix_world.copy()
                obj.parent = group
                obj.matrix_world = world
        fit_report = None
        if rough:
            from converter.rough_layout import geometry_for, fit_translation
            vertices, joints = geometry_for(rigs[0], meshes)
            try:
                translation, fit_report = fit_translation(vertices, joints, row["observation"], camera)
            except ValueError as exc:
                raise SceneInputError("rough observations could not be fitted") from exc
            tx, ty, tz = translation
            row = {**row, "position": [tx, tz, -ty], "scale": 1.0, "rotation_y_degrees": 0.0}
        x, y, z = row["position"]
        group.matrix_world = (Matrix.Translation((x, -z, y))
                              @ Matrix.Rotation(math.radians(row["rotation_y_degrees"]), 4, "Z")
                              @ Matrix.Scale(row["scale"], 4))
        bpy.context.view_layer.update()
        people.append({**row, "root_name": group.name, "armature_name": rigs[0].name,
                       "bone_count": len(rigs[0].data.bones), "mesh_count": len(meshes),
                       **({"layout_fit": fit_report} if fit_report else {})})
    framing = None
    if camera:
        from mathutils import Vector
        from converter.scene_camera import configure_camera
        data = bpy.data.cameras.new("rough_camera")
        obj = bpy.data.objects.new("rough_camera", data)
        bpy.context.collection.objects.link(obj)
        obj.location = camera["position_blender"]
        obj.rotation_euler = Vector((0, 1, 0)).to_track_quat("-Z", "Y").to_euler()
        framing = configure_camera(obj, camera)
        scene = bpy.context.scene
        scene.render.resolution_x = rough["width"]
        scene.render.resolution_y = rough["height"]
        scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
        scene.camera = obj
    before = _snapshot()
    bpy.ops.export_scene.fbx(
        filepath=str(output_path), use_selection=False, object_types={"EMPTY", "ARMATURE", "MESH", "CAMERA"},
        axis_forward="-Z", axis_up="Y", apply_unit_scale=True, global_scale=1.0,
        use_mesh_modifiers=True, add_leaf_bones=False, bake_anim=False,
        primary_bone_axis="Y", secondary_bone_axis="X", mesh_smooth_type="FACE",
        path_mode="AUTO", embed_textures=False, use_custom_props=True,
    )
    # Check the serialized file, not the still-resident composition.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    _import(output_path)
    bpy.context.view_layer.update()
    error = _verify(before, _snapshot())
    if framing:
        from converter.scene_camera import verify_camera_recipe
        cameras = [o for o in bpy.context.scene.objects if o.type == "CAMERA"]
        if len(cameras) != 1:
            raise ValueError("scene camera did not survive export")
        verify_camera_recipe(cameras[0], framing)
    return dict(ok=True, version=SCENE_VERSION, blender_version=bpy.app.version_string,
                person_count=len(people), persons=people, roundtrip_verified=True,
                max_world_error_m=error, tolerance_m=0.0001,
                **({"framing": framing} if framing else {}),
                **({"rough_layout": {"version": "rough-layout-v1", "camera": camera,
                    "depth_is_estimate": True, "manual_adjustment": False,
                    "csp_camera_preservation": False}} if rough else {}))
