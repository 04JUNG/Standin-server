"""Verified scene artifact and preview generated from the identical file."""

from pathlib import Path
import json, sys, math, hashlib, time, shutil
import bpy, numpy as np
from mathutils import Vector
from converter.scene import compose, _snapshot, _verify
from converter.scene_camera import configure_camera, framing_recipe, verify_camera_recipe
from converter import retarget as rt

COLORS = [
    (0.13, 0.40, 0.74, 1),
    (0.92, 0.33, 0.11, 1),
    (0.48, 0.22, 0.72, 1),
    (0.1, 0.55, 0.32, 1),
    (0.7, 0.5, 0.1, 1),
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def add_camera(cfg):
    scene = bpy.context.scene
    factor = max(1, math.ceil(1000 / max(cfg["width"], cfg["height"])))
    scene.render.resolution_x = round(cfg["width"] * factor)
    scene.render.resolution_y = round(cfg["height"] * factor)
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
    bpy.ops.object.camera_add(location=cfg["position_blender"])
    camera = bpy.context.object
    camera.name = "StoryboardCamera"
    camera.rotation_euler = (
        Vector(cfg["look_direction_blender"]).to_track_quat("-Z", "Y").to_euler()
    )
    camera.data.type = "PERSP"
    configure_camera(camera, cfg)
    camera.data.clip_start = 0.05
    camera.data.clip_end = 1000
    camera["physical_distance_is_estimate"] = True
    scene.camera = camera
    bpy.context.view_layer.update()
    return camera


def style():
    scene = bpy.context.scene
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        index = int(obj.name.split("_")[1])
        mat = bpy.data.materials.new("review_color_" + str(index))
        mat.diffuse_color = COLORS[index % len(COLORS)]
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        for poly in obj.data.polygons:
            poly.material_index = 0
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "BOTH"
    scene.display.shading.show_shadows = True
    scene.display.shading.background_type = "VIEWPORT"
    scene.display.shading.background_color = (0.91, 0.94, 0.98)
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.render.film_transparent = False


def export_with_camera(path, cfg):
    camera = add_camera(cfg)
    matrix = camera.matrix_world.copy()
    lens = camera.data.lens
    sensor = camera.data.sensor_width
    before = _snapshot()
    bpy.ops.export_scene.fbx(
        filepath=str(path),
        object_types={"ARMATURE", "MESH", "EMPTY", "CAMERA"},
        axis_forward="-Z",
        axis_up="Y",
        apply_unit_scale=True,
        global_scale=1,
        add_leaf_bones=False,
        bake_anim=False,
        use_mesh_modifiers=True,
        primary_bone_axis="Y",
        secondary_bone_axis="X",
        embed_textures=False,
        use_custom_props=True,
    )
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(
        filepath=str(path),
        ignore_leaf_bones=False,
        automatic_bone_orientation=False,
        use_anim=False,
    )
    bpy.context.view_layer.update()
    error = _verify(before, _snapshot())
    cameras = [o for o in bpy.context.scene.objects if o.type == "CAMERA"]
    assert len(cameras) == 1
    actual = cameras[0]
    verify_camera_recipe(actual, framing_recipe(cfg))
    actual.data.sensor_fit = "HORIZONTAL"
    factor = max(1, math.ceil(1000 / max(cfg["width"], cfg["height"])))
    bpy.context.scene.render.resolution_x = round(cfg["width"] * factor)
    bpy.context.scene.render.resolution_y = round(cfg["height"] * factor)
    bpy.context.scene.render.resolution_percentage = 100
    matrix_error = max(
        abs(a - b)
        for arow, brow in zip(matrix, actual.matrix_world)
        for a, b in zip(arow, brow)
    )
    assert matrix_error < 0.0001, (path, matrix_error)
    angle = 2 * math.atan(actual.data.sensor_width / (2 * actual.data.lens))
    assert abs(angle - math.radians(cfg["horizontal_fov_degrees"])) < 1e-5, (
        angle,
        lens,
        sensor,
        actual.data.lens,
        actual.data.sensor_width,
    )
    from bpy_extras.object_utils import world_to_camera_view

    points = [
        Vector(p)
        for row in _snapshot().values()
        if row["type"] == "ARMATURE"
        for p in row["points"]
    ]
    errors = []
    for p in points:
        depth = p.y - cfg["position_blender"][1]
        if depth <= 0:
            continue
        expected = np.array(
            [
                0.5 + cfg["focal_pixels"] * p.x / depth / cfg["width"],
                0.5
                + cfg["focal_pixels"]
                * (p.z - cfg["position_blender"][2])
                / depth
                / cfg["height"],
            ]
        )
        projected = world_to_camera_view(bpy.context.scene, actual, p)
        errors.append(
            float(
                np.linalg.norm(
                    (np.array(projected[:2]) - expected) * [cfg["width"], cfg["height"]]
                )
            )
        )
    maximum = max(errors)
    assert maximum < 1.0, (
        "camera_projection_error",
        maximum,
        actual.data.sensor_fit,
        actual.data.sensor_width,
        actual.data.sensor_height,
    )
    return actual, dict(
        camera_projection_max_error_source_pixels=maximum,
        blender_import_adapter="restore HORIZONTAL sensor fit from camera.json; Blender FBX importer resets it to AUTO",
        mesh_bone_roundtrip_max_error_m=error,
        camera_matrix_max_error=matrix_error,
        horizontal_fov_degrees=math.degrees(angle),
        embedded_camera=True,
        framing=framing_recipe(cfg),
    )


def render(camera, cfg, path):
    style()
    scene = bpy.context.scene
    factor = max(1, math.ceil(1000 / max(cfg["width"], cfg["height"])))
    scene.render.resolution_x = round(cfg["width"] * factor)
    scene.render.resolution_y = round(cfg["height"] * factor)
    scene.render.resolution_percentage = 100
    scene.camera = camera
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def render_angle(path):
    scene = bpy.context.scene
    points = [
        Vector(p)
        for row in _snapshot().values()
        if row["type"] == "MESH"
        for p in row["points"]
    ]
    lo = Vector([min(p[i] for p in points) for i in range(3)])
    hi = Vector([max(p[i] for p in points) for i in range(3)])
    center = (lo + hi) / 2
    direction = Vector((-1, -1, 0.45)).normalized()
    bpy.ops.object.camera_add(
        location=center + direction * max(12, (hi - lo).length * 2)
    )
    cam = bpy.context.object
    cam.name = "preview_only_angle"
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam.data.type = "ORTHO"
    cam.data.sensor_fit = "HORIZONTAL"
    basis = cam.rotation_euler.to_matrix()
    right = basis @ Vector((1, 0, 0))
    up = basis @ Vector((0, 1, 0))
    xs = [p.dot(right) for p in points]
    ys = [p.dot(up) for p in points]
    cam.location += right * ((min(xs) + max(xs)) / 2 - center.dot(right)) + up * (
        (min(ys) + max(ys)) / 2 - center.dot(up)
    )
    cam.data.ortho_scale = max(
        (max(xs) - min(xs)) * 1.15, (max(ys) - min(ys)) * 1100 / 720 * 1.15
    )
    scene.camera = cam
    scene.render.resolution_x = 1100
    scene.render.resolution_y = 720
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def run(ctx):
    ROOT = ctx.work
    BASE = ctx.baseline
    records = []
    for item in json.loads((ROOT / "inputs.json").read_text()):
        cut = item["cut_id"]
        folder = ROOT / cut
        layout = json.loads((folder / "layout.json").read_text())
        cfg = layout["camera"]
        sources = [Path(p["source_fbx"]) for p in layout["people"]]
        results = {}
        begin = time.monotonic()
        for variant, name in [("raw", "scene-raw.fbx"), ("final", "scene.fbx")]:
            manifest = json.loads((folder / f"{variant}-scene-input.json").read_text())
            path = folder / name
            composition = compose(manifest, sources, path)
            camera, verification = export_with_camera(path, cfg)
            if not (folder / "artifact-only").exists():
                render(camera, cfg, folder / ("raw.png" if variant == "raw" else "final.png"))
            results[variant] = dict(
                composition=composition, verification=verification, fbx_sha256=sha(path)
            )
            if variant == "final" and not (folder / "artifact-only").exists():
                render_angle(folder / "angle.png")
        if (BASE / cut / "final.png").exists():
            shutil.copy2(BASE / cut / "final.png", folder / "old.png")
        if (BASE / cut / "scene.fbx").exists():
            shutil.copy2(BASE / cut / "scene.fbx", folder / "scene-old.fbx")
        record = dict(
            cut_id=cut,
            people=len(sources),
            layout_sha256=sha(folder / "layout.json"),
            same_source_pose_fbx=False,
            manual_pose_changes=False,
            manual_positions=False,
            results=results,
            elapsed_s=time.monotonic() - begin,
        )
        (folder / "scene-result.json").write_text(json.dumps(record, indent=2))
        records = [r for r in records if r["cut_id"] != cut] + [record]
        records.sort(key=lambda r: r["cut_id"])
        (ROOT / "scene-results.json").write_text(json.dumps(records, indent=2))
        print(
            "SCENE", cut, len(sources), "s", round(record["elapsed_s"], 1), flush=True
        )
