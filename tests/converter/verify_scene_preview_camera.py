"""Real Blender FBX roundtrip: portrait/square/landscape preserve projection.

Run with Blender --background --factory-startup --python-exit-code 1 --python ...
No customer assets, external models, rendering or paid API calls required.
"""

import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
from converter.final_preview_worker import configure_scene_camera

rows = []
with tempfile.TemporaryDirectory() as folder:
    for width, height in ((690, 1168), (690, 1926), (283, 163), (512, 512)):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene
        data = bpy.data.cameras.new("verified-scene-camera")
        camera = bpy.data.objects.new("verified-scene-camera", data)
        scene.collection.objects.link(camera)
        camera.location = (0, -5, 1)
        camera.rotation_euler = Vector((0, 1, 0)).to_track_quat("-Z", "Y").to_euler()
        data.sensor_fit = "HORIZONTAL"
        data.sensor_width = 36
        data.sensor_height = 36 * height / width
        data.lens = 40
        camera["source_image_width"] = width
        camera["source_image_height"] = height
        configure_scene_camera(scene, camera)
        bpy.context.view_layer.update()
        points = [Vector(p) for p in ((-1, 0, 1), (1, 0, 1), (0, 0, 2), (0, 2, 0))]
        expected = [world_to_camera_view(scene, camera, p) for p in points]
        path = str(Path(folder) / "scene.fbx")
        bpy.ops.export_scene.fbx(
            filepath=path,
            object_types={"CAMERA"},
            axis_forward="-Z",
            axis_up="Y",
            use_custom_props=True,
        )
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.fbx(
            filepath=path, ignore_leaf_bones=True, automatic_bone_orientation=False
        )
        scene = bpy.context.scene
        camera = next(o for o in scene.objects if o.type == "CAMERA")
        configure_scene_camera(scene, camera)
        bpy.context.view_layer.update()
        actual = [world_to_camera_view(scene, camera, p) for p in points]
        error = max(
            max(abs(a.x - b.x), abs(a.y - b.y)) for a, b in zip(actual, expected)
        )
        assert error < 1e-5, (width, height, error)
        assert camera.data.sensor_fit == "HORIZONTAL"
        rows.append({"width": width, "height": height, "normalized_max_error": error})
print("SCENE_PREVIEW_PROJECTION_OK " + json.dumps(rows))
