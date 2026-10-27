"""Real Blender FBX roundtrip: portrait/square/landscape preserve projection.

Run with Blender --background --factory-startup --python-exit-code 1 --python ...
No customer assets, external models, rendering or paid API calls required.
"""

import json
import math
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bpy
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view
from converter.final_preview_worker import configure_scene_camera
from converter.rough_layout import camera_for
from converter.scene_camera import configure_camera, verify_camera_recipe

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
        cfg = camera_for(width, height)
        cfg["position_blender"] = list(camera.location)
        recipe = configure_camera(camera, cfg)
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
        verify_camera_recipe(camera, recipe)
        configure_scene_camera(scene, camera)
        bpy.context.view_layer.update()
        actual = [world_to_camera_view(scene, camera, p) for p in points]
        error = max(
            max(abs(a.x - b.x), abs(a.y - b.y)) for a, b in zip(actual, expected)
        )
        assert error < 1e-5, (width, height, error)
        assert camera.data.sensor_fit == "HORIZONTAL"
        # Metadata must not conceal a different serialized camera projection.
        matrix = camera.matrix_world.copy()
        mutations = {
            "position": Matrix.Translation((1, 0, 0)) @ matrix,
            **{f"rotation_{axis}": matrix @ Matrix.Rotation(math.radians(10), 4, axis)
               for axis in ("X", "Y", "Z")},
        }
        rejected = []
        for name, changed in mutations.items():
            camera.matrix_world = changed
            bpy.context.view_layer.update()
            try:
                verify_camera_recipe(camera, recipe)
            except ValueError:
                rejected.append(name)
            else:
                raise AssertionError(f"changed camera {name} passed verification")
        camera.matrix_world = matrix
        bpy.context.view_layer.update()
        # A parent may move the world camera without changing local location.
        parent = bpy.data.objects.new("changed-camera-parent", None)
        scene.collection.objects.link(parent)
        camera.parent = parent
        camera.matrix_world = matrix
        parent.location.x = 1
        bpy.context.view_layer.update()
        try:
            verify_camera_recipe(camera, recipe)
        except ValueError:
            rejected.append("parent_transform")
        else:
            raise AssertionError("changed camera parent passed verification")
        camera.parent = None
        camera.matrix_world = matrix
        camera.data.shift_x = .1
        bpy.context.view_layer.update()
        try:
            verify_camera_recipe(camera, recipe)
        except ValueError:
            rejected.append("principal_shift")
        else:
            raise AssertionError("off-center camera passed verification")
        camera.data.shift_x = 0
        verify_camera_recipe(camera, recipe)
        rows.append({"width": width, "height": height, "normalized_max_error": error,
                     "rejected_projection_changes": rejected})
print("SCENE_PREVIEW_PROJECTION_OK " + json.dumps(rows))
