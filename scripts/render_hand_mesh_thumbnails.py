"""Render hand catalog PNGs from the library's actual default human mesh.

blender -b --factory-startup --python-exit-code 1 --python \
  scripts/render_hand_mesh_thumbnails.py -- --character standin-master-v2.fbx

Uses the same finger rotations as FBX export and the approved pose-library
studio. Only preview geometry is cropped; the source character is never saved.
The manifest is published after every requested PNG has rendered successfully.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STYLE = "approved-reference-anatomy-detail-dark-v1"
CHARACTER_SHA = "7c648b97a24a3bb4914b6e5d515708c33727979881d92ef916d5726e22301f3d"


def main():
    import bmesh
    import bpy
    import numpy as np
    from mathutils import Matrix, Quaternion, Vector
    from converter import retarget as rt
    from converter.hand_geometry import finger_rotations, hand_frame, matrix_to_quaternion
    from converter.hand_library import SIDES, bone_name, load_library
    from converter.hands import _prefix, _side_geometry, _hand_length
    from converter.preview_style import configure_workbench

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--character", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "converter/hand_thumbs")
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--presets", nargs="+", help="Preview subset; do not replace the catalog manifest")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    character_sha = hashlib.sha256(args.character.read_bytes()).hexdigest()
    if character_sha != CHARACTER_SHA:
        raise ValueError("Use the approved standin-master-v2 source character")
    if not 64 <= args.size <= 1024 or args.size % 2:
        raise ValueError("size must be an even number between 64 and 1024")
    library = load_library()
    ids = args.presets or library.preset_ids
    for preset_id in ids:
        library.preset(preset_id)
    args.out.mkdir(parents=True, exist_ok=True)
    rt.reset_scene()
    arm, meshes = rt.import_character(str(args.character))
    prefix = _prefix(arm)
    scene = bpy.context.scene
    configure_workbench(scene)
    shading = scene.display.shading
    shading.single_color = (.82, .84, .88)
    shading.show_cavity = True
    shading.cavity_type = "BOTH"
    shading.cavity_ridge_factor = shading.curvature_ridge_factor = 1.1
    shading.cavity_valley_factor = shading.curvature_valley_factor = 1.0
    shading.show_specular_highlight = True
    shading.show_shadows = True
    shading.background_type = "VIEWPORT"
    shading.background_color = (.045, .055, .075)
    scene.display.render_aa = "16"
    scene.render.resolution_x = scene.render.resolution_y = args.size
    scene.render.resolution_percentage = 100
    scene.render.dither_intensity = 0
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    camera_data = bpy.data.cameras.new("Hand thumbnail")
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    scene.collection.objects.link(camera)
    camera_data.type = "ORTHO"
    camera_data.clip_start = .001
    camera_data.clip_end = 1000
    scene.camera = camera
    files = {}
    # Rest geometry is shared between presets; no incremental pose accumulation.
    for side in SIDES:
        wrist, bases, rests = _side_geometry(arm, side, prefix)
        frame = hand_frame(side, wrist, bases)
        length = _hand_length(arm, side, prefix)
        world = arm.matrix_world
        axis = (world.to_3x3() @ Vector(frame.length_axis)).normalized()
        normal = (world.to_3x3() @ Vector(frame.palm_normal)).normalized()
        across = (world.to_3x3() @ Vector(frame.pinky_side)).normalized()
        wrist_world = world @ Vector(wrist)
        for preset_id in ids:
            preset = library.preset(preset_id)
            # Closed grips need the thumb-side view to reveal the thumb lying
            # across the fingers. Open gestures retain a broad palm silhouette.
            closed = (sum(preset.fingers["thumb"].curl) > 60 and
                      all(sum(preset.fingers[f].curl) > 120
                          for f in ("index", "middle", "ring", "pinky")))
            direction = (normal + across * (-.7 if closed else .32) + axis * .12).normalized()
            up = (axis - direction * axis.dot(direction)).normalized()
            right = up.cross(direction).normalized()
            basis = Matrix((right, up, direction)).transposed()
            camera.rotation_euler = basis.to_euler()
            for pb in arm.pose.bones:
                pb.rotation_mode = "QUATERNION"
                pb.rotation_quaternion = Quaternion()
            rotations = finger_rotations(side, preset, frame, rests, bases)
            for (finger, joint), local in rotations.items():
                arm.pose.bones[bone_name(side, finger, joint, prefix)].rotation_quaternion = Quaternion(matrix_to_quaternion(local))
            for mesh in meshes:
                mesh.hide_render = False
            bpy.context.view_layer.update()
            previews = []
            points = []
            depsgraph = bpy.context.evaluated_depsgraph_get()
            for source in meshes:
                evaluated = source.evaluated_get(depsgraph)
                data = bpy.data.meshes.new_from_object(evaluated)
                data.transform(source.matrix_world)
                bm = bmesh.new()
                bm.from_mesh(data)
                bmesh.ops.bisect_plane(bm, geom=list(bm.verts) + list(bm.edges) + list(bm.faces),
                    plane_co=wrist_world - axis * (length * .14), plane_no=axis,
                    clear_inner=True, clear_outer=False, dist=1e-6)
                bm.to_mesh(data)
                bm.free()
                source.hide_render = True
                if not len(data.polygons):
                    bpy.data.meshes.remove(data)
                    continue
                for polygon in data.polygons:
                    polygon.use_smooth = True
                obj = bpy.data.objects.new("Hand preview crop", data)
                scene.collection.objects.link(obj)
                previews.append(obj)
                points.extend(v.co.copy() for v in data.vertices)
            if not points:
                raise RuntimeError(f"Empty hand crop: {preset_id}/{side}")
            # Fit the actual posed surface, including bent fingertips and wrist.
            coords = [(p.dot(right), p.dot(up), p.dot(direction)) for p in points]
            low = np.min(coords, axis=0)
            high = np.max(coords, axis=0)
            center = (low + high) / 2
            camera.location = right * center[0] + up * center[1] + direction * (high[2] + length * 5)
            camera_data.ortho_scale = float(max(high[0] - low[0], high[1] - low[1]) * 1.22)
            name = f"{preset_id}__{side}.png"
            scene.render.filepath = str(args.out / name)
            bpy.ops.render.render(write_still=True)
            files[name] = hashlib.sha256((args.out / name).read_bytes()).hexdigest()
            for obj in previews:
                data = obj.data
                bpy.data.objects.remove(obj, do_unlink=True)
                bpy.data.meshes.remove(data)
            print(f"HAND_THUMBNAIL {preset_id} {side}", flush=True)
    if not args.presets:
        manifest = dict(library_version=library.library_version, library_sha256=library.sha256,
            size_css_px=args.size // 2, scale=2,
            view="palm-oblique, fingers-up; thumb-side for closed grips",
            renderer="default-human-hand-mesh-v1", style=STYLE,
            character_id="standin-master-v2", character_sha256=character_sha, files=files)
        pending = args.out / "manifest.json.tmp"
        pending.write_text(json.dumps(manifest, indent=1) + "\n")
        pending.replace(args.out / "manifest.json")
    print(f"Rendered {len(files)} hand thumbnails", flush=True)


if __name__ == "__main__":
    main()
