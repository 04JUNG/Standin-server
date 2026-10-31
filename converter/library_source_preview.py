"""Offline native library renderer using the same source fingers as live FBX."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from converter.source_fingers import VERSION as FINGER_VERSION

METHOD = "frozen-native-source-fingers-v1"
SCOPES = ("full", "half", "bust", "shoulder_head")
VIEWS = tuple(f"yaw_{a:03d}" for a in range(0, 360, 45))


def revision():
    root = Path(__file__).parent
    files = ("library_source_preview.py", "source_fingers.py", "framing.py",
             "preview_style.py", "preview_studio.sl", "protocol.py",
             "convert.py", "retarget.py", "bone_map.py", "preview_model.py", "SHA256SUMS.v325")
    return hashlib.sha256(b"".join((root / n).read_bytes().replace(b"\r\n", b"\n")
                                   for n in files)).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    pending.replace(path)


def render_pose(job):
    import bpy
    import numpy as np
    from mathutils import Vector
    from converter import retarget as rt
    from converter.framing import crop_meshes
    from converter.preview_style import configure_workbench
    from converter.protocol import EXPECTED_BLENDER_BUILD_HASH, SOLVER_MANIFEST_SHA256
    from converter.source_fingers import convert_source_pose
    from converter.preview_model import bake_fbx, model_key

    if bpy.app.build_hash.decode() != EXPECTED_BLENDER_BUILD_HASH:
        raise ValueError("library renderer requires the pinned Blender build")
    source, character = Path(job["bvh"]), Path(job["character"])
    digest = sha(source)
    character_digest = sha(character)
    if digest != job["source_bvh_sha256"]:
        raise ValueError("source BVH changed before rendering")
    output = Path(job["out"])
    folder = output / "reports" / digest
    folder.mkdir(parents=True, exist_ok=True)
    images = output / "thumbs" / "scope8"
    images.mkdir(parents=True, exist_ok=True)
    version = revision()
    report_path = folder / "report.json"
    if report_path.exists():
        prior = json.loads(report_path.read_text())
        model = prior.get("preview_model") or {}
        if (prior.get("ok") and prior.get("renderer_revision") == version
                and prior.get("source_bvh_sha256") == digest
                and prior.get("character_sha256") == character_digest
                and model.get("key") == model_key(digest, character_digest)
                and (output / model.get("path", "missing")).is_file()
                and sha(output / model["path"]) == model.get("sha256")
                and sha((output / model["path"]).with_suffix(".json")) == model.get("metadata_sha256")
                and len(prior.get("images", [])) == len(SCOPES) * len(VIEWS)
                and all((output / im["path"]).is_file()
                        and sha(output / im["path"]) == im["image_sha256"]
                        and sha((output / im["path"]).with_suffix(".json")) == im["metadata_sha256"]
                        for im in prior["images"])):
            return {**prior, "reused": True}
    fbx = folder / "human.fbx"
    conversion = convert_source_pose(bvh_path=str(source), character_fbx=str(character),
                                     out_path=str(fbx), embed_textures=False)
    if not conversion.ok:
        raise ValueError("body conversion failed")
    converted = conversion.as_dict()
    save(folder / "conversion.json", converted)
    rt.reset_scene()
    arm, originals = rt.import_character(str(fbx))
    for obj in originals:
        obj.hide_render = True
    records = []
    for scope in SCOPES:
        copies = []
        for original in originals:
            obj = original.copy()
            obj.data = original.data.copy()
            bpy.context.collection.objects.link(obj)
            obj.hide_render = False
            copies.append(obj.name)
        crop = crop_meshes(arm, [bpy.data.objects[n] for n in copies], scope)
        meshes = [bpy.data.objects[n] for n in copies if n in bpy.data.objects]
        points = [obj.matrix_world @ v.co for obj in meshes for v in obj.data.vertices]
        point_array = np.array([tuple(p) for p in points])
        center = Vector([(min(p[i] for p in points) + max(p[i] for p in points)) / 2
                         for i in range(3)])
        for view in VIEWS:
            angle = int(view[4:])
            yaw = math.radians(angle)
            direction = Vector((math.sin(yaw), -math.cos(yaw), 0))
            bpy.ops.object.camera_add()
            camera = bpy.context.object
            camera.location = center + direction * 10
            camera.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()
            camera.data.type = "ORTHO"
            camera.data.clip_end = 1000
            bpy.context.view_layer.update()
            basis = np.array([tuple(row) for row in camera.rotation_euler.to_matrix()])
            local = (point_array - np.array(tuple(camera.location))) @ basis[:, :2]
            low, high = local.min(axis=0), local.max(axis=0)
            camera.data.ortho_scale = max(high[i] - low[i] for i in range(2)) * 1.14
            camera.location += camera.rotation_euler.to_matrix() @ Vector(
                ((low[0] + high[0]) / 2, (low[1] + high[1]) / 2, 0))
            scene = bpy.context.scene
            scene.camera = camera
            configure_workbench(scene)
            shading = scene.display.shading
            shading.single_color = (.82, .84, .88)
            shading.show_cavity = True
            shading.cavity_type = "BOTH"
            shading.background_type = "VIEWPORT"
            shading.background_color = (.045, .055, .075)
            scene.display.render_aa = "16"
            scene.render.resolution_x = scene.render.resolution_y = 512
            scene.render.resolution_percentage = 100
            scene.render.image_settings.file_format = "JPEG"
            scene.render.image_settings.color_mode = "RGB"
            scene.render.image_settings.quality = 90
            scene.render.film_transparent = False
            target = images / f"{digest}__{scope}__{view}.jpg"
            scene.render.filepath = str(target)
            for _ in range(5):
                bpy.ops.render.render(write_still=True)
                raster = bpy.data.images.load(str(target), check_existing=False)
                try:
                    pixels = np.empty(512 * 512 * 4, dtype=np.float32)
                    raster.pixels.foreach_get(pixels)
                    pixels = pixels.reshape(512, 512, 4)[:, :, :3]
                    background = np.median([pixels[0, 0], pixels[0, -1],
                                            pixels[-1, 0], pixels[-1, -1]], axis=0)
                    yy, xx = np.where(np.linalg.norm(pixels - background, axis=2) > .09)
                    if len(xx) < 2621:
                        raise ValueError("empty/insufficient thumbnail foreground")
                    border = int(min(xx.min(), yy.min(), 511 - xx.max(), 511 - yy.max()))
                finally:
                    bpy.data.images.remove(raster)
                if border >= 12:
                    break
                camera.data.ortho_scale *= 1.2
            else:
                raise ValueError("thumbnail touches image border")
            metadata = dict(schema_version=2, render_method=METHOD, renderer_revision=version,
                source_bvh_sha256=digest, character_id="standin-master-v2",
                character_sha256=character_digest, scope=scope, view=view, azimuth_degrees=angle,
                native_camera_yaw=(180-angle) % 360, pitch_degrees=0, projection="orthographic",
                projection_matches_search=False, selected_body=False, hands_applied=False,
                source_fingers_version=FINGER_VERSION, source_fingers=converted["source_fingers"],
                source_frame=0, preview_only=True, anatomy_approval=False, width=512, height=512,
                format="jpeg", quality=90, style="approved-anatomy-detail-dark-5scope-8view-v1",
                solver_manifest_sha256=SOLVER_MANIFEST_SHA256, framing=crop,
                border_pixels=border, image_sha256=sha(target), bytes=target.stat().st_size)
            save(target.with_suffix(".json"), metadata)
            records.append(dict(scope=scope, view=view, path=str(target.relative_to(output)),
                image_sha256=metadata["image_sha256"], metadata_sha256=sha(target.with_suffix(".json")),
                renderer_revision=version))
            bpy.data.objects.remove(camera, do_unlink=True)
        for obj in meshes:
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if data.users == 0:
                bpy.data.meshes.remove(data)
    if sha(source) != digest:
        raise ValueError("source BVH changed during rendering")
    if sha(character) != character_digest:
        raise ValueError("character changed during rendering")
    key = model_key(digest, character_digest)
    glb = output / "pose-previews" / (key + ".glb")
    glb.parent.mkdir(parents=True, exist_ok=True)
    preview = bake_fbx(fbx, glb, dict(source_bvh_sha256=digest,
        character_sha256=character_digest, character_id="standin-master-v2"))
    save(glb.with_suffix(".json"), preview)
    report = dict(ok=True, pose_id=job["pose_id"], source_bvh_sha256=digest,
        renderer_revision=version, character_sha256=character_digest, fbx_sha256=sha(fbx),
        source_fingers=converted["source_fingers"], images=records,
        preview_model=dict(key=key, path=str(glb.relative_to(output)), sha256=sha(glb),
                           metadata_sha256=sha(glb.with_suffix(".json"))))
    save(report_path, report)
    return report


if __name__ == "__main__":
    import traceback
    jobs = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())
    failed = 0
    for job in jobs:
        try:
            result = render_pose(job)
            print("SOURCE_HANDS_PREVIEW", result["pose_id"], len(result["images"]), flush=True)
        except Exception as exc:
            failed += 1
            traceback.print_exc()
            folder = Path(job["out"]) / "reports" / job["source_bvh_sha256"]
            folder.mkdir(parents=True, exist_ok=True)
            save(folder / "report.json", dict(ok=False, pose_id=job["pose_id"],
                source_bvh_sha256=job["source_bvh_sha256"], renderer_revision=revision(),
                error=f"{type(exc).__name__}: {exc}"))
    raise SystemExit(1 if failed else 0)
