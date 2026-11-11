"""Render only the immutable, verified final FBX; never retarget or export it."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def configure_scene_camera(scene, camera):
    # Scene FBXs are calibrated with horizontal sensor fit. FBX import resets it
    # to AUTO, which chooses a vertical fit for portrait panels and crops actors.
    # Restore the exporter convention before rendering the same immutable FBX.
    camera.data.sensor_fit = "HORIZONTAL"
    scene.camera = camera
    width = float(camera.get("source_image_width", 512))
    height = float(camera.get("source_image_height", 512))
    scene.render.resolution_x = max(1, round(512 * width / max(width, height)))
    scene.render.resolution_y = max(1, round(512 * height / max(width, height)))
    scene.render.resolution_percentage = 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    job_path = Path(args.job).resolve(strict=True)
    root = job_path.parent
    job = json.loads(job_path.read_text())
    source = root / "final.fbx"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != job["artifact_sha256"]:
        raise ValueError("final FBX changed")
    from converter import retarget as rt
    from converter.framing import render_preview

    rt.reset_scene()
    output = root / "preview.png"
    if job.get("scene_camera"):
        import bpy
        from converter.preview_style import configure_workbench

        bpy.ops.import_scene.fbx(
            filepath=str(source),
            ignore_leaf_bones=True,
            automatic_bone_orientation=False,
        )
        cameras = [obj for obj in bpy.context.scene.objects if obj.type == "CAMERA"]
        if len(cameras) != 1:
            raise ValueError("expected verified scene camera")
        scene = bpy.context.scene
        configure_scene_camera(scene, cameras[0])
        configure_workbench(scene)
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = str(output)
        bpy.ops.render.render(write_still=True)
    else:
        _, meshes = rt.import_character(str(source))
        render_preview(meshes, output, "front", aligned=job["aligned"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError("preview modified final FBX")
    data = output.read_bytes()
    (root / "report.json").write_text(
        json.dumps(
            {
                "ok": True,
                "artifact_sha256": digest,
                "job_sha256": hashlib.sha256(job_path.read_bytes()).hexdigest(),
                "preview_sha256": hashlib.sha256(data).hexdigest(),
                "preview_size": len(data),
            }
        )
    )


if __name__ == "__main__":
    main()
