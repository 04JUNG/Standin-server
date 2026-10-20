"""Blender entry point. Communicates through JSON files, never HTTP or AWS."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time
import traceback

# Blender runs this file as a script, outside the application's Python runtime.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bpy

from converter import retarget
from converter.convert import convert
from pose_curation.rendering.profiles import register
from pose_curation.rendering.fingers import apply_channels, rest_bases
from pose_curation.rendering.scene import render_views
from pose_curation.rendering.anatomy import rest_frames, inspect
from pose_curation.rendering.props import add_guides
from pose_curation.storage import sha256, write_json


def render_one(job: dict, pose: dict, finger_bases: dict, body_rest: dict) -> dict:
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="pose-review-") as temp:
        fbx = str(Path(temp) / "converted.fbx")
        report = convert(bvh_path=pose["bvh"], character_fbx=job["character"],
                         out_path=fbx, src_profile=pose.get("rig_profile", "100style"), dst_profile="mixamo",
                         embed_textures=False)
        payload = report.as_dict()
        if not report.ok:
            raise ValueError(f"retarget failed: {payload}")
        # A safety rollback may restore the FBX without restoring the scene.
        # Always render the final exported artifact, including any rollback.
        retarget.reset_scene()
        arm, _ = retarget.import_character(fbx)
        fingers = apply_channels(arm, pose["bvh"], finger_bases)
        anatomy = inspect(arm, body_rest)
        guides = add_guides(arm, pose.get('prop_guides', ''), pose.get('support', ''))
        images = render_views(Path(job["output_dir"]), pose["pose_id"], resolution=512)
    return {"pose_id": pose["pose_id"], "bvh_sha256": pose["bvh_sha256"],
            "thumbnails": {view: {"path": str(path), "sha256": sha256(path)} for view, path in images.items()},
            "conversion": payload, "fingers": fingers, "anatomy": anatomy, "prop_guides": guides,
            "scene_spec": {k: pose.get(k, '') for k in ('prop_guides', 'support')}, "blender_version": bpy.app.version_string,
            "blender_build_hash": bpy.app.build_hash.decode(), "seconds": round(time.monotonic() - started, 3)}


def main() -> None:
    job = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text(encoding="utf-8"))
    if bpy.app.version[:3] != (5, 2, 0):
        raise RuntimeError("character previews require Blender 5.2.0, matching the converter")
    register()
    finger_bases = rest_bases(job["character"])
    body_rest = rest_frames(retarget._armatures()[0])
    for index, pose in enumerate(job["poses"], 1):
        result_path = Path(job["result_dir"]) / f"{pose['pose_id']}.json"
        try:
            result = {"ok": True, **render_one(job, pose, finger_bases, body_rest)}
        except Exception as exc:
            result = {"ok": False, "pose_id": pose["pose_id"], "error": f"{type(exc).__name__}: {exc}"}
            traceback.print_exc()
        result["render_fingerprint"] = job["fingerprint"]
        write_json(result_path, result)
        print(f"PREVIEW {index}/{len(job['poses'])} {pose['pose_id']} ok={result['ok']}", flush=True)


if __name__ == "__main__":
    main()
