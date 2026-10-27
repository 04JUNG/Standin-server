"""One private local job per Blender process; only the host reads its result."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from converter.local.contract import LocalError, versions, sha256  # noqa: E402


def run(job):
    import bpy
    from converter import retarget as rt
    from converter.local import inspection

    if bpy.app.version[:3] != (5, 2, 0):
        raise LocalError("RUNTIME_MISMATCH", "This local converter requires Blender 5.2.0")
    if not (sys.dont_write_bytecode and sys.flags.no_user_site and sys.flags.safe_path):
        raise LocalError("RUNTIME_MISMATCH", "The local child must disable bytecode writes and unsafe Python paths")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    work = Path(job["workDir"])
    if job["action"] == "import":
        source = Path(job["sourcePath"])
        if source.suffix.lower() == ".cs3c":
            from converter.local.cs3c import import_character
            import_character(source)
        arm, meshes = inspection.import_fbx(source)
        character, capabilities = inspection.inspect(arm, meshes)
        # .blend is the common representation, not a lossy preview GLB. It keeps
        # original hierarchy, bind/rest transforms, mesh transforms and weights.
        normalized = work / "character.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(normalized), check_existing=False)
        preview = work / "preview.glb"
        try:
            bpy.ops.export_scene.gltf(filepath=str(preview), export_format="GLB", export_animations=False,
                                      export_extras=False)
        except Exception:
            preview.unlink(missing_ok=True)
        validation = inspection.roundtrip(arm, meshes, work / "registration-check.fbx", character["mapping"])
        return {"schemaVersion": 1, "localModelId": job["localModelId"],
                "sourceHash": job["sourceHash"], "sourceFormat": source.suffix[1:].lower(),
                "versions": versions(), "character": character, "capabilities": capabilities,
                "normalizedHash": sha256(normalized), "previewPath": "preview.glb" if preview.is_file() else None,
                "previewHash": sha256(preview) if preview.is_file() else None, "validation": validation}
    bpy.ops.wm.open_mainfile(filepath=job["normalizedPath"], load_ui=False, use_scripts=False)
    arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    inspection.require(len(arms) == 1, "UNSUPPORTED_RIG", "Stored character is invalid")
    arm = arms[0]
    character, _ = inspection.inspect(arm, meshes)
    src = rt.import_bvh(job["posePath"], frame=0)
    inspection.require(len(src.data.bones) <= 512, "LIMIT_EXCEEDED", "Pose skeleton exceeds the local limit")
    report = rt.retarget(src, arm, dst_profile=character["profile"], apply_root_translation=False)
    inspection.require(report.ok, "CONVERSION_FAILED", "Retarget rejected the selected pose")
    bpy.data.objects.remove(src, do_unlink=True)
    rt.apply_output_mode(arm, meshes, "rigged_rest")
    validation = inspection.roundtrip(arm, meshes, work / "result.fbx", character["mapping"])
    return {"snapshot": job["snapshot"], "snapshotHash": job["snapshotHash"],
            "artifactHash": sha256(work / "result.fbx"), "validation": validation,
            "retarget": report.as_dict(), "selectionId": job["selectionId"]}


def main():
    job = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())
    try:
        result = {"ok": True, "result": run(job)}
    except LocalError as exc:
        result = {"ok": False, "error": exc.payload()}
    except Exception:
        # Blender traces contain model names and local paths. Keep private child
        # diagnostics out of IPC, telemetry and persisted logs.
        result = {"ok": False, "error": LocalError("CONVERSION_FAILED", "Local conversion failed").payload()}
    Path(job["resultPath"]).write_text(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
