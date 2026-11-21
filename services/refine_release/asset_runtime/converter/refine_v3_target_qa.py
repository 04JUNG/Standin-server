"""Offline Blender QA for an exact character/shape/hands snapshot.

Run with Blender --background --python-exit-code 1 --python this_file -- --bvh ... --character ...
--character-id ... --output-dir /private/tmp/new-dir [--shape-registry ...]
This runs the real frozen worker, full-body shadow audit, and FBX roundtrip. It
DOES NOT register a fast FK adapter or issue a product mesh safety certificate.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

# External authoring controllers and assets are read-only QA inputs.
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(path):
    import numpy as np
    from converter import retarget as rt
    from converter.mesh_validation import extract_evaluated_skin, topology_sha256
    from converter.bone_map import resolve_profile, PROFILES
    rt.reset_scene()
    arm, meshes = rt.import_character(str(path))
    vertices, triangles = extract_evaluated_skin(meshes)
    profile = resolve_profile([b.name for b in arm.data.bones])
    joints = {key: list(arm.matrix_world @ arm.data.bones[name].head_local)
              for key, name in PROFILES[profile].items() if name in arm.data.bones}
    bones = {b.name: {"parent": b.parent.name if b.parent else None,
                      "matrix": [list(row) for row in b.matrix_local]}
             for b in arm.data.bones}
    return {"rig_profile": profile, "joints_m": joints, "bones": bones,
            "vertices": np.asarray(vertices), "triangles": triangles,
            "topology_sha256": topology_sha256(triangles, len(vertices))}


def main():
    import numpy as np
    import bpy
    from converter import retarget as rt
    from converter.worker import run_job, load_job
    from converter.protocol import JOB_SCHEMA_VERSION
    from converter.shape_spec import resolve_shape
    from converter.final_artifact import validate
    from src.target_profile import TargetProfile, _digest
    p = argparse.ArgumentParser()
    p.add_argument("--bvh", required=True)
    p.add_argument("--character", required=True)
    p.add_argument("--character-id", required=True)
    p.add_argument("--shape-registry")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--hand-left", default="relaxed")
    p.add_argument("--hand-right", default="relaxed")
    a = p.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    output = Path(a.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source.bvh"
    source.write_bytes(Path(a.bvh).read_bytes())
    character = Path(a.character).resolve()
    asset_sha = digest(character)
    if a.shape_registry:
        os.environ["CONVERTER_SHAPE_REGISTRY"] = str(Path(a.shape_registry).resolve())
    recipe = resolve_shape(a.character_id, asset_sha)
    os.environ["CONVERTER_REFINE_V3_MESH_SHADOW"] = "1"
    job = dict(schema_version=JOB_SCHEMA_VERSION, conversion_id="refine-v3-offline-qa",
               temp_dir=str(output), bvh_path=str(source), character_fbx=str(character),
               character_id=a.character_id, character_sha256=asset_sha,
               output_path=str(output/"final.fbx"), report_path=str(output/"worker-report.json"),
               frame=0, mirror=False, output_mode="rigged_rest", apply_root_translation=False,
               embed_textures=False, force_exact_v324=False, thumbnail=None,
               hands={"left": a.hand_left, "right": a.hand_right})
    if recipe is not None:
        job["shape_recipe"] = recipe
    job_path = output/"job.json"
    job_path.write_text(json.dumps(job))
    report = run_job(load_job(job_path))
    (output/"worker-report.json").write_text(json.dumps(report, indent=2))
    if not report.get("ok"):
        raise RuntimeError("actual Converter rejected input")
    original_digest = digest(output/"final.fbx")
    before = snapshot(output/"final.fbx")
    rt.export_fbx(str(output/"roundtrip.fbx"), embed_textures=False)
    after = snapshot(output/"roundtrip.fbx")
    same_topology = before["topology_sha256"] == after["topology_sha256"]
    joint_error = max(float(np.linalg.norm(np.asarray(value)-after["joints_m"][key]))
                      for key, value in before["joints_m"].items())
    vertex_error = (float(np.max(np.linalg.norm(before["vertices"]-after["vertices"], axis=1)))
                    if same_topology else None)
    shape_hash = _digest({"values": recipe["values"], "rules": recipe["rules"],
                          "blend_sha256": recipe["blend_sha256"]}) if recipe else _digest({"shape": "asset-default"})
    if recipe:
        neutral_candidates = [p for p in (output/"neutral").rglob("model.fbx")
                              if digest(p) == report["shape_bake"]["neutral_fbx_sha256"]]
        if len(neutral_candidates) != 1:
            raise ValueError("neutral shape rest artifact identity mismatch")
        neutral_path = neutral_candidates[0]
    else:
        neutral_path = character
    neutral = snapshot(neutral_path)
    target = TargetProfile(profile_id=a.character_id, character_sha256=asset_sha,
        shape_sha256=shape_hash, rig_sha256=_digest({k: v["parent"] for k,v in neutral["bones"].items()}),
        rest_sha256=_digest(neutral["bones"]), bone_map_sha256=digest(ROOT/"converter/bone_map.py"),
        hand_sha256=_digest({"requested": job["hands"], "library_sha256": report["hands"]["library_sha256"]}),
        retarget_sha256=report["retarget_sha256"],
        converter_manifest_sha256=report["solver_manifest_sha256"],
        controller_sha256=recipe["controller_sha256"] if recipe else _digest({"controller": None}),
        revision="offline-snapshot-1", hand_revision="offline-hands-1", adapter_version="converter-reference-only-v1",
        converter_version=report["solver_version"], blender_build_hash=report["blender_build_hash"])
    final_integrity = validate(output/"final.fbx", "full")
    if digest(output/"final.fbx") != original_digest:
        raise RuntimeError("QA modified immutable final bytes")
    result = {"schema": "refine-v3-target-qa-v1", "target_profile": target.__dict__,
              "profile_sha256": target.digest, "source_sha256": digest(source),
              "artifact_sha256": original_digest, "blender_build_hash": bpy.app.build_hash.decode(),
              "joints_world_m": before["joints_m"], "coordinate_frame": "right-handed-z-up",
              "roundtrip_joint_error_m": joint_error, "roundtrip_vertex_error_m": vertex_error,
              "roundtrip_topology_match": same_topology, "artifact_integrity": final_integrity,
              "mesh_audit": report["refine_v3_mesh_shadow"], "shape_applied": recipe is not None,
              "hand_applied_bones": report["hands"]["applied_bones"],
              "fast_fk_equivalence_certified": False, "production_supported": False}
    (output/"qa-report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("artifact_sha256", "roundtrip_joint_error_m",
          "roundtrip_vertex_error_m", "production_supported")}, sort_keys=True))


if __name__ == "__main__":
    main()
