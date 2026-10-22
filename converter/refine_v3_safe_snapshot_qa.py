"""Blender-only, finite actual FBX safety QA. No production registration.

Makes a new derived neutral asset to fix demonstrated local source crossings,
then runs the real frozen body converter and requested hand presets. Original
rest normals/areas survive repair/body/hand stages; output normals are never a
reference. An explicit --neutral-source flag creates a QA posture, never changes
a selected product pose. All artifacts stay in a fresh output directory.
"""
from pathlib import Path
from dataclasses import asdict
import argparse
import hashlib
import json
import shutil
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_RUN_OUTPUT = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(path):
    import numpy as np
    from converter import retarget as rt
    from converter.mesh_validation import extract_evaluated_skin
    rt.reset_scene()
    arm, meshes = rt.import_character(str(path))
    if len(meshes) != 1:
        raise ValueError("finite_skin_profile_requires_single_mesh")
    mesh = meshes[0]
    if (len(mesh.modifiers) != 1
            or any(mod.type != "ARMATURE" or mod.use_deform_preserve_volume
                   or mod.use_bone_envelopes or not mod.use_vertex_groups or mod.vertex_group
                   for mod in mesh.modifiers)):
        raise ValueError("finite_snapshot_modifier_unsupported")
    vertices, faces = extract_evaluated_skin(meshes)
    names = sorted(bone.name for bone in arm.data.bones)
    indices = {name: i for i, name in enumerate(names)}
    groups = {g.index: g.name for g in mesh.vertex_groups}
    weights = np.zeros((len(vertices), len(names)))
    for vertex in mesh.data.vertices:
        for group in vertex.groups:
            if groups[group.group] not in indices:
                raise ValueError("unmapped_skin_group")
            weights[vertex.index, indices[groups[group.group]]] = group.weight
    bones = np.asarray([arm.matrix_world @ arm.data.bones[name].matrix_local for name in names], float)
    return {"vertices": np.asarray(vertices), "faces": np.asarray(faces), "weights": weights,
            "bones": bones, "names": names,
            "parents": {b.name: b.parent.name if b.parent else None for b in arm.data.bones}}


def same_rig(before, after, *, require_rest=False):
    import numpy as np
    if (before["names"] != after["names"] or before["parents"] != after["parents"]
            or not np.array_equal(before["faces"], after["faces"])
            or np.max(abs(before["weights"]-after["weights"])) > 1e-5
            or (require_rest and np.max(abs(before["bones"]-after["bones"])) > 1e-5)):
        raise ValueError("snapshot_rig_topology_or_weights_changed")


def main():
    global _RUN_OUTPUT
    import bpy
    import numpy as np
    from mathutils import Vector
    from converter import retarget as rt
    from converter.shape_bake import import_full_character, _load
    from converter.shape_spec import resolve_shape
    from converter.convert import convert
    from converter.hands import bake_hands
    from converter.hand_library import load_library
    from converter.worker import _frozen_lineage, EXPECTED_BLENDER_BUILD_HASH, SOLVER_VERSION
    from converter.neutral_mesh_repair import repair_neutral, VERSION as REPAIR_VERSION
    from converter.mesh_validation import audit_mesh, MeshPolicy, topology_sha256
    from converter.deformation_witness import build_lbs_witness, compose_lbs_witness, verify_witness
    from converter.final_artifact import validate
    from converter.artifact_receipt import issue_receipt, verify_receipt
    from src.target_profile import TargetProfile, _digest
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--character", required=True)
    p.add_argument("--character-id", required=True)
    p.add_argument("--shape-registry", required=True)
    p.add_argument("--source-bvh", required=True)
    p.add_argument("--neutral-source", action="store_true")
    p.add_argument("--hand-left", default="open")
    p.add_argument("--hand-right", default="open")
    p.add_argument("--output-dir", required=True)
    a = p.parse_args(sys.argv[sys.argv.index("--")+1:])
    if bpy.app.build_hash.decode() != EXPECTED_BLENDER_BUILD_HASH:
        raise ValueError("unsupported_blender_build")
    lineage = _frozen_lineage()
    output = Path(a.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    _RUN_OUTPUT = output
    character = Path(a.character).resolve()
    original_asset_sha = sha(character)
    import os
    os.environ["CONVERTER_SHAPE_REGISTRY"] = str(Path(a.shape_registry).resolve())
    recipe = resolve_shape(a.character_id, original_asset_sha)
    if recipe is None:
        raise ValueError("actual_native_shape_recipe_required")
    # Existing shape terminal-bone policy, including weighted Head/ToeBase leaves.
    rt.import_character = import_full_character
    controller = _load(recipe)
    native_neutral = controller.export_fixed(str(output/"native-neutral")) / "model.fbx"
    neutral = snapshot(native_neutral)
    original_faces = neutral["faces"]
    neutral_normals = np.cross(neutral["vertices"][original_faces[:, 1]]-neutral["vertices"][original_faces[:, 0]],
                              neutral["vertices"][original_faces[:, 2]]-neutral["vertices"][original_faces[:, 0]])
    original_areas = np.linalg.norm(neutral_normals, axis=1)
    # This released pure mannequin has one weighted skin mesh, no garments/hair.
    regions = tuple("skin:"+neutral["names"][i] for i in np.argmax(neutral["weights"][original_faces].mean(1), axis=1))
    shape_spec = {key: recipe[key] for key in ("values", "rules", "blend_sha256", "controller_sha256")}
    baseline_profile = _digest({"original_asset_sha256": original_asset_sha, "shape": shape_spec,
                                "repair_policy": REPAIR_VERSION})
    topology = topology_sha256(original_faces, len(neutral["vertices"]))
    def mesh_check(vertices, *, profile=baseline_profile, directors=neutral_normals, areas=original_areas):
        policy = MeshPolicy(profile, topology, "finite-original-rest-directors-v2", regions)
        return audit_mesh(vertices, original_faces, policy=policy, profile_sha256=profile,
                          expected_normals=directors, reference_double_areas=areas)
    baseline_audit = mesh_check(neutral["vertices"])
    repaired, repair = repair_neutral(neutral["vertices"], original_faces, baseline_audit, mesh_check)
    # Never modify the source/release asset. Only this newly identified asset changes.
    rt.reset_scene()
    _, meshes = rt.import_character(str(native_neutral))
    mesh = meshes[0]
    for index in repair["changed_vertices"]:
        mesh.data.vertices[index].co = mesh.matrix_world.inverted() @ Vector(repaired[index])
    mesh.data.update()
    bpy.context.view_layer.update()
    derived = output/"derived-neutral.fbx"
    rt.export_fbx(str(derived), embed_textures=False)
    derived_sha = sha(derived)
    rest = snapshot(derived)
    same_rig(neutral, rest, require_rest=True)
    source = output/"source.bvh"
    if a.neutral_source:
        from src.bvh import parse_bvh, single_frame_bvh_text
        _, frames = parse_bvh(a.source_bvh)
        source.write_text(single_frame_bvh_text(a.source_bvh, np.zeros_like(frames[0])))
    else:
        shutil.copyfile(a.source_bvh, source)
    source_sha = sha(source)
    hands = {"left": a.hand_left, "right": a.hand_right}
    target = TargetProfile(a.character_id+".offline-neutral-repair-v1", original_asset_sha,
        _digest({"native_shape": shape_spec, "derived_neutral_sha256": derived_sha, "repair": repair}),
        _digest(rest["parents"]), _digest(rest["bones"].tolist()), sha(ROOT/"converter/bone_map.py"),
        _digest({"hands": hands, "library_sha256": load_library().sha256}), lineage["retarget_sha256"],
        lineage["solver_manifest_sha256"], recipe["controller_sha256"], "finite-snapshot-1", "hands-1",
        "finite-normalized-lbs-shell-witness-v1", SOLVER_VERSION, bpy.app.build_hash.decode())
    identity = dict(source_asset_sha256=original_asset_sha, profile_sha256=target.digest, source_sha256=source_sha)
    witness = build_lbs_witness(neutral_vertices=neutral["vertices"], corrected_rest_vertices=repaired,
        faces=original_faces, vertex_weights=neutral["weights"],
        bone_world_transforms=np.repeat(np.eye(4)[None], len(rest["names"]), axis=0),
        mesh_world_transform=np.eye(4), stage="neutral-repair", stage_input_sha256=sha(native_neutral), **identity)
    stages = []
    def validate_stage(path, data, witness):
        replay = verify_witness(witness, vertices=data["vertices"], faces=data["faces"],
                                stage_input_sha256=witness.stage_input_sha256, **identity)
        audit = mesh_check(data["vertices"], profile=target.digest,
                           directors=witness.expected_directors, areas=witness.reference_double_areas)
        audit.update(full_body_artifact_sha256=sha(path), deformation_witness_sha256=witness.digest,
                     fold_reference_kind="independent-original-rest-lbs-three-directors")
        row = {"stage": witness.stage, "artifact_sha256": sha(path), "witness": witness.manifest(),
               "witness_sha256": witness.digest, "replay": replay, "mesh_audit": audit}
        stages.append(row)
        (output/"stages.json").write_text(json.dumps(stages, indent=2))
        if audit["status"] != "mesh_validated":
            raise ValueError("actual_stage_mesh_not_safe:"+witness.stage)
        return audit
    validate_stage(derived, rest, witness)
    body_path = output/"body.fbx"
    # Shape was really baked above; regenerating it here would discard the repair.
    converted = convert(bvh_path=str(source), character_fbx=str(derived), out_path=str(body_path), embed_textures=False)
    (output/"body-report.json").write_text(json.dumps(converted.as_dict(), indent=2))
    if not converted.ok:
        raise ValueError("existing_converter_gate_rejected_source")
    body = snapshot(body_path)
    same_rig(rest, body)
    witness = compose_lbs_witness(witness, faces=original_faces, vertex_weights=rest["weights"],
        bone_world_transforms=body["bones"] @ np.linalg.inv(rest["bones"]),
        stage="body", stage_input_sha256=derived_sha)
    validate_stage(body_path, body, witness)
    final = output/"final.fbx"
    shutil.copyfile(body_path, final)
    hand_report = bake_hands(final, hands)
    actual = snapshot(final)
    same_rig(body, actual)
    witness = compose_lbs_witness(witness, faces=original_faces, vertex_weights=body["weights"],
        bone_world_transforms=actual["bones"] @ np.linalg.inv(body["bones"]),
        stage="hands", stage_input_sha256=sha(body_path))
    audit = validate_stage(final, actual, witness)
    integrity = validate(final, "full", len(rest["names"]))
    context = _digest({"profile": target.digest, "source": source_sha,
                       "hand_witness": witness.digest, "scope": "full", "revision": "1"})
    receipt = issue_receipt(final.read_bytes(), source_sha256=source_sha, context_sha256=context,
        profile_sha256=target.digest, revision="1", scope="full", mesh_audit=audit, integrity_audit=integrity)
    if not verify_receipt(receipt, final.read_bytes(), revision="1", context_sha256=context, profile_sha256=target.digest):
        raise ValueError("final_receipt_mismatch")
    if sha(character) != original_asset_sha:
        raise ValueError("original_asset_changed")
    result = {"schema": "finite-actual-fbx-safety-qa-v1", "profile": asdict(target), "profile_sha256": target.digest,
        "original_asset_sha256": original_asset_sha, "native_neutral_sha256": sha(native_neutral),
        "derived_neutral_sha256": derived_sha, "source_sha256": source_sha,
        "neutral_qa_source": a.neutral_source, "source_template_sha256": sha(a.source_bvh),
        "original_neutral_audit": baseline_audit, "repair": repair,
        "stages": stages, "hands": hand_report, "mesh_audit": audit, "artifact_integrity": integrity,
        "receipt": asdict(receipt), "receipt_sha256": receipt.digest, "artifact_sha256": sha(final),
        "artifact_size": final.stat().st_size, "safe_final_snapshot_count": 1,
        "production_registered": False, "continuous_fk_certified": False, "deployed": False}
    (output/"qa-report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"mesh_status": audit["status"], "artifact_sha256": sha(final),
                      "receipt_sha256": receipt.digest, "production_registered": False}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Failed actual rough poses need reviewable evidence too. Never issue a
        # receipt, turn an exception into success, or overwrite a prior run.
        if _RUN_OUTPUT is not None:
            stage_file = _RUN_OUTPUT/"stages.json"
            stages = json.loads(stage_file.read_text()) if stage_file.exists() else []
            hard = str(error) in {"discrete_shell_orientation_reversed",
                                  "discrete_shell_area_collapse", "nonpositive_skin_determinant"}
            status = "validation_blocked" if hard else "mesh_unverified"
            if stages and stages[-1]["mesh_audit"]["status"] != "mesh_validated":
                status = stages[-1]["mesh_audit"]["status"]
            failure = {"schema": "finite-actual-fbx-safety-failure-v1", "status": status,
                "reason": str(error), "error_type": type(error).__name__,
                "last_validated_stage": next((row["stage"] for row in reversed(stages)
                    if row["mesh_audit"]["status"] == "mesh_validated"), None),
                "artifacts": {name: sha(_RUN_OUTPUT/name) for name in
                    ("source.bvh", "native-neutral/model.fbx", "derived-neutral.fbx", "body.fbx", "final.fbx")
                    if (_RUN_OUTPUT/name).is_file()},
                "final_receipt_issued": False, "production_registered": False, "deployed": False}
            (_RUN_OUTPUT/"qa-failure.json").write_text(json.dumps(failure, indent=2))
        raise
