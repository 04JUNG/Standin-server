#!/usr/bin/env python3
"""Local real-rough application: actual refine handler → derived skin → safe FBX.

Replays an existing live analysis, including its exact candidate cameras. It does
not call a VLM, fabricate observed-joint authority, change a user-selected pose
inside /refine, register a production profile, or deploy anything. Candidate
comparison is explicit in this offline job. Only verified final bytes are adopted.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--character", type=Path, required=True)
    p.add_argument("--character-id", required=True)
    p.add_argument("--shape-registry", type=Path, required=True)
    p.add_argument("--hand-left", required=True)
    p.add_argument("--hand-right", required=True)
    p.add_argument("--blender", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--depth-safe", action="store_true", help="Fit the actual target rig with depth/angle/full-mesh guards")
    a = p.parse_args()
    for key in ("analysis", "data_root", "character", "shape_registry", "blender", "output_dir"):
        setattr(a, key, getattr(a, key).resolve())
    a.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ.update(REFINE_ENABLED="1", REFINE_V2_ENABLED="1", REFINE_V3_MODE="off",
        DATA_DIR=str(a.data_root), REFINE_POSE_QUARANTINE_PATH=str(ROOT/"config/refine_pose_quarantine.v1.json"))
    os.chdir(a.data_root.parent)
    import numpy as np
    from api import app as api
    from api.models import RefineRequest
    from src.repo import get_pose_meta
    from src.bvh import parse_bvh, coco17_from_fk, fk
    from src.refine_projection import FixedProjection, LEFT_RIGHT
    from converter.artifact_receipt import issue_receipt, verify_receipt
    api.STATE.update(db_path=str(a.data_root/"poses.db"), pose_library={})
    record = json.loads(a.analysis.read_text())
    people = record["analyze"]["people"]
    if len(people) != 1:
        raise ValueError("offline_apply_requires_one_unambiguous_person")
    person = people[0]
    if (person["coverage_class"] != "full" or person["lower_body_observed"] is not True
            or person["refine_allowed"] is not True):
        raise ValueError("offline_apply_requires_existing_full_body_refine_authority")
    query = np.asarray(person["keypoints"], float)
    scale = float(np.linalg.norm(query[[5,6]].mean(0)-query[[11,12]].mean(0)))
    if query.shape != (17,2) or not np.isfinite(query).all() or scale < 1e-6:
        raise ValueError("invalid_rough_query")
    target = (query-query[[11,12]].mean(0))/scale
    fields = {k: person[k] for k in ("keypoints", "scores", "refine_allowed", "refinable_limbs",
        "lower_body_observed", "skeleton_state", "coverage_class", "slot_origin", "skeleton_source",
        "search_stability", "distance_metric", "confidence_threshold")}
    results = []
    for rank, chosen in enumerate(person["dynamic_search"]["selected"][:5], 1):
        candidate = next((c for c in person["refine_candidates"] if c["pose_id"] == chosen["pose_id"]), None)
        meta = get_pose_meta(api.STATE["db_path"], chosen["pose_id"])
        if (candidate is None or not meta or meta.get("set_id") or chosen.get("review_required") is True
                or chosen.get("constraints_satisfied") is False):
            continue
        camera = candidate["camera"]
        if not all(camera["observed_joint_mask"][5:]):
            continue
        folder = a.output_dir/f"candidate-{rank}"
        folder.mkdir()
        req = RefineRequest(pose_id=chosen["pose_id"], view=candidate["view"],
            search_distance=candidate["distance"], camera=camera, render_thumbnail=False, **fields)
        response = api.refine(req).model_dump(mode="json")
        (folder/"response.json").write_text(json.dumps(response, indent=2))
        base = Path(meta["bvh_path"])
        selected = folder/"selected.bvh"
        selected.write_text(response["bvh"] if response.get("bvh") else base.read_text())
        joints, frames = parse_bvh(str(base))
        baseline, _ = coco17_from_fk(joints, fk(joints, frames[0]))
        rotation = np.asarray(camera["rotation"])
        projection = FixedProjection.from_aligned_base(baseline@rotation.T, camera["distance_torso"])
        joints, frames = parse_bvh(str(selected))
        points, _ = coco17_from_fk(joints, fk(joints, frames[0]))
        xy = projection.project(points@rotation.T).astype(float)
        if camera["labels_swapped"]:
            xy = xy[LEFT_RIGHT]
        xy = (xy-xy[[11,12]].mean(0))/np.linalg.norm(xy[[5,6]].mean(0)-xy[[11,12]].mean(0))
        error = float(np.linalg.norm(xy[5:]-target[5:], axis=1).mean())
        if not np.isfinite(error):
            continue
        results.append(dict(pose_id=chosen["pose_id"], original_rank=rank, source_path=str(selected),
            source_sha256=sha(selected), source_joint_nme=error, camera=camera,
            refined=response["refined"], reason=response["reason"]))
    if not results:
        raise ValueError("no_eligible_actual_candidate")
    results.sort(key=lambda row: (row["source_joint_nme"], row["original_rank"]))
    (a.output_dir/"candidate-comparison.json").write_text(json.dumps(results, indent=2))
    selected = results[0]
    print("SELECTED", selected["pose_id"], selected["source_joint_nme"], flush=True)

    def run(script, args, log, *, blender=False, allow_failure=False):
        command = ([str(a.blender), "-b", "--python-exit-code", "1", "--python", str(ROOT/script), "--"]
                   if blender else [sys.executable, str(ROOT/script)])
        with (a.output_dir/log).open("w") as handle:
            process = subprocess.run([*command, *map(str, args)], stdout=handle, stderr=subprocess.STDOUT,
                                     timeout=240, cwd=ROOT)
        if process.returncode and not allow_failure:
            raise ValueError("application_stage_failed:"+log)
        return process.returncode

    common = ["--character", a.character, "--character-id", a.character_id,
        "--shape-registry", a.shape_registry, "--source-bvh", selected["source_path"],
        "--hand-left", a.hand_left, "--hand-right", a.hand_right]
    baseline = a.output_dir/"baseline-qa"
    run("converter/refine_v3_safe_snapshot_qa.py", [*common, "--output-dir", baseline],
        "baseline-qa.log", blender=True, allow_failure=True)
    final_dir = baseline
    plan_dir = None
    if not (baseline/"qa-report.json").is_file():
        planning = a.output_dir/"planning-inputs"
        run("converter/refine_skin_plan_inputs.py", ["--qa-dir", baseline, "--character", a.character,
            "--hand-left", a.hand_left, "--hand-right", a.hand_right, "--output-dir", planning],
            "planning-inputs.log", blender=True)
        plan_dir = a.output_dir/"skin-plan"
        run("scripts/plan_refine_skin_repair.py", ["--inputs", planning/"inputs.npz", "--output-dir", plan_dir],
            "skin-plan.log")
        final_dir = a.output_dir/"final-qa"
        run("converter/refine_v3_safe_snapshot_qa.py", [*common, "--skin-plan", plan_dir/"skin-plan.npz",
            "--output-dir", final_dir], "final-qa.log", blender=True)
    if a.depth_safe:
        if plan_dir is None:
            raise ValueError("depth_safe_requires_derived_skin_planning_inputs")
        proposal = a.output_dir/"depth-proposal"
        run("scripts/refine_target_depth.py", ["--inputs", planning/"inputs.npz",
            "--skin-plan", plan_dir/"skin-plan.npz", "--analysis", a.analysis,
            "--candidate-comparison", a.output_dir/"candidate-comparison.json", "--output-dir", proposal],
            "depth-proposal.log")
        realization = a.output_dir/"depth-realization"
        realization_status = run("converter/realize_pose_plan.py", ["--proposal", proposal/"pose-plan.npz",
            "--rest-fbx", final_dir/"derived-neutral.fbx", "--hand-left", a.hand_left,
            "--hand-right", a.hand_right, "--output-dir", realization], "depth-realization.log", blender=True,
            allow_failure=True)
        if realization_status:
            # A proposal is not an artifact. One smaller proposal may be tried;
            # every depth, angle, serialization and mesh gate still applies.
            proposal = a.output_dir/"depth-proposal-smaller"
            run("scripts/refine_target_depth.py", ["--inputs", planning/"inputs.npz",
                "--skin-plan", plan_dir/"skin-plan.npz", "--analysis", a.analysis,
                "--candidate-comparison", a.output_dir/"candidate-comparison.json",
                "--max-step", ".25", "--uniform-only", "--output-dir", proposal], "depth-proposal-smaller.log")
            realization = a.output_dir/"depth-realization-smaller"
            run("converter/realize_pose_plan.py", ["--proposal", proposal/"pose-plan.npz",
                "--rest-fbx", final_dir/"derived-neutral.fbx", "--hand-left", a.hand_left,
                "--hand-right", a.hand_right, "--output-dir", realization], "depth-realization-smaller.log", blender=True)
        depth_plan = a.output_dir/"depth-plan"
        run("scripts/plan_realized_pose_skin.py", ["--inputs", planning/"inputs.npz",
            "--skin-plan", proposal/"skin-plan.npz", "--realized-pose", realization/"realized-pose.npz",
            "--output-dir", depth_plan], "depth-plan.log")
        final_dir = a.output_dir/"depth-final-qa"
        run("converter/refine_v3_safe_snapshot_qa.py", [*common,
            "--skin-plan", depth_plan/"skin-plan.npz", "--pose-plan", depth_plan/"pose-plan.npz",
            "--pose-rig", realization/"unapproved-body-rig.fbx", "--output-dir", final_dir],
            "depth-final-qa.log", blender=True)
    report = json.loads((final_dir/"qa-report.json").read_text())
    artifact = final_dir/"final.fbx"
    if report["neutral_qa_source"] or report["source_sha256"] != selected["source_sha256"]:
        raise ValueError("application_source_mismatch")
    receipt = issue_receipt(artifact.read_bytes(), source_sha256=selected["source_sha256"],
        context_sha256=report["receipt"]["context_sha256"], profile_sha256=report["profile_sha256"],
        revision=report["receipt"]["revision"], scope="full", mesh_audit=report["mesh_audit"],
        integrity_audit=report["artifact_integrity"])
    if (receipt.digest != report["receipt_sha256"] or not verify_receipt(receipt, artifact.read_bytes(),
            revision=receipt.revision, context_sha256=receipt.context_sha256, profile_sha256=receipt.profile_sha256)):
        raise ValueError("application_receipt_mismatch")
    result = dict(schema="actual-rough-apply-v1", applied=True, selected=selected,
        artifact=str(artifact), artifact_sha256=sha(artifact), receipt_sha256=receipt.digest,
        profile_sha256=receipt.profile_sha256, mesh_status="mesh_validated",
        analysis_sha256=sha(a.analysis), image_sha256=record["image_sha256"],
        authority="existing-full-body-refine-contract", regional_observation_authority_promoted=False,
        provider_requests=0, production_registered=False, deployed=False)
    result["depth_safe"] = bool(a.depth_safe)
    if a.depth_safe:
        checks = report.get("pose_safety") or {}
        if checks.get("final_body", {}).get("status") != "pose_validated":
            raise ValueError("final_depth_safety_missing")
        result["pose_safety"] = checks
    (a.output_dir/"applied.json").write_text(json.dumps(result, indent=2))
    print("APPLIED", artifact, result["artifact_sha256"], flush=True)


if __name__ == "__main__":
    main()
