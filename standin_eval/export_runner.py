"""Explicit opt-in local adapters around existing Blender QA entrypoints.

Constructing a plan or importing evidence never runs Blender. Commands use an
argument vector, no shell; inputs are hash-pinned and output directories must be
fresh. Public results contain IDs/hashes, never command arguments/private paths.
"""
from __future__ import annotations

import math
import subprocess
import time
from pathlib import Path

from .artifact_lineage import is_sha
from .mesh_evidence_adapter import adapt_actual_snapshot, adapt_roundtrip, adapt_scene_roundtrip
from .util import REPO_ROOT, hash_json, read_json, sha256_file

SCRIPTS = {"single": "scripts/verify_generated_fbx_roundtrip.py",
           "scene": "tests/converter/verify_scene_roundtrip.py",
           "snapshot": "converter/refine_v3_safe_snapshot_qa.py"}


def blender_plan(kind, *, blender, output_dir, inputs, options=None):
    """inputs = {role: {path, sha256}}. Snapshot source is never neutralized."""
    if kind not in SCRIPTS:
        raise ValueError("unsupported Blender evaluation kind")
    options = dict(options or {})
    allowed = {"single": {"scope", "character_id", "left_hand", "right_hand", "controls"},
               "scene": set(), "snapshot": {"character_id", "hand_left", "hand_right"}}
    if set(options) - allowed[kind]:
        raise ValueError("unsupported Blender option")
    required = {"single": {"bvh", "character"}, "scene": {"source"},
                "snapshot": {"source_bvh", "character", "shape_registry"}}[kind]
    if not required <= set(inputs):
        raise ValueError("missing Blender input roles")
    if kind == "snapshot" and not options.get("character_id"):
        raise ValueError("snapshot character_id required")
    script = REPO_ROOT / SCRIPTS[kind]
    if not script.is_file():
        return {"kind": kind, "state": "missing_dependency", "reason": "missing_converter_tool"}
    for values in inputs.values():
        for item in values if isinstance(values, list) else [values]:
            path = Path(item["path"])
            if not is_sha(item.get("sha256")) or not path.is_file() or sha256_file(path) != item["sha256"]:
                raise ValueError("Blender input hash mismatch")
    if kind == "scene" and (not isinstance(inputs["source"], list) or len(inputs["source"]) < 2):
        raise ValueError("scene roundtrip tool requires at least two sources")
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("Blender evaluation output must be fresh")
    command = [str(blender), "--background", "--factory-startup", "--python", str(script), "--"]
    for role, values in inputs.items():
        for item in values if isinstance(values, list) else [values]:
            command += ["--" + role.replace("_", "-"), str(Path(item["path"]).resolve())]
    for key, value in options.items():
        if key == "controls":
            for control, number in sorted(value.items()):
                command += ["--control", f"{control}={number}"]
        else:
            command += ["--" + key.replace("_", "-"), str(value)]
    command += ["--output-dir" if kind == "snapshot" else "--out", str(output)]
    plan = {"kind": kind, "state": "planned", "command": command, "output_dir": str(output),
            "script_sha256": sha256_file(script), "inputs": inputs,
            "options": options, "options_sha256": hash_json(options), "execution_kind": "local_blender"}
    return {**plan, "plan_sha256": hash_json(plan)}


def execute_blender(plan, *, expected, timeout_seconds, allow_blender=False):
    """Never called by default CLI/offline fixture evaluation."""
    if not allow_blender:
        return {"state": "unsupported", "reason": "local_blender_execution_not_enabled"}
    if plan.get("state") != "planned":
        return {"state": plan.get("state", "error"), "reason": plan.get("reason", "invalid_plan")}
    if hash_json({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan.get("plan_sha256"):
        raise ValueError("Blender execution plan changed after sealing")
    if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("explicit positive Blender timeout required")
    # Rebuild and compare immediately before execution: changed inputs, script,
    # options or argv cannot silently bypass the frozen plan.
    kind = plan["kind"]
    if sha256_file(REPO_ROOT / SCRIPTS[kind]) != plan.get("script_sha256"):
        raise ValueError("Blender checker changed after plan")
    for values in plan["inputs"].values():
        for item in values if isinstance(values, list) else [values]:
            if sha256_file(item["path"]) != item["sha256"]:
                raise ValueError("Blender input changed after plan")
    output = Path(plan["output_dir"])
    if output.exists():
        raise ValueError("Blender evaluation output must be fresh")
    started = time.monotonic()
    common = {"execution_kind": "local_blender", "checker_sha256": plan["script_sha256"],
              "unit_id": expected.get("unit_id"), "artifact_id": expected.get("artifact_id")}
    try:
        completed = subprocess.run(plan["command"], cwd=REPO_ROOT, capture_output=True,
                                   timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired:
        return {**common, "state": "timeout", "elapsed_seconds": time.monotonic() - started}
    except FileNotFoundError:
        return {**common, "state": "missing_dependency", "reason": "blender_executable_missing"}
    except OSError:
        return {**common, "state": "error", "reason": "blender_execution_error"}
    common["elapsed_seconds"] = time.monotonic() - started
    common["process_returncode"] = completed.returncode
    if completed.returncode:
        failure = output / "qa-failure.json"
        if kind == "snapshot" and failure.is_file():
            return {**common, **adapt_actual_snapshot(read_json(failure), expected=expected, execution_kind="local_blender")}
        return {**common, "state": "error", "reason": "converter_tool_failed"}
    try:
        if kind == "scene":
            return {**common, **adapt_scene_roundtrip(read_json(output / "verification.json"))}
        artifact = output / ("final.fbx" if kind == "snapshot" else "model.fbx")
        digest = sha256_file(artifact)
        if expected.get("artifact_sha256") and digest != expected["artifact_sha256"]:
            return {**common, "state": "fail", "reason": "output_artifact_hash_mismatch"}
        if kind == "snapshot":
            bound = {**expected, "artifact_sha256": digest, "artifact_readback_verified": True}
            report = read_json(output / "qa-report.json")
            if report.get("artifact_size") != artifact.stat().st_size:
                return {**common, "state": "fail", "reason": "output_artifact_size_mismatch"}
            return {**common, **adapt_actual_snapshot(report, expected=bound, execution_kind="local_blender")}
        required_checks = []
        if "shape_registry" in plan["inputs"]:
            required_checks += ["shape_controls_match", "shape_digest_matches"]
        if plan["options"].get("left_hand") or plan["options"].get("right_hand"):
            required_checks.append("hand_body_bones_preserved")
        structural = adapt_roundtrip(read_json(output / "result.json"), artifact_sha256=digest,
                                     scope=expected["scope"], execution_kind="local_blender", required_checks=required_checks)
        return {**common, **structural, **{k: expected.get(k) for k in ("profile_sha256", "scope", "selection_revision")}}
    except (OSError, ValueError, KeyError, TypeError):
        return {**common, "state": "error", "reason": "converter_report_missing_or_malformed"}
