"""Immutable, private-path-free lineage for evaluation artifacts (not product state)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .util import hash_json, sha256_file

SCHEMA = "evaluation-extension-v1"
STAGES = ("selected_bvh", "body_fbx", "hands_fbx", "individual_fbx", "scene_fbx", "download", "native_save")
SELECTION_FIELDS = ("unit_id", "cut_id", "gt_person_id", "candidate_id", "arm", "selection_revision",
                    "body_revision", "hand_revision", "scope", "camera_sha256", "source_sha256",
                    "profile_sha256")
BINDINGS = ("model_sha256", "db_sha256", "asset_sha256", "controller_sha256", "converter_sha256",
            "blender_build", "bone_map_sha256", "hand_library_sha256")


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def selection_errors(selection):
    errors = ["missing_selection:" + key for key in SELECTION_FIELDS if not selection.get(key)]
    for key in ("camera_sha256", "source_sha256", "profile_sha256"):
        if not is_sha(selection.get(key)):
            errors.append("invalid_selection_hash:" + key)
    if type(selection.get("person_index")) is not int or selection["person_index"] < 0:
        errors.append("explicit_person_index_required")
    for key in BINDINGS:
        value = selection.get("bindings", {}).get(key)
        if not value or (key.endswith("sha256") and not is_sha(value)):
            errors.append("missing_binding:" + key)
    backend = selection.get("backend", {})
    if not backend.get("requested") or not backend.get("actual"):
        errors.append("backend_unbound")
    elif backend["requested"] != backend["actual"]:
        errors.append("backend_fallback")
    return errors


def artifact_node(path, *, artifact_id, stage, parents, selections, execution_kind="fixture"):
    """Hash bytes locally; never serialize source paths or infer permanent person IDs."""
    path = Path(path)
    if stage not in STAGES or not artifact_id or not selections:
        raise ValueError("artifact stage, ID and explicit selections are required")
    return {"schema": SCHEMA, "artifact_id": artifact_id, "stage": stage,
            "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "parents": list(parents), "selection_sha256s": [hash_json(s) for s in selections],
            "execution_kind": execution_kind}


def validate_lineage(selections, nodes, *, require_native=True):
    """Validate the exact selected BVH -> body -> hands -> file -> saved bytes DAG.

    Missing stages stay unknown. Contradictory hashes, revision bindings, parents
    or person mappings fail; regenerated FBX bytes are permitted until download.
    """
    failures, unknown = [], []
    if not selections:
        unknown.append("selections_missing")
    selection_keys = [s.get("gt_person_id") for s in selections]
    if len(set(selection_keys)) != len(selection_keys):
        failures.append("duplicate_permanent_person")
    indices = [s.get("person_index") for s in selections]
    if len(set(indices)) != len(indices):
        failures.append("duplicate_request_person_index")
    for selection in selections:
        unknown.extend(selection_errors(selection))
    expected = {hash_json(s): s for s in selections}
    ids = [n.get("artifact_id") for n in nodes]
    if len(set(ids)) != len(ids):
        failures.append("duplicate_artifact_id")
    by_id = {n.get("artifact_id"): n for n in nodes}
    counts = Counter()
    ranks = {stage: i for i, stage in enumerate(STAGES)}
    for node in nodes:
        stage, ident = node.get("stage"), node.get("artifact_id")
        if stage not in STAGES or not ident:
            failures.append("invalid_artifact_stage_or_id")
            continue
        counts[stage] += 1
        if not is_sha(node.get("sha256")) or type(node.get("bytes")) is not int or node["bytes"] <= 0:
            failures.append("invalid_artifact_bytes:" + str(ident))
        keys = node.get("selection_sha256s", [])
        if not keys or len(set(keys)) != len(keys) or any(k not in expected for k in keys):
            failures.append("selection_binding_mismatch:" + str(ident))
        if stage in STAGES[:4] and len(keys) != 1:
            failures.append("individual_selection_count:" + str(ident))
        if stage in ("scene_fbx", "download", "native_save") and set(keys) != set(expected):
            failures.append("scene_person_coverage:" + str(ident))
        parents = node.get("parents", [])
        if len(set(parents)) != len(parents):
            failures.append("duplicate_parent:" + str(ident))
        if stage == "selected_bvh":
            if parents or (len(keys) == 1 and keys[0] in expected
                           and node.get("sha256") != expected[keys[0]].get("source_sha256")):
                failures.append("selected_bvh_mismatch:" + str(ident))
            continue
        if not parents:
            unknown.append("parent_missing:" + str(ident))
        parent_keys = set()
        for parent_id in parents:
            parent = by_id.get(parent_id)
            if parent is None:
                failures.append("dangling_parent:" + str(ident))
                continue
            parent_stage = parent.get("stage")
            allowed = {"body_fbx": {"selected_bvh"}, "hands_fbx": {"body_fbx"},
                       "individual_fbx": {"hands_fbx"}, "scene_fbx": {"individual_fbx"},
                       "download": {"individual_fbx", "scene_fbx"}, "native_save": {"download"}}
            if parent_stage not in allowed[stage] or ranks.get(parent_stage, 99) >= ranks[stage]:
                failures.append("invalid_parent_stage:" + str(ident))
            parent_keys.update(parent.get("selection_sha256s", []))
            if stage in ("download", "native_save") and (
                    node.get("sha256") != parent.get("sha256") or node.get("bytes") != parent.get("bytes")):
                failures.append("delivery_bytes_mismatch:" + str(ident))
        if parents and parent_keys != set(keys):
            failures.append("parent_selection_mismatch:" + str(ident))
        if stage != "scene_fbx" and len(parents) != 1:
            failures.append("parent_count:" + str(ident))
    for stage in STAGES[:4]:
        for key in expected:
            matching = [n for n in nodes if n.get("stage") == stage and n.get("selection_sha256s") == [key]]
            if not matching:
                unknown.append("stage_missing:" + stage)
            elif len(matching) > 1:
                failures.append("ambiguous_stage:" + stage)
    for stage in (["scene_fbx"] if len(selections) > 1 else []) + ["download"] + (["native_save"] if require_native else []):
        if counts[stage] == 0:
            unknown.append("stage_missing:" + stage)
        elif counts[stage] > 1:
            failures.append("ambiguous_stage:" + stage)
    return {"state": "fail" if failures else "unknown" if unknown else "pass",
            "failures": sorted(set(failures)), "unknown_reasons": sorted(set(unknown)),
            "person_count": len(selections), "artifact_count": len(nodes)}
