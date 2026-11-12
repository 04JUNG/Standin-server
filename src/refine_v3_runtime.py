"""Default-off shadow integration. Experiments never replace F25/output bytes."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import os
import time

from .refine_v3_contracts import canonical_sha256, sign_binding, verify_binding
from .refine_projection import query_sha256

VERSION = "refine-shadow-v1"


def mode():
    # No application mode is exposed until actual profile/mesh promotion.
    return "shadow" if os.getenv("REFINE_V3_MODE", "off").lower() == "shadow" else "off"


def _key():
    return os.getenv("REFINE_V3_BINDING_SECRET", "").encode("utf-8")


def capabilities():
    return {"version": VERSION, "mode": mode(), "regional_apply": False,
            "target_apply": False, "certified_profiles": 0,
            "binding_available": len(_key()) >= 32,
            "mesh_validation_required": True}


def image_digest(image):
    """Hash exact pipeline input representation; never retain or log pixels."""
    if isinstance(image, (str, os.PathLike)):
        with open(image, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    import numpy as np
    array = np.ascontiguousarray(np.asarray(image))
    return hashlib.sha256(canonical_sha256({"shape": array.shape, "dtype": str(array.dtype)}).encode()
                          + array.tobytes()).hexdigest()


def issue_for_slot(desc, slot, report, camera_index, *, image_sha256):
    """Issue only from server-owned slot/index inputs, never request certificates."""
    if mode() == "off":
        return None
    trace = {"version": VERSION, "mode": "shadow", "applied": False,
             "mesh_status": "mesh_unverified", "reason": "binding_key_unavailable"}
    desc.quality_trace["refine_v3"] = trace
    if len(_key()) < 32 or slot.skeleton is None or camera_index is None:
        return None
    from .refine_regional import camera_from_verified_hit, build_regional_binding
    points = slot.skeleton.keypoints.tolist()
    observations = getattr(slot.skeleton, "joint_provenance", None) or ()
    provenance = [r.to_serializable() if hasattr(r, "to_serializable") else dict(r) for r in observations]
    if len(provenance) != 17:
        trace["reason"] = "joint_provenance_unavailable"
        return None
    # Canonicalize only this exact stable slot's identifier. A neighboring
    # person's record keeps its different identifier and is rejected by policy.
    provenance = [{**r, "person_id": str(slot.slot_id)}
                  if r.get("person_id") == slot.slot_id else r for r in provenance]
    # Structural checks do not certify visibility or anatomical labels. Missing
    # independent side verification remains unknown and policy must abstain.
    ownership = {"person_id": str(slot.slot_id), "slot_origin": slot.slot_origin,
                 "ownership_valid": bool(slot.slot_origin == "vlm" and not slot.reasons
                                         and slot.state in ("valid", "partial")),
                 "side_labels_known": desc.quality_trace.get("side_labels_verified") is True}
    candidates = []
    for row in report.get("selected", [])[:5]:
        geometry = (row.get("evidence") or {}).get("geometry") or {}
        index = geometry.get("camera_index")
        if type(index) is not int or not 0 <= index < len(camera_index.cameras):
            continue
        source_sha = camera_index.hashes.get(row.get("pose_id"))
        if not source_sha:
            continue
        try:
            camera = camera_from_verified_hit(candidate=row, geometry=geometry,
                indexed_camera=asdict(camera_index.cameras[index]), source_bvh_sha256=source_sha,
                keypoints=points)
            decision = build_regional_binding(pose_id=row["pose_id"], source_bvh_sha256=source_sha,
                keypoints=points, provenance=provenance, camera=camera,
                ownership=ownership, enabled=True)
        except (ValueError, TypeError, KeyError):
            continue
        candidates.append({"pose_id": row["pose_id"], "source_bvh_sha256": source_sha,
                           "reason": decision.reason, "binding": decision.binding})
    payload = {"version": VERSION, "person_id": str(slot.slot_id),
               "image_sha256": image_sha256, "query_sha256": query_sha256(points),
               "provenance": provenance, "ownership": ownership, "candidates": candidates,
               "refine_evidence": getattr(desc, "refine_evidence", None),
               "evidence_sha256": canonical_sha256(getattr(desc, "refine_evidence", None) or {"status": "absent"})}
    trace.update(reason="shadow_only", candidate_count=len(candidates),
                 eligible_count=sum(c["binding"] is not None for c in candidates))
    return sign_binding(payload, _key())


def shadow_for_request(req, source_path, *, deadline):
    """Return bounded diagnostics only; no candidate bytes cross this boundary."""
    if mode() == "off":
        return None
    diagnostic = {"version": VERSION, "mode": "shadow", "applied": False,
                  "mesh_status": "mesh_unverified", "reason": "binding_unavailable"}
    envelope = getattr(req, "refine_v3", None)
    payload = verify_binding(envelope, _key())
    if payload is None:
        return diagnostic
    if payload.get("version") != VERSION or payload.get("query_sha256") != query_sha256(req.keypoints):
        diagnostic["reason"] = "query_mismatch"
        return diagnostic
    with open(source_path, "rb") as source:
        digest = hashlib.sha256(source.read()).hexdigest()
    selected = [c for c in payload.get("candidates", []) if c.get("pose_id") == req.pose_id]
    if len(selected) != 1 or selected[0].get("source_bvh_sha256") != digest:
        diagnostic["reason"] = "source_binding_mismatch"
        return diagnostic
    if not selected[0].get("binding"):
        diagnostic["reason"] = selected[0].get("reason", "regional_policy")
        return diagnostic
    camera = getattr(req, "camera", None)
    bound_camera = selected[0]["binding"].get("camera", {})
    if camera is not None:
        import numpy as np
        if (not np.array_equal(camera.rotation, bound_camera.get("rotation"))
                or (camera.projection or "orthographic") != bound_camera.get("projection")
                or camera.distance_torso != bound_camera.get("distance_torso")
                or camera.labels_swapped is True):
            diagnostic["reason"] = "camera_context_mismatch"
            return diagnostic
    remaining = deadline - time.monotonic() - .1
    if remaining <= 0:
        diagnostic["reason"] = "shared_deadline"
        return diagnostic
    from .refine_v3 import refine_regional_bvh
    try:
        result = refine_regional_bvh(source_path, req.keypoints, selected[0]["binding"],
            payload["provenance"], payload["ownership"], enabled=True,
            deadline_seconds=min(1., remaining))
        # Deliberate allowlist: no BVH, coordinates or raw VLM text in diagnostics.
        values = result.to_dict()
        values.update(result.diagnostics)
        diagnostic.update({key: values[key] for key in
                           ("reason", "refined", "gain", "loss_base", "loss_final", "elapsed_ms") if key in values})
        diagnostic["source_experiment"] = True
        diagnostic["baseline"] = "selected_source_b0"
        diagnostic["compared_to_f25"] = False
    except (ValueError, TypeError, KeyError, OSError, AssertionError):
        diagnostic["reason"] = "shadow_failed"
    return diagnostic
