"""Legs first, anatomy-attached arms second, then one atomic final decision.

Callbacks edit pose matrices only. Skin topology repair is an authored asset
operation, not an implicit side effect of this runtime. Blender is lazy-loaded.
"""
from dataclasses import dataclass
import numpy as np
from . import VERSION


@dataclass(frozen=True)
class Stage:
    kind: str
    sides: tuple
    apply: object

    def roots(self):
        if self.kind not in ("legs", "arms") or not self.sides or len(set(self.sides)) != len(self.sides):
            raise ValueError("explicit stage and unique anatomical sides required")
        if not set(self.sides) <= {"left", "right"}:
            raise ValueError("screen-side labels cannot select joints")
        return {"mixamorig:" + s.title() + ("UpLeg" if self.kind == "legs" else "Arm") for s in self.sides}


def snapshot(arm):
    return {p.name: np.array(arm.matrix_world @ p.matrix) for p in arm.pose.bones}


def moving_bones(arm, stage):
    roots = stage.roots()
    if not roots <= {p.name for p in arm.pose.bones}:
        raise ValueError("stage chain missing from destination")
    return {p.name for p in arm.pose.bones if p.name in roots or
            any(parent.name in roots for parent in p.parent_recursive)}


def skin_nonregression(before, after, tolerance_mm=2.):
    """Require complete measurements; missing skin evidence cannot pass."""
    for value in (before, after):
        if not isinstance(value, dict) or not isinstance(value.get("collisions"), list) or not value.get("regions"):
            return {"accepted": False, "reasons": ["skin_measurement_missing"]}
        if value.get("region_closure_failures"):
            return {"accepted": False, "reasons": ["skin_region_not_closed"]}
        for c in value["collisions"]:
            if not np.isfinite(c.get("max_depth_mm", float("nan"))) or c["max_depth_mm"] < 0:
                return {"accepted": False, "reasons": ["invalid_skin_measurement"]}
    baseline = {tuple(sorted(c["regions"])): c["max_depth_mm"] for c in before["collisions"]}
    new = [c for c in after["collisions"] if c["max_depth_mm"] >
           baseline.get(tuple(sorted(c["regions"])), 0.) + tolerance_mm]
    return {"accepted": not new, "reasons": ["new_skin_penetration"] if new else [],
            "new_collisions": new, "scope": "measured_skin_nonregression_not_pose_similarity"}


def run_refine_v4(arm, meshes, *, leg_stage=None, arm_stage=None, final_check=None):
    """On any rejected stage/check restore all initial pose bases, including fingers.

    final_check(arm, meshes, receipts) must independently remeasure skin and any
    required hand contact after both stages. No quality score substitutes for it.
    Mesh changes are outside the callback contract and rejected by the runner.
    """
    import bpy
    stages = [s for s in (leg_stage, arm_stage) if s is not None]
    if leg_stage and leg_stage.kind != "legs" or arm_stage and arm_stage.kind != "arms":
        raise ValueError("invalid leg/arm stage order")
    for stage in stages:
        stage.roots()
    result = {"version": VERSION, "accepted": False, "status": "not_applied",
              "order": [s.kind for s in stages], "stages": [], "reasons": [],
              "score_status": "not_scored", "release_allowed": False}
    if not stages:
        result.update(status="unchanged", reasons=["no_stage_requested"])
        return result
    if not callable(final_check):
        result["reasons"] = ["final_geometry_and_contact_check_required"]
        return result
    if arm.animation_data or any(p.constraints for p in arm.pose.bones):
        result["reasons"] = ["bake_animation_and_constraints_first"]
        return result
    before = snapshot(arm)
    bases = {p.name: p.matrix_basis.copy() for p in arm.pose.bones}
    object_matrix = arm.matrix_world.copy()
    lengths = {p.name: float((arm.matrix_world @ p.tail - arm.matrix_world @ p.head).length)
               for p in arm.pose.bones}
    # Keep pristine data for exact recovery if a callback violates its pose-only contract.
    saved_meshes = [(m, m.data, m.data.copy(), m.matrix_world.copy()) for m in meshes]
    try:
        for stage in stages:
            prior = snapshot(arm)
            allowed = moving_bones(arm, stage)
            receipt = stage.apply(arm, meshes)
            if not isinstance(receipt, dict):
                raise ValueError("stage_receipt_missing")
            result["stages"].append({"kind": stage.kind, "sides": list(stage.sides), "receipt": receipt})
            if receipt.get("accepted") is not True:
                raise ValueError(stage.kind + ":stage_rejected")
            current = snapshot(arm)
            if set(current) != set(prior):
                raise ValueError("joint_hierarchy_changed")
            error = max((float(np.max(abs(current[n] - v))) for n, v in prior.items() if n not in allowed), default=0.)
            result["stages"][-1]["protected_matrix_delta"] = error
            result["stages"][-1]["changed_joints"] = [n for n,v in prior.items()
                if float(np.max(abs(current[n]-v))) > 1e-6]
            if not np.isfinite(error) or error > 2e-5:
                raise ValueError(stage.kind + ":protected_joint_changed")
        after = snapshot(arm)
        if not all(np.isfinite(v).all() for v in after.values()):
            raise ValueError("nonfinite_pose")
        length_error = max(abs(float((arm.matrix_world @ p.tail - arm.matrix_world @ p.head).length) - lengths[p.name])
                           for p in arm.pose.bones)
        result["bone_length_error_m"] = length_error
        if length_error > 2e-5:
            raise ValueError("bone_length_changed")
        for m, original, saved, matrix in saved_meshes:
            if m.data != original or len(m.data.vertices) != len(saved.vertices) or len(m.data.polygons) != len(saved.polygons):
                raise ValueError("mesh_edit_outside_pose_contract")
            if any((a.co - b.co).length > 1e-9 for a, b in zip(m.data.vertices, saved.vertices)):
                raise ValueError("mesh_edit_outside_pose_contract")
            if np.max(abs(np.array(m.matrix_world) - np.array(matrix))) > 1e-9:
                raise ValueError("mesh_object_transform_changed")
        final = final_check(arm, meshes, result["stages"])
        result["final_check"] = final
        if not isinstance(final, dict) or final.get("accepted") is not True:
            raise ValueError("final_geometry_or_contact_rejected")
        if max(float(np.max(abs(snapshot(arm)[n]-v))) for n,v in after.items()) > 1e-9:
            raise ValueError("final_checker_changed_pose")
        changed = max(float(np.max(abs(after[n] - v))) for n, v in before.items()) > 1e-6
        result.update(accepted=True, status="applied" if changed else "unchanged")
    except Exception as exc:
        result["reasons"].append(str(exc))
        arm.matrix_world = object_matrix
        for p in arm.pose.bones:
            p.matrix_basis = bases[p.name]
        for m, original, saved, matrix in saved_meshes:
            m.data = saved
            m.matrix_world = matrix
        bpy.context.view_layer.update()
        result["rollback_matrix_delta"] = max(float(np.max(abs(snapshot(arm)[n] - v))) for n, v in before.items())
    finally:
        for _, _, saved, _ in saved_meshes:
            if saved.users == 0:
                bpy.data.meshes.remove(saved)
    return result
