"""V3.2.3 promoted runtime stronger swing selector."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

import bpy
from mathutils import Matrix


ROOT = Path(__file__).resolve().parent
PARENT_PATH = ROOT / "ankle_swing.py"
POLICY_PATH = ROOT / "ankle_clearance_policy.json"
_SIDES = ("L", "R")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_v322():
    from . import ankle_swing
    return ankle_swing


def _number(value, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise RuntimeError(f"{label} must be finite")
    return result


def load_policy(path: str | os.PathLike[str] = POLICY_PATH) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    policy = json.loads(raw)
    if policy.get("schema_version") != 1 or policy.get("status") != "PROMOTED_RUNTIME":
        raise RuntimeError("invalid promoted V3.2.3 policy")
    parent_hash = _sha256(PARENT_PATH)
    if parent_hash != policy.get("parent_module_sha256"):
        raise RuntimeError(
            f"V3.2.2 parent hash mismatch: expected {policy.get('parent_module_sha256')}, "
            f"got {parent_hash}"
        )
    candidates = [_number(value, "candidate_mu") for value in policy["candidate_mu"]]
    if candidates != sorted(set(candidates)) or candidates[0] != 0.0:
        raise RuntimeError("candidate_mu must be sorted, unique, and start at zero")
    if any(value < 0.0 or value > 1.5 for value in candidates):
        raise RuntimeError("candidate_mu must stay in the reviewed [0, 1.5] QA range")
    if 1.0 not in candidates:
        raise RuntimeError("candidate_mu must contain the V3.2.2 unit reference")
    policy["candidate_mu"] = candidates
    for group in ("activation", "eligibility", "measurement", "stronger_swing"):
        policy[group] = {
            key: _number(value, f"{group}.{key}")
            for key, value in policy[group].items()
        }
    return policy, hashlib.sha256(raw).hexdigest()


def _gain_rotation(rotation: Matrix, gain: float) -> Matrix:
    """Scale the minimal swing beyond one without adding a new axis."""
    if gain <= 0.0:
        return Matrix.Identity(4)
    axis, angle = rotation.to_quaternion().normalized().to_axis_angle()
    scaled = angle * gain
    if scaled >= math.radians(60.0):
        raise RuntimeError("stronger ankle swing exceeded 60 degree hard guard")
    return Matrix.Rotation(scaled, 4, axis)


def _length_diagnostics(v322, rt, bone_map, src_arm, dst_arm, src_table,
                        dst_table, side: str, mirror: bool) -> dict:
    nodes = (f"upleg.{side}", f"leg.{side}", f"foot.{side}", f"toe.{side}")
    source = [
        v322._source_point(
            rt, bone_map, src_arm, src_table, canonical, rest=True, mirror=mirror
        ) for canonical in nodes
    ]
    target = [
        v322._target_point(rt, dst_arm, dst_table, canonical, rest=True)
        for canonical in nodes
    ]
    source_lengths = [(source[i + 1] - source[i]).length for i in range(3)]
    target_lengths = [(target[i + 1] - target[i]).length for i in range(3)]

    def ratios(values):
        thigh, shin, foot = values
        total = max(thigh + shin + foot, 1.0e-12)
        return {
            "thigh_chain_ratio": thigh / total,
            "shin_chain_ratio": shin / total,
            "foot_chain_ratio": foot / total,
            "foot_to_shin": foot / max(shin, 1.0e-12),
        }

    source_ratios = ratios(source_lengths)
    target_ratios = ratios(target_lengths)
    return {
        "source_rest_lengths": dict(zip(("thigh", "shin", "foot"), source_lengths)),
        "target_rest_lengths": dict(zip(("thigh", "shin", "foot"), target_lengths)),
        "source_ratios": source_ratios,
        "target_ratios": target_ratios,
        "target_over_source_foot_to_shin": (
            target_ratios["foot_to_shin"] / max(source_ratios["foot_to_shin"], 1.0e-12)
        ),
        "used_as_rotation_formula": False,
        "interpretation": "proportion diagnostic only; actual mesh selects gain",
    }


def _surface_from_row(v322, row: dict, side: str):
    return v322.SurfaceMetrics(**row["surface"][side])


def _meets_rescue(surface, policy: dict) -> bool:
    gate = policy["stronger_swing"]
    return (
        surface.measurable
        and surface.clearance_p01_leg_ratio >= gate["rescue_clearance_p01_leg_ratio"]
        and surface.area_ratio_p01 >= gate["rescue_area_ratio_p01"]
        and surface.edge_ratio_min >= gate["rescue_edge_ratio_min"]
        and surface.log_edge_strain_p99 <= gate["rescue_log_edge_strain_p99"]
        and surface.dihedral_change_p99_deg <= gate["rescue_dihedral_change_p99_deg"]
        and surface.new_sharp_fold_count == 0
    )


def _rescue_score(surface, policy: dict) -> float:
    gate = policy["stronger_swing"]
    if not surface.measurable:
        return 1.0e9
    return (
        max(0.0, gate["rescue_clearance_p01_leg_ratio"] - surface.clearance_p01_leg_ratio)
        / gate["rescue_clearance_p01_leg_ratio"]
        + max(0.0, gate["rescue_area_ratio_p01"] - surface.area_ratio_p01)
        / gate["rescue_area_ratio_p01"]
        + max(0.0, gate["rescue_edge_ratio_min"] - surface.edge_ratio_min)
        / gate["rescue_edge_ratio_min"]
        + max(0.0, surface.log_edge_strain_p99 - gate["rescue_log_edge_strain_p99"])
        / gate["rescue_log_edge_strain_p99"]
        + max(0.0, surface.dihedral_change_p99_deg
              - gate["rescue_dihedral_change_p99_deg"])
        / gate["rescue_dihedral_change_p99_deg"]
        + float(surface.new_sharp_fold_count)
    )


class ClearanceReport:
    def __init__(self, parent, selection):
        self.parent = parent
        self.ankle_clearance_selection = selection

    def __getattr__(self, name):
        return getattr(self.parent, name)

    def as_dict(self):
        payload = self.parent.as_dict()
        payload["ankle_clearance_selection"] = self.ankle_clearance_selection
        return payload


def convert_safe(
    *, bvh_path: str, character_fbx: str, out_path: str,
    frame: int = 0, mirror: bool = False, output_mode: str = "rigged_rest",
    apply_root_translation: bool = False, embed_textures: bool = True,
    src_profile: str | None = None, dst_profile: str | None = None,
    policy_path: str | os.PathLike[str] = POLICY_PATH,
):
    started = time.time()
    policy, policy_hash = load_policy(policy_path)
    v322 = _load_v322()
    rt, bone_map = v322._load_parent()
    if output_mode not in {"rigged_rest", "static_mesh"}:
        raise ValueError("V3.2.3 supports rigged_rest/static_mesh only")
    rt.reset_scene()
    dst_arm, meshes = rt.import_character(character_fbx)
    src_arm = rt.import_bvh(bvh_path, frame=frame)
    resolved_src = src_profile or bone_map.resolve_profile(
        [bone.name for bone in src_arm.data.bones]
    )
    resolved_dst = dst_profile or bone_map.resolve_profile(
        [bone.name for bone in dst_arm.data.bones]
    )
    palm_mu = float(policy["parent_palm_roll_mu"])
    zero = {side: 0.0 for side in _SIDES}
    rows = []
    activation = {side: {"active": False, "reasons": ["MESH_COUNT_NOT_ONE"]}
                  for side in _SIDES}
    length_diagnostics = {}
    selected_mu = dict(zero)
    mesh = meshes[0] if len(meshes) == 1 else None
    rest_positions = v322._positions(mesh, dst_arm, evaluated=False) if mesh else None

    original_scaler = v322._scaled_rotation
    v322._scaled_rotation = _gain_rotation
    try:
        if mesh is None:
            report = v322._run_parent(
                rt, src_arm, dst_arm, src_profile=resolved_src,
                dst_profile=resolved_dst, mirror=mirror,
                apply_root_translation=apply_root_translation,
                palm_mu=palm_mu, output_mode=output_mode, frame=frame,
            )
        else:
            baseline, report = v322._candidate_row(
                rt, bone_map, src_arm, dst_arm, mesh, rest_positions, policy,
                resolved_src, resolved_dst, mirror, output_mode, frame,
                apply_root_translation, palm_mu, zero,
            )
            rows.append(baseline)
            baseline_surfaces = {
                side: _surface_from_row(v322, baseline, side) for side in _SIDES
            }
            for side in _SIDES:
                geometry = v322.SwingGeometry(**baseline["geometry"][side])
                active, reasons = v322._activation(
                    geometry, baseline_surfaces[side], policy
                )
                activation[side] = {"active": active, "reasons": reasons}
            src_table = bone_map.PROFILES[report.src_profile]
            dst_table = bone_map.PROFILES[report.dst_profile]
            length_diagnostics = {
                side: _length_diagnostics(
                    v322, rt, bone_map, src_arm, dst_arm, src_table, dst_table,
                    side, mirror,
                ) for side in _SIDES
            }
            ladders = [
                policy["candidate_mu"] if activation[side]["active"] else [0.0]
                for side in _SIDES
            ]
            for left, right in itertools.product(*ladders):
                gains = {"L": float(left), "R": float(right)}
                if gains == zero:
                    continue
                row, _ = v322._candidate_row(
                    rt, bone_map, src_arm, dst_arm, mesh, rest_positions, policy,
                    resolved_src, resolved_dst, mirror, output_mode, frame,
                    apply_root_translation, palm_mu, gains,
                    baseline_surfaces=baseline_surfaces, activation=activation,
                )
                rows.append(row)

            unit_reference = {}
            for side in _SIDES:
                if activation[side]["active"]:
                    wanted = {"L": 0.0, "R": 0.0}
                    wanted[side] = 1.0
                    unit_reference[side] = next(
                        row for row in rows if row["mu"] == wanted
                    )
            max_overshoot = policy["stronger_swing"]["maximum_target_overshoot_deg"]
            for row in rows:
                if row is baseline:
                    row["rescue_met"] = False
                    row["rescue_score"] = sum(
                        _rescue_score(_surface_from_row(v322, row, side), policy)
                        for side in _SIDES if activation[side]["active"]
                    )
                    continue
                extra_reasons = []
                rescue_by_side = {}
                score = 0.0
                for side in _SIDES:
                    gain = row["mu"][side]
                    if gain <= 0.0 or not activation[side]["active"]:
                        continue
                    surface = _surface_from_row(v322, row, side)
                    rescue_by_side[side] = _meets_rescue(surface, policy)
                    score += _rescue_score(surface, policy)
                    result_error = row["applied"][side]["result_bend_error_deg"]
                    if gain > 1.0:
                        if result_error > max_overshoot:
                            extra_reasons.append(f"{side}:TARGET_OVERSHOOT_HARD_GUARD")
                        unit_surface = _surface_from_row(v322, unit_reference[side], side)
                        continued, reasons, improvements = v322._surface_nonregression(
                            surface, unit_surface, policy
                        )
                        row.setdefault("improvement_beyond_mu1", {})[side] = improvements
                        if not continued:
                            extra_reasons.extend(
                                f"{side}:NO_SAFE_IMPROVEMENT_BEYOND_MU1:{reason}"
                                for reason in reasons
                            )
                if extra_reasons:
                    row["eligible"] = False
                    row.setdefault("eligibility_reasons", []).extend(extra_reasons)
                row["rescue_met_by_side"] = rescue_by_side
                row["rescue_met"] = bool(rescue_by_side) and all(rescue_by_side.values())
                row["rescue_score"] = score

            eligible = [row for row in rows if row.get("eligible")]
            rescued = [row for row in eligible if row.get("rescue_met")]
            pool = rescued or eligible
            selected = min(
                pool,
                key=lambda row: (
                    0 if row.get("rescue_met") else 1,
                    row.get("rescue_score", 1.0e9),
                    sum(row["mu"].values()),
                    row["mu"]["L"], row["mu"]["R"],
                ),
            )
            selected_mu = selected["mu"]
            report = v322._run_parent(
                rt, src_arm, dst_arm, src_profile=resolved_src,
                dst_profile=resolved_dst, mirror=mirror,
                apply_root_translation=apply_root_translation,
                palm_mu=palm_mu, output_mode=output_mode, frame=frame,
            )
            if report.ok:
                src_table = bone_map.PROFILES[report.src_profile]
                dst_table = bone_map.PROFILES[report.dst_profile]
                for side in _SIDES:
                    geometry, context = v322._swing_geometry(
                        rt, bone_map, src_arm, dst_arm, src_table, dst_table,
                        side, mirror,
                    )
                    if selected_mu[side] > 0.0:
                        if not geometry.measurable or context is None:
                            raise RuntimeError("selected V3.2.3 swing became unmeasurable")
                        v322._apply_side_swing(
                            rt, dst_arm, dst_table, side, context, selected_mu[side]
                        )
    finally:
        v322._scaled_rotation = original_scaler

    if not report.ok:
        return report
    report.pose_fidelity_rmse = rt.pose_fidelity_rmse(
        src_arm, dst_arm, report.src_profile, report.dst_profile, mirror=mirror
    )
    report.skeleton_baseline_rmse = rt.skeleton_baseline_rmse(
        src_arm, dst_arm, report.src_profile, report.dst_profile
    )
    report.pose_fidelity_delta = report.pose_fidelity_rmse - report.skeleton_baseline_rmse
    selection = {
        "selector": "V3.2.3_ANKLE_STRONGER_SWING_ACTUAL_MESH_CLEARANCE",
        "status": policy["status"],
        "policy_sha256": policy_hash,
        "parent_variant": policy["parent_variant"],
        "parent_module_sha256": policy["parent_module_sha256"],
        "activation": activation,
        "length_diagnostics": length_diagnostics,
        "candidates": rows,
        "selected_mu": selected_mu,
        "fallback_to_exact_v321": selected_mu == zero,
        "stronger_than_v322": any(value > 1.0 for value in selected_mu.values()),
        "changed_bones_only": ["foot.L", "toe.L", "foot.R", "toe.R"],
        "availability_contract": "unmeasurable_or_unsafe_exports_exact_v321",
        "final_application_count": 1,
    }
    selected_positions = v322._positions(mesh, dst_arm, evaluated=True) if mesh else None
    bpy.data.objects.remove(src_arm, do_unlink=True)
    rt.apply_output_mode(dst_arm, meshes, output_mode)
    if mesh and selected_positions is not None:
        baked = v322._positions(mesh, dst_arm, evaluated=False)
        selection["post_bake_max_vertex_error"] = max(
            (a - b).length for a, b in zip(selected_positions, baked)
        )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    rt.export_fbx(out_path, embed_textures=embed_textures, bake_anim=False)
    report.warnings.append(f"ankle_clearance_qa_elapsed_sec={time.time() - started:.2f}")
    return ClearanceReport(report, selection)


def math_self_test() -> dict:
    rotation = Matrix.Rotation(math.radians(20.0), 4, "X")
    identity = _gain_rotation(rotation, 0.0)
    unit = _gain_rotation(rotation, 1.0)
    strong = _gain_rotation(rotation, 1.5)

    def angle(matrix):
        return math.degrees(matrix.to_quaternion().angle)

    result = {
        "mu0_deg": angle(identity),
        "mu1_deg": angle(unit),
        "mu1_5_deg": angle(strong),
        "axis_dot": abs(unit.to_quaternion().axis.dot(strong.to_quaternion().axis)),
    }
    # mathutils float32 equality tolerance; not a product acceptance margin.
    angle_tolerance_deg = 1.0e-4
    axis_tolerance = 5.0e-6
    result["angle_tolerance_deg"] = angle_tolerance_deg
    result["axis_tolerance"] = axis_tolerance
    result["passed"] = (
        result["mu0_deg"] == 0.0
        and abs(result["mu1_deg"] - 20.0) <= angle_tolerance_deg
        and abs(result["mu1_5_deg"] - 30.0) <= angle_tolerance_deg
        and abs(result["axis_dot"] - 1.0) <= axis_tolerance
    )
    if not result["passed"]:
        raise RuntimeError(f"V3.2.3 stronger swing self-test failed: {result}")
    return result
